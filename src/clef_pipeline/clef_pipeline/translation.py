"""Translation utilities for query normalization."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

from deep_translator import GoogleTranslator
from tqdm import tqdm


class GoogleTranslateClient:
    """Translate arbitrary text into a target language via Google Translate."""

    API_URL = "https://translation.googleapis.com/language/translate/v2"

    def __init__(self, api_key: str | None = None, target_language: str = "fr"):
        self.api_key = api_key
        self.target_language = target_language
        self._cache: dict[str, str] = {}
        self._public_translator = GoogleTranslator(source="auto", target=target_language)

    def translate(self, text: str) -> str:
        """Translate text to the configured target language."""
        if text in self._cache:
            return self._cache[text]
        if not text:
            return text
        if not self.api_key:
            translated_text = self._public_translator.translate(text)
            self._cache[text] = translated_text
            return translated_text

        payload = urllib.parse.urlencode(
            {"q": text, "target": self.target_language, "format": "text"}
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.API_URL}?key={urllib.parse.quote(self.api_key)}",
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"Google Translate API HTTP {exc.code}: {body or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Google Translate API request failed: {exc}") from exc

        translated_text = (
            data.get("data", {})
            .get("translations", [{}])[0]
            .get("translatedText", text)
        )
        self._cache[text] = translated_text
        return translated_text


def build_google_translate_client(target_language: str = "fr") -> GoogleTranslateClient:
    """Create a translator using API key if available, else no-key mode."""
    return GoogleTranslateClient(
        api_key=os.environ.get("GOOGLE_TRANSLATE_API_KEY"),
        target_language=target_language,
    )


class HunyuanMTTranslator:
    """Batch translator powered by ``tencent/Hunyuan-MT-7B``."""

    def __init__(
        self,
        model_name: str = "tencent/Hunyuan-MT-7B",
        target_language: str = "French",
        max_new_tokens: int = 256,
    ):
        self.model_name = model_name
        self.target_language = target_language
        self.max_new_tokens = max_new_tokens
        self._cache: dict[str, str] = {}
        self._model = None
        self._tokenizer = None
        self._torch = None

    def _ensure_loaded(self):
        if self._model is not None and self._tokenizer is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left"
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        if self._model.config.pad_token_id is None and self._tokenizer.pad_token_id is not None:
            self._model.config.pad_token_id = self._tokenizer.pad_token_id

    def _prompt(self, text: str) -> str:
        return (
            f"Translate the following segment into {self.target_language}, "
            "without additional explanation.\n\n"
            f"{text}"
        )

    def translate(self, text: str | None) -> str:
        source_text = "" if text is None else str(text)
        if not source_text.strip():
            return source_text
        if source_text in self._cache:
            return self._cache[source_text]

        self._ensure_loaded()
        prompt = self._prompt(source_text)
        tokenizer = self._tokenizer
        model = self._model
        torch = self._torch
        assert tokenizer is not None and model is not None and torch is not None

        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )

        decoded = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        if decoded.startswith(prompt):
            translated = decoded[len(prompt) :].strip()
        else:
            translated = decoded
        if not translated:
            translated = source_text
        translated = str(translated)
        self._cache[source_text] = translated
        return translated

    def translate_texts(
        self,
        texts: list[str | None],
        progress_desc: str = "Translating tweets",
        batch_size: int = 32,
    ) -> list[str]:
        normalized_texts = ["" if text is None else str(text) for text in texts]
        outputs = list(normalized_texts)

        if batch_size <= 1:
            return [self.translate(text) for text in tqdm(texts, desc=progress_desc)]

        self._ensure_loaded()
        tokenizer = self._tokenizer
        model = self._model
        torch = self._torch
        assert tokenizer is not None and model is not None and torch is not None

        missing: list[tuple[int, str]] = []
        for idx, text in enumerate(normalized_texts):
            if not text.strip():
                continue
            cached = self._cache.get(text)
            if cached is not None:
                outputs[idx] = cached
                continue
            missing.append((idx, text))

        for start in tqdm(
            range(0, len(missing), batch_size),
            desc=progress_desc,
        ):
            chunk = missing[start : start + batch_size]
            prompts = [self._prompt(text) for _idx, text in chunk]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(model.device)

            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                )

            decoded_batch = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for (original_idx, source_text), prompt, decoded in zip(
                chunk, prompts, decoded_batch, strict=False
            ):
                decoded = decoded.strip()
                if decoded.startswith(prompt):
                    translated = decoded[len(prompt) :].strip()
                else:
                    translated = decoded
                if not translated:
                    translated = source_text
                translated = str(translated)
                self._cache[source_text] = translated
                outputs[original_idx] = translated

        return outputs
