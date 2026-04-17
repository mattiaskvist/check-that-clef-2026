from .interfaces import BaseReranker


class Gemma2BReranker(BaseReranker):
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-gemma"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch

        print(f"Loading Cross-Encoder Reranker ({model_name}) to GPU...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map="auto"
        )
        self.model.eval()
        self.yes_loc = self.tokenizer("Yes", add_special_tokens=False)["input_ids"][0]

    def _get_inputs(self, pairs: list[tuple[str, str]], max_length: int = 1024):
        prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
        sep = "\n"
        prompt_inputs = self.tokenizer(
            prompt, return_tensors=None, add_special_tokens=False
        )["input_ids"]
        sep_inputs = self.tokenizer(sep, return_tensors=None, add_special_tokens=False)[
            "input_ids"
        ]

        inputs = []
        for query, passage in pairs:
            query_inputs = self.tokenizer(
                f"Query A: {query}",
                return_tensors=None,
                add_special_tokens=False,
                max_length=max_length * 3 // 4,
                truncation=True,
            )
            passage_inputs = self.tokenizer(
                f"Passage B: {passage}\nAnswer:",
                return_tensors=None,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )

            # --- Bypass prepare_for_model using manual concatenation ---
            bos = (
                [self.tokenizer.bos_token_id]
                if self.tokenizer.bos_token_id is not None
                else []
            )
            q_ids = bos + query_inputs["input_ids"]
            p_ids = sep_inputs + passage_inputs["input_ids"]

            if len(q_ids) + len(p_ids) > max_length:
                p_ids = p_ids[: max_length - len(q_ids)]

            item = {"input_ids": q_ids + p_ids + sep_inputs + prompt_inputs}
            item["attention_mask"] = [1] * len(item["input_ids"])
            # ----------------------------------------------------------------

            inputs.append(item)

        return self.tokenizer.pad(
            inputs,
            padding=True,
            max_length=max_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors="pt",
        )

    def _last_logit_pool(self, logits, attention_mask):
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return logits[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = logits.shape[0]
            return self.torch.stack(
                [logits[i, sequence_lengths[i]] for i in range(batch_size)], dim=0
            )

    def rerank(
        self, query: str, doc_indices: list[int], corpus: list[str]
    ) -> list[tuple[int, float]]:
        pairs = [[query, corpus[doc_id]] for doc_id in doc_indices]
        inputs = self._get_inputs(pairs).to(self.model.device)

        with self.torch.inference_mode():
            outputs = self.model(**inputs)
            pooled_logits = self._last_logit_pool(
                outputs.logits, inputs["attention_mask"]
            )
            scores = pooled_logits[:, self.yes_loc].cpu().float().tolist()

        results = list(zip(doc_indices, scores))
        results.sort(key=lambda x: x[1], reverse=True)
        return results


class NemotronReranker(BaseReranker):
    def __init__(
        self,
        model_name: str = "nvidia/llama-nemotron-rerank-1b-v2",
        max_length: int = 2048,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.model = None
        self.tokenizer = None

    def _ensure_loaded(self):
        if self.model is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch

        print(f"Loading Cross-Encoder Reranker ({self.model_name}) to GPU...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        ).eval()

        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def rerank(
        self, query: str, doc_indices: list[int], corpus: list[str]
    ) -> list[tuple[int, float]]:
        self._ensure_loaded()

        texts = [
            f"question:{query} \n \n passage:{corpus[doc_id]}" for doc_id in doc_indices
        ]

        batch_dict = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )

        batch_dict = {k: v.to(self.model.device) for k, v in batch_dict.items()}

        with self.torch.inference_mode():
            logits = self.model(**batch_dict).logits
            scores = logits.view(-1).cpu().float().tolist()

        if not isinstance(scores, list):
            scores = [scores]

        results = list(zip(doc_indices, scores))
        results.sort(key=lambda x: x[1], reverse=True)

        return results
