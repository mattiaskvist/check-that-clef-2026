"""Cross-encoder reranker implementations used after candidate retrieval."""

from .interfaces import BaseReranker


class Gemma2BReranker(BaseReranker):
    """Gemma-based generative reranker producing Yes/No relevance logits."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-gemma"):
        """Load tokenizer and causal LM weights for reranking.

        Args:
            model_name: Hugging Face model id for the reranker.
        """
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
        """Build padded model inputs for query/passage pairs.

        Args:
            pairs: Query/passage text pairs.
            max_length: Maximum token budget for one encoded pair.

        Returns:
            Tokenizer batch dictionary as PyTorch tensors.
        """
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
        """Select final-token logits for each sequence in a padded batch."""
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
        """Rerank candidate documents by the model's Yes-token score.

        Args:
            query: Query text.
            doc_indices: Candidate document indices.
            corpus: Document text corpus aligned to indices.

        Returns:
            Candidate indices paired with scores sorted descending.
        """
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
    """Sequence-classification reranker backed by Nemotron model weights."""

    def __init__(
        self,
        model_name: str = "nvidia/llama-nemotron-rerank-1b-v2",
        max_length: int = 2048,
    ):
        """Store model settings and defer heavy loading until first use.

        Args:
            model_name: Hugging Face model id for the reranker.
            max_length: Maximum sequence length for tokenizer truncation.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.model = None
        self.tokenizer = None

    def _ensure_loaded(self):
        """Lazily load tokenizer/model weights onto available GPU resources."""
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
        """Rerank candidate documents by sequence-classification logits.

        Args:
            query: Query text.
            doc_indices: Candidate document indices.
            corpus: Document text corpus aligned to indices.

        Returns:
            Candidate indices paired with scores sorted descending.
        """
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


class JinaReranker(BaseReranker):
    """Sequence-classification reranker backed by Jina model weights."""

    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v3",
        max_length: int = 2048,
    ):
        """Store model settings and defer heavy loading until first use.

        Args:
            model_name: Hugging Face model id for the reranker.
            max_length: Maximum sequence length for truncation.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.model = None

    def _ensure_loaded(self):
        """Lazily load model weights onto available GPU resources."""
        if self.model is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification

        self.torch = torch

        print(f"Loading Cross-Encoder Reranker ({self.model_name}) to GPU...")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        ).eval()

    def rerank(
        self, query: str, doc_indices: list[int], corpus: list[str]
    ) -> list[tuple[int, float]]:
        """Rerank candidate documents using Jina's compute_score method.

        Args:
            query: Query text.
            doc_indices: Candidate document indices.
            corpus: Document text corpus aligned to indices.

        Returns:
            Candidate indices paired with scores sorted descending.
        """
        self._ensure_loaded()

        pairs = [[query, corpus[doc_id]] for doc_id in doc_indices]

        with self.torch.inference_mode():
            scores = self.model.compute_score(pairs, max_length=self.max_length)

        if not isinstance(scores, list):
            scores = [scores]

        results = list(zip(doc_indices, scores))
        results.sort(key=lambda x: x[1], reverse=True)

        return results

class Qwen3Reranker(BaseReranker):
    """Qwen3-Reranker cross-encoder.

    Qwen3-Reranker is an instruction-aware causal LM reranker that dominates
    the 2026 multilingual reranking benchmarks (MMTEB-R 72.74 at 4B, leads
    the Qwen3 family at 8B). Scores are derived from a yes/no logit softmax
    on the final generated token, following the official Qwen inference
    recipe. Supports 100+ languages including strong German.
    """

    _PREFIX = (
        '<|im_start|>system\nJudge whether the Document meets the '
        'requirements based on the Query and the Instruct provided. '
        'Note that the answer can only be "yes" or "no".<|im_end|>\n'
        '<|im_start|>user\n'
    )
    _SUFFIX = (
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    _DEFAULT_INSTRUCTION = (
        "Given a scientific claim or social media post, retrieve the "
        "scientific paper that the claim refers to or is supported by."
    )

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-8B",
        max_length: int = 2048,
        micro_batch_size: int = 8,
        instruction: str | None = None,
    ):
        """Store model settings and defer heavy loading until first use.

        Args:
            model_name: Hugging Face model id. Use ``Qwen/Qwen3-Reranker-0.6B``,
                ``Qwen/Qwen3-Reranker-4B`` or ``Qwen/Qwen3-Reranker-8B``.
            max_length: Token budget per (query, passage) pair. Qwen3
                supports 32k but 2048 is plenty for title+abstract and much
                faster.
            micro_batch_size: Pairs per forward pass. Default 8 is tuned
                for B200 (192 GB HBM) running the 8B variant at fp16; drop
                this if running on A100-40GB or using flash_attention_2.
            instruction: Task instruction injected into the prompt. If
                None, a scientific-paper-retrieval default is used.
        """
        self.model_name = model_name
        self.max_length = int(max_length)
        self.micro_batch_size = max(1, int(micro_batch_size))
        self.instruction = instruction or self._DEFAULT_INSTRUCTION
        self.model = None
        self.tokenizer = None

    def _ensure_loaded(self):
        """Lazily load tokenizer/model weights and pre-tokenize the prompt."""
        if self.model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch

        print(f"Loading Qwen3 Reranker ({self.model_name}) to GPU...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        ).eval()

        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")

        self.prefix_tokens = self.tokenizer.encode(
            self._PREFIX, add_special_tokens=False
        )
        self.suffix_tokens = self.tokenizer.encode(
            self._SUFFIX, add_special_tokens=False
        )

    def _format_pair(self, query: str, doc: str) -> str:
        """Apply the Qwen3 instruction template to a (query, doc) pair."""
        return (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {doc}"
        )

    def _process_inputs(self, pairs: list[str]):
        """Tokenize pairs and splice the system/user/assistant scaffold."""
        budget = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=budget,
        )
        for i, ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ids + self.suffix_tokens
        inputs = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    def rerank(
        self, query: str, doc_indices: list[int], corpus: list[str]
    ) -> list[tuple[int, float]]:
        """Rerank candidates by the softmaxed yes/no logit probability.

        Args:
            query: Query text.
            doc_indices: Candidate document indices.
            corpus: Document text corpus aligned to indices.

        Returns:
            Candidate indices paired with scores sorted descending.
        """
        if not doc_indices:
            return []

        self._ensure_loaded()

        scores: list[float] = []
        for start in range(0, len(doc_indices), self.micro_batch_size):
            batch_ids = doc_indices[start : start + self.micro_batch_size]
            pairs = [self._format_pair(query, corpus[doc_id]) for doc_id in batch_ids]
            inputs = self._process_inputs(pairs)
            with self.torch.inference_mode():
                last_logits = self.model(**inputs).logits[:, -1, :]
                true_logits = last_logits[:, self.token_true_id]
                false_logits = last_logits[:, self.token_false_id]
                stacked = self.torch.stack([false_logits, true_logits], dim=1)
                log_probs = self.torch.nn.functional.log_softmax(stacked, dim=1)
                batch_scores = log_probs[:, 1].exp().cpu().float().tolist()
            scores.extend(batch_scores)

        results = list(zip(doc_indices, scores))
        results.sort(key=lambda x: x[1], reverse=True)
        return results
