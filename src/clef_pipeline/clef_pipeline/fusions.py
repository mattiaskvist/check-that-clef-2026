"""Pluggable fusion strategies for combining retrieval results.

Provides a unified interface (BaseFuser.fuse) so the pipeline can swap
between static RRF and learned Random-Forest fusion with a one-line change.
"""

import hashlib
import os
import pickle
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklearn.ensemble import RandomForestClassifier

import numpy as np

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


class FeatureGenerator:
    """Builds per-candidate feature vectors from retrieval scores and ranks."""

    FEATURE_NAMES_WITH_RRF = [
        "dense_score",
        "dense_rank",
        "sparse_score",
        "sparse_rank",
        "rrf_score",
        "rrf_rank",
    ]
    FEATURE_NAMES_WITHOUT_RRF = [
        "dense_score",
        "dense_rank",
        "sparse_score",
        "sparse_rank",
    ]

    @staticmethod
    def build_query_features(
        candidates: list[int],
        dense_scores: list[float],
        dense_ranked: list[int],
        sparse_scores: list[float],
        sparse_ranked: list[int],
        rrf_scores: dict[int, float],
        rrf_ranked: list[int],
        include_rrf: bool = True,
    ) -> list[list[float]]:
        """Build a feature vector for every candidate document.

        Args:
            candidates: Document indices to build features for.
            dense_scores: Full score array from dense retriever (indexed by doc_id).
            dense_ranked: Dense ranked list (doc indices sorted by score desc).
            sparse_scores: Full score array from sparse retriever (indexed by doc_id).
            sparse_ranked: Sparse ranked list (doc indices sorted by score desc).
            rrf_scores: Dict mapping doc_id → RRF score.
            rrf_ranked: RRF ranked list (doc indices sorted by score desc).
            include_rrf: Whether to include RRF score/rank as features.

        Returns:
            List of feature vectors, one per candidate.
        """
        if not candidates:
            return []

        # Build O(1) rank lookup arrays
        n_dense = len(dense_scores)
        dense_rank_array = np.empty(n_dense, dtype=np.int32)
        for rank, doc_id in enumerate(dense_ranked):
            if doc_id < n_dense:
                dense_rank_array[doc_id] = rank

        n_sparse = len(sparse_scores)
        sparse_rank_array = np.empty(n_sparse, dtype=np.int32)
        for rank, doc_id in enumerate(sparse_ranked):
            if doc_id < n_sparse:
                sparse_rank_array[doc_id] = rank

        rrf_rank_lookup = {doc_id: rank for rank, doc_id in enumerate(rrf_ranked)}

        features = []
        for doc_id in candidates:
            feat = [
                float(dense_scores[doc_id]),
                float(dense_rank_array[doc_id]),
                float(sparse_scores[doc_id]),
                float(sparse_rank_array[doc_id]),
            ]
            if include_rrf:
                feat.extend(
                    [
                        float(rrf_scores.get(doc_id, 0.0)),
                        float(rrf_rank_lookup.get(doc_id, len(rrf_ranked))),
                    ]
                )
            features.append(feat)

        return features

    @staticmethod
    def get_feature_names(include_rrf: bool = True) -> list[str]:
        if include_rrf:
            return list(FeatureGenerator.FEATURE_NAMES_WITH_RRF)
        return list(FeatureGenerator.FEATURE_NAMES_WITHOUT_RRF)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseFuser(ABC):
    """Abstract base for all fusion strategies."""

    @abstractmethod
    def fuse(
        self,
        ranked_lists: list[list[int]],
        scores_lists: list[list[float]] | None = None,
        top_k: int = 10,
        lang: str | None = None,
    ) -> list[int]:
        """Fuse multiple ranked lists into a single ranked list.

        Args:
            ranked_lists: List of ranked document-index lists (one per retriever).
            scores_lists: Optional parallel list of score arrays (one per retriever).
            top_k: Number of candidates to return.
            lang: Language code (used by per-language models).

        Returns:
            Fused ranked list of document indices, length <= top_k.
        """
        ...


# ---------------------------------------------------------------------------
# Static RRF fusion (drop-in wrapper around existing logic)
# ---------------------------------------------------------------------------


class RRFFuser(BaseFuser):
    """Reciprocal Rank Fusion — the existing static baseline."""

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k

    def fuse(
        self,
        ranked_lists: list[list[int]],
        scores_lists: list[list[float]] | None = None,
        top_k: int = 10,
        lang: str | None = None,
    ) -> list[int]:
        """RRF ignores scores_lists and lang — uses only rank positions."""
        rrf_scores: dict[int, float] = {}
        for ranked_list in ranked_lists:
            for rank, doc_id in enumerate(ranked_list):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (
                    self.rrf_k + rank + 1
                )
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [int(doc_id) for doc_id, _ in sorted_docs[:top_k]]

    @staticmethod
    def _rrf_with_scores(
        ranked_lists: list[list[int]],
        k: int = 60,
        top_k: int | None = None,
    ) -> tuple[list[int], dict[int, float]]:
        """Return (ranked_doc_ids, {doc_id: rrf_score}). Used internally for feature gen."""
        rrf_scores: dict[int, float] = {}
        for ranked_list in ranked_lists:
            for rank, doc_id in enumerate(ranked_list):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        if top_k is not None:
            sorted_docs = sorted_docs[:top_k]
        top_ids = [int(doc_id) for doc_id, _ in sorted_docs]
        top_scores = {doc_id: score for doc_id, score in sorted_docs}
        return top_ids, top_scores


# ---------------------------------------------------------------------------
# Learned Random Forest fusion
# ---------------------------------------------------------------------------


class RandomForestFuser(BaseFuser):
    """Learned fusion using per-language (or global) Random Forest classifiers.

    The classifier predicts P(relevant | features) for each candidate and
    returns candidates sorted by that probability.
    """

    DEFAULT_RF_PARAMS = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(
        self,
        global_model: bool = True,
        rf_params: dict | None = None,
        candidate_top_k: int = 500,
        hf_repo_id: str | None = None,
        hf_token: str | None = None,
    ):
        """
        Args:
            global_model: When true, train a single model over all languages.
                When false, train and use one model per language.
            rf_params: Override RF hyperparameters. Merges with defaults.
            candidate_top_k: How many candidates per retriever to consider
                when building the union set during training.
            hf_repo_id: Hugging Face repository ID to push/pull models from.
            hf_token: Optional Hugging Face token.
        """
        self.global_model = global_model
        self.candidate_top_k = candidate_top_k
        self.rf_params = {**self.DEFAULT_RF_PARAMS, **(rf_params or {})}
        self.model: RandomForestClassifier | None = None
        self.models: dict[str, RandomForestClassifier] = {}
        self._trained = False
        self._fingerprint: str | None = None
        self.hf_repo_id = hf_repo_id
        self.hf_token = hf_token

    # -- Cache fingerprint --------------------------------------------------

    @staticmethod
    def _compute_fingerprint(
        dense_model_name: str,
        sparse_config: dict,
        train_split: str,
        rf_params: dict,
        global_model: bool,
        candidate_top_k: int,
    ) -> str:
        """Deterministic hash of all config that affects the trained model."""
        parts = [
            f"dense={dense_model_name}",
            f"sparse_k1={sparse_config.get('k1', '')}",
            f"sparse_b={sparse_config.get('b', '')}",
            f"sparse_stemmer={sparse_config.get('stemmer', '')}",
            f"split={train_split}",
            f"rf={sorted(rf_params.items())}",
            f"global_model={int(bool(global_model))}",
            f"candidate_top_k={candidate_top_k}",
        ]
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _cache_path(self, cache_dir: str) -> str:
        if self._fingerprint is None:
            raise RuntimeError("Fingerprint not set — call train() or load() first.")
        os.makedirs(os.path.join(cache_dir, "rf_fusion_models"), exist_ok=True)
        return os.path.join(
            cache_dir, "rf_fusion_models", f"rf_{self._fingerprint}.pkl"
        )

    # -- Save / Load --------------------------------------------------------

    def save(self, cache_dir: str) -> str:
        """Pickle trained models to cache_dir and upload to Hugging Face if configured. Returns the written path."""
        path = self._cache_path(cache_dir)
        payload = {
            "global_model": self.global_model,
            "model": self.model,
            "models": self.models,
            "rf_params": self.rf_params,
            "candidate_top_k": self.candidate_top_k,
            "fingerprint": self._fingerprint,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"[RF Fuser] Saved trained models locally to {path}")

        if self.hf_repo_id:
            from huggingface_hub import HfApi

            api = HfApi(token=self.hf_token)
            api.create_repo(repo_id=self.hf_repo_id, exist_ok=True, repo_type="model")
            filename = f"rf_{self._fingerprint}.pkl"
            print(f"[RF Fuser] Uploading to Hugging Face: {self.hf_repo_id}/{filename}")
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=filename,
                repo_id=self.hf_repo_id,
                repo_type="model",
            )
            print("[RF Fuser] Upload complete.")

        return path

    def load(
        self,
        cache_dir: str,
        dense_model_name: str,
        sparse_config: dict,
        train_split: str = "train",
    ) -> bool:
        """Try loading cached models from Hugging Face or locally. Returns True if successful."""
        self._fingerprint = self._compute_fingerprint(
            dense_model_name,
            sparse_config,
            train_split,
            self.rf_params,
            self.global_model,
            self.candidate_top_k,
        )
        path = self._cache_path(cache_dir)

        if self.hf_repo_id:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import (
                EntryNotFoundError,
                RepositoryNotFoundError,
            )

            filename = f"rf_{self._fingerprint}.pkl"
            try:
                print(
                    f"[RF Fuser] Attempting to download {filename} from {self.hf_repo_id}..."
                )
                path = hf_hub_download(
                    repo_id=self.hf_repo_id,
                    filename=filename,
                    token=self.hf_token,
                    cache_dir=os.path.join(cache_dir, "hf_hub"),
                )
            except (EntryNotFoundError, RepositoryNotFoundError):
                print(
                    f"[RF Fuser] File {filename} not found in repository {self.hf_repo_id}."
                )
                return False
            except Exception as e:
                print(f"[RF Fuser] Failed to download from Hugging Face: {e}")
                return False
        elif not os.path.exists(path):
            print(f"[RF Fuser] No cached model at {path}")
            return False

        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.global_model = payload.get("global_model", self.global_model)
        self.model = payload.get("model")
        self.models = payload.get("models", {}) or {}
        self._trained = True
        print(f"[RF Fuser] Loaded cached models from {path}")
        return True

    # -- Training -----------------------------------------------------------

    def train(
        self,
        dense_retriever,
        sparse_retriever,
        train_tweets_by_lang: dict[str, list[dict]],
        article_pubkeys: list[str],
        dense_model_name: str,
        sparse_config: dict,
        train_split: str = "train",
    ):
        """Run retrieval on train data, build features, train RF classifiers.

        Args:
            dense_retriever: Indexed HarrierRetriever with train queries pre-encoded.
            sparse_retriever: Indexed SparseRetriever with train queries pre-encoded.
            train_tweets_by_lang: {lang: [tweet_dict]} from the train split.
            article_pubkeys: List of pubkeys aligned with collection indices.
            dense_model_name: For cache fingerprint.
            sparse_config: For cache fingerprint (dict with k1, b, stemmer keys).
            train_split: Split name for cache fingerprint.
        """
        from sklearn.ensemble import RandomForestClassifier
        from tqdm import tqdm

        self._fingerprint = self._compute_fingerprint(
            dense_model_name,
            sparse_config,
            train_split,
            self.rf_params,
            self.global_model,
            self.candidate_top_k,
        )

        languages = list(train_tweets_by_lang.keys())

        # Collect features per language
        lang_features: dict[str, tuple[list, list]] = {}  # lang -> (X, y)

        for lang in languages:
            tweets = train_tweets_by_lang[lang]
            X_lang, y_lang = [], []

            for i in tqdm(
                range(len(tweets)), desc=f"[RF Train] Building features {lang.upper()}"
            ):
                true_pubkey = tweets[i]["pubkey"]

                # Retrieve with scores
                dense_ranks, dense_scores = dense_retriever.search_with_scores(
                    i, cache_name=f"queries_train_{lang}"
                )
                sparse_ranks, sparse_scores = sparse_retriever.search_with_scores(
                    i, cache_name=f"sparse_queries_train_{lang}"
                )

                # Build candidate union
                union_candidates = list(
                    dict.fromkeys(
                        dense_ranks[: self.candidate_top_k]
                        + sparse_ranks[: self.candidate_top_k]
                    )
                )

                # RRF over union for features
                rrf_ranked, rrf_scores = RRFFuser._rrf_with_scores(
                    [dense_ranks, sparse_ranks], top_k=len(union_candidates)
                )

                # Feature vectors
                features = FeatureGenerator.build_query_features(
                    union_candidates,
                    dense_scores,
                    dense_ranks,
                    sparse_scores,
                    sparse_ranks,
                    rrf_scores,
                    rrf_ranked,
                    include_rrf=True,
                )

                # Labels
                labels = [
                    1 if article_pubkeys[doc_id] == true_pubkey else 0
                    for doc_id in union_candidates
                ]

                X_lang.extend(features)
                y_lang.extend(labels)

            lang_features[lang] = (X_lang, y_lang)

        self.models = {}
        self.model = None
        if self.global_model:
            X_all, y_all = [], []
            for lang in languages:
                X_lang, y_lang = lang_features[lang]
                X_all.extend(X_lang)
                y_all.extend(y_lang)

            print(f"[RF Train] Training unified fuser model on {len(X_all)} samples...")
            clf = RandomForestClassifier(**self.rf_params)
            clf.fit(np.array(X_all, dtype=np.float32), np.array(y_all, dtype=np.float32))
            self.model = clf
            self.models["global"] = clf
        else:
            for lang in languages:
                X_lang, y_lang = lang_features[lang]
                print(
                    f"[RF Train] Training {lang.upper()} fuser model on {len(X_lang)} samples..."
                )
                clf = RandomForestClassifier(**self.rf_params)
                clf.fit(
                    np.array(X_lang, dtype=np.float32),
                    np.array(y_lang, dtype=np.float32),
                )
                self.models[lang] = clf

        self._trained = True
        print("[RF Train] Training complete.")

    # -- Fusion (inference) -------------------------------------------------

    def fuse(
        self,
        ranked_lists: list[list[int]],
        scores_lists: list[list[float]] | None = None,
        top_k: int = 10,
        lang: str | None = None,
    ) -> list[int]:
        """Fuse by predicting P(relevant) for each candidate.

        Args:
            ranked_lists: [dense_ranks, sparse_ranks]
            scores_lists: [dense_scores, sparse_scores] — required for RF.
            top_k: Number of candidates to return.
            lang: Language code to select the per-language model.

        Returns:
            Document indices sorted by predicted relevance, length <= top_k.
        """
        if not self._trained:
            raise RuntimeError(
                "RandomForestFuser is not trained. Call train() or load() first."
            )
        if scores_lists is None:
            raise ValueError("RandomForestFuser requires scores_lists.")
        if len(ranked_lists) != 2 or len(scores_lists) != 2:
            raise ValueError("Expected exactly 2 ranked_lists and 2 scores_lists.")

        dense_ranks, sparse_ranks = ranked_lists
        dense_scores, sparse_scores = scores_lists

        # Build candidate union (same approach as training)
        union_candidates = list(
            dict.fromkeys(
                dense_ranks[: self.candidate_top_k]
                + sparse_ranks[: self.candidate_top_k]
            )
        )

        # RRF for features
        rrf_ranked, rrf_scores = RRFFuser._rrf_with_scores(
            [dense_ranks, sparse_ranks], top_k=len(union_candidates)
        )

        # Build features
        features = FeatureGenerator.build_query_features(
            union_candidates,
            dense_scores,
            dense_ranks,
            sparse_scores,
            sparse_ranks,
            rrf_scores,
            rrf_ranked,
            include_rrf=True,
        )

        # Predict
        if self.global_model:
            model = self.models.get("global") or self.model
        else:
            if lang is None:
                raise ValueError("lang is required when global_model is False.")
            model = self.models.get(lang)

        if model is None:
            raise RuntimeError("RandomForestFuser model is not available for inference.")

        probs = model.predict_proba(np.array(features, dtype=np.float32))
        relevance_scores = probs[:, 1]

        # Sort by predicted relevance
        scored = list(zip(union_candidates, relevance_scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [int(doc_id) for doc_id, _ in scored[:top_k]]
