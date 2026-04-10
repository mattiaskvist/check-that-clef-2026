import numpy as np


class FusionProcessor:
    @staticmethod
    def reciprocal_rank_fusion(
        ranked_lists: list[list[int]], k: int = 60, top_k: int = 10
    ) -> tuple[list[int], dict[int, float]]:
        """Return (ranked_doc_ids, {doc_id: rrf_score}) for the top-k candidates."""
        rrf_scores = {}
        for ranked_list in ranked_lists:
            for rank, doc_id in enumerate(ranked_list):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = sorted_docs[:top_k]
        top_ids = [doc_id for doc_id, score in top_docs]
        top_scores = {doc_id: score for doc_id, score in top_docs}
        return top_ids, top_scores


class ScoreFusionProcessor:
    """XGBoost-based learned fusion layer.

    Trains a binary classifier on per-candidate feature vectors built from
    the scores and ranks produced by the upstream retrieval systems.
    At inference time, candidates are re-ranked by predicted P(correct=1).
    """

    FEATURE_NAMES = [
        "dense_score",
        "dense_rank",
        "sparse_score",
        "sparse_rank",
        "rrf_score",
        "rrf_rank",
        #"hard_indicator_score",
    ]

    def __init__(self, lgb_params: dict | None = None):
        self.model = None
        self.params = lgb_params or {
            "objective": "lambdarank",
            "metric": "ndcg",
            "learning_rate": 0.1,
            "n_estimators": 200,
            "num_leaves": 31,
            "min_data_in_leaf": 10,
        }

    @staticmethod
    def build_query_features(
        candidates: list[int],
        dense_scores: np.ndarray,
        dense_ranked: list[int],
        sparse_scores: np.ndarray,
        sparse_ranked: list[int],
        rrf_scores: dict[int, float],
        rrf_ranked: list[int],
        #hard_indicator_scores: dict[int, float],
    ) -> list[list[float]]:
        """Build feature vectors for all candidates of a single query.

        Args:
            candidates: list of candidate doc IDs (the RRF top-k).
            dense_scores: numpy array where dense_scores[doc_id] = cosine_sim.
            dense_ranked: full list of doc IDs sorted by dense score descending.
            sparse_scores: numpy array where sparse_scores[doc_id] = BM25 score.
            sparse_ranked: full list of doc IDs sorted by BM25 score descending.
            rrf_scores: dict {doc_id: rrf_score} for the top-k candidates.
            rrf_ranked: list of doc IDs sorted by RRF score descending.
            hard_indicator_scores: dict {doc_id: float} of extracted feature scores.

        Returns:
            list of feature vectors, one per candidate, in the same order as
            `candidates`.
        """
        # Pre-compute rank lookup dicts for O(1) access
        # Only compute for the top portion of dense/sparse to avoid huge dicts
        # (candidates are from RRF top-k, so their dense/sparse ranks are bounded)
        dense_rank_lookup = {}
        for rank, doc_id in enumerate(dense_ranked):
            dense_rank_lookup[doc_id] = rank
            if len(dense_rank_lookup) > 10000:
                break

        sparse_rank_lookup = {}
        for rank, doc_id in enumerate(sparse_ranked):
            sparse_rank_lookup[doc_id] = rank
            if len(sparse_rank_lookup) > 10000:
                break

        rrf_rank_lookup = {doc_id: rank for rank, doc_id in enumerate(rrf_ranked)}

        # Default rank for docs not found (very large → low relevance signal)
        max_rank = len(dense_ranked)

        features = []
        for doc_id in candidates:
            feat = [
                float(dense_scores[doc_id]),
                float(dense_rank_lookup.get(doc_id, max_rank)),
                float(sparse_scores[doc_id]),
                float(sparse_rank_lookup.get(doc_id, max_rank)),
                float(rrf_scores.get(doc_id, 0.0)),
                float(rrf_rank_lookup.get(doc_id, len(rrf_ranked))),
                #float(hard_indicator_scores.get(doc_id, 0.0)),
            ]
            features.append(feat)

        return features

    def train(self, X: np.ndarray, y: np.ndarray, group: np.ndarray | list[int]):
        """Train LightGBM LambdaMART ranker.

        Args:
            X: feature matrix of shape (n_samples, 8).
            y: binary labels of shape (n_samples,).
            group: array of group sizes (number of candidates per query).
        """
        import lightgbm as lgb
        
        # update deprecated params to new names to avoid warnings
        params = self.params.copy()
        if "min_data_in_leaf" in params:
            params["min_child_samples"] = params.pop("min_data_in_leaf")

        clf = lgb.LGBMRanker(**params)
        # Convert X to DataFrame with feature names if possible, else pass feature_name param
        try:
            import pandas as pd
            X_df = pd.DataFrame(X, columns=self.FEATURE_NAMES)
            clf.fit(X_df, y, group=group)
        except ImportError:
            clf.fit(X, y, group=group, feature_name=self.FEATURE_NAMES)
            
        self.model = clf

        # Print feature importances
        importances = clf.feature_importances_
        print("\n[LightGBM] Feature importances:")
        for name, imp in sorted(
            zip(self.FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True
        ):
            print(f"  {name:20s} {imp}")

    def predict_and_rerank(
        self, candidate_doc_ids: list[int], features: list[list[float]]
    ) -> list[tuple[int, float]]:
        """Predict LambdaMART score for each candidate and return sorted descending.

        Args:
            candidate_doc_ids: list of doc IDs.
            features: list of feature vectors (same order as candidate_doc_ids).

        Returns:
            list of (doc_id, score) sorted by score descending.
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet — call train() first.")

        try:
            import pandas as pd
            X = pd.DataFrame(features, columns=self.FEATURE_NAMES)
        except ImportError:
            X = np.array(features)

        scores = self.model.predict(X)

        results = list(zip(candidate_doc_ids, scores.tolist()))
        results.sort(key=lambda x: x[1], reverse=True)
        return results