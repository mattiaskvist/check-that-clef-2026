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

class FeatureGenerator:
    @staticmethod
    def build_query_features(
        candidates: list[int],
        dense_scores: np.ndarray,
        dense_ranked: list[int],
        sparse_scores: np.ndarray,
        sparse_ranked: list[int],
        rrf_scores: dict[int, float],
        rrf_ranked: list[int],
        include_rrf: bool = True,
    ) -> list[list[float]]:
        
        if not candidates:
            return []

        # Invert the ranked lists to get O(1) exact ranks for every single candidate
        dense_rank_array = np.empty(len(dense_scores), dtype=np.int32)
        dense_rank_array[dense_ranked] = np.arange(len(dense_ranked))

        sparse_rank_array = np.empty(len(sparse_scores), dtype=np.int32)
        sparse_rank_array[sparse_ranked] = np.arange(len(sparse_ranked))

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
                feat.extend([
                    float(rrf_scores.get(doc_id, 0.0)),
                    float(rrf_rank_lookup.get(doc_id, len(rrf_ranked))),
                ])
            features.append(feat)

        return features

    @staticmethod
    def get_feature_names(include_rrf: bool = True) -> list[str]:
        names = ["dense_score", "dense_rank", "sparse_score", "sparse_rank"]
        if include_rrf:
            names.extend(["rrf_score", "rrf_rank"])
        return names


class BaseMLFuser:
    def __init__(self, include_rrf: bool = True):
        self.model = None
        self.include_rrf = include_rrf
        
    @property
    def feature_names(self):
        return FeatureGenerator.get_feature_names(self.include_rrf)

    def train(self, X: np.ndarray, y: np.ndarray, group: np.ndarray | list[int] = None):
        raise NotImplementedError

    def predict_and_rerank(self, candidate_doc_ids: list[int], features: list[list[float]]) -> list[tuple[int, float]]:
        if self.model is None:
            raise RuntimeError("Model not trained yet — call train() first.")
        
        scores = self._predict(features)
        results = list(zip(candidate_doc_ids, scores))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _predict(self, features: list[list[float]]) -> list[float]:
        raise NotImplementedError

class RandomForestFuser(BaseMLFuser):
    def __init__(self, include_rrf: bool = True, rf_params: dict | None = None):
        super().__init__(include_rrf)
        self.rf_params = rf_params if rf_params is not None else {
            "n_estimators": 100, "max_depth": 10, "random_state": 42, "n_jobs": -1
        }
        if "random_state" not in self.rf_params: self.rf_params["random_state"] = 42
        if "n_jobs" not in self.rf_params: self.rf_params["n_jobs"] = -1

    def train(self, X: np.ndarray, y: np.ndarray, group: np.ndarray | list[int] = None):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(**self.rf_params)
        self.model.fit(np.array(X), np.array(y))

    def _predict(self, features: list[list[float]]) -> list[float]:
        import numpy as np
        probs = self.model.predict_proba(np.array(features))
        return probs[:, 1].tolist()