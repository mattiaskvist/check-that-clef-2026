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

class LGBMFuser(BaseMLFuser):
    def __init__(self, include_rrf: bool = True, lgb_params: dict | None = None):
        super().__init__(include_rrf)
        self.params = lgb_params or {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.1,
            "n_estimators": 100,
            "num_leaves": 15,
            "min_data_in_leaf": 10,
            "verbose": -1,
        }

    def train(self, X: np.ndarray, y: np.ndarray, group: np.ndarray | list[int] = None):
        import lightgbm as lgb
        params = self.params.copy()
        if "min_data_in_leaf" in params:
            params["min_child_samples"] = params.pop("min_data_in_leaf")

        clf = lgb.LGBMClassifier(**params)
        try:
            import pandas as pd
            X_df = pd.DataFrame(X, columns=self.feature_names)
            clf.fit(X_df, y)
        except ImportError:
            clf.fit(X, y, feature_name=self.feature_names)
            
        self.model = clf

    def _predict(self, features: list[list[float]]) -> list[float]:
        try:
            import pandas as pd
            X = pd.DataFrame(features, columns=self.feature_names)
        except ImportError:
            X = np.array(features)
        probs = self.model.predict_proba(X)
        return probs[:, 1].tolist()

class XGBFuser(BaseMLFuser):
    def train(self, X: np.ndarray, y: np.ndarray, group: np.ndarray | list[int] = None):
        import xgboost as xgb
        self.model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)
        self.model.fit(np.array(X), np.array(y))

    def _predict(self, features: list[list[float]]) -> list[float]:
        import numpy as np
        probs = self.model.predict_proba(np.array(features))
        return probs[:, 1].tolist()

class RandomForestFuser(BaseMLFuser):
    def train(self, X: np.ndarray, y: np.ndarray, group: np.ndarray | list[int] = None):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        self.model.fit(np.array(X), np.array(y))

    def _predict(self, features: list[list[float]]) -> list[float]:
        import numpy as np
        probs = self.model.predict_proba(np.array(features))
        return probs[:, 1].tolist()

class LogisticRegressionFuser(BaseMLFuser):
    def train(self, X: np.ndarray, y: np.ndarray, group: np.ndarray | list[int] = None):
        from sklearn.linear_model import LogisticRegression
        # Scale inputs just in case to help convergence (though we min-max scaled rank bounds are huge)
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(np.array(X))
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X_scaled, np.array(y))

    def _predict(self, features: list[list[float]]) -> list[float]:
        import numpy as np
        X_scaled = self.scaler.transform(np.array(features))
        probs = self.model.predict_proba(X_scaled)
        return probs[:, 1].tolist()