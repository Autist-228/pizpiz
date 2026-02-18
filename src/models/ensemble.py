import logging
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

MODELS_DIR = Path("data/models")


class LightGBMModel:
    def __init__(self):
        self.model = None
        self.name = "lightgbm"
        self.feature_names: list[str] = []

    def predict(self, features: dict) -> float:
        if self.model is None:
            return self._heuristic_predict(features)
        try:
            import lightgbm as lgb
            X = np.array([[features.get(f, 0) for f in self.feature_names]])
            pred = self.model.predict(X)[0]
            return float(np.clip(pred, 0.01, 0.99))
        except Exception as e:
            logger.debug(f"LightGBM predict error: {e}")
            return self._heuristic_predict(features)

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]):
        try:
            import lightgbm as lgb
            self.feature_names = feature_names
            train_data = lgb.Dataset(X, label=y, feature_name=feature_names)
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
            }
            self.model = lgb.train(params, train_data, num_boost_round=200)
            self._save()
        except Exception as e:
            logger.error(f"LightGBM train error: {e}")

    def load(self) -> bool:
        path = MODELS_DIR / f"{self.name}.pkl"
        if path.exists():
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                self.model = data["model"]
                self.feature_names = data["feature_names"]
                return True
            except Exception as e:
                logger.error(f"LightGBM load error: {e}")
        return False

    def _save(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = MODELS_DIR / f"{self.name}.pkl"
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "feature_names": self.feature_names}, f)

    @staticmethod
    def _heuristic_predict(features: dict) -> float:
        score = 0.5
        edge = features.get("edge_raw", 0)
        score += edge * 0.4
        elo_exp = features.get("elo_expected", 0.5)
        score = score * 0.6 + elo_exp * 0.4
        wr_diff = features.get("win_rate_diff", 0)
        score += wr_diff * 0.15
        momentum = features.get("price_momentum", 0)
        score += momentum * 0.1
        return float(np.clip(score, 0.05, 0.95))


class XGBoostModel:
    def __init__(self):
        self.model = None
        self.name = "xgboost"
        self.feature_names: list[str] = []

    def predict(self, features: dict) -> float:
        if self.model is None:
            return self._heuristic_predict(features)
        try:
            import xgboost as xgb
            X = np.array([[features.get(f, 0) for f in self.feature_names]])
            dmat = xgb.DMatrix(X, feature_names=self.feature_names)
            pred = self.model.predict(dmat)[0]
            return float(np.clip(pred, 0.01, 0.99))
        except Exception as e:
            logger.debug(f"XGBoost predict error: {e}")
            return self._heuristic_predict(features)

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]):
        try:
            import xgboost as xgb
            self.feature_names = feature_names
            dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names)
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "verbosity": 0,
            }
            self.model = xgb.train(params, dtrain, num_boost_round=200)
            self._save()
        except Exception as e:
            logger.error(f"XGBoost train error: {e}")

    def load(self) -> bool:
        path = MODELS_DIR / f"{self.name}.pkl"
        if path.exists():
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                self.model = data["model"]
                self.feature_names = data["feature_names"]
                return True
            except Exception as e:
                logger.error(f"XGBoost load error: {e}")
        return False

    def _save(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = MODELS_DIR / f"{self.name}.pkl"
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "feature_names": self.feature_names}, f)

    @staticmethod
    def _heuristic_predict(features: dict) -> float:
        score = 0.5
        edge = features.get("edge_raw", 0)
        score += edge * 0.35
        elo_exp = features.get("elo_expected", 0.5)
        score = score * 0.55 + elo_exp * 0.45
        recent_diff = features.get("recent_wr_diff", 0)
        score += recent_diff * 0.15
        bid_pressure = features.get("bid_pressure", 0.5)
        score += (bid_pressure - 0.5) * 0.1
        return float(np.clip(score, 0.05, 0.95))


class CatBoostModel:
    def __init__(self):
        self.model = None
        self.name = "catboost"
        self.feature_names: list[str] = []

    def predict(self, features: dict) -> float:
        if self.model is None:
            return self._heuristic_predict(features)
        try:
            X = np.array([[features.get(f, 0) for f in self.feature_names]])
            pred = self.model.predict_proba(X)[0][1]
            return float(np.clip(pred, 0.01, 0.99))
        except Exception as e:
            logger.debug(f"CatBoost predict error: {e}")
            return self._heuristic_predict(features)

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]):
        try:
            from catboost import CatBoostClassifier
            self.feature_names = feature_names
            self.model = CatBoostClassifier(
                iterations=200,
                learning_rate=0.05,
                depth=6,
                verbose=0,
                allow_writing_files=False,
            )
            self.model.fit(X, y)
            self._save()
        except Exception as e:
            logger.error(f"CatBoost train error: {e}")

    def load(self) -> bool:
        path = MODELS_DIR / f"{self.name}.pkl"
        if path.exists():
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                self.model = data["model"]
                self.feature_names = data["feature_names"]
                return True
            except Exception as e:
                logger.error(f"CatBoost load error: {e}")
        return False

    def _save(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = MODELS_DIR / f"{self.name}.pkl"
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "feature_names": self.feature_names}, f)

    @staticmethod
    def _heuristic_predict(features: dict) -> float:
        score = 0.5
        edge = features.get("edge_raw", 0)
        score += edge * 0.45
        elo_exp = features.get("elo_expected", 0.5)
        score = score * 0.5 + elo_exp * 0.5
        net_diff = features.get("net_rating_diff", 0)
        score += net_diff * 0.005
        sentiment = features.get("sentiment_diff", 0)
        score += sentiment * 0.05
        return float(np.clip(score, 0.05, 0.95))


class LogisticModel:
    def __init__(self):
        self.model = None
        self.name = "logistic"
        self.feature_names: list[str] = []
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def predict(self, features: dict) -> float:
        if self.model is None:
            return self._heuristic_predict(features)
        try:
            X = np.array([[features.get(f, 0) for f in self.feature_names]])
            if self.mean is not None and self.std is not None:
                X = (X - self.mean) / np.maximum(self.std, 1e-8)
            pred = self.model.predict_proba(X)[0][1]
            return float(np.clip(pred, 0.01, 0.99))
        except Exception as e:
            logger.debug(f"Logistic predict error: {e}")
            return self._heuristic_predict(features)

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]):
        try:
            self.feature_names = feature_names
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0)
            X_scaled = (X - self.mean) / np.maximum(self.std, 1e-8)
            self.model = LogisticRegression(
                max_iter=1000, C=1.0, solver="lbfgs"
            )
            self.model.fit(X_scaled, y)
            self._save()
        except Exception as e:
            logger.error(f"Logistic train error: {e}")

    def load(self) -> bool:
        path = MODELS_DIR / f"{self.name}.pkl"
        if path.exists():
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                self.model = data["model"]
                self.feature_names = data["feature_names"]
                self.mean = data.get("mean")
                self.std = data.get("std")
                return True
            except Exception as e:
                logger.error(f"Logistic load error: {e}")
        return False

    def _save(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = MODELS_DIR / f"{self.name}.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_names": self.feature_names,
                "mean": self.mean,
                "std": self.std,
            }, f)

    @staticmethod
    def _heuristic_predict(features: dict) -> float:
        score = 0.5
        edge = features.get("edge_raw", 0)
        score += edge * 0.5
        wr_diff = features.get("win_rate_diff", 0)
        score += wr_diff * 0.25
        elo_exp = features.get("elo_expected", 0.5)
        score = score * 0.7 + elo_exp * 0.3
        return float(np.clip(score, 0.05, 0.95))


class ModelEnsemble:
    def __init__(self):
        self.lgbm = LightGBMModel()
        self.xgb = XGBoostModel()
        self.catboost = CatBoostModel()
        self.logistic = LogisticModel()
        self.models = [self.lgbm, self.xgb, self.catboost, self.logistic]
        self.model_weights = [0.3, 0.3, 0.25, 0.15]

    def load_models(self):
        for model in self.models:
            loaded = model.load()
            if loaded:
                logger.info(f"Loaded trained {model.name} model")
            else:
                logger.info(f"{model.name}: using heuristic mode (no trained model)")

    def predict(self, features: dict) -> dict:
        predictions = {}
        for model in self.models:
            pred = model.predict(features)
            predictions[model.name] = pred

        weighted_sum = sum(
            predictions[m.name] * w
            for m, w in zip(self.models, self.model_weights)
        )

        preds_list = list(predictions.values())
        agreement = 1.0 - float(np.std(preds_list)) * 2
        agreement = max(0.0, min(1.0, agreement))

        return {
            "predictions": predictions,
            "ensemble_prob": weighted_sum,
            "model_agreement": agreement,
            "model_spread": max(preds_list) - min(preds_list),
            "model_std": float(np.std(preds_list)),
            "median_pred": float(np.median(preds_list)),
        }

    def train_all(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]):
        for model in self.models:
            logger.info(f"Training {model.name}...")
            model.train(X, y, feature_names)
            logger.info(f"Trained {model.name}")
