import logging
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.models.ensemble import ModelEnsemble
from src.features.builder import FeatureBuilder

logger = logging.getLogger(__name__)

HISTORY_DB = "data/historical.db"
MODELS_DIR = Path("data/models")


class TrainingPipeline:
    def __init__(self):
        self.ensemble = ModelEnsemble()
        self.feature_builder = FeatureBuilder()
        self.min_samples = 50
        self.feature_names: list[str] = []

    def run_training(self) -> dict:
        logger.info("Starting ML training pipeline...")

        raw_data = self._load_training_data()
        if len(raw_data) < self.min_samples:
            logger.warning(
                f"Not enough training data: {len(raw_data)} < {self.min_samples}. "
                f"Need more resolved markets."
            )
            return {
                "status": "insufficient_data",
                "samples": len(raw_data),
                "min_required": self.min_samples,
            }

        X, y, feature_names = self._prepare_features(raw_data)
        self.feature_names = feature_names

        if len(X) < self.min_samples:
            logger.warning(f"After feature prep only {len(X)} valid samples")
            return {
                "status": "insufficient_data",
                "samples": len(X),
                "min_required": self.min_samples,
            }

        logger.info(f"Training on {len(X)} samples, {len(feature_names)} features")

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        self.ensemble.train_all(X_train, y_train, feature_names)

        train_metrics = self._evaluate(X_train, y_train, "train")
        test_metrics = self._evaluate(X_test, y_test, "test")

        importance = self._get_feature_importance(feature_names)

        self._save_training_log(len(X), train_metrics, test_metrics, importance)

        logger.info(
            f"Training complete: train_acc={train_metrics['accuracy']:.3f}, "
            f"test_acc={test_metrics['accuracy']:.3f}"
        )

        return {
            "status": "success",
            "total_samples": len(X),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "features": len(feature_names),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "top_features": importance[:10],
        }

    def _load_training_data(self) -> list[dict]:
        if not Path(HISTORY_DB).exists():
            return []
        conn = sqlite3.connect(HISTORY_DB)
        rows = conn.execute("""
            SELECT market_id, platform, question, category, tags,
                   outcome_yes_price, volume, end_date, result,
                   home_team, away_team, sport,
                   bookmaker_home_prob, bookmaker_away_prob, n_bookmakers
            FROM resolved_markets
            WHERE result IS NOT NULL
            ORDER BY end_date ASC
        """).fetchall()
        conn.close()

        data = []
        for r in rows:
            data.append({
                "market_id": r[0],
                "platform": r[1],
                "question": r[2],
                "category": r[3],
                "tags": r[4],
                "yes_price": r[5],
                "volume": r[6],
                "end_date": r[7],
                "result": r[8],
                "home_team": r[9],
                "away_team": r[10],
                "sport": r[11],
                "bookmaker_home_prob": r[12],
                "bookmaker_away_prob": r[13],
                "n_bookmakers": r[14],
            })
        return data

    def _prepare_features(self, raw_data: list[dict]) -> tuple:
        feature_rows = []
        labels = []

        for entry in raw_data:
            try:
                yes_price = entry["yes_price"]
                if yes_price <= 0.01 or yes_price >= 0.99:
                    continue

                bk_home = entry.get("bookmaker_home_prob", 0)
                bk_away = entry.get("bookmaker_away_prob", 0)
                if bk_home <= 0:
                    bk_home = 0.5
                if bk_away <= 0:
                    bk_away = 0.5

                bookmaker_odds = {
                    "home_prob": bk_home,
                    "away_prob": bk_away,
                    "draw_prob": 0.0,
                    "n_bookmakers": entry.get("n_bookmakers", 0),
                    "has_sharp": False,
                }

                features = self.feature_builder.build_features(
                    market_price=yes_price,
                    bookmaker_odds=bookmaker_odds,
                    home_stats=None,
                    away_stats=None,
                    home_news_sentiment={"injury": 0, "positive": 0, "negative": 0},
                    away_news_sentiment={"injury": 0, "positive": 0, "negative": 0},
                    orderbook={"bids": [], "asks": []},
                    price_history=[],
                    hours_to_event=6.0,
                    platform=entry.get("platform", "polymarket"),
                    sport=entry.get("sport", "unknown"),
                )

                feature_rows.append(features)
                labels.append(entry["result"])

            except Exception as e:
                logger.debug(f"Feature prep error for {entry.get('market_id')}: {e}")
                continue

        if not feature_rows:
            return np.array([]), np.array([]), []

        feature_names = sorted(feature_rows[0].keys())
        X = np.array([
            [row.get(f, 0) for f in feature_names]
            for row in feature_rows
        ], dtype=np.float64)

        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

        y = np.array(labels, dtype=np.float64)

        return X, y, feature_names

    def _evaluate(self, X: np.ndarray, y: np.ndarray, split: str) -> dict:
        if len(X) == 0:
            return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "brier": 1.0}

        predictions = []
        for i in range(len(X)):
            feat_dict = {
                name: X[i][j]
                for j, name in enumerate(self.feature_names)
            }
            result = self.ensemble.predict(feat_dict)
            predictions.append(result["ensemble_prob"])

        preds = np.array(predictions)
        pred_labels = (preds > 0.5).astype(int)

        accuracy = np.mean(pred_labels == y)
        tp = np.sum((pred_labels == 1) & (y == 1))
        fp = np.sum((pred_labels == 1) & (y == 0))
        fn = np.sum((pred_labels == 0) & (y == 1))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        brier = np.mean((preds - y) ** 2)

        logger.info(
            f"{split}: acc={accuracy:.3f} prec={precision:.3f} "
            f"rec={recall:.3f} f1={f1:.3f} brier={brier:.3f}"
        )

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "brier": float(brier),
        }

    def _get_feature_importance(self, feature_names: list[str]) -> list[dict]:
        importance = []

        try:
            if self.ensemble.lgbm.model is not None:
                imp = self.ensemble.lgbm.model.feature_importance(importance_type="gain")
                total = sum(imp)
                if total > 0:
                    for name, val in zip(feature_names, imp):
                        importance.append({
                            "feature": name,
                            "importance": float(val / total),
                            "model": "lightgbm",
                        })
        except Exception as e:
            logger.debug(f"LightGBM importance error: {e}")

        if not importance:
            for name in feature_names:
                importance.append({
                    "feature": name,
                    "importance": 1.0 / max(len(feature_names), 1),
                    "model": "uniform",
                })

        importance.sort(key=lambda x: x["importance"], reverse=True)
        return importance

    def _save_training_log(
        self,
        n_samples: int,
        train_metrics: dict,
        test_metrics: dict,
        importance: list[dict],
    ):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_samples": n_samples,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "top_features": importance[:20],
        }
        log_path = MODELS_DIR / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        logger.info(f"Training log saved to {log_path}")

    def get_last_training_log(self) -> dict | None:
        log_path = MODELS_DIR / "training_log.json"
        if log_path.exists():
            with open(log_path) as f:
                return json.load(f)
        return None
