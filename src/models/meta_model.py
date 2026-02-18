import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


class MetaModel:
    def __init__(self):
        self.calibration_history: list[dict] = []

    def combine_signals(
        self,
        ensemble_result: dict,
        bookmaker_edge: float,
        nlp_sentiment: float,
        microstructure_signal: float,
        bayesian_prob: float,
        features: dict,
    ) -> dict:
        ensemble_prob = ensemble_result["ensemble_prob"]
        agreement = ensemble_result["model_agreement"]

        weights = self._dynamic_weights(ensemble_result, features)

        combined = (
            weights["ensemble"] * ensemble_prob +
            weights["bookmaker"] * (features.get("market_yes_price", 0.5) + bookmaker_edge) +
            weights["nlp"] * self._nlp_adjustment(ensemble_prob, nlp_sentiment) +
            weights["microstructure"] * self._micro_adjustment(ensemble_prob, microstructure_signal) +
            weights["bayesian"] * bayesian_prob
        )

        combined = float(np.clip(combined, 0.02, 0.98))

        market_price = features.get("market_yes_price", 0.5)
        final_edge = combined - market_price

        confidence = self._calculate_confidence(
            ensemble_result, bookmaker_edge, final_edge, features
        )

        return {
            "meta_probability": combined,
            "meta_edge": final_edge,
            "meta_confidence": confidence,
            "signal_weights": weights,
            "component_signals": {
                "ensemble": ensemble_prob,
                "bookmaker_edge": bookmaker_edge,
                "nlp_sentiment": nlp_sentiment,
                "microstructure": microstructure_signal,
                "bayesian": bayesian_prob,
            },
        }

    def _dynamic_weights(self, ensemble_result: dict, features: dict) -> dict:
        agreement = ensemble_result["model_agreement"]
        has_sharp = features.get("has_sharp_bookmaker", 0)
        hours = features.get("hours_to_event", 12)
        has_orderbook = features.get("has_orderbook", 0)
        any_injury = features.get("any_injury_news", 0)

        w_ensemble = 0.35
        w_bookmaker = 0.25
        w_nlp = 0.10
        w_micro = 0.10
        w_bayesian = 0.20

        if agreement > 0.8:
            w_ensemble += 0.05
            w_bookmaker -= 0.05

        if has_sharp:
            w_bookmaker += 0.05
            w_ensemble -= 0.05

        if hours < 3:
            w_micro += 0.05
            w_bayesian += 0.05
            w_nlp -= 0.05
            w_bookmaker -= 0.05

        if any_injury:
            w_nlp += 0.05
            w_ensemble -= 0.05

        if not has_orderbook:
            w_micro = 0.02
            extra = 0.08
            w_ensemble += extra * 0.5
            w_bookmaker += extra * 0.3
            w_bayesian += extra * 0.2

        total = w_ensemble + w_bookmaker + w_nlp + w_micro + w_bayesian
        return {
            "ensemble": w_ensemble / total,
            "bookmaker": w_bookmaker / total,
            "nlp": w_nlp / total,
            "microstructure": w_micro / total,
            "bayesian": w_bayesian / total,
        }

    @staticmethod
    def _nlp_adjustment(base_prob: float, sentiment: float) -> float:
        adjustment = sentiment * 0.15
        return float(np.clip(base_prob + adjustment, 0.05, 0.95))

    @staticmethod
    def _micro_adjustment(base_prob: float, signal: float) -> float:
        adjustment = signal * 0.1
        return float(np.clip(base_prob + adjustment, 0.05, 0.95))

    def _calculate_confidence(
        self,
        ensemble_result: dict,
        bookmaker_edge: float,
        final_edge: float,
        features: dict,
    ) -> float:
        confidence = 0.5

        agreement = ensemble_result["model_agreement"]
        confidence += (agreement - 0.5) * 0.3

        if abs(bookmaker_edge) > 0.05:
            edge_sign = 1 if bookmaker_edge > 0 else -1
            final_sign = 1 if final_edge > 0 else -1
            if edge_sign == final_sign:
                confidence += 0.15
            else:
                confidence -= 0.1

        model_spread = ensemble_result["model_spread"]
        if model_spread < 0.1:
            confidence += 0.1
        elif model_spread > 0.25:
            confidence -= 0.15

        n_bk = features.get("n_bookmakers", 0)
        if n_bk >= 5:
            confidence += 0.1
        elif n_bk >= 3:
            confidence += 0.05

        depth = features.get("total_depth", 0)
        if depth > 10000:
            confidence += 0.05

        return float(np.clip(confidence, 0.1, 0.95))

    def update_calibration(self, predicted_prob: float, actual_outcome: int):
        self.calibration_history.append({
            "predicted": predicted_prob,
            "actual": actual_outcome,
        })

    def get_calibration_stats(self) -> dict:
        if len(self.calibration_history) < 10:
            return {"n_samples": len(self.calibration_history), "calibrated": False}

        bins = np.linspace(0, 1, 11)
        bin_counts = np.zeros(10)
        bin_correct = np.zeros(10)

        for entry in self.calibration_history:
            pred = entry["predicted"]
            actual = entry["actual"]
            bin_idx = min(int(pred * 10), 9)
            bin_counts[bin_idx] += 1
            bin_correct[bin_idx] += actual

        calibration_error = 0
        valid_bins = 0
        for i in range(10):
            if bin_counts[i] >= 3:
                expected = (bins[i] + bins[i + 1]) / 2
                actual_rate = bin_correct[i] / bin_counts[i]
                calibration_error += abs(expected - actual_rate)
                valid_bins += 1

        avg_error = calibration_error / max(valid_bins, 1)

        return {
            "n_samples": len(self.calibration_history),
            "calibrated": True,
            "avg_calibration_error": avg_error,
            "well_calibrated": avg_error < 0.1,
        }
