import logging
import math
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)


class BayesianUpdater:
    def __init__(self):
        self.priors: dict[str, dict] = {}

    def get_posterior(
        self,
        event_id: str,
        prior_prob: float,
        new_evidence: list[dict],
    ) -> dict:
        if event_id not in self.priors:
            self.priors[event_id] = {
                "initial_prior": prior_prob,
                "current_posterior": prior_prob,
                "updates": [],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

        posterior = self.priors[event_id]["current_posterior"]

        for evidence in new_evidence:
            posterior = self._apply_update(posterior, evidence)

        posterior = float(np.clip(posterior, 0.02, 0.98))
        self.priors[event_id]["current_posterior"] = posterior

        return {
            "bayesian_prob": posterior,
            "initial_prior": self.priors[event_id]["initial_prior"],
            "n_updates": len(self.priors[event_id]["updates"]),
            "shift_from_prior": posterior - self.priors[event_id]["initial_prior"],
            "confidence": self._update_confidence(event_id),
        }

    def _apply_update(self, prior: float, evidence: dict) -> float:
        ev_type = evidence.get("type", "")
        strength = evidence.get("strength", 0.0)
        direction = evidence.get("direction", 0)

        likelihood_ratios = {
            "price_move": 1.0 + strength * direction * 0.3,
            "volume_spike": 1.0 + strength * direction * 0.2,
            "news_injury": 1.0 + strength * direction * 0.4,
            "news_positive": 1.0 + strength * direction * 0.15,
            "bookmaker_shift": 1.0 + strength * direction * 0.35,
            "smart_money": 1.0 + strength * direction * 0.25,
            "orderbook_imbalance": 1.0 + strength * direction * 0.15,
            "model_update": 1.0 + strength * direction * 0.2,
        }

        lr = likelihood_ratios.get(ev_type, 1.0 + strength * direction * 0.1)
        lr = max(lr, 0.1)

        odds = prior / max(1 - prior, 0.001)
        posterior_odds = odds * lr
        posterior = posterior_odds / (1 + posterior_odds)

        return float(np.clip(posterior, 0.02, 0.98))

    def build_evidence_from_signals(
        self,
        features: dict,
        prev_features: dict | None,
        micro_signals: dict,
        nlp_signals: dict,
    ) -> list[dict]:
        evidence: list[dict] = []

        if prev_features:
            price_change = (
                features.get("market_yes_price", 0.5) -
                prev_features.get("market_yes_price", 0.5)
            )
            if abs(price_change) > 0.02:
                evidence.append({
                    "type": "price_move",
                    "strength": min(abs(price_change) * 5, 1.0),
                    "direction": 1 if price_change > 0 else -1,
                })

            bk_shift = (
                features.get("bookmaker_prob", 0.5) -
                prev_features.get("bookmaker_prob", 0.5)
            )
            if abs(bk_shift) > 0.02:
                evidence.append({
                    "type": "bookmaker_shift",
                    "strength": min(abs(bk_shift) * 5, 1.0),
                    "direction": 1 if bk_shift > 0 else -1,
                })

        if micro_signals.get("smart_money_signal", 0) != 0:
            sm = micro_signals["smart_money_signal"]
            evidence.append({
                "type": "smart_money",
                "strength": abs(sm),
                "direction": 1 if sm > 0 else -1,
            })

        if micro_signals.get("large_move_detected", False):
            evidence.append({
                "type": "volume_spike",
                "strength": 0.6,
                "direction": 1 if micro_signals.get("flow_signal", 0) > 0 else -1,
            })

        depth_imbalance = micro_signals.get("depth_imbalance", 0)
        if abs(depth_imbalance) > 0.3:
            evidence.append({
                "type": "orderbook_imbalance",
                "strength": min(abs(depth_imbalance), 1.0),
                "direction": 1 if depth_imbalance > 0 else -1,
            })

        injury = nlp_signals.get("injury_impact", 0)
        if abs(injury) > 0.2:
            evidence.append({
                "type": "news_injury",
                "strength": abs(injury),
                "direction": 1 if injury > 0 else -1,
            })

        news_sent = nlp_signals.get("news_sentiment", 0)
        if abs(news_sent) > 0.1:
            evidence.append({
                "type": "news_positive",
                "strength": abs(news_sent),
                "direction": 1 if news_sent > 0 else -1,
            })

        return evidence

    def _update_confidence(self, event_id: str) -> float:
        entry = self.priors.get(event_id)
        if not entry:
            return 0.3

        n_updates = len(entry.get("updates", []))
        shift = abs(entry["current_posterior"] - entry["initial_prior"])

        conf = 0.4
        conf += min(n_updates * 0.05, 0.3)
        if shift < 0.05:
            conf += 0.1
        elif shift > 0.2:
            conf -= 0.1

        return float(np.clip(conf, 0.1, 0.9))

    def cleanup_old_events(self, active_event_ids: set[str]):
        to_remove = [eid for eid in self.priors if eid not in active_event_ids]
        for eid in to_remove:
            del self.priors[eid]
