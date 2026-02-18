import logging
import math

import numpy as np

from config import config

logger = logging.getLogger(__name__)


class KellyCriterion:
    def __init__(self, bankroll: float | None = None):
        self.bankroll = bankroll or config.initial_bankroll
        self.max_bet_fraction = config.max_bet_fraction

    def calculate_bet_size(
        self,
        model_prob: float,
        market_price: float,
        confidence: float,
        bankroll: float | None = None,
    ) -> dict:
        br = bankroll or self.bankroll
        edge = model_prob - market_price

        if edge <= 0:
            return {
                "bet_size": 0,
                "kelly_fraction": 0,
                "adjusted_fraction": 0,
                "edge": edge,
                "expected_value": 0,
                "recommended": False,
                "reason": "No positive edge",
            }

        odds = (1.0 / market_price) - 1.0 if market_price > 0 else 0
        if odds <= 0:
            return {
                "bet_size": 0,
                "kelly_fraction": 0,
                "adjusted_fraction": 0,
                "edge": edge,
                "expected_value": 0,
                "recommended": False,
                "reason": "Invalid odds",
            }

        kelly_f = (model_prob * odds - (1 - model_prob)) / odds
        kelly_f = max(kelly_f, 0)

        fractional_kelly = kelly_f * 0.25
        confidence_adjusted = fractional_kelly * confidence
        capped = min(confidence_adjusted, self.max_bet_fraction)

        bet_size = round(br * capped, 2)

        ev = edge * (1.0 / market_price)
        ev_per_dollar = ev * capped

        min_edge = config.min_edge
        recommended = (
            edge >= min_edge and
            confidence >= 0.4 and
            kelly_f > 0.01 and
            bet_size >= 1.0
        )

        reason = "Signal meets all criteria" if recommended else self._rejection_reason(
            edge, min_edge, confidence, kelly_f, bet_size
        )

        return {
            "bet_size": bet_size if recommended else 0,
            "kelly_fraction": kelly_f,
            "adjusted_fraction": capped,
            "edge": edge,
            "expected_value": ev_per_dollar,
            "odds_decimal": 1.0 / market_price if market_price > 0 else 0,
            "recommended": recommended,
            "reason": reason,
            "bankroll": br,
        }

    def update_bankroll(self, pnl: float):
        self.bankroll += pnl
        logger.info(f"Bankroll updated: ${self.bankroll:.2f} (PnL: ${pnl:+.2f})")

    @staticmethod
    def _rejection_reason(
        edge: float, min_edge: float, confidence: float, kelly_f: float, bet_size: float,
    ) -> str:
        reasons = []
        if edge < min_edge:
            reasons.append(f"Edge {edge:.1%} < min {min_edge:.1%}")
        if confidence < 0.4:
            reasons.append(f"Confidence {confidence:.1%} < 40%")
        if kelly_f <= 0.01:
            reasons.append(f"Kelly fraction too small ({kelly_f:.3f})")
        if bet_size < 1.0:
            reasons.append(f"Bet size ${bet_size:.2f} < $1")
        return "; ".join(reasons) if reasons else "Unknown"
