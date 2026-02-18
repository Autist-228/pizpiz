import logging
from dataclasses import dataclass

import numpy as np

from src.data.polymarket import PolymarketMarket
from src.data.kalshi import KalshiMarket

logger = logging.getLogger(__name__)


@dataclass
class CrossMarketOpportunity:
    event_description: str
    polymarket_price: float | None
    kalshi_price: float | None
    bookmaker_prob: float
    price_diff: float
    best_platform: str
    best_price: float
    arb_opportunity: bool
    arb_profit: float
    recommendation: str
    confidence: float


class CrossMarketAnalyzer:
    def __init__(self):
        self.match_cache: dict[str, dict] = {}

    def analyze(
        self,
        poly_markets: list[PolymarketMarket],
        kalshi_markets: list[KalshiMarket],
        bookmaker_odds: dict,
    ) -> list[CrossMarketOpportunity]:
        matched = self._match_markets(poly_markets, kalshi_markets)
        opportunities: list[CrossMarketOpportunity] = []

        for match in matched:
            opp = self._analyze_pair(match, bookmaker_odds)
            if opp:
                opportunities.append(opp)

        opportunities.sort(key=lambda x: abs(x.price_diff), reverse=True)
        return opportunities

    def select_best_platform(
        self,
        event_question: str,
        poly_price: float | None,
        kalshi_price: float | None,
        model_prob: float,
        poly_volume: float = 0,
        kalshi_volume: int = 0,
        poly_spread: float = 0,
        kalshi_spread: float = 0,
    ) -> dict:
        scores: dict[str, float] = {}

        if poly_price is not None:
            poly_edge = model_prob - poly_price
            poly_score = (
                abs(poly_edge) * 0.4 +
                min(poly_volume / 100000, 1.0) * 0.3 +
                max(0, 0.05 - poly_spread) * 5 * 0.15 +
                0.15
            )
            scores["polymarket"] = poly_score

        if kalshi_price is not None:
            kalshi_edge = model_prob - kalshi_price
            kalshi_score = (
                abs(kalshi_edge) * 0.4 +
                min(kalshi_volume / 50000, 1.0) * 0.3 +
                max(0, 0.05 - kalshi_spread) * 5 * 0.15 +
                0.1
            )
            scores["kalshi"] = kalshi_score

        if not scores:
            return {
                "best_platform": "none",
                "scores": {},
                "recommendation": "No platform available",
            }

        best = max(scores, key=scores.get)

        if poly_price is not None and kalshi_price is not None:
            cross_arb = self._check_cross_arb(poly_price, kalshi_price, model_prob)
        else:
            cross_arb = {"arb_exists": False, "arb_profit": 0}

        return {
            "best_platform": best,
            "scores": scores,
            "cross_arb": cross_arb,
            "recommendation": self._generate_recommendation(
                poly_price, kalshi_price, model_prob, best, cross_arb.get("arb_exists", False)
            ),
        }

    def _match_markets(
        self,
        poly: list[PolymarketMarket],
        kalshi: list[KalshiMarket],
    ) -> list[dict]:
        matches: list[dict] = []
        used_kalshi: set[str] = set()

        for pm in poly:
            best_match = None
            best_score = 0.0

            for km in kalshi:
                if km.ticker in used_kalshi:
                    continue
                score = self._similarity_score(pm.question, km.title)
                if score > best_score and score > 0.3:
                    best_score = score
                    best_match = km

            match_entry: dict = {"polymarket": pm, "kalshi": None, "match_score": 0}
            if best_match:
                match_entry["kalshi"] = best_match
                match_entry["match_score"] = best_score
                used_kalshi.add(best_match.ticker)

            matches.append(match_entry)

        for km in kalshi:
            if km.ticker not in used_kalshi:
                matches.append({
                    "polymarket": None,
                    "kalshi": km,
                    "match_score": 0,
                })

        return matches

    def _analyze_pair(self, match: dict, bookmaker_odds: dict) -> CrossMarketOpportunity | None:
        pm: PolymarketMarket | None = match.get("polymarket")
        km: KalshiMarket | None = match.get("kalshi")

        if pm is None and km is None:
            return None

        poly_price = pm.outcome_yes_price if pm else None
        kalshi_price = km.yes_price if km else None
        description = pm.question if pm else (km.title if km else "")

        bk_prob = bookmaker_odds.get("home_prob", 0.5)

        if poly_price is not None and kalshi_price is not None:
            price_diff = abs(poly_price - kalshi_price)
        elif poly_price is not None:
            price_diff = abs(poly_price - bk_prob)
        elif kalshi_price is not None:
            price_diff = abs(kalshi_price - bk_prob)
        else:
            return None

        if poly_price is not None and kalshi_price is not None:
            arb, arb_profit = self._check_pure_arb(poly_price, kalshi_price)
        else:
            arb, arb_profit = False, 0.0

        if poly_price is not None and kalshi_price is not None:
            if bk_prob > 0.5:
                best_platform = "polymarket" if poly_price < kalshi_price else "kalshi"
                best_price = min(poly_price, kalshi_price)
            else:
                best_platform = "polymarket" if (1 - poly_price) < (1 - kalshi_price) else "kalshi"
                best_price = max(poly_price, kalshi_price)
        elif poly_price is not None:
            best_platform = "polymarket"
            best_price = poly_price
        else:
            best_platform = "kalshi"
            best_price = kalshi_price

        confidence = 0.5
        if price_diff > 0.05:
            confidence += 0.2
        if arb:
            confidence += 0.3

        recommendation = self._generate_recommendation(
            poly_price, kalshi_price, bk_prob, best_platform, arb
        )

        return CrossMarketOpportunity(
            event_description=description,
            polymarket_price=poly_price,
            kalshi_price=kalshi_price,
            bookmaker_prob=bk_prob,
            price_diff=price_diff,
            best_platform=best_platform,
            best_price=best_price,
            arb_opportunity=arb,
            arb_profit=arb_profit,
            recommendation=recommendation,
            confidence=confidence,
        )

    @staticmethod
    def _check_pure_arb(price_a: float, price_b: float) -> tuple[bool, float]:
        cost_yes_a = price_a
        cost_no_b = 1.0 - price_b
        total_cost = cost_yes_a + cost_no_b
        if total_cost < 1.0:
            profit = 1.0 - total_cost
            return True, profit

        cost_no_a = 1.0 - price_a
        cost_yes_b = price_b
        total_cost = cost_no_a + cost_yes_b
        if total_cost < 1.0:
            profit = 1.0 - total_cost
            return True, profit

        return False, 0.0

    @staticmethod
    def _check_cross_arb(poly_price: float, kalshi_price: float, model_prob: float) -> dict:
        _, arb_profit = CrossMarketAnalyzer._check_pure_arb(poly_price, kalshi_price)
        return {
            "arb_exists": arb_profit > 0,
            "arb_profit": arb_profit,
            "price_discrepancy": abs(poly_price - kalshi_price),
            "model_favors": "polymarket" if abs(model_prob - poly_price) < abs(model_prob - kalshi_price) else "kalshi",
        }

    @staticmethod
    def _similarity_score(text_a: str, text_b: str) -> float:
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        stopwords = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "will", "?", "!"}
        words_a -= stopwords
        words_b -= stopwords
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    @staticmethod
    def _generate_recommendation(
        poly_price: float | None,
        kalshi_price: float | None,
        bk_prob: float,
        best_platform: str,
        arb: bool,
    ) -> str:
        if arb:
            return f"ARBITRAGE: Cross-market arb detected between Polymarket and Kalshi"

        best_price = poly_price if best_platform == "polymarket" else kalshi_price
        if best_price is None:
            return f"Trade on {best_platform}"

        edge = abs(bk_prob - best_price)
        if edge > 0.1:
            return f"STRONG: {best_platform} price significantly off from bookmakers (edge: {edge:.1%})"
        elif edge > 0.05:
            return f"MODERATE: {best_platform} shows decent edge vs bookmakers ({edge:.1%})"
        else:
            return f"WEAK: Small edge on {best_platform} ({edge:.1%})"
