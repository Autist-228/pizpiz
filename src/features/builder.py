import logging
import math
from datetime import datetime, timezone
from dataclasses import dataclass

import numpy as np

from src.data.polymarket import PolymarketMarket
from src.data.kalshi import KalshiMarket
from src.data.odds_api import BookmakerOdds, OddsAPIClient
from src.data.sports_stats import TeamStats
from src.data.news import NewsItem

logger = logging.getLogger(__name__)


@dataclass
class MatchFeatures:
    event_id: str
    platform: str
    question: str
    market_yes_price: float
    market_no_price: float
    end_date: datetime | None
    volume: float
    features: dict
    home_team: str
    away_team: str
    sport: str
    raw_market: dict


class FeatureBuilder:
    def __init__(self):
        self.odds_client = OddsAPIClient()

    def build_features(
        self,
        market_price: float,
        bookmaker_odds: dict,
        home_stats: TeamStats | None,
        away_stats: TeamStats | None,
        home_news_sentiment: dict,
        away_news_sentiment: dict,
        orderbook: dict,
        price_history: list[dict],
        hours_to_event: float,
        platform: str,
        sport: str,
    ) -> dict:
        f: dict = {}

        f.update(self._market_features(market_price, bookmaker_odds))
        f.update(self._team_features(home_stats, away_stats))
        f.update(self._news_features(home_news_sentiment, away_news_sentiment))
        f.update(self._orderbook_features(orderbook))
        f.update(self._price_history_features(price_history, market_price))
        f.update(self._temporal_features(hours_to_event))
        f.update(self._meta_features(platform, sport))

        return f

    def _market_features(self, market_price: float, bookmaker_odds: dict) -> dict:
        bk_prob = bookmaker_odds.get("home_prob", 0.5)
        n_bk = bookmaker_odds.get("n_bookmakers", 0)
        has_sharp = bookmaker_odds.get("has_sharp", False)

        edge = bk_prob - market_price if bk_prob > 0 else 0
        edge_pct = edge / max(market_price, 0.01)
        implied_overround = bk_prob + bookmaker_odds.get("away_prob", 0.5) + bookmaker_odds.get("draw_prob", 0)

        return {
            "market_yes_price": market_price,
            "bookmaker_prob": bk_prob,
            "bookmaker_away_prob": bookmaker_odds.get("away_prob", 0.5),
            "bookmaker_draw_prob": bookmaker_odds.get("draw_prob", 0),
            "edge_raw": edge,
            "edge_pct": edge_pct,
            "abs_edge": abs(edge),
            "edge_direction": 1 if edge > 0 else (-1 if edge < 0 else 0),
            "n_bookmakers": n_bk,
            "has_sharp_bookmaker": int(has_sharp),
            "implied_overround": implied_overround,
            "market_price_squared": market_price ** 2,
            "market_price_log": math.log(max(market_price, 0.001)),
            "market_price_complement": 1.0 - market_price,
            "price_distance_from_50": abs(market_price - 0.5),
            "is_favorite": int(market_price > 0.55),
            "is_underdog": int(market_price < 0.35),
            "is_coinflip": int(0.45 <= market_price <= 0.55),
        }

    def _team_features(self, home: TeamStats | None, away: TeamStats | None) -> dict:
        if home is None:
            home = TeamStats("", "", 0, 0, 0, 0.5, [], 0.5, 0, 0, 0.5, 0.5, 0, "N", 1500, {})
        if away is None:
            away = TeamStats("", "", 0, 0, 0, 0.5, [], 0.5, 0, 0, 0.5, 0.5, 0, "N", 1500, {})

        elo_diff = home.elo - away.elo
        elo_expected = 1.0 / (1.0 + 10 ** (-elo_diff / 400))

        return {
            "home_win_rate": home.win_rate,
            "away_win_rate": away.win_rate,
            "home_recent_wr": home.recent_win_rate,
            "away_recent_wr": away.recent_win_rate,
            "home_home_wr": home.home_win_rate,
            "away_away_wr": away.away_win_rate,
            "win_rate_diff": home.win_rate - away.win_rate,
            "recent_wr_diff": home.recent_win_rate - away.recent_win_rate,
            "home_streak": home.streak * (1 if home.streak_type == "W" else -1),
            "away_streak": away.streak * (1 if away.streak_type == "W" else -1),
            "streak_diff": (
                home.streak * (1 if home.streak_type == "W" else -1) -
                away.streak * (1 if away.streak_type == "W" else -1)
            ),
            "home_elo": home.elo,
            "away_elo": away.elo,
            "elo_diff": elo_diff,
            "elo_expected": elo_expected,
            "home_ppg": home.avg_points_for,
            "away_ppg": away.avg_points_for,
            "home_papg": home.avg_points_against,
            "away_papg": away.avg_points_against,
            "ppg_diff": home.avg_points_for - away.avg_points_for,
            "offensive_matchup": home.avg_points_for - away.avg_points_against,
            "defensive_matchup": away.avg_points_for - home.avg_points_against,
            "home_net_rating": home.avg_points_for - home.avg_points_against,
            "away_net_rating": away.avg_points_for - away.avg_points_against,
            "net_rating_diff": (
                (home.avg_points_for - home.avg_points_against) -
                (away.avg_points_for - away.avg_points_against)
            ),
            "home_games_played": home.wins + home.losses + home.draws,
            "away_games_played": away.wins + away.losses + away.draws,
            "total_games": home.wins + home.losses + away.wins + away.losses,
        }

    def _news_features(self, home_sentiment: dict, away_sentiment: dict) -> dict:
        return {
            "home_injury_signals": home_sentiment.get("injury_signals", 0),
            "away_injury_signals": away_sentiment.get("injury_signals", 0),
            "home_positive_news": home_sentiment.get("positive_signals", 0),
            "away_positive_news": away_sentiment.get("positive_signals", 0),
            "home_negative_news": home_sentiment.get("negative_signals", 0),
            "away_negative_news": away_sentiment.get("negative_signals", 0),
            "home_net_sentiment": home_sentiment.get("net_sentiment", 0),
            "away_net_sentiment": away_sentiment.get("net_sentiment", 0),
            "sentiment_diff": (
                home_sentiment.get("net_sentiment", 0) - away_sentiment.get("net_sentiment", 0)
            ),
            "any_injury_news": int(
                home_sentiment.get("has_injury_news", False) or
                away_sentiment.get("has_injury_news", False)
            ),
            "home_has_injury": int(home_sentiment.get("has_injury_news", False)),
            "away_has_injury": int(away_sentiment.get("has_injury_news", False)),
        }

    def _orderbook_features(self, orderbook: dict) -> dict:
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        bid_depth = sum(float(b.get("size", 0)) for b in bids[:10]) if bids else 0
        ask_depth = sum(float(a.get("size", 0)) for a in asks[:10]) if asks else 0

        best_bid = float(bids[0].get("price", 0)) if bids else 0
        best_ask = float(asks[0].get("price", 1)) if asks else 1

        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2 if (best_bid + best_ask) > 0 else 0.5

        total_depth = bid_depth + ask_depth
        bid_pressure = bid_depth / total_depth if total_depth > 0 else 0.5

        top_bid_size = float(bids[0].get("size", 0)) if bids else 0
        top_ask_size = float(asks[0].get("size", 0)) if asks else 0
        size_imbalance = (top_bid_size - top_ask_size) / max(top_bid_size + top_ask_size, 1)

        return {
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "spread": spread,
            "spread_pct": spread / max(mid_price, 0.001),
            "mid_price": mid_price,
            "bid_pressure": bid_pressure,
            "ask_pressure": 1.0 - bid_pressure,
            "depth_ratio": bid_depth / max(ask_depth, 1),
            "total_depth": total_depth,
            "n_bid_levels": len(bids),
            "n_ask_levels": len(asks),
            "size_imbalance": size_imbalance,
            "top_bid_size": top_bid_size,
            "top_ask_size": top_ask_size,
            "has_orderbook": int(bool(bids or asks)),
        }

    def _price_history_features(self, history: list[dict], current_price: float) -> dict:
        if not history:
            return {
                "price_change_1h": 0, "price_change_6h": 0, "price_change_24h": 0,
                "price_volatility": 0, "price_trend": 0, "price_momentum": 0,
                "price_mean_reversion": 0, "max_price": current_price,
                "min_price": current_price, "price_range": 0,
                "volume_trend": 0, "smart_money_signal": 0,
            }

        prices = []
        for h in history:
            p = h.get("p") or h.get("price") or h.get("yes")
            if p is not None:
                prices.append(float(p))

        if not prices:
            return {
                "price_change_1h": 0, "price_change_6h": 0, "price_change_24h": 0,
                "price_volatility": 0, "price_trend": 0, "price_momentum": 0,
                "price_mean_reversion": 0, "max_price": current_price,
                "min_price": current_price, "price_range": 0,
                "volume_trend": 0, "smart_money_signal": 0,
            }

        n = len(prices)
        last_price = prices[-1] if prices else current_price

        change_1h = current_price - prices[-min(6, n)] if n >= 2 else 0
        change_6h = current_price - prices[-min(36, n)] if n >= 2 else 0
        change_24h = current_price - prices[-min(144, n)] if n >= 2 else 0

        returns = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        volatility = float(np.std(returns)) if len(returns) > 1 else 0

        if n >= 5:
            recent = prices[-5:]
            x = list(range(5))
            slope = np.polyfit(x, recent, 1)[0] if len(recent) == 5 else 0
            trend = float(slope)
        else:
            trend = 0

        short_ma = np.mean(prices[-min(5, n):]) if n >= 2 else current_price
        long_ma = np.mean(prices[-min(20, n):]) if n >= 2 else current_price
        momentum = float(short_ma - long_ma)

        mean_price = np.mean(prices)
        mean_reversion = float(mean_price - current_price)

        max_price = max(prices)
        min_price = min(prices)

        if n >= 10:
            recent_vol = sum(abs(r) for r in returns[-5:]) / 5 if len(returns) >= 5 else 0
            older_vol = sum(abs(r) for r in returns[:5]) / 5 if len(returns) >= 5 else 0
            vol_trend = recent_vol - older_vol
        else:
            vol_trend = 0

        if n >= 10 and volatility > 0:
            recent_move = abs(change_1h)
            smart_money = recent_move / max(volatility, 0.001)
        else:
            smart_money = 0

        return {
            "price_change_1h": change_1h,
            "price_change_6h": change_6h,
            "price_change_24h": change_24h,
            "price_volatility": volatility,
            "price_trend": trend,
            "price_momentum": momentum,
            "price_mean_reversion": mean_reversion,
            "max_price": max_price,
            "min_price": min_price,
            "price_range": max_price - min_price,
            "volume_trend": vol_trend,
            "smart_money_signal": smart_money,
        }

    def _temporal_features(self, hours_to_event: float) -> dict:
        now = datetime.now(timezone.utc)
        hour = now.hour
        day_of_week = now.weekday()

        return {
            "hours_to_event": hours_to_event,
            "minutes_to_event": hours_to_event * 60,
            "log_hours_to_event": math.log(max(hours_to_event, 0.1)),
            "is_close_to_event": int(hours_to_event < 2),
            "is_far_from_event": int(hours_to_event > 24),
            "current_hour_utc": hour,
            "is_us_daytime": int(hour >= 13 or hour <= 4),
            "is_us_evening": int(0 <= hour <= 6),
            "day_of_week": day_of_week,
            "is_weekend": int(day_of_week >= 5),
            "is_game_day_evening": int(
                (day_of_week in [0, 2, 4] and 23 <= hour) or
                (day_of_week in [1, 3, 5] and hour <= 5)
            ),
        }

    def _meta_features(self, platform: str, sport: str) -> dict:
        sport_liquidity = {
            "basketball_nba": 0.9,
            "americanfootball_nfl": 0.95,
            "soccer_epl": 0.8,
            "soccer_usa_mls": 0.5,
            "mma_mixed_martial_arts": 0.6,
            "icehockey_nhl": 0.7,
        }

        return {
            "platform_polymarket": int(platform == "polymarket"),
            "platform_kalshi": int(platform == "kalshi"),
            "sport_nba": int("nba" in sport.lower()),
            "sport_nfl": int("nfl" in sport.lower()),
            "sport_soccer": int("soccer" in sport.lower() or "epl" in sport.lower()),
            "sport_mma": int("mma" in sport.lower() or "ufc" in sport.lower()),
            "sport_nhl": int("nhl" in sport.lower()),
            "sport_liquidity_score": sport_liquidity.get(sport, 0.5),
        }

    def extract_teams_from_question(self, question: str) -> tuple[str, str]:
        separators = [" vs ", " vs. ", " v ", " @ ", " at "]
        q_lower = question.lower()

        for sep in separators:
            if sep in q_lower:
                idx = q_lower.index(sep)
                parts_before = question[:idx].strip()
                parts_after = question[idx + len(sep):].strip()

                home = self._clean_team_name(parts_after)
                away = self._clean_team_name(parts_before)
                if home and away:
                    return home, away

        words = question.split()
        for i, word in enumerate(words):
            if word.lower() in ("win", "beat", "defeat", "over"):
                team_a = " ".join(words[max(0, i - 3):i])
                team_b = " ".join(words[i + 1:min(len(words), i + 4)])
                if team_a and team_b:
                    return self._clean_team_name(team_a), self._clean_team_name(team_b)

        return question[:50], ""

    @staticmethod
    def _clean_team_name(name: str) -> str:
        remove_words = [
            "will", "the", "to", "win", "beat", "defeat", "?", "!", ".",
            "game", "match", "tonight", "today", "tomorrow",
        ]
        words = name.split()
        cleaned = [w for w in words if w.lower().strip("?!.,") not in remove_words]
        return " ".join(cleaned).strip("?!., ")
