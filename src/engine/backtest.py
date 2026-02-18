import logging
import json
import sqlite3
import math
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from config import config
from src.models.ensemble import ModelEnsemble
from src.features.builder import FeatureBuilder
from src.engine.kelly import KellyCriterion

logger = logging.getLogger(__name__)

HISTORY_DB = "data/historical.db"


@dataclass
class BacktestTrade:
    market_id: str
    platform: str
    question: str
    side: str
    entry_price: float
    model_prob: float
    edge: float
    confidence: float
    bet_size: float
    result: int
    pnl: float
    bankroll_after: float


@dataclass
class BacktestResult:
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    roi: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_edge: float = 0.0
    avg_confidence: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    final_bankroll: float = 0.0
    initial_bankroll: float = 0.0
    equity_curve: list[float] = field(default_factory=list)
    daily_returns: list[float] = field(default_factory=list)
    trades: list[BacktestTrade] = field(default_factory=list)
    per_sport: dict = field(default_factory=dict)


class BacktestEngine:
    def __init__(self):
        self.ensemble = ModelEnsemble()
        self.feature_builder = FeatureBuilder()
        self.kelly = KellyCriterion()

    def run_backtest(
        self,
        initial_bankroll: float = 0,
        min_edge: float = 0,
        max_bet_fraction: float = 0,
    ) -> BacktestResult:
        bankroll = initial_bankroll or config.initial_bankroll
        min_edge = min_edge or config.min_edge
        max_bet_frac = max_bet_fraction or config.max_bet_fraction

        self.ensemble.load_models()

        raw_data = self._load_data()
        if not raw_data:
            logger.warning("No historical data for backtesting")
            return BacktestResult(initial_bankroll=bankroll, final_bankroll=bankroll)

        logger.info(f"Backtesting on {len(raw_data)} resolved markets")

        result = BacktestResult(
            initial_bankroll=bankroll,
            final_bankroll=bankroll,
        )
        result.equity_curve.append(bankroll)

        current_bankroll = bankroll
        peak_bankroll = bankroll

        for entry in raw_data:
            try:
                trade = self._process_entry(
                    entry, current_bankroll, min_edge, max_bet_frac
                )
                if trade is None:
                    continue

                result.trades.append(trade)
                result.total_trades += 1
                current_bankroll = trade.bankroll_after
                result.equity_curve.append(current_bankroll)

                if trade.pnl > 0:
                    result.wins += 1
                else:
                    result.losses += 1

                peak_bankroll = max(peak_bankroll, current_bankroll)
                drawdown = peak_bankroll - current_bankroll
                drawdown_pct = drawdown / max(peak_bankroll, 1)
                result.max_drawdown = max(result.max_drawdown, drawdown)
                result.max_drawdown_pct = max(result.max_drawdown_pct, drawdown_pct)

                sport = entry.get("sport", "unknown")
                if sport not in result.per_sport:
                    result.per_sport[sport] = {
                        "trades": 0, "wins": 0, "losses": 0, "pnl": 0.0,
                    }
                result.per_sport[sport]["trades"] += 1
                result.per_sport[sport]["pnl"] += trade.pnl
                if trade.pnl > 0:
                    result.per_sport[sport]["wins"] += 1
                else:
                    result.per_sport[sport]["losses"] += 1

                if current_bankroll <= bankroll * 0.3:
                    logger.warning("Backtest stopped: bankroll below 30% of initial")
                    break

            except Exception as e:
                logger.debug(f"Backtest entry error: {e}")
                continue

        result.final_bankroll = current_bankroll
        result.total_pnl = current_bankroll - bankroll
        result.roi = result.total_pnl / max(bankroll, 1)
        result.win_rate = result.wins / max(result.total_trades, 1)

        win_pnls = [t.pnl for t in result.trades if t.pnl > 0]
        loss_pnls = [t.pnl for t in result.trades if t.pnl <= 0]

        result.avg_win = sum(win_pnls) / max(len(win_pnls), 1)
        result.avg_loss = sum(loss_pnls) / max(len(loss_pnls), 1)
        result.best_trade = max(win_pnls) if win_pnls else 0
        result.worst_trade = min(loss_pnls) if loss_pnls else 0

        gross_profit = sum(win_pnls)
        gross_loss = abs(sum(loss_pnls))
        result.profit_factor = gross_profit / max(gross_loss, 0.01)

        result.avg_edge = (
            sum(t.edge for t in result.trades) / max(result.total_trades, 1)
        )
        result.avg_confidence = (
            sum(t.confidence for t in result.trades) / max(result.total_trades, 1)
        )

        if len(result.equity_curve) > 2:
            returns = []
            for i in range(1, len(result.equity_curve)):
                prev = result.equity_curve[i - 1]
                curr = result.equity_curve[i]
                if prev > 0:
                    returns.append((curr - prev) / prev)
            result.daily_returns = returns
            if returns and np.std(returns) > 0:
                result.sharpe_ratio = (
                    float(np.mean(returns) / np.std(returns)) * math.sqrt(252)
                )

        for sport_key, sport_data in result.per_sport.items():
            sport_data["win_rate"] = (
                sport_data["wins"] / max(sport_data["trades"], 1)
            )

        logger.info(
            f"Backtest complete: {result.total_trades} trades, "
            f"WR={result.win_rate:.1%}, PnL=${result.total_pnl:+.2f}, "
            f"Sharpe={result.sharpe_ratio:.2f}, MaxDD={result.max_drawdown_pct:.1%}"
        )

        return result

    def _process_entry(
        self,
        entry: dict,
        bankroll: float,
        min_edge: float,
        max_bet_frac: float,
    ) -> BacktestTrade | None:
        yes_price = entry.get("yes_price", 0.5)
        if yes_price <= 0.02 or yes_price >= 0.98:
            return None

        actual_result = entry.get("result")
        if actual_result is None:
            return None

        bk_home = entry.get("bookmaker_home_prob", 0)
        if bk_home <= 0:
            bk_home = 0.5

        bookmaker_odds = {
            "home_prob": bk_home,
            "away_prob": entry.get("bookmaker_away_prob", 0.5),
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

        ensemble_result = self.ensemble.predict(features)
        model_prob = ensemble_result["ensemble_prob"]
        confidence = ensemble_result["model_agreement"]

        edge = model_prob - yes_price

        if edge > 0:
            side = "YES"
            price = yes_price
            prob = model_prob
        else:
            side = "NO"
            price = 1.0 - yes_price
            prob = 1.0 - model_prob
            edge = abs(edge)

        if edge < min_edge:
            return None

        kelly_result = self.kelly.calculate_bet_size(prob, price, confidence)
        bet_size = min(kelly_result["bet_size"], bankroll * max_bet_frac)
        bet_size = min(bet_size, bankroll * 0.95)

        if bet_size <= 0:
            return None

        if side == "YES":
            won = actual_result == 1
        else:
            won = actual_result == 0

        if won:
            pnl = bet_size * (1.0 / price - 1)
        else:
            pnl = -bet_size

        return BacktestTrade(
            market_id=entry.get("market_id", ""),
            platform=entry.get("platform", ""),
            question=entry.get("question", ""),
            side=side,
            entry_price=price,
            model_prob=prob,
            edge=edge,
            confidence=confidence,
            bet_size=bet_size,
            result=1 if won else 0,
            pnl=pnl,
            bankroll_after=bankroll + pnl,
        )

    def _load_data(self) -> list[dict]:
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

        return [
            {
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
            }
            for r in rows
        ]

    def format_report(self, result: BacktestResult) -> str:
        lines = [
            "BACKTEST REPORT",
            "=" * 40,
            f"Period: {result.total_trades} markets",
            f"Initial: ${result.initial_bankroll:.2f}",
            f"Final: ${result.final_bankroll:.2f}",
            "",
            "PERFORMANCE",
            "-" * 30,
            f"Total PnL: ${result.total_pnl:+.2f}",
            f"ROI: {result.roi:+.1%}",
            f"Win Rate: {result.win_rate:.1%} ({result.wins}W/{result.losses}L)",
            f"Sharpe Ratio: {result.sharpe_ratio:.2f}",
            f"Max Drawdown: {result.max_drawdown_pct:.1%} (${result.max_drawdown:.2f})",
            f"Profit Factor: {result.profit_factor:.2f}",
            "",
            "TRADE STATS",
            "-" * 30,
            f"Avg Win: ${result.avg_win:.2f}",
            f"Avg Loss: ${result.avg_loss:.2f}",
            f"Best Trade: ${result.best_trade:.2f}",
            f"Worst Trade: ${result.worst_trade:.2f}",
            f"Avg Edge: {result.avg_edge:.1%}",
            f"Avg Confidence: {result.avg_confidence:.0%}",
        ]

        if result.per_sport:
            lines.append("")
            lines.append("PER SPORT")
            lines.append("-" * 30)
            for sport, data in sorted(
                result.per_sport.items(),
                key=lambda x: x[1]["pnl"],
                reverse=True,
            ):
                wr = data["win_rate"]
                lines.append(
                    f"{sport}: {data['trades']}T {wr:.0%}WR ${data['pnl']:+.2f}"
                )

        return "\n".join(lines)
