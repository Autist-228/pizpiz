import logging
from datetime import datetime, timezone, timedelta
from collections import defaultdict

from config import config
from src.engine.paper_trading import PaperTradingEngine

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, paper_engine: PaperTradingEngine):
        self.paper_engine = paper_engine
        self.max_drawdown_pct = 0.20
        self.max_daily_trades = 15
        self.max_daily_loss = config.initial_bankroll * 0.10
        self.max_single_sport_pct = 0.40
        self.max_correlated_exposure = 0.30
        self.cooldown_after_losses = 3
        self.cooldown_minutes = 60
        self._last_loss_streak = 0
        self._cooldown_until: datetime | None = None

    def check_trade_allowed(self, signal: dict) -> tuple[bool, str]:
        if self._cooldown_until and datetime.now(timezone.utc) < self._cooldown_until:
            remaining = (self._cooldown_until - datetime.now(timezone.utc)).seconds // 60
            return False, f"Cooldown active ({remaining}min remaining after {self.cooldown_after_losses} consecutive losses)"

        stats = self.paper_engine.get_stats()

        drawdown_pct = self._calculate_drawdown(stats)
        if drawdown_pct >= self.max_drawdown_pct:
            return False, f"Max drawdown reached: {drawdown_pct:.1%} >= {self.max_drawdown_pct:.1%}"

        daily = self._get_today_stats()
        if daily["trades"] >= self.max_daily_trades:
            return False, f"Daily trade limit reached: {daily['trades']} >= {self.max_daily_trades}"

        if daily["loss"] >= self.max_daily_loss:
            return False, f"Daily loss limit reached: ${daily['loss']:.2f} >= ${self.max_daily_loss:.2f}"

        sport = signal.get("sport", "unknown")
        sport_exposure = self._get_sport_exposure(sport)
        bankroll = stats.get("bankroll", config.initial_bankroll)
        bet_size = signal.get("bet_size", 0)

        if bankroll > 0:
            new_exposure = (sport_exposure + bet_size) / bankroll
            if new_exposure > self.max_single_sport_pct:
                return False, f"Sport exposure limit: {sport} at {new_exposure:.1%} > {self.max_single_sport_pct:.1%}"

        total_open = sum(t.bet_size for t in self.paper_engine.get_open_trades())
        if bankroll > 0 and (total_open + bet_size) / bankroll > self.max_correlated_exposure * 3:
            return False, f"Total exposure too high: ${total_open + bet_size:.2f} / ${bankroll:.2f}"

        return True, "OK"

    def update_after_resolution(self, won: bool):
        if not won:
            self._last_loss_streak += 1
            if self._last_loss_streak >= self.cooldown_after_losses:
                self._cooldown_until = (
                    datetime.now(timezone.utc) + timedelta(minutes=self.cooldown_minutes)
                )
                logger.warning(
                    f"Cooldown activated: {self.cooldown_after_losses} consecutive losses. "
                    f"Pausing for {self.cooldown_minutes}min"
                )
        else:
            self._last_loss_streak = 0
            self._cooldown_until = None

    def adjust_bet_size(self, signal: dict) -> float:
        original = signal.get("bet_size", 0)
        stats = self.paper_engine.get_stats()
        bankroll = stats.get("bankroll", config.initial_bankroll)

        drawdown = self._calculate_drawdown(stats)
        if drawdown > 0.10:
            factor = 1.0 - (drawdown - 0.10) * 5
            factor = max(factor, 0.3)
            adjusted = original * factor
            logger.info(
                f"Risk adjustment: drawdown={drawdown:.1%}, "
                f"factor={factor:.2f}, ${original:.2f} -> ${adjusted:.2f}"
            )
            return adjusted

        if stats.get("win_rate", 0.5) > 0.65 and stats.get("resolved", 0) > 10:
            factor = min(1.0 + (stats["win_rate"] - 0.65) * 2, 1.3)
            adjusted = original * factor
            return min(adjusted, bankroll * config.max_bet_fraction)

        return original

    def get_risk_report(self) -> dict:
        stats = self.paper_engine.get_stats()
        open_trades = self.paper_engine.get_open_trades()

        sport_exposure: dict[str, float] = defaultdict(float)
        for t in open_trades:
            sport_exposure[t.sport] += t.bet_size

        bankroll = stats.get("bankroll", config.initial_bankroll)
        total_exposure = sum(t.bet_size for t in open_trades)

        return {
            "drawdown_pct": self._calculate_drawdown(stats),
            "max_drawdown_pct": self.max_drawdown_pct,
            "daily_trades": self._get_today_stats()["trades"],
            "max_daily_trades": self.max_daily_trades,
            "daily_loss": self._get_today_stats()["loss"],
            "max_daily_loss": self.max_daily_loss,
            "total_exposure": total_exposure,
            "exposure_pct": total_exposure / max(bankroll, 1),
            "sport_exposure": dict(sport_exposure),
            "loss_streak": self._last_loss_streak,
            "cooldown_active": (
                self._cooldown_until is not None
                and datetime.now(timezone.utc) < self._cooldown_until
            ),
            "bankroll": bankroll,
        }

    def _calculate_drawdown(self, stats: dict) -> float:
        bankroll = stats.get("bankroll", config.initial_bankroll)
        initial = config.initial_bankroll
        peak = max(initial, bankroll)

        if stats.get("total_pnl", 0) > 0:
            peak = initial + stats["total_pnl"]
            bankroll = peak

        if peak <= 0:
            return 0.0

        drawdown = (peak - bankroll) / peak
        return max(drawdown, 0.0)

    def _get_today_stats(self) -> dict:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        daily = self.paper_engine.get_daily_breakdown()

        for d in daily:
            if d.get("date") == today:
                return {
                    "trades": d.get("total", 0),
                    "loss": abs(min(d.get("pnl", 0), 0)),
                }

        open_today = [
            t for t in self.paper_engine.get_all_trades(limit=50)
            if t.timestamp.startswith(today)
        ]
        return {"trades": len(open_today), "loss": 0.0}

    def _get_sport_exposure(self, sport: str) -> float:
        open_trades = self.paper_engine.get_open_trades()
        return sum(t.bet_size for t in open_trades if t.sport == sport)
