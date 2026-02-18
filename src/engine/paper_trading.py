import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from pathlib import Path

from config import config

logger = logging.getLogger(__name__)


@dataclass
class PaperTrade:
    trade_id: str
    event_id: str
    platform: str
    question: str
    side: str
    entry_price: float
    model_prob: float
    meta_edge: float
    confidence: float
    bet_size: float
    kelly_fraction: float
    timestamp: str
    event_end_time: str
    status: str
    exit_price: float | None = None
    pnl: float | None = None
    result: str | None = None
    resolved_at: str | None = None
    sport: str = ""
    home_team: str = ""
    away_team: str = ""
    signals: str = ""


class PaperTradingEngine:
    def __init__(self):
        self.db_path = config.db_path
        self._init_db()

    def _init_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                event_id TEXT,
                platform TEXT,
                question TEXT,
                side TEXT,
                entry_price REAL,
                model_prob REAL,
                meta_edge REAL,
                confidence REAL,
                bet_size REAL,
                kelly_fraction REAL,
                timestamp TEXT,
                event_end_time TEXT,
                status TEXT,
                exit_price REAL,
                pnl REAL,
                result TEXT,
                resolved_at TEXT,
                sport TEXT,
                home_team TEXT,
                away_team TEXT,
                signals TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_stats (
                date TEXT PRIMARY KEY,
                total_trades INTEGER,
                wins INTEGER,
                losses INTEGER,
                pending INTEGER,
                total_pnl REAL,
                win_rate REAL,
                avg_edge REAL,
                avg_confidence REAL,
                bankroll REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scan_log (
                scan_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                poly_markets_found INTEGER,
                kalshi_markets_found INTEGER,
                signals_generated INTEGER,
                trades_placed INTEGER,
                errors TEXT
            )
        """)
        conn.commit()
        conn.close()

    def place_trade(self, trade: PaperTrade) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO trades VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                trade.trade_id, trade.event_id, trade.platform, trade.question,
                trade.side, trade.entry_price, trade.model_prob, trade.meta_edge,
                trade.confidence, trade.bet_size, trade.kelly_fraction,
                trade.timestamp, trade.event_end_time, trade.status,
                trade.exit_price, trade.pnl, trade.result, trade.resolved_at,
                trade.sport, trade.home_team, trade.away_team, trade.signals,
            ))
            conn.commit()
            conn.close()
            logger.info(f"Paper trade placed: {trade.trade_id} | {trade.question[:50]} | ${trade.bet_size:.2f}")
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"Trade {trade.trade_id} already exists")
            return False
        except Exception as e:
            logger.error(f"Error placing trade: {e}")
            return False

    def resolve_trade(self, trade_id: str, outcome: bool) -> PaperTrade | None:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT * FROM trades WHERE trade_id = ?", (trade_id,)).fetchone()
        if not row:
            conn.close()
            return None

        trade = self._row_to_trade(row)

        if trade.side == "YES":
            won = outcome
        else:
            won = not outcome

        if won:
            payout = trade.bet_size * (1.0 / trade.entry_price - 1)
            pnl = payout
            result = "WIN"
        else:
            pnl = -trade.bet_size
            result = "LOSS"

        now = datetime.now(timezone.utc).isoformat()
        exit_price = 1.0 if outcome else 0.0

        conn.execute("""
            UPDATE trades SET
                status = 'resolved',
                exit_price = ?,
                pnl = ?,
                result = ?,
                resolved_at = ?
            WHERE trade_id = ?
        """, (exit_price, pnl, result, now, trade_id))
        conn.commit()
        conn.close()

        trade.status = "resolved"
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.result = result
        trade.resolved_at = now

        return trade

    def get_open_trades(self) -> list[PaperTrade]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT * FROM trades WHERE status = 'open' ORDER BY timestamp DESC"
        ).fetchall()
        conn.close()
        return [self._row_to_trade(r) for r in rows]

    def get_all_trades(self, limit: int = 100) -> list[PaperTrade]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [self._row_to_trade(r) for r in rows]

    def get_resolved_trades(self, limit: int = 100) -> list[PaperTrade]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT * FROM trades WHERE status = 'resolved' ORDER BY resolved_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [self._row_to_trade(r) for r in rows]

    def get_stats(self) -> dict:
        conn = sqlite3.connect(self.db_path)

        total = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        resolved = conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'resolved'").fetchone()[0]
        wins = conn.execute("SELECT COUNT(*) FROM trades WHERE result = 'WIN'").fetchone()[0]
        losses = conn.execute("SELECT COUNT(*) FROM trades WHERE result = 'LOSS'").fetchone()[0]
        pending = conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'open'").fetchone()[0]

        pnl_row = conn.execute("SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE status = 'resolved'").fetchone()
        total_pnl = pnl_row[0] if pnl_row else 0

        avg_edge_row = conn.execute("SELECT COALESCE(AVG(meta_edge), 0) FROM trades").fetchone()
        avg_edge = avg_edge_row[0] if avg_edge_row else 0

        avg_conf_row = conn.execute("SELECT COALESCE(AVG(confidence), 0) FROM trades").fetchone()
        avg_conf = avg_conf_row[0] if avg_conf_row else 0

        biggest_win_row = conn.execute(
            "SELECT COALESCE(MAX(pnl), 0) FROM trades WHERE result = 'WIN'"
        ).fetchone()
        biggest_win = biggest_win_row[0] if biggest_win_row else 0

        biggest_loss_row = conn.execute(
            "SELECT COALESCE(MIN(pnl), 0) FROM trades WHERE result = 'LOSS'"
        ).fetchone()
        biggest_loss = biggest_loss_row[0] if biggest_loss_row else 0

        conn.close()

        win_rate = wins / max(resolved, 1)
        roi = total_pnl / max(config.initial_bankroll, 1)

        return {
            "total_trades": total,
            "resolved": resolved,
            "wins": wins,
            "losses": losses,
            "pending": pending,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "roi": roi,
            "avg_edge": avg_edge,
            "avg_confidence": avg_conf,
            "biggest_win": biggest_win,
            "biggest_loss": biggest_loss,
            "bankroll": config.initial_bankroll + total_pnl,
            "initial_bankroll": config.initial_bankroll,
        }

    def get_daily_breakdown(self) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT
                DATE(timestamp) as day,
                COUNT(*) as total,
                SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                COALESCE(SUM(pnl), 0) as pnl
            FROM trades
            WHERE status = 'resolved'
            GROUP BY DATE(timestamp)
            ORDER BY day DESC
            LIMIT 30
        """).fetchall()
        conn.close()

        return [
            {
                "date": r[0],
                "total": r[1],
                "wins": r[2],
                "losses": r[3],
                "pnl": r[4],
                "win_rate": r[2] / max(r[1], 1),
            }
            for r in rows
        ]

    def log_scan(
        self,
        poly_found: int,
        kalshi_found: int,
        signals: int,
        trades: int,
        errors: str = "",
    ):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO scan_log (timestamp, poly_markets_found, kalshi_markets_found,
                                  signals_generated, trades_placed, errors)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            poly_found, kalshi_found, signals, trades, errors,
        ))
        conn.commit()
        conn.close()

    def has_existing_trade(self, event_id: str, platform: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE event_id = ? AND platform = ? AND status = 'open'",
            (event_id, platform),
        ).fetchone()
        conn.close()
        return row[0] > 0

    @staticmethod
    def _row_to_trade(row: tuple) -> PaperTrade:
        return PaperTrade(
            trade_id=row[0], event_id=row[1], platform=row[2], question=row[3],
            side=row[4], entry_price=row[5], model_prob=row[6], meta_edge=row[7],
            confidence=row[8], bet_size=row[9], kelly_fraction=row[10],
            timestamp=row[11], event_end_time=row[12], status=row[13],
            exit_price=row[14], pnl=row[15], result=row[16], resolved_at=row[17],
            sport=row[18], home_team=row[19], away_team=row[20], signals=row[21],
        )
