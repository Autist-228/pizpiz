import asyncio
import logging
from datetime import datetime, timezone

from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes

from config import config
from src.engine.paper_trading import PaperTradingEngine

logger = logging.getLogger(__name__)


class TelegramBot:
    def __init__(self, paper_engine: PaperTradingEngine):
        self.paper_engine = paper_engine
        self.bot: Bot | None = None
        self.app: Application | None = None
        self.chat_id = config.telegram_chat_id

    async def initialize(self):
        if not config.telegram_bot_token:
            logger.warning("No TELEGRAM_BOT_TOKEN set, Telegram bot disabled")
            return

        self.app = Application.builder().token(config.telegram_bot_token).build()
        self.bot = self.app.bot

        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("stats", self._cmd_stats))
        self.app.add_handler(CommandHandler("trades", self._cmd_trades))
        self.app.add_handler(CommandHandler("open", self._cmd_open))
        self.app.add_handler(CommandHandler("daily", self._cmd_daily))
        self.app.add_handler(CommandHandler("help", self._cmd_help))

        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()
        logger.info("Telegram bot started")

    async def shutdown(self):
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()

    async def send_signal(self, signal: dict):
        if not self.bot or not self.chat_id:
            return

        side = signal.get("side", "YES")
        edge = signal.get("meta_edge", 0)
        confidence = signal.get("confidence", 0)
        bet_size = signal.get("bet_size", 0)
        platform = signal.get("platform", "")
        question = signal.get("question", "")
        model_prob = signal.get("model_prob", 0)
        market_price = signal.get("market_price", 0)
        home_team = signal.get("home_team", "")
        away_team = signal.get("away_team", "")
        sport = signal.get("sport", "")

        strength = "STRONG" if edge > 0.12 else ("MODERATE" if edge > 0.08 else "WEAK")

        signals_detail = signal.get("signals_detail", {})
        ensemble = signals_detail.get("ensemble_prob", model_prob)
        bookmaker = signals_detail.get("bookmaker_edge", 0)
        nlp_sig = signals_detail.get("nlp_signal", 0)
        micro_sig = signals_detail.get("micro_signal", 0)
        bayesian = signals_detail.get("bayesian_prob", model_prob)

        msg = (
            f"{'🔥' if strength == 'STRONG' else '⚡' if strength == 'MODERATE' else '📊'} "
            f"NEW SIGNAL — {strength}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📋 {question[:80]}\n"
            f"🏟 {sport.upper()} | {platform.upper()}\n"
            f"⚔️ {home_team} vs {away_team}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📌 Side: {side}\n"
            f"💰 Market: {market_price:.1%} → Model: {model_prob:.1%}\n"
            f"📈 Edge: {edge:+.1%} | Conf: {confidence:.0%}\n"
            f"💵 Bet: ${bet_size:.2f}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🧠 L1 Arb edge: {bookmaker:+.1%}\n"
            f"🤖 L2-3 ML ensemble: {ensemble:.1%}\n"
            f"📰 L4 NLP: {nlp_sig:+.2f}\n"
            f"📊 L5 Micro: {micro_sig:+.2f}\n"
            f"🔄 L6 Bayesian: {bayesian:.1%}\n"
            f"🌐 L7 Platform: {platform}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )

        await self._send(msg)

    async def send_trade_result(self, trade):
        if not self.bot or not self.chat_id:
            return

        emoji = "✅" if trade.result == "WIN" else "❌"
        pnl_emoji = "💚" if (trade.pnl or 0) > 0 else "💔"

        msg = (
            f"{emoji} TRADE RESOLVED\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📋 {trade.question[:80]}\n"
            f"📌 {trade.side} @ {trade.entry_price:.1%}\n"
            f"🎯 Result: {trade.result}\n"
            f"{pnl_emoji} PnL: ${trade.pnl:+.2f}\n"
            f"📊 Edge was: {trade.meta_edge:+.1%}\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )

        await self._send(msg)

    async def send_daily_summary(self, stats: dict):
        if not self.bot or not self.chat_id:
            return

        wr = stats.get("win_rate", 0)
        pnl = stats.get("total_pnl", 0)
        bankroll = stats.get("bankroll", config.initial_bankroll)
        roi = stats.get("roi", 0)

        msg = (
            f"📊 DAILY SUMMARY\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📈 Total trades: {stats.get('total_trades', 0)}\n"
            f"✅ Wins: {stats.get('wins', 0)} | ❌ Losses: {stats.get('losses', 0)}\n"
            f"⏳ Pending: {stats.get('pending', 0)}\n"
            f"🎯 Win rate: {wr:.1%}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Total PnL: ${pnl:+.2f}\n"
            f"📊 ROI: {roi:+.1%}\n"
            f"🏦 Bankroll: ${bankroll:.2f}\n"
            f"📉 Biggest win: ${stats.get('biggest_win', 0):.2f}\n"
            f"📉 Biggest loss: ${stats.get('biggest_loss', 0):.2f}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Avg edge: {stats.get('avg_edge', 0):.1%}\n"
            f"🎯 Avg confidence: {stats.get('avg_confidence', 0):.0%}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )

        await self._send(msg)

    async def send_scan_summary(self, poly_count: int, kalshi_count: int, signals: int, trades: int):
        if not self.bot or not self.chat_id:
            return

        msg = (
            f"🔍 SCAN COMPLETE\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🟣 Polymarket: {poly_count} markets\n"
            f"🔵 Kalshi: {kalshi_count} markets\n"
            f"⚡ Signals: {signals}\n"
            f"💰 New trades: {trades}\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )

        await self._send(msg)

    async def send_error(self, error_msg: str):
        if not self.bot or not self.chat_id:
            return

        msg = f"⚠️ ERROR\n━━━━━━━━━━━━━━━━━━━━\n{error_msg[:500]}"
        await self._send(msg)

    async def _send(self, text: str):
        if not self.bot or not self.chat_id:
            logger.info(f"[TG disabled] {text[:100]}...")
            return
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=None,
            )
        except Exception as e:
            logger.error(f"Telegram send error: {e}")

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_chat:
            chat_id = update.effective_chat.id
            await update.message.reply_text(
                f"🤖 ML Prediction Bot Active\n"
                f"Your chat ID: {chat_id}\n\n"
                f"Commands:\n"
                f"/stats - Overall statistics\n"
                f"/trades - Recent trades\n"
                f"/open - Open positions\n"
                f"/daily - Daily breakdown\n"
                f"/help - Help"
            )

    async def _cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        stats = self.paper_engine.get_stats()
        await self.send_daily_summary(stats)

    async def _cmd_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        trades = self.paper_engine.get_resolved_trades(limit=10)
        if not trades:
            await update.message.reply_text("No resolved trades yet.")
            return

        msg = "📋 RECENT TRADES\n━━━━━━━━━━━━━━━━━━━━\n"
        for t in trades:
            emoji = "✅" if t.result == "WIN" else "❌"
            msg += (
                f"{emoji} {t.question[:40]}...\n"
                f"   {t.side} @ {t.entry_price:.0%} → ${t.pnl:+.2f}\n"
            )

        await update.message.reply_text(msg)

    async def _cmd_open(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        trades = self.paper_engine.get_open_trades()
        if not trades:
            await update.message.reply_text("No open trades.")
            return

        msg = "📋 OPEN TRADES\n━━━━━━━━━━━━━━━━━━━━\n"
        for t in trades:
            msg += (
                f"⏳ {t.question[:40]}...\n"
                f"   {t.side} @ {t.entry_price:.0%} | ${t.bet_size:.2f}\n"
                f"   Edge: {t.meta_edge:+.1%} | Conf: {t.confidence:.0%}\n"
            )

        await update.message.reply_text(msg)

    async def _cmd_daily(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        daily = self.paper_engine.get_daily_breakdown()
        if not daily:
            await update.message.reply_text("No daily data yet.")
            return

        msg = "📅 DAILY BREAKDOWN\n━━━━━━━━━━━━━━━━━━━━\n"
        for d in daily[:7]:
            msg += (
                f"📆 {d['date']}: {d['wins']}W/{d['losses']}L "
                f"({d['win_rate']:.0%}) ${d['pnl']:+.2f}\n"
            )

        await update.message.reply_text(msg)

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🤖 ML Sports Prediction Bot\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "7-Level Architecture:\n"
            "L1: Bookmaker arbitrage\n"
            "L2: ML Ensemble (LightGBM, XGBoost, CatBoost, LogReg)\n"
            "L3: Meta-model combining all signals\n"
            "L4: NLP/GPT sentiment analysis\n"
            "L5: Market microstructure\n"
            "L6: Bayesian real-time updating\n"
            "L7: Cross-market analysis\n\n"
            "Platforms: Polymarket + Kalshi\n"
            "Mode: Paper trading\n\n"
            "Commands:\n"
            "/stats - Overall statistics\n"
            "/trades - Recent resolved trades\n"
            "/open - Open positions\n"
            "/daily - Daily breakdown"
        )
