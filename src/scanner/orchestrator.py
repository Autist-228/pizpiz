import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone, timedelta

import numpy as np

from config import config
from src.data.polymarket import PolymarketClient, PolymarketMarket
from src.data.kalshi import KalshiClient, KalshiMarket
from src.data.odds_api import OddsAPIClient
from src.data.sports_stats import SportsStatsClient
from src.data.news import NewsClient
from src.features.builder import FeatureBuilder
from src.models.ensemble import ModelEnsemble
from src.models.meta_model import MetaModel
from src.nlp.sentiment import SentimentAnalyzer
from src.market.microstructure import MicrostructureAnalyzer
from src.engine.bayesian import BayesianUpdater
from src.engine.kelly import KellyCriterion
from src.engine.paper_trading import PaperTradingEngine, PaperTrade
from src.crossmarket.analyzer import CrossMarketAnalyzer
from src.telegram_bot.bot import TelegramBot
from src.data.historical import HistoricalDataCollector
from src.models.trainer import TrainingPipeline
from src.engine.backtest import BacktestEngine
from src.engine.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self):
        self.polymarket = PolymarketClient()
        self.kalshi = KalshiClient()
        self.odds_api = OddsAPIClient()
        self.sports_stats = SportsStatsClient()
        self.news_client = NewsClient()

        self.feature_builder = FeatureBuilder()
        self.ensemble = ModelEnsemble()
        self.meta_model = MetaModel()
        self.sentiment = SentimentAnalyzer()
        self.microstructure = MicrostructureAnalyzer()
        self.bayesian = BayesianUpdater()
        self.kelly = KellyCriterion()
        self.paper_engine = PaperTradingEngine()
        self.cross_market = CrossMarketAnalyzer()
        self.telegram = TelegramBot(self.paper_engine)
        self.historical = HistoricalDataCollector()
        self.trainer = TrainingPipeline()
        self.backtest_engine = BacktestEngine()
        self.risk_manager = RiskManager(self.paper_engine)

        self.prev_features: dict[str, dict] = {}
        self.running = False
        self._scan_count = 0
        self._retrain_every_n_scans = 48

    async def initialize(self):
        logger.info("Initializing orchestrator...")

        self.telegram._risk_manager = self.risk_manager

        logger.info("Collecting historical data...")
        try:
            total = await self.historical.collect_all(days_back=180)
            logger.info(f"Historical data: {total} total records")
        except Exception as e:
            logger.error(f"Historical collection error: {e}")

        logger.info("Training ML models...")
        try:
            train_result = self.trainer.run_training()
            if train_result.get("status") == "success":
                logger.info("ML models trained successfully")
                await self.telegram.send_training_report(train_result)
            else:
                logger.info(f"Training: {train_result.get('status')}")
        except Exception as e:
            logger.error(f"Training error: {e}")

        self.ensemble.load_models()

        logger.info("Running backtest...")
        try:
            bt_result = self.backtest_engine.run_backtest()
            if bt_result.total_trades > 0:
                report = self.backtest_engine.format_report(bt_result)
                logger.info(f"Backtest: {bt_result.total_trades} trades, WR={bt_result.win_rate:.1%}")
                await self.telegram.send_backtest_report(report)
        except Exception as e:
            logger.error(f"Backtest error: {e}")

        await self.telegram.initialize()
        logger.info("Orchestrator initialized")

    async def shutdown(self):
        self.running = False
        await self.polymarket.close()
        await self.kalshi.close()
        await self.odds_api.close()
        await self.sports_stats.close()
        await self.news_client.close()
        await self.historical.close()
        await self.telegram.shutdown()
        logger.info("Orchestrator shut down")

    async def run_loop(self):
        self.running = True
        interval = config.scan_interval_minutes * 60
        logger.info(f"Starting scan loop (every {config.scan_interval_minutes} min)")

        while self.running:
            try:
                await self.run_single_scan()
            except Exception as e:
                logger.error(f"Scan error: {e}", exc_info=True)
                await self.telegram.send_error(f"Scan error: {str(e)[:200]}")

            await self._check_resolutions()

            self._scan_count += 1
            if self._scan_count % self._retrain_every_n_scans == 0:
                await self._auto_retrain()

            stats = self.paper_engine.get_stats()
            if stats["total_trades"] > 0 and stats["total_trades"] % 10 == 0:
                await self.telegram.send_daily_summary(stats)

            logger.info(f"Sleeping {config.scan_interval_minutes} min until next scan...")
            await asyncio.sleep(interval)

    async def run_single_scan(self):
        logger.info("=" * 60)
        logger.info("Starting scan...")
        scan_start = datetime.now(timezone.utc)

        poly_markets, kalshi_markets, all_odds = await asyncio.gather(
            self.polymarket.fetch_sports_markets(),
            self.kalshi.fetch_sports_markets(),
            self.odds_api.fetch_all_sports_odds(),
            return_exceptions=True,
        )

        if isinstance(poly_markets, Exception):
            logger.error(f"Polymarket fetch failed: {poly_markets}")
            poly_markets = []
        if isinstance(kalshi_markets, Exception):
            logger.error(f"Kalshi fetch failed: {kalshi_markets}")
            kalshi_markets = []
        if isinstance(all_odds, Exception):
            logger.error(f"Odds fetch failed: {all_odds}")
            all_odds = []

        logger.info(
            f"Fetched: {len(poly_markets)} Polymarket, "
            f"{len(kalshi_markets)} Kalshi, "
            f"{len(all_odds)} odds entries"
        )

        signals: list[dict] = []
        trades_placed = 0

        for market in poly_markets:
            try:
                signal = await self._process_polymarket(market, all_odds)
                if signal and signal.get("recommended"):
                    signals.append(signal)
                    placed = await self._execute_signal(signal)
                    if placed:
                        trades_placed += 1
            except Exception as e:
                logger.error(f"Error processing PM {market.question[:50]}: {e}")

        for market in kalshi_markets:
            try:
                signal = await self._process_kalshi(market, all_odds)
                if signal and signal.get("recommended"):
                    signals.append(signal)
                    placed = await self._execute_signal(signal)
                    if placed:
                        trades_placed += 1
            except Exception as e:
                logger.error(f"Error processing KL {market.title[:50]}: {e}")

        cross_opps = self.cross_market.analyze(poly_markets, kalshi_markets, {})
        arb_count = sum(1 for o in cross_opps if o.arb_opportunity)
        if arb_count:
            logger.info(f"Cross-market: {arb_count} arb opportunities detected")

        self.paper_engine.log_scan(
            len(poly_markets), len(kalshi_markets), len(signals), trades_placed
        )

        await self.telegram.send_scan_summary(
            len(poly_markets), len(kalshi_markets), len(signals), trades_placed
        )

        elapsed = (datetime.now(timezone.utc) - scan_start).total_seconds()
        logger.info(
            f"Scan complete in {elapsed:.1f}s: "
            f"{len(signals)} signals, {trades_placed} trades placed"
        )

    async def _process_polymarket(
        self, market: PolymarketMarket, all_odds: list,
    ) -> dict | None:
        if self.paper_engine.has_existing_trade(market.condition_id, "polymarket"):
            return None

        if market.volume < config.min_volume:
            return None

        home_team, away_team = self.feature_builder.extract_teams_from_question(market.question)
        if not home_team:
            return None

        sport = self._detect_sport(market.question, market.tags)

        orderbook_task = self.polymarket.fetch_market_orderbook(market.token_id_yes)
        history_task = self.polymarket.fetch_price_history(market.condition_id)
        home_stats_task = self.sports_stats.get_team_stats(home_team, sport)
        away_stats_task = self.sports_stats.get_team_stats(away_team, sport)
        home_news_task = self.news_client.search_team_news(home_team, sport)
        away_news_task = self.news_client.search_team_news(away_team, sport)

        results = await asyncio.gather(
            orderbook_task, history_task,
            home_stats_task, away_stats_task,
            home_news_task, away_news_task,
            return_exceptions=True,
        )

        orderbook = results[0] if not isinstance(results[0], Exception) else {"bids": [], "asks": []}
        price_history = results[1] if not isinstance(results[1], Exception) else []
        home_stats = results[2] if not isinstance(results[2], Exception) else None
        away_stats = results[3] if not isinstance(results[3], Exception) else None
        home_news = results[4] if not isinstance(results[4], Exception) else []
        away_news = results[5] if not isinstance(results[5], Exception) else []

        bookmaker_odds = self.odds_api.get_consensus_odds(all_odds, home_team, away_team)

        home_sentiment = self.news_client.extract_sentiment_keywords(home_news)
        away_sentiment = self.news_client.extract_sentiment_keywords(away_news)

        hours_to_event = 12.0
        if market.end_date:
            delta = (market.end_date - datetime.now(timezone.utc)).total_seconds() / 3600
            hours_to_event = max(delta, 0.1)

        features = self.feature_builder.build_features(
            market_price=market.outcome_yes_price,
            bookmaker_odds=bookmaker_odds,
            home_stats=home_stats,
            away_stats=away_stats,
            home_news_sentiment=home_sentiment,
            away_news_sentiment=away_sentiment,
            orderbook=orderbook,
            price_history=price_history,
            hours_to_event=hours_to_event,
            platform="polymarket",
            sport=sport,
        )

        ensemble_result = self.ensemble.predict(features)

        nlp_result = await self.sentiment.analyze_matchup(
            home_team, away_team, sport, home_news + away_news, market.outcome_yes_price
        )

        micro_result = self.microstructure.analyze(
            orderbook, price_history, market.outcome_yes_price, market.volume,
            market.condition_id,
        )

        prev = self.prev_features.get(market.condition_id)
        evidence = self.bayesian.build_evidence_from_signals(features, prev, micro_result, nlp_result)
        bayesian_result = self.bayesian.get_posterior(
            market.condition_id, ensemble_result["ensemble_prob"], evidence
        )
        self.prev_features[market.condition_id] = features

        meta_result = self.meta_model.combine_signals(
            ensemble_result=ensemble_result,
            bookmaker_edge=features.get("edge_raw", 0),
            nlp_sentiment=nlp_result.get("combined_nlp_signal", 0),
            microstructure_signal=micro_result.get("combined_micro_signal", 0),
            bayesian_prob=bayesian_result["bayesian_prob"],
            features=features,
        )

        model_prob = meta_result["meta_probability"]
        edge = meta_result["meta_edge"]
        confidence = meta_result["meta_confidence"]

        if edge > 0:
            side = "YES"
            price = market.outcome_yes_price
        else:
            side = "NO"
            price = market.outcome_no_price
            model_prob = 1.0 - model_prob
            edge = abs(edge)

        cross = self.cross_market.select_best_platform(
            market.question, market.outcome_yes_price, None, model_prob
        )

        kelly_result = self.kelly.calculate_bet_size(model_prob, price, confidence)

        return {
            "event_id": market.condition_id,
            "platform": "polymarket",
            "question": market.question,
            "home_team": home_team,
            "away_team": away_team,
            "sport": sport,
            "side": side,
            "market_price": price,
            "model_prob": model_prob,
            "meta_edge": edge,
            "confidence": confidence,
            "bet_size": kelly_result["bet_size"],
            "kelly_fraction": kelly_result["kelly_fraction"],
            "recommended": kelly_result["recommended"],
            "reason": kelly_result["reason"],
            "end_time": market.end_date.isoformat() if market.end_date else "",
            "volume": market.volume,
            "signals_detail": {
                "ensemble_prob": ensemble_result["ensemble_prob"],
                "model_predictions": ensemble_result["predictions"],
                "model_agreement": ensemble_result["model_agreement"],
                "bookmaker_edge": features.get("edge_raw", 0),
                "nlp_signal": nlp_result.get("combined_nlp_signal", 0),
                "micro_signal": micro_result.get("combined_micro_signal", 0),
                "bayesian_prob": bayesian_result["bayesian_prob"],
                "cross_platform": cross["best_platform"],
            },
        }

    async def _process_kalshi(
        self, market: KalshiMarket, all_odds: list,
    ) -> dict | None:
        if self.paper_engine.has_existing_trade(market.ticker, "kalshi"):
            return None

        if market.volume < config.min_volume:
            return None

        home_team, away_team = self.feature_builder.extract_teams_from_question(market.title)
        if not home_team:
            return None

        sport = self._detect_sport(market.title, [market.category, market.sub_category])

        orderbook_task = self.kalshi.fetch_market_orderbook(market.ticker)
        history_task = self.kalshi.fetch_market_history(market.ticker)
        home_stats_task = self.sports_stats.get_team_stats(home_team, sport)
        away_stats_task = self.sports_stats.get_team_stats(away_team, sport)
        home_news_task = self.news_client.search_team_news(home_team, sport)
        away_news_task = self.news_client.search_team_news(away_team, sport)

        results = await asyncio.gather(
            orderbook_task, history_task,
            home_stats_task, away_stats_task,
            home_news_task, away_news_task,
            return_exceptions=True,
        )

        orderbook = results[0] if not isinstance(results[0], Exception) else {"yes": [], "no": []}
        price_history = results[1] if not isinstance(results[1], Exception) else []
        home_stats = results[2] if not isinstance(results[2], Exception) else None
        away_stats = results[3] if not isinstance(results[3], Exception) else None
        home_news = results[4] if not isinstance(results[4], Exception) else []
        away_news = results[5] if not isinstance(results[5], Exception) else []

        bookmaker_odds = self.odds_api.get_consensus_odds(all_odds, home_team, away_team)

        home_sentiment = self.news_client.extract_sentiment_keywords(home_news)
        away_sentiment = self.news_client.extract_sentiment_keywords(away_news)

        hours_to_event = 12.0
        if market.end_date:
            delta = (market.end_date - datetime.now(timezone.utc)).total_seconds() / 3600
            hours_to_event = max(delta, 0.1)

        features = self.feature_builder.build_features(
            market_price=market.yes_price,
            bookmaker_odds=bookmaker_odds,
            home_stats=home_stats,
            away_stats=away_stats,
            home_news_sentiment=home_sentiment,
            away_news_sentiment=away_sentiment,
            orderbook=orderbook,
            price_history=price_history,
            hours_to_event=hours_to_event,
            platform="kalshi",
            sport=sport,
        )

        ensemble_result = self.ensemble.predict(features)

        nlp_result = await self.sentiment.analyze_matchup(
            home_team, away_team, sport, home_news + away_news, market.yes_price
        )

        micro_result = self.microstructure.analyze(
            orderbook, price_history, market.yes_price, float(market.volume),
            market.ticker,
        )

        prev = self.prev_features.get(market.ticker)
        evidence = self.bayesian.build_evidence_from_signals(features, prev, micro_result, nlp_result)
        bayesian_result = self.bayesian.get_posterior(
            market.ticker, ensemble_result["ensemble_prob"], evidence
        )
        self.prev_features[market.ticker] = features

        meta_result = self.meta_model.combine_signals(
            ensemble_result=ensemble_result,
            bookmaker_edge=features.get("edge_raw", 0),
            nlp_sentiment=nlp_result.get("combined_nlp_signal", 0),
            microstructure_signal=micro_result.get("combined_micro_signal", 0),
            bayesian_prob=bayesian_result["bayesian_prob"],
            features=features,
        )

        model_prob = meta_result["meta_probability"]
        edge = meta_result["meta_edge"]
        confidence = meta_result["meta_confidence"]

        if edge > 0:
            side = "YES"
            price = market.yes_price
        else:
            side = "NO"
            price = market.no_price
            model_prob = 1.0 - model_prob
            edge = abs(edge)

        cross = self.cross_market.select_best_platform(
            market.title, None, market.yes_price, model_prob
        )

        kelly_result = self.kelly.calculate_bet_size(model_prob, price, confidence)

        return {
            "event_id": market.ticker,
            "platform": "kalshi",
            "question": market.title,
            "home_team": home_team,
            "away_team": away_team,
            "sport": sport,
            "side": side,
            "market_price": price,
            "model_prob": model_prob,
            "meta_edge": edge,
            "confidence": confidence,
            "bet_size": kelly_result["bet_size"],
            "kelly_fraction": kelly_result["kelly_fraction"],
            "recommended": kelly_result["recommended"],
            "reason": kelly_result["reason"],
            "end_time": market.end_date.isoformat() if market.end_date else "",
            "volume": float(market.volume),
            "signals_detail": {
                "ensemble_prob": ensemble_result["ensemble_prob"],
                "model_predictions": ensemble_result["predictions"],
                "model_agreement": ensemble_result["model_agreement"],
                "bookmaker_edge": features.get("edge_raw", 0),
                "nlp_signal": nlp_result.get("combined_nlp_signal", 0),
                "micro_signal": micro_result.get("combined_micro_signal", 0),
                "bayesian_prob": bayesian_result["bayesian_prob"],
                "cross_platform": cross["best_platform"],
            },
        }

    async def _auto_retrain(self):
        logger.info("Auto-retrain: collecting new data and retraining...")
        try:
            await self.historical.collect_all(days_back=30)
            result = self.trainer.run_training()
            if result.get("status") == "success":
                self.ensemble.load_models()
                await self.telegram.send_training_report(result)
                logger.info("Auto-retrain complete")
            else:
                logger.info(f"Auto-retrain: {result.get('status')}")
        except Exception as e:
            logger.error(f"Auto-retrain error: {e}")

    async def _execute_signal(self, signal: dict) -> bool:
        allowed, reason = self.risk_manager.check_trade_allowed(signal)
        if not allowed:
            logger.info(f"Risk manager blocked trade: {reason}")
            await self.telegram.send_risk_alert(reason)
            return False

        signal["bet_size"] = self.risk_manager.adjust_bet_size(signal)

        trade = PaperTrade(
            trade_id=f"PT-{uuid.uuid4().hex[:8]}",
            event_id=signal["event_id"],
            platform=signal["platform"],
            question=signal["question"],
            side=signal["side"],
            entry_price=signal["market_price"],
            model_prob=signal["model_prob"],
            meta_edge=signal["meta_edge"],
            confidence=signal["confidence"],
            bet_size=signal["bet_size"],
            kelly_fraction=signal["kelly_fraction"],
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_end_time=signal.get("end_time", ""),
            status="open",
            sport=signal.get("sport", ""),
            home_team=signal.get("home_team", ""),
            away_team=signal.get("away_team", ""),
            signals=json.dumps(signal.get("signals_detail", {})),
        )

        placed = self.paper_engine.place_trade(trade)
        if placed:
            await self.telegram.send_signal(signal)
            self.kelly.update_bankroll(0)

        return placed

    async def _check_resolutions(self):
        open_trades = self.paper_engine.get_open_trades()
        now = datetime.now(timezone.utc)

        for trade in open_trades:
            if not trade.event_end_time:
                continue

            try:
                end_time = datetime.fromisoformat(trade.event_end_time.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue

            if now < end_time + timedelta(hours=1):
                continue

            resolved = await self._resolve_trade(trade)
            if resolved:
                await self.telegram.send_trade_result(resolved)

    async def _resolve_trade(self, trade: PaperTrade) -> PaperTrade | None:
        outcome = await self._check_outcome(trade)
        if outcome is None:
            return None

        resolved = self.paper_engine.resolve_trade(trade.trade_id, outcome)
        if resolved:
            self.kelly.update_bankroll(resolved.pnl or 0)
            self.meta_model.update_calibration(trade.model_prob, int(outcome))
            won = resolved.result == "WIN"
            self.risk_manager.update_after_resolution(won)
        return resolved

    async def _check_outcome(self, trade: PaperTrade) -> bool | None:
        if trade.platform == "polymarket":
            try:
                session = await self.polymarket._get_session()
                url = f"{self.polymarket.gamma_url}/markets/{trade.event_id}"
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("closed"):
                            result = data.get("resolvedBy") or data.get("resolution")
                            if result:
                                return str(result).lower() in ("yes", "1", "true")
            except Exception as e:
                logger.debug(f"Could not check PM outcome for {trade.event_id}: {e}")

        elif trade.platform == "kalshi":
            try:
                session = await self.kalshi._get_session()
                url = f"{self.kalshi.base_url}/markets/{trade.event_id}"
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        market_data = data.get("market", data)
                        result = market_data.get("result")
                        if result:
                            return result.lower() in ("yes", "1", "true")
            except Exception as e:
                logger.debug(f"Could not check KL outcome for {trade.event_id}: {e}")

        return None

    @staticmethod
    def _detect_sport(text: str, tags: list) -> str:
        search = text.lower() + " " + " ".join(str(t).lower() for t in tags)

        sport_map = {
            "basketball_nba": ["nba", "basketball", "lakers", "celtics", "warriors", "nets", "bucks", "76ers"],
            "americanfootball_nfl": ["nfl", "football", "chiefs", "eagles", "cowboys", "patriots", "super bowl"],
            "soccer_epl": ["epl", "premier league", "arsenal", "chelsea", "liverpool", "man city", "manchester"],
            "soccer_usa_mls": ["mls", "inter miami", "lafc", "galaxy", "sounders"],
            "mma_mixed_martial_arts": ["ufc", "mma", "bellator", "fight", "bout"],
            "icehockey_nhl": ["nhl", "hockey", "bruins", "rangers", "penguins", "oilers"],
        }

        for sport_key, keywords in sport_map.items():
            if any(kw in search for kw in keywords):
                return sport_key

        return "sports_other"
