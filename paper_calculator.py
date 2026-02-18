import asyncio
import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

Path("data/logs").mkdir(parents=True, exist_ok=True)
Path("data/models").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/logs/paper_calc.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

import numpy as np

from config import config
from src.data.odds_api import OddsAPIClient, BookmakerOdds
from src.data.sports_stats import SportsStatsClient
from src.features.builder import FeatureBuilder
from src.models.ensemble import ModelEnsemble
from src.models.meta_model import MetaModel
from src.nlp.sentiment import SentimentAnalyzer
from src.market.microstructure import MicrostructureAnalyzer
from src.engine.bayesian import BayesianUpdater

BANKROLL = 500.0
BET_SIZE = 10.0
MIN_EDGE = 0.04
USE_CLAUDE = False
CLAUDE_TOP_N = 5


def extract_games_within_window(all_odds, now, deadline):
    grace = timedelta(hours=1)
    event_map = {}
    for o in all_odds:
        if o.commence_time < (now - grace) or o.commence_time > deadline:
            continue
        key = f"{o.home_team}|{o.away_team}|{o.sport}"
        if key not in event_map:
            event_map[key] = {
                "home_team": o.home_team,
                "away_team": o.away_team,
                "sport": o.sport,
                "commence": o.commence_time,
                "odds_entries": [],
            }
        event_map[key]["odds_entries"].append(o)

    games = sorted(event_map.values(), key=lambda g: g["commence"])
    for g in games:
        entries = g["odds_entries"]
        n = len(entries)
        avg_home = sum(e.home_win_prob for e in entries) / n
        avg_away = sum(e.away_win_prob for e in entries) / n
        avg_draw = sum(e.draw_prob for e in entries) / n
        total = avg_home + avg_away + avg_draw
        if total > 0:
            avg_home /= total
            avg_away /= total
            avg_draw /= total

        sharp_bks = ["pinnacle", "betfair", "matchbook", "smarkets"]
        sharp = [e for e in entries if e.bookmaker in sharp_bks]
        if sharp:
            s_home = sum(e.home_win_prob for e in sharp) / len(sharp)
            s_away = sum(e.away_win_prob for e in sharp) / len(sharp)
            s_draw = sum(e.draw_prob for e in sharp) / len(sharp)
            s_total = s_home + s_away + s_draw
            if s_total > 0:
                s_home /= s_total
                s_away /= s_total
                s_draw /= s_total
            g["sharp_home"] = s_home
            g["sharp_away"] = s_away
            g["sharp_draw"] = s_draw
        else:
            g["sharp_home"] = avg_home
            g["sharp_away"] = avg_away
            g["sharp_draw"] = avg_draw

        g["consensus_home"] = avg_home
        g["consensus_away"] = avg_away
        g["consensus_draw"] = avg_draw
        g["n_bookmakers"] = n
        g["has_sharp"] = len(sharp) > 0

        probs = [e.home_win_prob for e in entries]
        g["prob_spread"] = float(max(probs) - min(probs)) if len(probs) > 1 else 0
        g["prob_std"] = float(np.std(probs)) if len(probs) > 1 else 0

    return games


async def process_game(game, stats_client, feature_builder, ensemble,
                       meta_model, sentiment, micro, bayesian):
    home = game["home_team"]
    away = game["away_team"]
    sport = game["sport"]
    market_price = game["consensus_home"]

    home_stats, away_stats = None, None
    try:
        results = await asyncio.gather(
            stats_client.get_team_stats(home, sport),
            stats_client.get_team_stats(away, sport),
            return_exceptions=True,
        )
        home_stats = results[0] if not isinstance(results[0], Exception) else None
        away_stats = results[1] if not isinstance(results[1], Exception) else None
    except Exception:
        pass

    bookmaker_odds = {
        "home_prob": game["consensus_home"],
        "away_prob": game["consensus_away"],
        "draw_prob": game["consensus_draw"],
        "n_bookmakers": game["n_bookmakers"],
        "has_sharp": game["has_sharp"],
    }

    hours_to_event = max(
        (game["commence"] - datetime.now(timezone.utc)).total_seconds() / 3600, 0.1
    )

    features = feature_builder.build_features(
        market_price=market_price,
        bookmaker_odds=bookmaker_odds,
        home_stats=home_stats,
        away_stats=away_stats,
        home_news_sentiment={"net_sentiment": 0},
        away_news_sentiment={"net_sentiment": 0},
        orderbook={"bids": [], "asks": []},
        price_history=[],
        hours_to_event=hours_to_event,
        platform="bookmaker",
        sport=sport,
    )

    ensemble_result = ensemble.predict(features)
    nlp_result = sentiment._rule_based_analysis([], market_price)

    micro_result = micro.analyze(
        {"bids": [], "asks": []}, [], market_price, 0.0, f"{home}_vs_{away}"
    )

    evidence = bayesian.build_evidence_from_signals(
        features, None, micro_result, nlp_result
    )
    bayesian_result = bayesian.get_posterior(
        f"{home}_vs_{away}", ensemble_result["ensemble_prob"], evidence
    )

    meta_result = meta_model.combine_signals(
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
        side = "HOME"
        price = market_price
    else:
        side = "AWAY"
        price = game["consensus_away"]
        model_prob = 1.0 - model_prob
        edge = abs(edge)

    sharp_edge = 0.0
    if game["has_sharp"]:
        if side == "HOME":
            sharp_edge = game["sharp_home"] - game["consensus_home"]
        else:
            sharp_edge = game["sharp_away"] - game["consensus_away"]

    payout_if_win = BET_SIZE * (1.0 / max(price, 0.01) - 1)
    loss_if_lose = -BET_SIZE
    ev = model_prob * payout_if_win + (1 - model_prob) * loss_if_lose

    return {
        "home_team": home,
        "away_team": away,
        "sport": sport,
        "commence": game["commence"].isoformat(),
        "side": side,
        "market_price": price,
        "model_prob": model_prob,
        "edge": edge,
        "confidence": confidence,
        "consensus_home": game["consensus_home"],
        "consensus_away": game["consensus_away"],
        "consensus_draw": game["consensus_draw"],
        "sharp_home": game.get("sharp_home", 0),
        "sharp_away": game.get("sharp_away", 0),
        "sharp_edge": sharp_edge,
        "n_bookmakers": game["n_bookmakers"],
        "has_sharp": game["has_sharp"],
        "prob_spread": game.get("prob_spread", 0),
        "ensemble_prob": ensemble_result["ensemble_prob"],
        "ensemble_preds": ensemble_result["predictions"],
        "bayesian_prob": bayesian_result["bayesian_prob"],
        "model_agreement": ensemble_result.get("model_agreement", 0),
        "bet_size": BET_SIZE,
        "potential_win": payout_if_win,
        "potential_loss": loss_if_lose,
        "expected_value": ev,
        "nlp_source": "rule_based",
        "nlp_signal": nlp_result.get("combined_nlp_signal", 0),
        "home_stats_avail": bool(home_stats and home_stats.wins > 0),
        "away_stats_avail": bool(away_stats and away_stats.wins > 0),
        "home_elo": features.get("home_elo", 1500),
        "away_elo": features.get("away_elo", 1500),
        "meta_weights": meta_result.get("signal_weights", {}),
    }


async def run_paper_calculator():
    now = datetime.now(timezone.utc)
    deadline = now + timedelta(hours=48)

    logger.info("=" * 60)
    logger.info("PAPER TRADING CALCULATOR — REAL SPORTS (Odds API)")
    logger.info(f"Bankroll: ${BANKROLL:.2f} | Bet: ${BET_SIZE:.2f} | Min edge: {MIN_EDGE:.0%}")
    logger.info(f"Window: {now.strftime('%H:%M')} — {deadline.strftime('%d %H:%M')} UTC (48h)")
    nlp_mode = f"ON (top {CLAUDE_TOP_N})" if USE_CLAUDE else "OFF (rule-based)"
    logger.info(f"Claude NLP: {nlp_mode}")
    logger.info("=" * 60)

    odds_client = OddsAPIClient()
    stats_client = SportsStatsClient()
    feature_builder = FeatureBuilder()
    ensemble = ModelEnsemble()
    meta_model = MetaModel()
    sentiment = SentimentAnalyzer()
    micro = MicrostructureAnalyzer()
    bayesian = BayesianUpdater()

    ensemble.load_models()

    logger.info("Fetching bookmaker odds from Odds API...")
    all_odds = await odds_client.fetch_all_sports_odds()
    logger.info(f"Got {len(all_odds)} odds entries")

    games = extract_games_within_window(all_odds, now, deadline)
    logger.info(f"Found {len(games)} games in next 12h:")
    for g in games:
        logger.info(
            f"  [{g['sport']}] {g['home_team']} vs {g['away_team']} "
            f"@ {g['commence'].strftime('%H:%M UTC')}"
        )

    if not games:
        logger.info("No games found in 12h window.")
        report = generate_report([], games, now)
        print(report)
        Path("data/paper_calculator_report.txt").write_text(report)
        await send_telegram_report([], games, now)
        await odds_client.close()
        await stats_client.close()
        return

    logger.info("Processing games through 7-level ML pipeline...")
    predictions = []
    for game in games:
        try:
            pred = await process_game(
                game, stats_client, feature_builder,
                ensemble, meta_model, sentiment, micro, bayesian,
            )
            if pred is None:
                continue

            if pred["edge"] >= MIN_EDGE:
                predictions.append(pred)
                logger.info(
                    f"  PICK: {pred['side']} {pred['home_team']} vs "
                    f"{pred['away_team']} | edge={pred['edge']:.1%} "
                    f"conf={pred['confidence']:.0%} ev=${pred['expected_value']:+.2f}"
                )
            else:
                logger.info(
                    f"  LOW EDGE: {pred['home_team']} vs {pred['away_team']} "
                    f"| edge={pred['edge']:.1%} < {MIN_EDGE:.0%}"
                )
        except Exception as e:
            logger.error(
                f"  ERROR: {game['home_team']} vs {game['away_team']}: {e}"
            )

    if USE_CLAUDE and predictions:
        logger.info(f"Running Claude NLP on top {CLAUDE_TOP_N} picks...")
        predictions.sort(key=lambda x: x["edge"], reverse=True)
        for pred in predictions[:CLAUDE_TOP_N]:
            try:
                nlp_result = await sentiment.analyze_matchup(
                    pred["home_team"], pred["away_team"],
                    pred["sport"], [], pred["market_price"],
                )
                pred["nlp_source"] = nlp_result.get("source", "claude")
                pred["nlp_signal"] = nlp_result.get("combined_nlp_signal", 0)
                edge_adj = nlp_result.get("overall_edge", 0) * 0.05
                pred["edge"] = max(pred["edge"] + edge_adj, 0)
            except Exception as e:
                logger.warning(f"  Claude error: {e}")

    predictions.sort(key=lambda x: x["edge"], reverse=True)

    logger.info("=" * 60)
    logger.info(f"RESULT: {len(predictions)} picks from {len(games)} games")
    logger.info("=" * 60)

    report = generate_report(predictions, games, now)
    print(report)

    Path("data/paper_calculator_report.txt").write_text(report)
    Path("data/paper_predictions.json").write_text(
        json.dumps(predictions, indent=2, default=str)
    )
    logger.info("Files saved to data/")

    await send_telegram_report(predictions, games, now)

    await odds_client.close()
    await stats_client.close()


def generate_report(predictions, games, now):
    total_bets = len(predictions)
    total_wagered = total_bets * BET_SIZE

    lines = [
        "=" * 60,
        "PAPER TRADING CALCULATOR — 48 HOUR FORECAST",
        f"Generated: {now.strftime('%Y-%m-%d %H:%M UTC')}",
        f"Window: {now.strftime('%m/%d %H:%M')} — {(now + timedelta(hours=48)).strftime('%m/%d %H:%M')} UTC",
        "=" * 60,
        "",
        f"Bankroll:       ${BANKROLL:.2f}",
        f"Bet size:       ${BET_SIZE:.2f} (flat)",
        f"Games analyzed: {len(games)}",
        f"Picks made:     {total_bets}",
        f"Total wagered:  ${total_wagered:.2f}",
        f"Remaining cash: ${BANKROLL - total_wagered:.2f}",
        "",
    ]

    if not predictions:
        lines.append(f"No predictions with edge >= {MIN_EDGE:.0%} found.")
        lines.append("Bookmaker lines are tight — ML found no mispricing.")
        lines.append("")
        lines.append("ALL GAMES ANALYZED:")
        for g in games:
            ct = g["commence"].strftime("%H:%M UTC")
            lines.append(
                f"  [{g['sport']}] {g['home_team']} vs {g['away_team']} @ {ct}"
            )
            lines.append(
                f"    Consensus: H={g['consensus_home']:.0%} "
                f"D={g['consensus_draw']:.0%} A={g['consensus_away']:.0%} "
                f"({g['n_bookmakers']} bks)"
            )
        return "\n".join(lines)

    avg_edge = sum(p["edge"] for p in predictions) / total_bets
    avg_conf = sum(p["confidence"] for p in predictions) / total_bets
    avg_model_prob = sum(p["model_prob"] for p in predictions) / total_bets
    total_ev = sum(p["expected_value"] for p in predictions)

    lines.extend([
        "--- AGGREGATE ---",
        f"Avg edge:       {avg_edge:.1%}",
        f"Avg confidence: {avg_conf:.0%}",
        f"Avg win prob:   {avg_model_prob:.1%}",
        f"Total EV:       ${total_ev:+.2f}",
        "",
        "--- SCENARIOS ---",
    ])

    for label, wr in [("Worst (40%)", 0.40), ("Pessimistic (50%)", 0.50),
                       ("Conservative (60%)", 0.60), ("Target (70%)", 0.70),
                       ("Optimistic (80%)", 0.80)]:
        wins = int(total_bets * wr)
        losses = total_bets - wins
        sorted_p = sorted(predictions, key=lambda x: x["potential_win"], reverse=True)
        sim_pnl = sum(p["potential_win"] for p in sorted_p[:wins])
        sim_pnl += sum(p["potential_loss"] for p in sorted_p[wins:])
        roi = sim_pnl / total_wagered * 100 if total_wagered > 0 else 0
        lines.append(
            f"  {label}: {wins}W/{losses}L | PnL: ${sim_pnl:+.2f} "
            f"| Bankroll: ${BANKROLL + sim_pnl:.2f} | ROI: {roi:+.0f}%"
        )

    lines.extend(["", "--- PICKS (by edge) ---", ""])

    for i, p in enumerate(predictions, 1):
        sport_short = p["sport"].replace("soccer_", "").replace("_", " ").upper()
        try:
            ct = datetime.fromisoformat(p["commence"]).strftime("%H:%M UTC")
        except (ValueError, TypeError):
            ct = "?"

        sharp_info = ""
        if p.get("has_sharp"):
            sharp_info = f" | Sharp: {p['sharp_edge']:+.1%}"
        stats_info = "API-Football" if p.get("home_stats_avail") else "default"

        lines.extend([
            f"#{i} | {p['side']} | Edge: {p['edge']:.1%} | Conf: {p['confidence']:.0%}",
            f"   {p['home_team']} vs {p['away_team']}",
            f"   {sport_short} | Kickoff: {ct}",
            f"   Book consensus: H={p['consensus_home']:.0%} "
            f"D={p['consensus_draw']:.0%} A={p['consensus_away']:.0%} "
            f"({p['n_bookmakers']} bks){sharp_info}",
            f"   ML ensemble: {p['ensemble_prob']:.1%} | Bayesian: "
            f"{p['bayesian_prob']:.1%} | Final: {p['model_prob']:.1%}",
            f"   LGB={p['ensemble_preds'].get('lightgbm', 0):.2f} "
            f"XGB={p['ensemble_preds'].get('xgboost', 0):.2f} "
            f"CAT={p['ensemble_preds'].get('catboost', 0):.2f} "
            f"LOG={p['ensemble_preds'].get('logistic', 0):.2f}",
            f"   Bet: ${p['bet_size']:.2f} | Win: +${p['potential_win']:.2f} "
            f"| Lose: -${abs(p['potential_loss']):.2f} "
            f"| EV: ${p['expected_value']:+.2f}",
            f"   Team stats: {stats_info} | Elo: {p['home_elo']:.0f} vs "
            f"{p['away_elo']:.0f}",
            f"   Line spread: {p.get('prob_spread', 0):.1%}",
            "",
        ])

    lines.extend([
        "=" * 60,
        "HOW TO VERIFY:",
        "1. Check each game result after it finishes",
        "2. If our SIDE won -> add potential_win to bankroll",
        "3. If our SIDE lost -> subtract $10 from bankroll",
        "4. Calculate actual win rate and PnL",
        "=" * 60,
        "DISCLAIMER: Paper trading only. No real money at risk.",
    ])

    return "\n".join(lines)


async def send_telegram_report(predictions, games, now):
    if not config.telegram_bot_token or not config.telegram_chat_id:
        logger.info("Telegram not configured, skipping send")
        return

    try:
        from telegram import Bot
        bot = Bot(token=config.telegram_bot_token)

        total = len(predictions)
        if total == 0:
            msg = (
                f"PAPER CALC | {now.strftime('%Y-%m-%d %H:%M UTC')}\n"
                f"Games: {len(games)} | Picks: 0\n"
                f"No edge >= {MIN_EDGE:.0%} found.\n\n"
                "Games analyzed:\n"
            )
            for g in games:
                ct = g["commence"].strftime("%H:%M")
                msg += f"  {g['home_team']} vs {g['away_team']} @ {ct}\n"
        else:
            total_ev = sum(p["expected_value"] for p in predictions)
            avg_edge = sum(p["edge"] for p in predictions) / total

            msg = (
                f"PAPER CALC — 12H FORECAST\n"
                f"{'=' * 30}\n"
                f"Bankroll: ${BANKROLL:.0f} | Bet: ${BET_SIZE:.0f}\n"
                f"Games: {len(games)} | Picks: {total}\n"
                f"Wagered: ${total * BET_SIZE:.0f}\n"
                f"Avg edge: {avg_edge:.1%} | EV: ${total_ev:+.2f}\n"
                f"{'=' * 30}\n\n"
            )

            for i, p in enumerate(predictions, 1):
                try:
                    ct = datetime.fromisoformat(p["commence"]).strftime("%H:%M")
                except (ValueError, TypeError):
                    ct = "?"
                sport_short = p["sport"].replace("soccer_", "").replace("_", " ")
                msg += (
                    f"#{i} {p['side']} | {p['home_team']} vs {p['away_team']}\n"
                    f"  {sport_short} @ {ct} UTC\n"
                    f"  Edge:{p['edge']:.1%} Prob:{p['model_prob']:.0%} "
                    f"EV:${p['expected_value']:+.1f}\n\n"
                )

            msg += "SCENARIOS:\n"
            for label, wr in [("50%", 0.50), ("60%", 0.60), ("70%", 0.70)]:
                wins = int(total * wr)
                sorted_p = sorted(
                    predictions, key=lambda x: x["potential_win"], reverse=True
                )
                sim_pnl = sum(p["potential_win"] for p in sorted_p[:wins])
                sim_pnl += sum(p["potential_loss"] for p in sorted_p[wins:])
                msg += f"  {label}WR: ${sim_pnl:+.1f} -> ${BANKROLL + sim_pnl:.0f}\n"

        if len(msg) > 4000:
            msg = msg[:4000] + "\n..."

        await bot.send_message(chat_id=config.telegram_chat_id, text=msg)
        logger.info("Telegram report sent!")
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")


if __name__ == "__main__":
    asyncio.run(run_paper_calculator())
