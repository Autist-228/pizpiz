#!/usr/bin/env python3
"""
Sport Predictor — Clean signal generator for NBA + Soccer
Uses real team stats (balldontlie) + bookmaker consensus (Odds API)
to predict game outcomes and generate betting signals.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field

import aiohttp
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("predictor")

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
BALLDONTLIE_KEY = os.getenv("BALLDONTLIE_KEY", "f98c0d90-aa6e-49dc-8570-53386cb1f58a")
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

BANKROLL = 500.0
BET_SIZE = 10.0
MIN_CONFIDENCE = 0.55
WINDOW_HOURS = 24

SHARP_BOOKMAKERS = {"pinnacle", "betfair", "matchbook", "smarkets", "betfair_ex_eu"}

NBA_SPORTS = ["basketball_nba"]
SOCCER_SPORTS = [
    "soccer_epl", "soccer_spain_la_liga", "soccer_italy_serie_a",
    "soccer_germany_bundesliga", "soccer_france_ligue_one",
    "soccer_uefa_champs_league",
]
ALL_SPORTS = NBA_SPORTS + SOCCER_SPORTS


@dataclass
class TeamForm:
    name: str
    team_id: int = 0
    wins: int = 0
    losses: int = 0
    home_wins: int = 0
    home_losses: int = 0
    away_wins: int = 0
    away_losses: int = 0
    pts_for: float = 0
    pts_against: float = 0
    games: int = 0
    last10_wins: int = 0
    last10_losses: int = 0
    streak: int = 0
    last_game_date: str = ""
    recent_results: list = field(default_factory=list)

    @property
    def win_rate(self):
        return self.wins / max(self.games, 1)

    @property
    def home_wr(self):
        h = self.home_wins + self.home_losses
        return self.home_wins / max(h, 1)

    @property
    def away_wr(self):
        a = self.away_wins + self.away_losses
        return self.away_wins / max(a, 1)

    @property
    def ppg(self):
        return self.pts_for / max(self.games, 1)

    @property
    def opp_ppg(self):
        return self.pts_against / max(self.games, 1)

    @property
    def net_rating(self):
        return self.ppg - self.opp_ppg

    @property
    def last10_wr(self):
        t = self.last10_wins + self.last10_losses
        return self.last10_wins / max(t, 1)

    @property
    def days_rest(self):
        if not self.last_game_date:
            return 3
        try:
            last = datetime.strptime(self.last_game_date, "%Y-%m-%d")
            today = datetime.now()
            return max((today - last).days - 1, 0)
        except Exception:
            return 3


@dataclass
class GamePrediction:
    home_team: str
    away_team: str
    sport: str
    start_time: str
    pick: str
    pick_team: str
    confidence: float
    book_home_prob: float
    book_away_prob: float
    our_home_prob: float
    our_away_prob: float
    edge: float
    n_bookmakers: int
    has_sharp: bool
    home_form: dict
    away_form: dict
    factors: dict
    potential_win: float = 0.0
    potential_loss: float = 0.0
    expected_value: float = 0.0


async def fetch_nba_recent_games(session: aiohttp.ClientSession, days_back: int = 45) -> list[dict]:
    """Fetch recent NBA games with scores for form calculation."""
    all_games = []
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    cursor = None
    pages = 0
    while pages < 10:
        url = "https://api.balldontlie.io/v1/games"
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "per_page": 100,
        }
        if cursor:
            params["cursor"] = cursor

        try:
            async with session.get(url, params=params,
                                   headers={"Authorization": BALLDONTLIE_KEY}) as resp:
                if resp.status == 429:
                    logger.warning("balldontlie rate limited, waiting 2s...")
                    await asyncio.sleep(2)
                    continue
                if resp.status != 200:
                    logger.error(f"balldontlie games: {resp.status}")
                    break
                data = await resp.json()
                games = data.get("data", [])
                scored = [g for g in games if g.get("home_team_score", 0) > 0]
                all_games.extend(scored)
                next_cursor = data.get("meta", {}).get("next_cursor")
                if not next_cursor or len(games) < 100:
                    break
                cursor = next_cursor
                pages += 1
                await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"balldontlie fetch error: {e}")
            break

    logger.info(f"Fetched {len(all_games)} NBA games (last {days_back} days)")
    return all_games


def build_team_forms(games: list[dict]) -> dict[str, TeamForm]:
    """Build team form from historical games."""
    teams: dict[str, TeamForm] = {}
    sorted_games = sorted(games, key=lambda g: g.get("date", ""))

    for g in sorted_games:
        ht_name = g["home_team"]["full_name"]
        vt_name = g["visitor_team"]["full_name"]
        hs = g["home_team_score"]
        vs = g["visitor_team_score"]
        date = g.get("date", "")

        if ht_name not in teams:
            teams[ht_name] = TeamForm(name=ht_name, team_id=g["home_team"]["id"])
        if vt_name not in teams:
            teams[vt_name] = TeamForm(name=vt_name, team_id=g["visitor_team"]["id"])

        ht = teams[ht_name]
        vt = teams[vt_name]

        ht.games += 1
        vt.games += 1
        ht.pts_for += hs
        ht.pts_against += vs
        vt.pts_for += vs
        vt.pts_against += hs
        ht.last_game_date = date
        vt.last_game_date = date

        home_won = hs > vs
        if home_won:
            ht.wins += 1
            ht.home_wins += 1
            vt.losses += 1
            vt.away_losses += 1
            ht.recent_results.append(1)
            vt.recent_results.append(0)
        else:
            vt.wins += 1
            vt.away_wins += 1
            ht.losses += 1
            ht.home_losses += 1
            ht.recent_results.append(0)
            vt.recent_results.append(1)

    for t in teams.values():
        last10 = t.recent_results[-10:]
        t.last10_wins = sum(last10)
        t.last10_losses = len(last10) - t.last10_wins
        streak = 0
        for r in reversed(t.recent_results):
            if r == (1 if t.recent_results and t.recent_results[-1] else 0):
                streak += 1
            else:
                break
        t.streak = streak if t.recent_results and t.recent_results[-1] == 1 else -streak

    return teams


async def fetch_todays_nba_games(session: aiohttp.ClientSession) -> list[dict]:
    """Fetch today's NBA games."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")

    all_games = []
    for date in [today, tomorrow]:
        url = "https://api.balldontlie.io/v1/games"
        params = {"dates[]": date, "per_page": 100}
        try:
            async with session.get(url, params=params,
                                   headers={"Authorization": BALLDONTLIE_KEY}) as resp:
                if resp.status == 429:
                    await asyncio.sleep(2)
                    async with session.get(url, params=params,
                                           headers={"Authorization": BALLDONTLIE_KEY}) as resp2:
                        if resp2.status == 200:
                            data = await resp2.json()
                            all_games.extend(data.get("data", []))
                elif resp.status == 200:
                    data = await resp.json()
                    all_games.extend(data.get("data", []))
        except Exception as e:
            logger.error(f"Error fetching games for {date}: {e}")
        await asyncio.sleep(0.5)

    now = datetime.now(timezone.utc)
    deadline = now + timedelta(hours=WINDOW_HOURS)
    upcoming = []
    for g in all_games:
        dt_str = g.get("datetime", "")
        if not dt_str:
            continue
        try:
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            if dt > (now - timedelta(hours=1)) and dt < deadline:
                g["_parsed_dt"] = dt
                upcoming.append(g)
        except Exception:
            pass

    logger.info(f"Found {len(upcoming)} upcoming NBA games in {WINDOW_HOURS}h window")
    return upcoming


async def fetch_odds_api(session: aiohttp.ClientSession) -> dict:
    """Fetch bookmaker odds from Odds API, return dict keyed by 'home|away'."""
    if not ODDS_API_KEY:
        logger.warning("No ODDS_API_KEY, skipping odds fetch")
        return {}

    odds_data = {}
    for sport in ALL_SPORTS:
        url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "us,eu,uk",
            "markets": "h2h",
            "oddsFormat": "decimal",
        }
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 429:
                    logger.warning(f"Odds API rate limited on {sport}")
                    continue
                if resp.status != 200:
                    continue
                events = await resp.json()
                for ev in events:
                    home = ev.get("home_team", "")
                    away = ev.get("away_team", "")
                    commence = ev.get("commence_time", "")
                    key = f"{home}|{away}"

                    all_probs = []
                    sharp_probs = []
                    for bk in ev.get("bookmakers", []):
                        bk_name = bk.get("key", "")
                        for mkt in bk.get("markets", []):
                            if mkt.get("key") != "h2h":
                                continue
                            outcomes = {o["name"]: o["price"] for o in mkt.get("outcomes", [])}
                            h_odds = outcomes.get(home, 2.0)
                            a_odds = outcomes.get(away, 2.0)
                            d_odds = outcomes.get("Draw", 0)

                            h_p = 1.0 / h_odds if h_odds > 0 else 0
                            a_p = 1.0 / a_odds if a_odds > 0 else 0
                            d_p = 1.0 / d_odds if d_odds > 0 else 0
                            total = h_p + a_p + d_p
                            if total > 0:
                                h_p /= total
                                a_p /= total
                                d_p /= total
                            all_probs.append({"h": h_p, "a": a_p, "d": d_p, "bk": bk_name})
                            if bk_name in SHARP_BOOKMAKERS:
                                sharp_probs.append({"h": h_p, "a": a_p, "d": d_p, "bk": bk_name})

                    if all_probs:
                        source = sharp_probs if sharp_probs else all_probs
                        avg_h = np.mean([p["h"] for p in source])
                        avg_a = np.mean([p["a"] for p in source])
                        avg_d = np.mean([p["d"] for p in source])
                        total = avg_h + avg_a + avg_d
                        if total > 0:
                            avg_h /= total
                            avg_a /= total
                            avg_d /= total

                        all_h = [p["h"] for p in all_probs]
                        spread = max(all_h) - min(all_h) if len(all_h) > 1 else 0

                        odds_data[key] = {
                            "sport": sport,
                            "home": home,
                            "away": away,
                            "commence": commence,
                            "home_prob": float(avg_h),
                            "away_prob": float(avg_a),
                            "draw_prob": float(avg_d),
                            "n_bookmakers": len(all_probs),
                            "has_sharp": len(sharp_probs) > 0,
                            "line_spread": float(spread),
                            "sharp_home": float(np.mean([p["h"] for p in sharp_probs])) if sharp_probs else float(avg_h),
                            "sharp_away": float(np.mean([p["a"] for p in sharp_probs])) if sharp_probs else float(avg_a),
                        }
        except Exception as e:
            logger.error(f"Odds API error for {sport}: {e}")

    logger.info(f"Odds API: {len(odds_data)} events with odds")
    return odds_data


def match_odds_to_game(game: dict, odds_data: dict) -> dict | None:
    """Find matching odds for a game."""
    ht = game["home_team"]["full_name"]
    vt = game["visitor_team"]["full_name"]
    direct_key = f"{ht}|{vt}"
    if direct_key in odds_data:
        return odds_data[direct_key]

    ht_lower = ht.lower()
    vt_lower = vt.lower()
    for key, val in odds_data.items():
        oh = val["home"].lower()
        oa = val["away"].lower()
        if (oh in ht_lower or ht_lower in oh) and (oa in vt_lower or vt_lower in oa):
            return val
        parts_h = ht_lower.split()
        parts_a = vt_lower.split()
        if any(p in oh for p in parts_h if len(p) > 3) and any(p in oa for p in parts_a if len(p) > 3):
            return val
    return None


def predict_nba_game(
    home_form: TeamForm | None,
    away_form: TeamForm | None,
    book_home: float,
    book_away: float,
    has_sharp: bool,
    line_spread: float,
) -> dict:
    """
    Predict NBA game outcome using real stats + bookmaker odds.

    Strategy: Start with bookmaker consensus (they know more than us),
    then adjust based on factors they might under-weight.
    """
    base_home = book_home
    base_away = book_away
    adjustments = {}

    if home_form and away_form and home_form.games >= 5 and away_form.games >= 5:
        wr_diff = home_form.win_rate - away_form.win_rate
        adj = wr_diff * 0.08
        adjustments["overall_record"] = adj
        base_home += adj

        home_ha = home_form.home_wr - away_form.away_wr
        adj_ha = home_ha * 0.06
        adjustments["home_away_split"] = adj_ha
        base_home += adj_ha

        l10_diff = home_form.last10_wr - away_form.last10_wr
        adj_l10 = l10_diff * 0.10
        adjustments["last10_form"] = adj_l10
        base_home += adj_l10

        net_diff = home_form.net_rating - away_form.net_rating
        adj_net = np.clip(net_diff / 100, -0.05, 0.05)
        adjustments["net_rating"] = float(adj_net)
        base_home += adj_net

        h_rest = home_form.days_rest
        a_rest = away_form.days_rest
        rest_adj = 0.0
        if h_rest == 0:
            rest_adj -= 0.03
        elif h_rest >= 2:
            rest_adj += 0.01
        if a_rest == 0:
            rest_adj += 0.03
        elif a_rest >= 2:
            rest_adj -= 0.01
        adjustments["rest_days"] = rest_adj
        base_home += rest_adj

        h_streak = home_form.streak
        a_streak = away_form.streak
        streak_adj = 0.0
        if h_streak >= 3:
            streak_adj += 0.02
        elif h_streak <= -3:
            streak_adj -= 0.02
        if a_streak >= 3:
            streak_adj -= 0.02
        elif a_streak <= -3:
            streak_adj += 0.02
        adjustments["streak"] = streak_adj
        base_home += streak_adj
    else:
        adjustments["no_stats"] = 0.0

    base_home = float(np.clip(base_home, 0.05, 0.95))
    base_away = 1.0 - base_home

    data_quality = 0.0
    if home_form and home_form.games >= 5:
        data_quality += 0.3
    if away_form and away_form.games >= 5:
        data_quality += 0.3
    if has_sharp:
        data_quality += 0.2
    if line_spread < 0.05:
        data_quality += 0.2

    model_weight = 0.3 + data_quality * 0.4
    final_home = base_home * model_weight + book_home * (1 - model_weight)
    final_away = 1.0 - final_home

    return {
        "our_home": float(final_home),
        "our_away": float(final_away),
        "adjustments": adjustments,
        "data_quality": data_quality,
        "model_weight": model_weight,
    }


def generate_signal(
    home_team: str,
    away_team: str,
    sport: str,
    start_time: str,
    book_home: float,
    book_away: float,
    our_home: float,
    our_away: float,
    n_bookmakers: int,
    has_sharp: bool,
    home_form: TeamForm | None,
    away_form: TeamForm | None,
    adjustments: dict,
) -> GamePrediction | None:
    """Generate a betting signal if we have edge."""
    home_edge = our_home - book_home
    away_edge = our_away - book_away

    if home_edge > away_edge and home_edge > 0:
        pick = "HOME"
        pick_team = home_team
        confidence = our_home
        edge = home_edge
        price = book_home
    elif away_edge > 0:
        pick = "AWAY"
        pick_team = away_team
        confidence = our_away
        edge = away_edge
        price = book_away
    else:
        return None

    if confidence < MIN_CONFIDENCE:
        return None

    payout = BET_SIZE * (1.0 / max(price, 0.01) - 1)
    loss = -BET_SIZE
    ev = confidence * payout + (1 - confidence) * loss

    hf = {}
    if home_form and home_form.games > 0:
        hf = {
            "record": f"{home_form.wins}W-{home_form.losses}L",
            "wr": f"{home_form.win_rate:.0%}",
            "home_wr": f"{home_form.home_wr:.0%}",
            "l10": f"{home_form.last10_wins}-{home_form.last10_losses}",
            "ppg": f"{home_form.ppg:.1f}",
            "net": f"{home_form.net_rating:+.1f}",
            "streak": home_form.streak,
            "rest": home_form.days_rest,
        }
    af = {}
    if away_form and away_form.games > 0:
        af = {
            "record": f"{away_form.wins}W-{away_form.losses}L",
            "wr": f"{away_form.win_rate:.0%}",
            "away_wr": f"{away_form.away_wr:.0%}",
            "l10": f"{away_form.last10_wins}-{away_form.last10_losses}",
            "ppg": f"{away_form.ppg:.1f}",
            "net": f"{away_form.net_rating:+.1f}",
            "streak": away_form.streak,
            "rest": away_form.days_rest,
        }

    return GamePrediction(
        home_team=home_team,
        away_team=away_team,
        sport=sport,
        start_time=start_time,
        pick=pick,
        pick_team=pick_team,
        confidence=confidence,
        book_home_prob=book_home,
        book_away_prob=book_away,
        our_home_prob=our_home,
        our_away_prob=our_away,
        edge=edge,
        n_bookmakers=n_bookmakers,
        has_sharp=has_sharp,
        home_form=hf,
        away_form=af,
        factors=adjustments,
        potential_win=payout,
        potential_loss=loss,
        expected_value=ev,
    )


def format_report(predictions: list[GamePrediction], all_games_count: int) -> str:
    now = datetime.now(timezone.utc)
    lines = [
        "=" * 55,
        f"SPORT PREDICTOR — {WINDOW_HOURS}H SIGNALS",
        f"Generated: {now.strftime('%Y-%m-%d %H:%M UTC')}",
        "=" * 55,
        "",
        f"Bankroll:       ${BANKROLL:.0f}",
        f"Bet size:       ${BET_SIZE:.0f} per signal",
        f"Min confidence: {MIN_CONFIDENCE:.0%}",
        f"Games analyzed: {all_games_count}",
        f"Signals:        {len(predictions)}",
    ]

    if not predictions:
        lines.append("")
        lines.append("No signals generated. All games below confidence threshold.")
        return "\n".join(lines)

    total_wagered = len(predictions) * BET_SIZE
    avg_conf = np.mean([p.confidence for p in predictions])
    avg_edge = np.mean([p.edge for p in predictions])
    total_ev = sum(p.expected_value for p in predictions)

    lines.extend([
        f"Total wagered:  ${total_wagered:.0f}",
        "",
        f"Avg confidence: {avg_conf:.0%}",
        f"Avg edge:       {avg_edge:.1%}",
        f"Total EV:       ${total_ev:+.2f}",
        "",
        "--- SIGNALS (sorted by confidence) ---",
        "",
    ])

    sorted_preds = sorted(predictions, key=lambda p: p.confidence, reverse=True)
    for i, p in enumerate(sorted_preds, 1):
        star = " ***" if p.confidence >= 0.65 else " **" if p.confidence >= 0.60 else ""
        sport_short = p.sport.replace("basketball_", "").replace("soccer_", "").replace("_", " ").upper()
        lines.append(f"#{i} | {p.pick} {p.pick_team}{star}")
        lines.append(f"   {p.home_team} vs {p.away_team}")
        lines.append(f"   {sport_short} | {p.start_time}")
        lines.append(f"   Confidence: {p.confidence:.0%} | Edge: {p.edge:.1%} | EV: ${p.expected_value:+.2f}")
        lines.append(f"   Book: H={p.book_home_prob:.0%} A={p.book_away_prob:.0%} ({p.n_bookmakers} bks{', sharp' if p.has_sharp else ''})")
        lines.append(f"   Our:  H={p.our_home_prob:.0%} A={p.our_away_prob:.0%}")

        if p.home_form:
            hf = p.home_form
            lines.append(f"   {p.home_team}: {hf.get('record','-')} | L10:{hf.get('l10','-')} | H-WR:{hf.get('home_wr','-')} | PPG:{hf.get('ppg','-')} | Rest:{hf.get('rest','-')}d")
        if p.away_form:
            af = p.away_form
            lines.append(f"   {p.away_team}: {af.get('record','-')} | L10:{af.get('l10','-')} | A-WR:{af.get('away_wr','-')} | PPG:{af.get('ppg','-')} | Rest:{af.get('rest','-')}d")

        factors_str = " | ".join(f"{k}:{v:+.1%}" if isinstance(v, float) else f"{k}:{v}" for k, v in p.factors.items() if v != 0)
        if factors_str:
            lines.append(f"   Factors: {factors_str}")

        lines.append(f"   Bet ${BET_SIZE:.0f}: Win +${p.potential_win:.2f} | Lose -${BET_SIZE:.0f}")
        lines.append("")

    lines.append("--- SCENARIOS ---")
    n = len(predictions)
    tw = n * BET_SIZE
    for label, wr in [("Pessimistic (45%)", 0.45), ("Conservative (55%)", 0.55),
                       ("Target (60%)", 0.60), ("Good (65%)", 0.65), ("Great (70%)", 0.70)]:
        wins = round(n * wr)
        losses = n - wins
        sp = sorted(predictions, key=lambda x: x.potential_win, reverse=True)
        pnl = sum(p.potential_win for p in sp[:wins]) + sum(p.potential_loss for p in sp[wins:])
        roi = pnl / tw * 100 if tw > 0 else 0
        lines.append(f"  {label}: {wins}W/{losses}L | PnL: ${pnl:+.2f} | Bankroll: ${BANKROLL + pnl:.0f} | ROI: {roi:+.0f}%")

    lines.extend([
        "",
        "--- HOW TO VERIFY ---",
        "1. Check each game result after it finishes",
        "2. If pick won -> add potential_win",
        "3. If pick lost -> subtract $10",
        "4. Calculate actual win rate",
        "",
        "DISCLAIMER: Paper trading only. Not financial advice.",
    ])
    return "\n".join(lines)


async def send_telegram(predictions: list[GamePrediction], all_games_count: int):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.info("Telegram not configured")
        return

    try:
        from telegram import Bot
        bot = Bot(token=TELEGRAM_TOKEN)

        now = datetime.now(timezone.utc)
        n = len(predictions)

        if n == 0:
            msg = f"PREDICTOR | {now.strftime('%H:%M UTC')}\nGames: {all_games_count} | Signals: 0\nNo confident picks found."
        else:
            avg_conf = np.mean([p.confidence for p in predictions])
            total_ev = sum(p.expected_value for p in predictions)

            msg = (
                f"SPORT PREDICTOR — SIGNALS\n"
                f"{'=' * 30}\n"
                f"{now.strftime('%Y-%m-%d %H:%M UTC')}\n"
                f"Games: {all_games_count} | Signals: {n}\n"
                f"Avg conf: {avg_conf:.0%} | EV: ${total_ev:+.1f}\n"
                f"{'=' * 30}\n\n"
            )

            sorted_preds = sorted(predictions, key=lambda p: p.confidence, reverse=True)
            for i, p in enumerate(sorted_preds, 1):
                star = "***" if p.confidence >= 0.65 else "**" if p.confidence >= 0.60 else "*"
                sport_short = p.sport.replace("basketball_", "").replace("soccer_", "").replace("_", " ").upper()
                msg += (
                    f"#{i} {star} {p.pick} {p.pick_team}\n"
                    f"  {p.home_team} vs {p.away_team}\n"
                    f"  {sport_short} | {p.start_time}\n"
                    f"  Conf:{p.confidence:.0%} Edge:{p.edge:.1%} EV:${p.expected_value:+.1f}\n"
                )
                if p.home_form:
                    msg += f"  {p.home_team}: {p.home_form.get('record','-')} L10:{p.home_form.get('l10','-')}\n"
                if p.away_form:
                    msg += f"  {p.away_team}: {p.away_form.get('record','-')} L10:{p.away_form.get('l10','-')}\n"
                msg += "\n"

            msg += "SCENARIOS:\n"
            tw = n * BET_SIZE
            for label, wr in [("55%WR", 0.55), ("60%WR", 0.60), ("65%WR", 0.65)]:
                wins = int(n * wr)
                sp = sorted(predictions, key=lambda x: x.potential_win, reverse=True)
                pnl = sum(p.potential_win for p in sp[:wins]) + sum(p.potential_loss for p in sp[wins:])
                msg += f"  {label}: ${pnl:+.1f} -> ${BANKROLL + pnl:.0f}\n"

        if len(msg) > 4000:
            msg = msg[:4000] + "\n..."

        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
        logger.info("Telegram report sent!")
    except Exception as e:
        logger.error(f"Telegram failed: {e}")


async def run_predictor():
    now = datetime.now(timezone.utc)
    logger.info("=" * 55)
    logger.info("SPORT PREDICTOR — Starting")
    logger.info(f"Time: {now.strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info(f"Window: {WINDOW_HOURS}h | Min confidence: {MIN_CONFIDENCE:.0%}")
    logger.info("=" * 55)

    async with aiohttp.ClientSession() as session:
        logger.info("Step 1: Fetching recent NBA games for form data...")
        recent_games = await fetch_nba_recent_games(session, days_back=45)
        team_forms = build_team_forms(recent_games)
        logger.info(f"  Built form for {len(team_forms)} teams")
        for name, tf in sorted(team_forms.items(), key=lambda x: x[1].win_rate, reverse=True)[:10]:
            logger.info(f"  {name}: {tf.wins}W-{tf.losses}L ({tf.win_rate:.0%}) | L10:{tf.last10_wins}-{tf.last10_losses} | PPG:{tf.ppg:.1f} | Net:{tf.net_rating:+.1f}")

        logger.info("Step 2: Fetching today's NBA games...")
        todays_games = await fetch_todays_nba_games(session)
        for g in todays_games:
            dt = g.get("_parsed_dt", "?")
            logger.info(f"  {g['home_team']['full_name']} vs {g['visitor_team']['full_name']} @ {dt}")

        logger.info("Step 3: Fetching bookmaker odds...")
        odds_data = await fetch_odds_api(session)

        logger.info("Step 4: Generating predictions...")
        all_predictions = []
        total_analyzed = 0

        for game in todays_games:
            ht = game["home_team"]["full_name"]
            vt = game["visitor_team"]["full_name"]
            dt = game.get("_parsed_dt")
            start_str = dt.strftime("%H:%M UTC") if dt else "?"

            odds = match_odds_to_game(game, odds_data)
            if not odds:
                logger.warning(f"  No odds for {ht} vs {vt}, skipping")
                continue

            total_analyzed += 1
            home_form = team_forms.get(ht)
            away_form = team_forms.get(vt)

            pred = predict_nba_game(
                home_form=home_form,
                away_form=away_form,
                book_home=odds["home_prob"],
                book_away=odds["away_prob"],
                has_sharp=odds["has_sharp"],
                line_spread=odds["line_spread"],
            )

            signal = generate_signal(
                home_team=ht,
                away_team=vt,
                sport=odds["sport"],
                start_time=start_str,
                book_home=odds["home_prob"],
                book_away=odds["away_prob"],
                our_home=pred["our_home"],
                our_away=pred["our_away"],
                n_bookmakers=odds["n_bookmakers"],
                has_sharp=odds["has_sharp"],
                home_form=home_form,
                away_form=away_form,
                adjustments=pred["adjustments"],
            )

            if signal:
                all_predictions.append(signal)
                star = "***" if signal.confidence >= 0.65 else "**" if signal.confidence >= 0.60 else ""
                logger.info(f"  SIGNAL{star}: {signal.pick} {signal.pick_team} | conf={signal.confidence:.0%} edge={signal.edge:.1%} ev=${signal.expected_value:+.2f}")
            else:
                logger.info(f"  SKIP: {ht} vs {vt} | no edge or low confidence")

        for key, odds in odds_data.items():
            if odds["sport"] not in SOCCER_SPORTS:
                continue
            commence = odds.get("commence", "")
            if commence:
                try:
                    dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                    if dt < (now - timedelta(hours=1)) or dt > (now + timedelta(hours=WINDOW_HOURS)):
                        continue
                except Exception:
                    continue

            total_analyzed += 1
            home = odds["home"]
            away = odds["away"]
            start_str = dt.strftime("%H:%M UTC") if commence else "?"

            signal = generate_signal(
                home_team=home,
                away_team=away,
                sport=odds["sport"],
                start_time=start_str,
                book_home=odds["home_prob"],
                book_away=odds["away_prob"],
                our_home=odds["home_prob"],
                our_away=odds["away_prob"],
                n_bookmakers=odds["n_bookmakers"],
                has_sharp=odds["has_sharp"],
                home_form=None,
                away_form=None,
                adjustments={"no_soccer_stats": 0},
            )
            if signal:
                all_predictions.append(signal)
                logger.info(f"  SIGNAL: {signal.pick} {signal.pick_team} (soccer) | conf={signal.confidence:.0%}")

        logger.info(f"\nTotal analyzed: {total_analyzed} | Signals: {len(all_predictions)}")
        logger.info("=" * 55)

        report = format_report(all_predictions, total_analyzed)
        print(report)

        os.makedirs("data", exist_ok=True)
        with open("data/predictor_report.txt", "w") as f:
            f.write(report)
        preds_json = []
        for p in all_predictions:
            preds_json.append({
                "home_team": p.home_team,
                "away_team": p.away_team,
                "sport": p.sport,
                "start_time": p.start_time,
                "pick": p.pick,
                "pick_team": p.pick_team,
                "confidence": p.confidence,
                "edge": p.edge,
                "expected_value": p.expected_value,
                "book_home": p.book_home_prob,
                "book_away": p.book_away_prob,
                "our_home": p.our_home_prob,
                "our_away": p.our_away_prob,
                "home_form": p.home_form,
                "away_form": p.away_form,
                "factors": p.factors,
            })
        with open("data/predictor_signals.json", "w") as f:
            json.dump(preds_json, f, indent=2)
        logger.info("Files saved to data/")

        await send_telegram(all_predictions, total_analyzed)


if __name__ == "__main__":
    asyncio.run(run_predictor())
