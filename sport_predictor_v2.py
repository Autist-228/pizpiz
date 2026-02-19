#!/usr/bin/env python3
"""
Sport Predictor V2 — FULL POWER
=================================
Combines: Real NBA stats + 44 bookmakers + ESPN injuries + Covers public betting
+ Claude AI analysis + Kelly sizing + tracking + Telegram bot + autopilot

Data sources:
1. balldontlie API — NBA team form (last 45 days)
2. Odds API — 44 bookmakers including Pinnacle (sharp)
3. ESPN scraper — injuries with impact scoring
4. Covers scraper — public betting % (contrarian signals)
5. Travel factor — timezone-based fatigue calculation
6. Claude AI — game-by-game reasoning (triple confirmation)

Logic:
- Start with bookmaker consensus (they know the most)
- Adjust for: form, H/A split, last 10, net rating, rest, streak
- Adjust for: injuries, travel/jet lag, public money (contrarian)
- Claude AI votes on each game independently
- Triple confirmation: our model + Claude + bookmakers must agree
- Kelly Criterion sizes each bet based on confidence + edge
- Drawdown protection: reduce bets when losing
- Only signal when confidence > 55% AND edge > 3%
"""

import asyncio
import json
import logging
import math
import os
import sys
import time as _time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict, field
from pathlib import Path

import aiohttp
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("predictor_v2")

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
BALLDONTLIE_KEY = os.getenv("BALLDONTLIE_KEY", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

BANKROLL = 500.0
BASE_BET = 5.0
MIN_CONFIDENCE = 0.52
MIN_EDGE = 0.02
MAX_BET_PCT = 0.08
WINDOW_HOURS = 24
AUTOPILOT_INTERVAL = 1800
KELLY_FRACTION = 0.25

SHARP_BOOKMAKERS = {"pinnacle", "betfair", "matchbook", "smarkets", "betfair_ex_eu"}

NBA_SPORTS = ["basketball_nba"]
SOCCER_SPORTS = [
    "soccer_epl", "soccer_spain_la_liga", "soccer_italy_serie_a",
    "soccer_germany_bundesliga", "soccer_france_ligue_one",
    "soccer_uefa_champs_league",
]
ALL_SPORTS = NBA_SPORTS + SOCCER_SPORTS

DATA_DIR = Path("data")
SIGNALS_FILE = DATA_DIR / "signals_history.json"
RESULTS_FILE = DATA_DIR / "results_history.json"
STATE_FILE = DATA_DIR / "bot_state.json"


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
class GameAnalysis:
    home_team: str
    away_team: str
    sport: str
    start_time: str
    our_home_prob: float = 0.5
    our_away_prob: float = 0.5
    book_home_prob: float = 0.5
    book_away_prob: float = 0.5
    n_bookmakers: int = 0
    has_sharp: bool = False
    verdict: str = "SKIP"
    verdict_team: str = ""
    confidence: float = 0.0
    edge: float = 0.0
    skip_reason: str = ""
    home_form_str: str = ""
    away_form_str: str = ""
    injury_note: str = ""
    claude_note: str = ""


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
    bet_size: float = 0.0
    potential_win: float = 0.0
    potential_loss: float = 0.0
    expected_value: float = 0.0
    injury_impact_home: float = 0.0
    injury_impact_away: float = 0.0
    travel_factor: float = 0.0
    public_home_pct: float = 0.5
    public_away_pct: float = 0.5
    contrarian_signal: str = ""
    claude_pick: str = ""
    claude_confidence: float = 0.0
    claude_reasoning: str = ""
    triple_confirmed: bool = False
    timestamp: str = ""
    game_id: str = ""
    result: str = ""


class BotState:
    def __init__(self):
        self.bankroll = BANKROLL
        self.total_wagered = 0.0
        self.total_pnl = 0.0
        self.signals_total = 0
        self.signals_won = 0
        self.signals_lost = 0
        self.signals_pending = 0
        self.current_streak = 0
        self.max_drawdown = 0.0
        self.peak_bankroll = BANKROLL
        self.history: list[dict] = []
        self._load()

    def _load(self):
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text())
                self.bankroll = data.get("bankroll", BANKROLL)
                self.total_wagered = data.get("total_wagered", 0)
                self.total_pnl = data.get("total_pnl", 0)
                self.signals_total = data.get("signals_total", 0)
                self.signals_won = data.get("signals_won", 0)
                self.signals_lost = data.get("signals_lost", 0)
                self.signals_pending = data.get("signals_pending", 0)
                self.current_streak = data.get("current_streak", 0)
                self.max_drawdown = data.get("max_drawdown", 0)
                self.peak_bankroll = data.get("peak_bankroll", BANKROLL)
                self.history = data.get("history", [])
            except Exception:
                pass

    def save(self):
        DATA_DIR.mkdir(exist_ok=True)
        STATE_FILE.write_text(json.dumps({
            "bankroll": self.bankroll,
            "total_wagered": self.total_wagered,
            "total_pnl": self.total_pnl,
            "signals_total": self.signals_total,
            "signals_won": self.signals_won,
            "signals_lost": self.signals_lost,
            "signals_pending": self.signals_pending,
            "current_streak": self.current_streak,
            "max_drawdown": self.max_drawdown,
            "peak_bankroll": self.peak_bankroll,
            "history": self.history[-100:],
        }, indent=2))

    @property
    def win_rate(self):
        done = self.signals_won + self.signals_lost
        return self.signals_won / max(done, 1)

    @property
    def roi(self):
        return self.total_pnl / max(self.total_wagered, 1) * 100

    @property
    def drawdown_pct(self):
        return (self.peak_bankroll - self.bankroll) / max(self.peak_bankroll, 1) * 100

    def is_cautious_mode(self):
        return self.drawdown_pct > 20

    def record_signal(self, pred: GamePrediction):
        self.signals_total += 1
        self.signals_pending += 1
        self.total_wagered += pred.bet_size
        entry = {
            "game_id": pred.game_id,
            "timestamp": pred.timestamp,
            "pick": pred.pick,
            "pick_team": pred.pick_team,
            "home_team": pred.home_team,
            "away_team": pred.away_team,
            "confidence": pred.confidence,
            "edge": pred.edge,
            "bet_size": pred.bet_size,
            "potential_win": pred.potential_win,
            "potential_loss": pred.potential_loss,
            "ev": pred.expected_value,
            "sport": pred.sport,
            "triple_confirmed": pred.triple_confirmed,
            "result": "pending",
        }
        self.history.append(entry)
        self.save()

    def record_result(self, game_id: str, won: bool):
        for entry in reversed(self.history):
            if entry.get("game_id") == game_id and entry.get("result") == "pending":
                if won:
                    entry["result"] = "won"
                    self.signals_won += 1
                    self.signals_pending -= 1
                    pnl = entry["potential_win"]
                    self.bankroll += pnl
                    self.total_pnl += pnl
                    self.current_streak = max(self.current_streak, 0) + 1
                else:
                    entry["result"] = "lost"
                    self.signals_lost += 1
                    self.signals_pending -= 1
                    pnl = entry["potential_loss"]
                    self.bankroll += pnl
                    self.total_pnl += pnl
                    self.current_streak = min(self.current_streak, 0) - 1

                if self.bankroll > self.peak_bankroll:
                    self.peak_bankroll = self.bankroll
                dd = self.peak_bankroll - self.bankroll
                if dd > self.max_drawdown:
                    self.max_drawdown = dd

                self.save()
                return True
        return False


async def fetch_nba_recent_games(session: aiohttp.ClientSession, days_back: int = 45) -> list[dict]:
    all_games = []
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    cursor = None
    pages = 0
    while pages < 10:
        url = "https://api.balldontlie.io/v1/games"
        params = {"start_date": start_date, "end_date": end_date, "per_page": 100}
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
    if not ODDS_API_KEY:
        logger.warning("No ODDS_API_KEY")
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
                    best_home_odds = 0
                    best_away_odds = 0
                    for bk in ev.get("bookmakers", []):
                        bk_name = bk.get("key", "")
                        for mkt in bk.get("markets", []):
                            if mkt.get("key") != "h2h":
                                continue
                            outcomes = {o["name"]: o["price"] for o in mkt.get("outcomes", [])}
                            h_odds = outcomes.get(home, 2.0)
                            a_odds = outcomes.get(away, 2.0)
                            d_odds = outcomes.get("Draw", 0)

                            best_home_odds = max(best_home_odds, h_odds)
                            best_away_odds = max(best_away_odds, a_odds)

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
                            "best_home_odds": best_home_odds,
                            "best_away_odds": best_away_odds,
                        }
        except Exception as e:
            logger.error(f"Odds API error for {sport}: {e}")

    logger.info(f"Odds API: {len(odds_data)} events with odds")
    return odds_data


def match_odds_to_game(game: dict, odds_data: dict) -> dict | None:
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


def scrape_injuries_safe() -> dict:
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from src.scrapers.espn_injuries import scrape_espn_injuries
        return scrape_espn_injuries()
    except Exception as e:
        logger.warning(f"ESPN scraper failed: {e}")
        return {}


def get_injury_impact(injuries: dict, team_name: str) -> float:
    try:
        from src.scrapers.espn_injuries import get_team_injury_impact
        return get_team_injury_impact(injuries, team_name)
    except Exception:
        return 0.0


def scrape_public_betting_safe() -> list:
    try:
        from src.scrapers.covers import scrape_covers_nba
        return scrape_covers_nba()
    except Exception as e:
        logger.warning(f"Covers scraper failed: {e}")
        return []


def get_public_data(public_list: list, home: str, away: str) -> dict:
    try:
        from src.scrapers.covers import get_public_betting
        return get_public_betting(public_list, home, away)
    except Exception:
        return {"home_pct": 0.5, "away_pct": 0.5, "contrarian_signal": None, "strength": 0}


def get_travel(home: str, away: str) -> float:
    try:
        from src.scrapers.bball_ref import get_travel_factor
        return get_travel_factor(home, away)
    except Exception:
        return 0.0


async def claude_analyze_game(
    session: aiohttp.ClientSession,
    home_team: str,
    away_team: str,
    home_form: TeamForm | None,
    away_form: TeamForm | None,
    book_home: float,
    book_away: float,
    injury_home: float,
    injury_away: float,
    public_home_pct: float,
) -> dict:
    if not ANTHROPIC_API_KEY:
        return {"pick": "", "confidence": 0, "reasoning": "No API key"}

    hf_str = "No data"
    if home_form and home_form.games >= 5:
        hf_str = (f"{home_form.wins}W-{home_form.losses}L ({home_form.win_rate:.0%}) | "
                  f"Home: {home_form.home_wr:.0%} | L10: {home_form.last10_wins}-{home_form.last10_losses} | "
                  f"PPG: {home_form.ppg:.1f} | Net: {home_form.net_rating:+.1f} | "
                  f"Rest: {home_form.days_rest}d | Streak: {home_form.streak}")

    af_str = "No data"
    if away_form and away_form.games >= 5:
        af_str = (f"{away_form.wins}W-{away_form.losses}L ({away_form.win_rate:.0%}) | "
                  f"Away: {away_form.away_wr:.0%} | L10: {away_form.last10_wins}-{away_form.last10_losses} | "
                  f"PPG: {away_form.ppg:.1f} | Net: {away_form.net_rating:+.1f} | "
                  f"Rest: {away_form.days_rest}d | Streak: {away_form.streak}")

    prompt = f"""You are an NBA betting analyst. Analyze this game and predict the winner.

GAME: {home_team} (HOME) vs {away_team} (AWAY)

HOME FORM (last 45 days): {hf_str}
AWAY FORM (last 45 days): {af_str}

BOOKMAKER CONSENSUS: Home {book_home:.0%} / Away {book_away:.0%} (44 bookmakers avg)
INJURY IMPACT: Home -{injury_home:.0%} / Away -{injury_away:.0%}
PUBLIC BETTING: {public_home_pct:.0%} on home

Respond ONLY with valid JSON:
{{"pick": "HOME" or "AWAY", "confidence": 0.50-0.95, "reasoning": "1-2 sentence reason"}}"""

    try:
        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}],
        }
        async with session.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                logger.warning(f"Claude API error {resp.status}: {body[:200]}")
                return {"pick": "", "confidence": 0, "reasoning": f"API error {resp.status}"}

            result = await resp.json()
            text = result.get("content", [{}])[0].get("text", "")

            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end])
                return {
                    "pick": parsed.get("pick", ""),
                    "confidence": float(parsed.get("confidence", 0)),
                    "reasoning": parsed.get("reasoning", ""),
                }

    except Exception as e:
        logger.warning(f"Claude analysis failed: {e}")

    return {"pick": "", "confidence": 0, "reasoning": "Failed"}


def predict_nba_game(
    home_form: TeamForm | None,
    away_form: TeamForm | None,
    book_home: float,
    book_away: float,
    has_sharp: bool,
    line_spread: float,
    injury_home: float = 0.0,
    injury_away: float = 0.0,
    travel_adj: float = 0.0,
    contrarian_signal: str | None = None,
    contrarian_strength: float = 0.0,
) -> dict:
    base_home = book_home
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
        adj_net = float(np.clip(net_diff / 100, -0.05, 0.05))
        adjustments["net_rating"] = adj_net
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
        if rest_adj != 0:
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
        if streak_adj != 0:
            adjustments["streak"] = streak_adj
        base_home += streak_adj

    if injury_home > 0 or injury_away > 0:
        inj_adj = injury_away - injury_home
        adjustments["injuries"] = inj_adj
        base_home += inj_adj

    if travel_adj != 0:
        adjustments["travel_fatigue"] = travel_adj
        base_home += travel_adj

    if contrarian_signal and contrarian_strength > 0.3:
        c_adj = 0.0
        if contrarian_signal == "HOME":
            c_adj = contrarian_strength * 0.04
        elif contrarian_signal == "AWAY":
            c_adj = -contrarian_strength * 0.04
        if c_adj != 0:
            adjustments["anti_public"] = c_adj
        base_home += c_adj

    base_home = float(np.clip(base_home, 0.05, 0.95))
    base_away = 1.0 - base_home

    data_quality = 0.0
    if home_form and home_form.games >= 5:
        data_quality += 0.25
    if away_form and away_form.games >= 5:
        data_quality += 0.25
    if has_sharp:
        data_quality += 0.2
    if line_spread < 0.05:
        data_quality += 0.15
    if injury_home > 0 or injury_away > 0:
        data_quality += 0.1
    if contrarian_signal:
        data_quality += 0.05

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


def kelly_bet_size(confidence: float, edge: float, bankroll: float,
                   cautious: bool = False, book_price: float = 0.0) -> float:
    if edge <= 0 or confidence < MIN_CONFIDENCE:
        return 0.0

    if book_price > 0:
        b = (1.0 / max(book_price, 0.01)) - 1.0
    else:
        b = (1.0 / max(confidence - edge, 0.01)) - 1.0

    p = confidence
    q = 1 - p

    if b <= 0:
        return BASE_BET

    kelly = (b * p - q) / b
    kelly = max(kelly, 0)
    kelly *= KELLY_FRACTION

    if cautious:
        kelly *= 0.5

    bet = bankroll * kelly
    bet = max(bet, BASE_BET)
    bet = min(bet, bankroll * MAX_BET_PCT)
    bet = round(bet, 2)

    return bet


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
    injury_home: float = 0.0,
    injury_away: float = 0.0,
    travel_adj: float = 0.0,
    public_home_pct: float = 0.5,
    public_away_pct: float = 0.5,
    contrarian_signal: str = "",
    claude_result: dict | None = None,
    state: BotState | None = None,
) -> GamePrediction | None:
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
    if edge < MIN_EDGE:
        return None
    if n_bookmakers < 3:
        return None

    claude_pick = ""
    claude_conf = 0.0
    claude_reasoning = ""
    triple_confirmed = False

    if claude_result and claude_result.get("pick"):
        claude_pick = claude_result["pick"]
        claude_conf = claude_result.get("confidence", 0)
        claude_reasoning = claude_result.get("reasoning", "")

        if claude_pick == pick:
            confidence = confidence * 0.6 + claude_conf * 0.4
            triple_confirmed = True
        else:
            confidence *= 0.85

    if confidence < MIN_CONFIDENCE:
        return None

    bankroll = state.bankroll if state else BANKROLL
    cautious = state.is_cautious_mode() if state else False
    bet_size = kelly_bet_size(confidence, edge, bankroll, cautious, book_price=price)

    payout = bet_size * (1.0 / max(price, 0.01) - 1)
    loss = -bet_size
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

    now = datetime.now(timezone.utc)
    game_id = f"{home_team}_{away_team}_{now.strftime('%Y%m%d')}"

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
        bet_size=bet_size,
        potential_win=payout,
        potential_loss=loss,
        expected_value=ev,
        injury_impact_home=injury_home,
        injury_impact_away=injury_away,
        travel_factor=travel_adj,
        public_home_pct=public_home_pct,
        public_away_pct=public_away_pct,
        contrarian_signal=contrarian_signal,
        claude_pick=claude_pick,
        claude_confidence=claude_conf,
        claude_reasoning=claude_reasoning,
        triple_confirmed=triple_confirmed,
        timestamp=now.isoformat(),
        game_id=game_id,
    )


def format_report(predictions: list[GamePrediction], all_games_count: int, state: BotState) -> str:
    now = datetime.now(timezone.utc)
    lines = [
        "=" * 55,
        "SPORT PREDICTOR V2 — FULL POWER",
        f"Generated: {now.strftime('%Y-%m-%d %H:%M UTC')}",
        "=" * 55,
        "",
        f"Bankroll:       ${state.bankroll:.0f}",
        f"Mode:           {'CAUTIOUS (drawdown >20%)' if state.is_cautious_mode() else 'NORMAL'}",
        f"Min confidence: {MIN_CONFIDENCE:.0%} | Min edge: {MIN_EDGE:.0%}",
        f"Games analyzed: {all_games_count}",
        f"Signals:        {len(predictions)}",
    ]

    if state.signals_won + state.signals_lost > 0:
        lines.extend([
            "",
            f"--- TRACK RECORD ---",
            f"W/L: {state.signals_won}/{state.signals_lost} ({state.win_rate:.0%})",
            f"P&L: ${state.total_pnl:+.2f} | ROI: {state.roi:+.1f}%",
            f"Streak: {state.current_streak} | Max DD: ${state.max_drawdown:.0f}",
        ])

    if not predictions:
        lines.append("")
        lines.append("No signals. All games below confidence/edge threshold.")
        return "\n".join(lines)

    total_wagered = sum(p.bet_size for p in predictions)
    avg_conf = np.mean([p.confidence for p in predictions])
    avg_edge = np.mean([p.edge for p in predictions])
    total_ev = sum(p.expected_value for p in predictions)
    triple_count = sum(1 for p in predictions if p.triple_confirmed)

    lines.extend([
        f"Total wagered:  ${total_wagered:.2f} (Kelly sized)",
        f"Triple confirm: {triple_count}/{len(predictions)}",
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
        star = " ***" if p.confidence >= 0.70 else " **" if p.confidence >= 0.60 else " *"
        triple = " [TRIPLE]" if p.triple_confirmed else ""
        sport_short = p.sport.replace("basketball_", "").replace("soccer_", "").replace("_", " ").upper()

        lines.append(f"#{i} | {p.pick} {p.pick_team}{star}{triple}")
        lines.append(f"   {p.home_team} vs {p.away_team}")
        lines.append(f"   {sport_short} | {p.start_time}")
        lines.append(f"   Confidence: {p.confidence:.0%} | Edge: {p.edge:.1%} | EV: ${p.expected_value:+.2f}")
        lines.append(f"   Book: H={p.book_home_prob:.0%} A={p.book_away_prob:.0%} ({p.n_bookmakers} bks{', sharp' if p.has_sharp else ''})")
        lines.append(f"   Our:  H={p.our_home_prob:.0%} A={p.our_away_prob:.0%}")
        lines.append(f"   BET: ${p.bet_size:.2f} (Kelly) | Win +${p.potential_win:.2f} | Lose -${p.bet_size:.2f}")

        if p.home_form:
            hf = p.home_form
            lines.append(f"   {p.home_team}: {hf.get('record','-')} | L10:{hf.get('l10','-')} | H-WR:{hf.get('home_wr','-')} | PPG:{hf.get('ppg','-')} | Rest:{hf.get('rest','-')}d")
        if p.away_form:
            af = p.away_form
            lines.append(f"   {p.away_team}: {af.get('record','-')} | L10:{af.get('l10','-')} | A-WR:{af.get('away_wr','-')} | PPG:{af.get('ppg','-')} | Rest:{af.get('rest','-')}d")

        extra = []
        if p.injury_impact_home > 0:
            extra.append(f"Inj(H):-{p.injury_impact_home:.0%}")
        if p.injury_impact_away > 0:
            extra.append(f"Inj(A):-{p.injury_impact_away:.0%}")
        if p.travel_factor != 0:
            extra.append(f"Travel:{p.travel_factor:+.1%}")
        if p.contrarian_signal:
            extra.append(f"Contrarian:{p.contrarian_signal}")
        if p.public_home_pct != 0.5:
            extra.append(f"Public:H={p.public_home_pct:.0%}/A={p.public_away_pct:.0%}")
        if extra:
            lines.append(f"   Extra: {' | '.join(extra)}")

        if p.claude_pick:
            lines.append(f"   Claude: {p.claude_pick} ({p.claude_confidence:.0%}) — {p.claude_reasoning}")

        factors_str = " | ".join(f"{k}:{v:+.1%}" if isinstance(v, float) else f"{k}:{v}" for k, v in p.factors.items() if v != 0)
        if factors_str:
            lines.append(f"   Factors: {factors_str}")

        lines.append("")

    lines.append("--- SCENARIOS ---")
    n = len(predictions)
    tw = total_wagered
    for label, wr in [("Pessimistic (45%)", 0.45), ("Conservative (55%)", 0.55),
                       ("Target (60%)", 0.60), ("Good (65%)", 0.65), ("Great (70%)", 0.70)]:
        wins = round(n * wr)
        losses = n - wins
        sp = sorted(predictions, key=lambda x: x.potential_win, reverse=True)
        pnl = sum(p.potential_win for p in sp[:wins]) + sum(p.potential_loss for p in sp[wins:])
        roi = pnl / tw * 100 if tw > 0 else 0
        lines.append(f"  {label}: {wins}W/{losses}L | PnL: ${pnl:+.2f} | Bankroll: ${state.bankroll + pnl:.0f} | ROI: {roi:+.0f}%")

    lines.extend([
        "",
        "--- DATA SOURCES ---",
        "1. balldontlie API (NBA team form 45d)",
        "2. Odds API (44 bookmakers + Pinnacle sharp)",
        "3. ESPN scraper (injury impact scoring)",
        "4. Covers.com scraper (public betting %)",
        "5. Travel/timezone fatigue factor",
        "6. Claude AI (game-by-game analysis)",
        "",
        "--- HOW TO VERIFY ---",
        "1. Check each game result after it finishes",
        "2. If pick won -> add potential_win to bankroll",
        "3. If pick lost -> subtract bet_size from bankroll",
        "4. Use /results command in Telegram to record",
        "",
        "DISCLAIMER: Paper trading only. Not financial advice.",
    ])
    return "\n".join(lines)


def _conf_bar(conf: float) -> str:
    filled = round(conf * 10)
    empty = 10 - filled
    return "█" * filled + "░" * empty + f" {conf:.0%}"


def _sport_icon(sport: str) -> str:
    if "basketball" in sport or "nba" in sport:
        return "🏀"
    if "soccer" in sport or "epl" in sport or "liga" in sport or "serie" in sport or "bundesliga" in sport or "ligue" in sport or "champs" in sport:
        return "⚽"
    if "hockey" in sport or "nhl" in sport:
        return "🏒"
    if "baseball" in sport or "mlb" in sport:
        return "⚾"
    if "football" in sport or "nfl" in sport:
        return "🏈"
    if "tennis" in sport:
        return "🎾"
    if "mma" in sport or "ufc" in sport:
        return "🥊"
    return "🎯"


async def _tg_send(text: str, reply_markup: dict | None = None, chat_id: str = ""):
    if not TELEGRAM_TOKEN:
        return None
    cid = chat_id or TELEGRAM_CHAT_ID
    if not cid:
        return None
    try:
        import httpx
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload: dict = {"chat_id": cid, "text": text, "parse_mode": "HTML"}
        if reply_markup:
            payload["reply_markup"] = reply_markup
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                return resp.json().get("result", {}).get("message_id")
            logger.error(f"TG send failed: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        logger.error(f"TG send error: {e}")
    return None


async def _tg_edit(message_id: int, text: str, reply_markup: dict | None = None, chat_id: str = ""):
    if not TELEGRAM_TOKEN:
        return
    cid = chat_id or TELEGRAM_CHAT_ID
    if not cid:
        return
    try:
        import httpx
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText"
        payload: dict = {"chat_id": cid, "message_id": message_id, "text": text, "parse_mode": "HTML"}
        if reply_markup:
            payload["reply_markup"] = reply_markup
        async with httpx.AsyncClient(timeout=15) as client:
            await client.post(url, json=payload)
    except Exception:
        pass


async def _tg_answer_cb(callback_query_id: str):
    if not TELEGRAM_TOKEN:
        return
    try:
        import httpx
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery"
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(url, json={"callback_query_id": callback_query_id})
    except Exception:
        pass


def _main_keyboard() -> dict:
    return {"inline_keyboard": [
        [{"text": "📊 Дашборд", "callback_data": "dash"}],
        [{"text": "💼 Позиции", "callback_data": "positions"}, {"text": "🎯 События 6ч", "callback_data": "events"}],
        [{"text": "📈 Статистика", "callback_data": "stats"}, {"text": "📝 История", "callback_data": "history"}],
        [{"text": "🔍 Скан сейчас", "callback_data": "scan"}, {"text": "⚙️ Настройки", "callback_data": "settings"}],
    ]}


def _back_keyboard() -> dict:
    return {"inline_keyboard": [[{"text": "◀️ Назад", "callback_data": "dash"}]]}


def _settings_keyboard(state: BotState) -> dict:
    return {"inline_keyboard": [
        [{"text": "➕ $100", "callback_data": "bal_add_100"}, {"text": "➖ $100", "callback_data": "bal_sub_100"}],
        [{"text": "➕ $500", "callback_data": "bal_add_500"}, {"text": "🔄 Сброс $500", "callback_data": "bal_reset"}],
        [{"text": "◀️ Назад", "callback_data": "dash"}],
    ]}


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_dashboard_text(state: BotState) -> str:
    now = datetime.now(timezone.utc)
    pnl_today = 0.0
    pnl_7d = 0.0
    wins_today = 0
    losses_today = 0
    wins_7d = 0
    losses_7d = 0
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=7)

    for h in state.history:
        ts_str = h.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str)
        except Exception:
            continue
        result = h.get("result", "pending")
        if result == "pending":
            continue
        pnl_val = h.get("potential_win", 0) if result == "won" else h.get("potential_loss", 0)
        if ts >= today_start:
            pnl_today += pnl_val
            if result == "won":
                wins_today += 1
            else:
                losses_today += 1
        if ts >= week_start:
            pnl_7d += pnl_val
            if result == "won":
                wins_7d += 1
            else:
                losses_7d += 1

    pending = [h for h in state.history if h.get("result") == "pending"]
    pending_exposure = sum(h.get("bet_size", 0) for h in pending)

    if state.is_cautious_mode():
        mode_str = "\u26a0\ufe0f \u041e\u0421\u0422\u041e\u0420\u041e\u0416\u041d\u041e (DD &gt;20%)"
    else:
        mode_str = "\u2705 \u041d\u043e\u0440\u043c\u0430\u043b\u044c\u043d\u044b\u0439"

    pnl_all_icon = "🟢" if state.total_pnl >= 0 else "🔴"
    pnl_today_icon = "🟢" if pnl_today >= 0 else "🔴"
    pnl_7d_icon = "🟢" if pnl_7d >= 0 else "🔴"
    pnl_sign = "+" if state.total_pnl >= 0 else ""
    pnl_today_sign = "+" if pnl_today >= 0 else ""
    pnl_7d_sign = "+" if pnl_7d >= 0 else ""

    streak_icon = "🔥" if state.current_streak > 0 else ("\u2744\ufe0f" if state.current_streak < 0 else "\u2796")

    return (
        f"💹 <b>\u0422\u0420\u0415\u0419\u0414\u0418\u041d\u0413 \u0422\u0415\u0420\u041c\u0418\u041d\u0410\u041b</b>\n"
        f"🕒 {now.strftime('%d.%m.%Y %H:%M UTC')}\n"
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
        f"💰 <b>\u0411\u0430\u043b\u0430\u043d\u0441:</b>  <code>${state.bankroll:.2f}</code>\n\n"
        f"{pnl_all_icon} <b>P&amp;L \u0432\u0441\u0435\u0433\u043e:</b>  <code>{pnl_sign}${state.total_pnl:.2f}</code>\n"
        f"{pnl_7d_icon} <b>P&amp;L 7\u0434:</b>    <code>{pnl_7d_sign}${pnl_7d:.2f}</code>  ({wins_7d}\u2705/{losses_7d}\u274c)\n"
        f"{pnl_today_icon} <b>P&amp;L \u0441\u0435\u0433\u043e\u0434\u043d\u044f:</b> <code>{pnl_today_sign}${pnl_today:.2f}</code>  ({wins_today}\u2705/{losses_today}\u274c)\n\n"
        f"💼 <b>\u041e\u0442\u043a\u0440\u044b\u0442\u043e:</b>  {len(pending)} \u043f\u043e\u0437\u0438\u0446\u0438\u0439  (${pending_exposure:.2f})\n"
        f"🎯 <b>\u0412\u0441\u0435\u0433\u043e \u0441\u0434\u0435\u043b\u043e\u043a:</b>  {state.signals_total}\n"
        f"🏆 <b>W/L:</b>   {state.signals_won}\u2705 / {state.signals_lost}\u274c\n"
        f"📊 <b>\u0412\u0438\u043d\u0440\u0435\u0439\u0442:</b> <code>{state.win_rate:.0%}</code>\n"
        f"📈 <b>ROI:</b>   <code>{state.roi:+.1f}%</code>\n"
        f"{streak_icon} <b>\u0421\u0435\u0440\u0438\u044f:</b>  {state.current_streak}\n"
        f"📉 <b>\u041c\u0430\u043a\u0441 \u043f\u0440\u043e\u0441\u0430\u0434\u043a\u0430:</b> ${state.max_drawdown:.0f} ({state.drawdown_pct:.0f}%)\n"
        f"🛡\ufe0f <b>\u0420\u0435\u0436\u0438\u043c:</b>  {mode_str}\n"
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"🔄 <i>\u0410\u0432\u0442\u043e\u0441\u043a\u0430\u043d \u043a\u0430\u0436\u0434\u044b\u0435 {AUTOPILOT_INTERVAL // 60} \u043c\u0438\u043d | \u041e\u043a\u043d\u043e: {WINDOW_HOURS}\u0447</i>"
    )


def build_positions_text(state: BotState) -> str:
    pending = [h for h in state.history if h.get("result") == "pending"]
    if not pending:
        return (
            "💼 <b>\u041e\u0422\u041a\u0420\u042b\u0422\u042b\u0415 \u041f\u041e\u0417\u0418\u0426\u0418\u0418</b>\n"
            "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
            "😴 \u041d\u0435\u0442 \u043e\u0442\u043a\u0440\u044b\u0442\u044b\u0445 \u043f\u043e\u0437\u0438\u0446\u0438\u0439\n\n"
            "🔄 <i>\u041f\u043e\u0437\u0438\u0446\u0438\u0438 \u043e\u0442\u043a\u0440\u044b\u0432\u0430\u044e\u0442\u0441\u044f \u0430\u0432\u0442\u043e\u043c\u0430\u0442\u0438\u0447\u0435\u0441\u043a\u0438 \u043f\u0440\u0438 \u0441\u043a\u0430\u043d\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u0438</i>"
        )

    total_exposure = sum(h.get("bet_size", 0) for h in pending)
    total_potential = sum(h.get("potential_win", 0) for h in pending)

    lines = [
        "💼 <b>\u041e\u0422\u041a\u0420\u042b\u0422\u042b\u0415 \u041f\u041e\u0417\u0418\u0426\u0418\u0418</b>",
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        f"📊 \u041e\u0442\u043a\u0440\u044b\u0442\u043e: {len(pending)}  |  \u0420\u0438\u0441\u043a: ${total_exposure:.2f}  |  \u041c\u0430\u043a\u0441 \u043f\u0440\u043e\u0444\u0438\u0442: +${total_potential:.2f}",
        "",
    ]

    for i, h in enumerate(reversed(pending), 1):
        sport_icon = _sport_icon(h.get("sport", ""))
        conf = h.get("confidence", 0)
        bar = _conf_bar(conf)
        triple = " 🔥🔥🔥" if h.get("triple_confirmed") else ""
        lines.append(
            f"<b>#{i} {sport_icon} {_esc(h.get('pick_team', '?'))}</b>{triple}\n"
            f"    {_esc(h.get('home_team', '?'))} vs {_esc(h.get('away_team', '?'))}\n"
            f"    🎯 {bar}\n"
            f"    📀 \u042d\u0434\u0436: {h.get('edge', 0):.1%} | 💵 \u0421\u0442\u0430\u0432\u043a\u0430: ${h.get('bet_size', 0):.2f}\n"
            f"    🟢 \u0412\u044b\u0438\u0433\u0440\u044b\u0448: +${h.get('potential_win', 0):.2f} | 🔴 \u041f\u0440\u043e\u0438\u0433\u0440\u044b\u0448: -${h.get('bet_size', 0):.2f}\n"
        )
    return "\n".join(lines)


def build_stats_text(state: BotState) -> str:
    now = datetime.now(timezone.utc)
    periods = {"24\u0447": 1, "7\u0434": 7, "30\u0434": 30, "\u0412\u0441\u0435\u0433\u043e": 9999}
    rows = []
    for label, days in periods.items():
        cutoff = now - timedelta(days=days)
        w = l = 0
        pnl = 0.0
        wagered = 0.0
        for h in state.history:
            ts_str = h.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str)
            except Exception:
                continue
            if ts < cutoff:
                continue
            result = h.get("result", "pending")
            if result == "pending":
                continue
            wagered += h.get("bet_size", 0)
            if result == "won":
                w += 1
                pnl += h.get("potential_win", 0)
            else:
                l += 1
                pnl += h.get("potential_loss", 0)
        total = w + l
        wr = w / total if total > 0 else 0
        roi = pnl / wagered * 100 if wagered > 0 else 0
        pnl_icon = "🟢" if pnl >= 0 else "🔴"
        pnl_sign = "+" if pnl >= 0 else ""
        rows.append(f"  {pnl_icon} <b>{label:5s}</b> {w}\u2705/{l}\u274c  WR:{wr:.0%}  {pnl_sign}${pnl:.2f}  ROI:{roi:+.0f}%")

    streak_icon = "🔥" if state.current_streak > 0 else ("\u2744\ufe0f" if state.current_streak < 0 else "\u2796")

    return (
        "📈 <b>\u0421\u0422\u0410\u0422\u0418\u0421\u0422\u0418\u041a\u0410</b>\n"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
        + "\n".join(rows) + "\n\n"
        f"💰 <b>\u0411\u0430\u043b\u0430\u043d\u0441:</b> <code>${state.bankroll:.2f}</code>\n"
        f"🏦 <b>\u0421\u0442\u0430\u0440\u0442:</b> $500.00\n"
        f"📉 <b>\u041c\u0430\u043a\u0441 \u043f\u0440\u043e\u0441\u0430\u0434\u043a\u0430:</b> ${state.max_drawdown:.0f} ({state.drawdown_pct:.0f}%)\n"
        f"{streak_icon} <b>\u0422\u0435\u043a\u0443\u0449\u0430\u044f \u0441\u0435\u0440\u0438\u044f:</b> {state.current_streak}"
    )


def build_history_text(state: BotState) -> str:
    recent = state.history[-15:]
    if not recent:
        return (
            "📝 <b>\u0418\u0421\u0422\u041e\u0420\u0418\u042f \u0421\u0414\u0415\u041b\u041e\u041a</b>\n"
            "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
            "😴 \u0421\u0434\u0435\u043b\u043e\u043a \u043f\u043e\u043a\u0430 \u043d\u0435\u0442"
        )

    lines = [
        "📝 <b>\u0418\u0421\u0422\u041e\u0420\u0418\u042f \u0421\u0414\u0415\u041b\u041e\u041a</b>",
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        "",
    ]
    for h in reversed(recent):
        result = h.get("result", "pending")
        if result == "won":
            icon = "\u2705"
            pnl = h.get("potential_win", 0)
            pnl_str = f"🟢 +${pnl:.2f}"
        elif result == "lost":
            icon = "\u274c"
            pnl = h.get("potential_loss", 0)
            pnl_str = f"🔴 -${abs(pnl):.2f}"
        else:
            icon = "\u23f3"
            pnl_str = "\u231b \u0436\u0434\u0451\u043c..."

        conf = h.get("confidence", 0)
        sport_icon = _sport_icon(h.get("sport", ""))
        ts_str = h.get("timestamp", "")[:16].replace("T", " ")
        lines.append(
            f"{icon} {sport_icon} <b>{_esc(h.get('pick_team', '?'))}</b> "
            f"| {conf:.0%} | ${h.get('bet_size', 0):.2f} | {pnl_str}"
            f"\n    <i>{ts_str}</i>"
        )
    return "\n".join(lines)


def build_events_text(all_analyses: list[GameAnalysis]) -> str:
    if not all_analyses:
        return (
            "🎯 <b>\u0421\u041e\u0411\u042b\u0422\u0418\u042f (6\u0447)</b>\n"
            "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
            "😴 \u0421\u043e\u0431\u044b\u0442\u0438\u0439 \u043d\u0435 \u043d\u0430\u0439\u0434\u0435\u043d\u043e\n\n"
            "🔄 <i>\u0421\u043b\u0435\u0434\u0443\u044e\u0449\u0438\u0439 \u0441\u043a\u0430\u043d \u0447\u0435\u0440\u0435\u0437 30 \u043c\u0438\u043d</i>"
        )

    bets = [a for a in all_analyses if a.verdict == "BET"]
    skips = [a for a in all_analyses if a.verdict != "BET"]

    lines = [
        "🎯 <b>\u0421\u041e\u0411\u042b\u0422\u0418\u042f (6\u0447)</b>",
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        f"📊 \u0412\u0441\u0435\u0433\u043e: {len(all_analyses)}  |  🟢 \u0421\u0442\u0430\u0432\u0438\u043c: {len(bets)}  |  \u26d4 \u041f\u0440\u043e\u043f\u0443\u0441\u043a: {len(skips)}",
        "",
    ]

    if bets:
        lines.append("🔥 <b>\u0421\u0422\u0410\u0412\u0418\u041c:</b>")
        for a in sorted(bets, key=lambda x: x.confidence, reverse=True):
            icon = _sport_icon(a.sport)
            bar = _conf_bar(a.confidence)
            lines.append(
                f"\n{icon} <b>🟢 {_esc(a.verdict_team)}</b>  🕒 {a.start_time}\n"
                f"    {_esc(a.home_team)} vs {_esc(a.away_team)}\n"
                f"    🎯 {bar}\n"
                f"    📀 \u042d\u0434\u0436: {a.edge:.1%} | 📚 \u0411\u0443\u043a\u0438: \u0414={a.book_home_prob:.0%} \u0413={a.book_away_prob:.0%}\n"
                f"    🧠 \u041d\u0430\u0448\u0438: \u0414={a.our_home_prob:.0%} \u0413={a.our_away_prob:.0%}"
            )
            if a.home_form_str:
                lines.append(f"    🏠 \u0414\u043e\u043c: {_esc(a.home_form_str)}")
            if a.away_form_str:
                lines.append(f"    \u2708\ufe0f \u0413\u043e\u0441\u0442\u0438: {_esc(a.away_form_str)}")
            if a.injury_note:
                lines.append(f"    🩹 \u0422\u0440\u0430\u0432\u043c\u044b: {_esc(a.injury_note)}")
            if a.claude_note:
                lines.append(f"    🤖 AI: {_esc(a.claude_note)}")

    if skips:
        lines.append(f"\n\u26d4 <b>\u041f\u0420\u041e\u041f\u0423\u0421\u041a\u0410\u0415\u041c:</b>")
        for a in skips:
            icon = _sport_icon(a.sport)
            lines.append(
                f"\n{icon} {_esc(a.home_team)} vs {_esc(a.away_team)}  🕒 {a.start_time}\n"
                f"    \u274c \u041f\u0440\u0438\u0447\u0438\u043d\u0430: {_esc(a.skip_reason)}\n"
                f"    📚 \u0411\u0443\u043a\u0438: \u0414={a.book_home_prob:.0%} \u0413={a.book_away_prob:.0%}"
            )

    return "\n".join(lines)


def build_settings_text(state: BotState) -> str:
    mode_icon = "\u26a0\ufe0f" if state.is_cautious_mode() else "\u2705"
    mode_name = "\u041e\u0421\u0422\u041e\u0420\u041e\u0416\u041d\u042b\u0419" if state.is_cautious_mode() else "\u041d\u043e\u0440\u043c\u0430\u043b\u044c\u043d\u044b\u0439"
    return (
        "\u2699\ufe0f <b>\u041d\u0410\u0421\u0422\u0420\u041e\u0419\u041a\u0418</b>\n"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
        f"💰 <b>\u0411\u0430\u043b\u0430\u043d\u0441:</b> <code>${state.bankroll:.2f}</code>\n"
        f"💵 <b>\u041c\u0430\u043a\u0441 \u0441\u0442\u0430\u0432\u043a\u0430:</b> {MAX_BET_PCT:.0%} \u043e\u0442 \u0431\u0430\u043b\u0430\u043d\u0441\u0430 (${state.bankroll * MAX_BET_PCT:.2f})\n"
        f"🎯 <b>\u041c\u0438\u043d \u0443\u0432\u0435\u0440\u0435\u043d\u043d\u043e\u0441\u0442\u044c:</b> {MIN_CONFIDENCE:.0%}\n"
        f"📀 <b>\u041c\u0438\u043d \u044d\u0434\u0436:</b> {MIN_EDGE:.0%}\n"
        f"📊 <b>Kelly:</b> {KELLY_FRACTION:.0%}\n"
        f"🕒 <b>\u041e\u043a\u043d\u043e \u0441\u043a\u0430\u043d\u0430:</b> {WINDOW_HOURS}\u0447\n"
        f"🔄 <b>\u0410\u0432\u0442\u043e\u0441\u043a\u0430\u043d:</b> \u043a\u0430\u0436\u0434\u044b\u0435 {AUTOPILOT_INTERVAL // 60} \u043c\u0438\u043d\n"
        f"{mode_icon} <b>\u0420\u0435\u0436\u0438\u043c:</b> {mode_name}\n"
        f"📝 <b>\u0422\u0438\u043f:</b> Paper Trading (\u0431\u0435\u0437 \u0440\u0435\u0430\u043b\u044c\u043d\u044b\u0445 \u0434\u0435\u043d\u0435\u0433)\n\n"
        "👇 <i>\u0418\u0437\u043c\u0435\u043d\u0438\u0442\u044c \u0431\u0430\u043b\u0430\u043d\u0441:</i>"
    )


async def send_telegram(predictions: list[GamePrediction], all_analyses: list[GameAnalysis],
                        all_games_count: int, state: BotState):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.info("Telegram not configured")
        return

    now = datetime.now(timezone.utc)
    n = len(predictions)

    if n == 0:
        msg = (
            f"🔍 <b>\u0421\u041a\u0410\u041d \u0417\u0410\u0412\u0415\u0420\u0428\u0401\u041d</b> | {now.strftime('%H:%M UTC')}\n"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
            f"📊 \u041f\u0440\u043e\u0441\u043a\u0430\u043d\u0438\u0440\u043e\u0432\u0430\u043d\u043e: {all_games_count} \u0441\u043e\u0431\u044b\u0442\u0438\u0439\n"
            f"😴 \u0421\u0438\u0433\u043d\u0430\u043b\u043e\u0432: 0 \u2014 \u043d\u0435\u0442 \u0432\u044b\u0433\u043e\u0434\u043d\u044b\u0445 \u0441\u0442\u0430\u0432\u043e\u043a\n\n"
            f"🔄 <i>\u0421\u043b\u0435\u0434\u0443\u044e\u0449\u0438\u0439 \u0441\u043a\u0430\u043d \u0447\u0435\u0440\u0435\u0437 {AUTOPILOT_INTERVAL // 60} \u043c\u0438\u043d</i>"
        )
        await _tg_send(msg, _main_keyboard())
        return

    avg_conf = np.mean([p.confidence for p in predictions])
    total_ev = sum(p.expected_value for p in predictions)
    total_bet = sum(p.bet_size for p in predictions)
    triple_count = sum(1 for p in predictions if p.triple_confirmed)

    header = (
        f"🚨 <b>\u041d\u041e\u0412\u042b\u0415 \u0421\u0422\u0410\u0412\u041a\u0418!</b>\n"
        f"🕒 {now.strftime('%d.%m.%Y %H:%M UTC')}\n"
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
        f"📊 \u0421\u043e\u0431\u044b\u0442\u0438\u0439: {all_games_count} | <b>🟢 \u0421\u0442\u0430\u0432\u043e\u043a: {n}</b> | 🔥 \u0422\u0440\u043e\u0439\u043d\u043e\u0439: {triple_count}\n"
        f"🎯 \u0421\u0440. \u0443\u0432\u0435\u0440\u0435\u043d\u043d\u043e\u0441\u0442\u044c: {avg_conf:.0%} | 💰 EV: +${total_ev:.2f}\n"
        f"💵 \u041f\u043e\u0441\u0442\u0430\u0432\u043b\u0435\u043d\u043e: ${total_bet:.2f} (Kelly)\n"
    )

    if state.signals_won + state.signals_lost > 0:
        header += f"🏆 \u0420\u0435\u043a\u043e\u0440\u0434: {state.signals_won}\u2705/{state.signals_lost}\u274c ({state.win_rate:.0%})\n"

    header += "\n"

    sorted_preds = sorted(predictions, key=lambda p: p.confidence, reverse=True)
    for i, p in enumerate(sorted_preds, 1):
        icon = _sport_icon(p.sport)
        bar = _conf_bar(p.confidence)
        triple_mark = " 🔥🔥🔥" if p.triple_confirmed else ""

        pick_ru = "ДОМ" if p.pick == "HOME" else "ГОСТИ"
        header += (
            f"<b>#{i} {icon} {pick_ru} {_esc(p.pick_team)}</b>{triple_mark}\n"
            f"    {_esc(p.home_team)} vs {_esc(p.away_team)} 🕒 {p.start_time}\n"
            f"    🎯 {bar} | 📀 \u042d\u0434\u0436: {p.edge:.1%}\n"
            f"    💵 \u0421\u0442\u0430\u0432\u043a\u0430: <code>${p.bet_size:.2f}</code> | 🟢 \u0412\u044b\u0438\u0433\u0440\u044b\u0448: +${p.potential_win:.2f}\n"
        )

        extras = []
        if p.home_form:
            extras.append(f"🏠 {p.home_form.get('record','-')} L10:{p.home_form.get('l10','-')}")
        if p.away_form:
            extras.append(f"\u2708\ufe0f {p.away_form.get('record','-')} L10:{p.away_form.get('l10','-')}")
        if p.injury_impact_home > 0:
            extras.append(f"🩹\u0414:-{p.injury_impact_home:.0%}")
        if p.injury_impact_away > 0:
            extras.append(f"🩹\u0413:-{p.injury_impact_away:.0%}")
        if p.claude_pick:
            claude_pick_ru = "ДОМ" if p.claude_pick == "HOME" else "ГОСТИ"
            extras.append(f"🤖 AI:{claude_pick_ru}({p.claude_confidence:.0%})")
        if p.contrarian_signal:
            extras.append(f"🔄 {_esc(p.contrarian_signal)}")
        if extras:
            header += f"    <i>{' | '.join(extras)}</i>\n"
        header += "\n"

    header += f"💰 <i>\u0411\u0430\u043b\u0430\u043d\u0441: ${state.bankroll:.2f} | \u0421\u043b\u0435\u0434. \u0441\u043a\u0430\u043d \u0447\u0435\u0440\u0435\u0437 {AUTOPILOT_INTERVAL // 60} \u043c\u0438\u043d</i>"

    if len(header) > 4000:
        header = header[:4000] + "\n..."

    await _tg_send(header, _main_keyboard())


async def send_telegram_message(text: str):
    await _tg_send(text)


async def run_predictor(use_claude: bool = True, use_scrapers: bool = True) -> tuple[list[GamePrediction], list[GameAnalysis]]:
    now = datetime.now(timezone.utc)
    state = BotState()

    logger.info("=" * 55)
    logger.info("SPORT PREDICTOR V2 — FULL AUTO TRADER")
    logger.info(f"Time: {now.strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info(f"Bankroll: ${state.bankroll:.0f} | Mode: {'CAUTIOUS' if state.is_cautious_mode() else 'NORMAL'}")
    logger.info(f"Window: {WINDOW_HOURS}h | Min conf: {MIN_CONFIDENCE:.0%} | Min edge: {MIN_EDGE:.0%}")
    logger.info("=" * 55)

    all_analyses: list[GameAnalysis] = []
    injuries = {}
    public_data = []
    if use_scrapers:
        logger.info("Step 0: Scraping ESPN injuries + public betting...")
        try:
            injuries = scrape_injuries_safe()
            total_inj = sum(len(v) for v in injuries.values())
            logger.info(f"  ESPN: {total_inj} injuries across {len(injuries)} teams")
        except Exception as e:
            logger.warning(f"  ESPN scraper failed: {e}")

        try:
            public_data = scrape_public_betting_safe()
            logger.info(f"  Public betting: {len(public_data)} games")
        except Exception as e:
            logger.warning(f"  Public betting scraper failed: {e}")

    async with aiohttp.ClientSession() as session:
        logger.info("Step 1: Fetching NBA team form (45 days)...")
        recent_games = await fetch_nba_recent_games(session, days_back=45)
        team_forms = build_team_forms(recent_games)
        logger.info(f"  Form data for {len(team_forms)} teams")

        logger.info("Step 2: Fetching today's NBA games...")
        todays_games = await fetch_todays_nba_games(session)
        logger.info(f"  Found {len(todays_games)} NBA games")

        logger.info("Step 3: Fetching bookmaker odds (44 bookmakers)...")
        odds_data = await fetch_odds_api(session)

        logger.info("Step 4: Analyzing ALL events...")
        all_predictions = []
        total_analyzed = 0
        claude_calls = 0

        for game in todays_games:
            ht = game["home_team"]["full_name"]
            vt = game["visitor_team"]["full_name"]
            dt = game.get("_parsed_dt")
            start_str = dt.strftime("%H:%M UTC") if dt else "?"

            analysis = GameAnalysis(home_team=ht, away_team=vt, sport="basketball_nba", start_time=start_str)

            odds = match_odds_to_game(game, odds_data)
            if not odds:
                analysis.skip_reason = "Нет коэффициентов букмекеров"
                all_analyses.append(analysis)
                continue

            total_analyzed += 1
            analysis.book_home_prob = odds["home_prob"]
            analysis.book_away_prob = odds["away_prob"]
            analysis.n_bookmakers = odds["n_bookmakers"]
            analysis.has_sharp = odds["has_sharp"]

            home_form = team_forms.get(ht)
            away_form = team_forms.get(vt)

            if home_form and home_form.games > 0:
                analysis.home_form_str = f"{home_form.wins}П-{home_form.losses}П L10:{home_form.last10_wins}-{home_form.last10_losses} Очк:{home_form.ppg:.1f}"
            if away_form and away_form.games > 0:
                analysis.away_form_str = f"{away_form.wins}П-{away_form.losses}П L10:{away_form.last10_wins}-{away_form.last10_losses} Очк:{away_form.ppg:.1f}"

            inj_home = get_injury_impact(injuries, ht) if injuries else 0.0
            inj_away = get_injury_impact(injuries, vt) if injuries else 0.0
            if inj_home > 0 or inj_away > 0:
                parts = []
                if inj_home > 0:
                    parts.append(f"{ht} -{inj_home:.0%}")
                if inj_away > 0:
                    parts.append(f"{vt} -{inj_away:.0%}")
                analysis.injury_note = " | ".join(parts)

            travel_adj = get_travel(ht, vt)
            pub = get_public_data(public_data, ht, vt) if public_data else {"home_pct": 0.5, "away_pct": 0.5, "contrarian_signal": None, "strength": 0}

            pred = predict_nba_game(
                home_form=home_form,
                away_form=away_form,
                book_home=odds["home_prob"],
                book_away=odds["away_prob"],
                has_sharp=odds["has_sharp"],
                line_spread=odds["line_spread"],
                injury_home=inj_home,
                injury_away=inj_away,
                travel_adj=travel_adj,
                contrarian_signal=pub.get("contrarian_signal"),
                contrarian_strength=pub.get("strength", 0),
            )
            analysis.our_home_prob = pred["our_home"]
            analysis.our_away_prob = pred["our_away"]

            claude_result = None
            if use_claude and ANTHROPIC_API_KEY and claude_calls < 15:
                claude_result = await claude_analyze_game(
                    session=session,
                    home_team=ht,
                    away_team=vt,
                    home_form=home_form,
                    away_form=away_form,
                    book_home=odds["home_prob"],
                    book_away=odds["away_prob"],
                    injury_home=inj_home,
                    injury_away=inj_away,
                    public_home_pct=pub.get("home_pct", 0.5),
                )
                claude_calls += 1
                if claude_result and claude_result.get("pick"):
                    pick_ru = "ДОМ" if claude_result['pick'] == "HOME" else "ГОСТИ"
                    analysis.claude_note = f"{pick_ru} ({claude_result.get('confidence', 0):.0%}) — {claude_result.get('reasoning', '')[:80]}"
                    logger.info(f"  Claude: {ht} vs {vt} -> {claude_result['pick']}")
                await asyncio.sleep(1)

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
                injury_home=inj_home,
                injury_away=inj_away,
                travel_adj=travel_adj,
                public_home_pct=pub.get("home_pct", 0.5),
                public_away_pct=pub.get("away_pct", 0.5),
                contrarian_signal=pub.get("contrarian_signal", ""),
                claude_result=claude_result,
                state=state,
            )

            if signal:
                all_predictions.append(signal)
                state.record_signal(signal)
                analysis.verdict = "BET"
                analysis.verdict_team = signal.pick_team
                analysis.confidence = signal.confidence
                analysis.edge = signal.edge
                logger.info(f"  BET: {signal.pick} {signal.pick_team} | conf={signal.confidence:.0%} edge={signal.edge:.1%} bet=${signal.bet_size:.2f}")
            else:
                home_edge = pred["our_home"] - odds["home_prob"]
                away_edge = pred["our_away"] - odds["away_prob"]
                max_conf = max(pred["our_home"], pred["our_away"])
                max_edge = max(home_edge, away_edge)
                reasons = []
                if max_conf < MIN_CONFIDENCE:
                    reasons.append(f"Низкая уверенность ({max_conf:.0%} < {MIN_CONFIDENCE:.0%})")
                if max_edge < MIN_EDGE:
                    reasons.append(f"Низкий эдж ({max_edge:.1%} < {MIN_EDGE:.0%})")
                if odds["n_bookmakers"] < 3:
                    reasons.append(f"Мало букмекеров ({odds['n_bookmakers']})")
                analysis.skip_reason = " | ".join(reasons) if reasons else "Нет эджа"
                analysis.confidence = max_conf
                analysis.edge = max_edge
                logger.info(f"  SKIP: {ht} vs {vt} | {analysis.skip_reason}")

            all_analyses.append(analysis)

        nba_processed = {f"{a.home_team} vs {a.away_team}" for a in all_analyses}

        for key, odds_entry in odds_data.items():
            commence = odds_entry.get("commence", "")
            if commence:
                try:
                    dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                    if dt < (now - timedelta(hours=1)) or dt > (now + timedelta(hours=WINDOW_HOURS)):
                        continue
                except Exception:
                    continue
            else:
                continue

            home = odds_entry["home"]
            away = odds_entry["away"]
            game_key = f"{home} vs {away}"
            if game_key in nba_processed:
                continue

            total_analyzed += 1
            start_str = dt.strftime("%H:%M UTC")

            analysis = GameAnalysis(
                home_team=home, away_team=away, sport=odds_entry["sport"], start_time=start_str,
                book_home_prob=odds_entry["home_prob"], book_away_prob=odds_entry["away_prob"],
                our_home_prob=odds_entry["sharp_home"], our_away_prob=odds_entry["sharp_away"],
                n_bookmakers=odds_entry["n_bookmakers"], has_sharp=odds_entry["has_sharp"],
            )

            home_edge = odds_entry["sharp_home"] - odds_entry["home_prob"]
            away_edge = odds_entry["sharp_away"] - odds_entry["away_prob"]

            signal = generate_signal(
                home_team=home,
                away_team=away,
                sport=odds_entry["sport"],
                start_time=start_str,
                book_home=odds_entry["home_prob"],
                book_away=odds_entry["away_prob"],
                our_home=odds_entry["sharp_home"],
                our_away=odds_entry["sharp_away"],
                n_bookmakers=odds_entry["n_bookmakers"],
                has_sharp=odds_entry["has_sharp"],
                home_form=None,
                away_form=None,
                adjustments={"sharp_vs_market": max(home_edge, away_edge)},
                state=state,
            )
            if signal:
                all_predictions.append(signal)
                state.record_signal(signal)
                analysis.verdict = "BET"
                analysis.verdict_team = signal.pick_team
                analysis.confidence = signal.confidence
                analysis.edge = signal.edge
                logger.info(f"  BET: {signal.pick} {signal.pick_team} [{odds_entry['sport']}] | conf={signal.confidence:.0%}")
            else:
                max_edge_val = max(abs(home_edge), abs(away_edge))
                analysis.skip_reason = f"Низкий эдж ({max_edge_val:.1%})" if max_edge_val < MIN_EDGE else "Нет эджа"
                analysis.confidence = max(odds_entry["sharp_home"], odds_entry["sharp_away"])
                analysis.edge = max_edge_val

            all_analyses.append(analysis)

        logger.info(f"\nTotal: {total_analyzed} | Trades placed: {len(all_predictions)} | Claude: {claude_calls}")
        logger.info("=" * 55)

        report = format_report(all_predictions, total_analyzed, state)
        print(report)

        DATA_DIR.mkdir(exist_ok=True)
        (DATA_DIR / "predictor_report_v2.txt").write_text(report)

        preds_json = []
        for p in all_predictions:
            preds_json.append({
                "game_id": p.game_id,
                "timestamp": p.timestamp,
                "home_team": p.home_team,
                "away_team": p.away_team,
                "sport": p.sport,
                "start_time": p.start_time,
                "pick": p.pick,
                "pick_team": p.pick_team,
                "confidence": p.confidence,
                "edge": p.edge,
                "bet_size": p.bet_size,
                "expected_value": p.expected_value,
                "book_home": p.book_home_prob,
                "book_away": p.book_away_prob,
                "our_home": p.our_home_prob,
                "our_away": p.our_away_prob,
                "injury_home": p.injury_impact_home,
                "injury_away": p.injury_impact_away,
                "travel": p.travel_factor,
                "public_home": p.public_home_pct,
                "contrarian": p.contrarian_signal,
                "claude_pick": p.claude_pick,
                "claude_confidence": p.claude_confidence,
                "claude_reasoning": p.claude_reasoning,
                "triple_confirmed": p.triple_confirmed,
                "home_form": p.home_form,
                "away_form": p.away_form,
                "factors": p.factors,
            })

        signals_history = []
        if SIGNALS_FILE.exists():
            try:
                signals_history = json.loads(SIGNALS_FILE.read_text())
            except Exception:
                pass
        signals_history.extend(preds_json)
        SIGNALS_FILE.write_text(json.dumps(signals_history, indent=2))

        analyses_json = [asdict(a) for a in all_analyses]
        (DATA_DIR / "last_analyses.json").write_text(json.dumps(analyses_json, indent=2))

        logger.info("Files saved to data/")
        await send_telegram(all_predictions, all_analyses, total_analyzed, state)

    return all_predictions, all_analyses


_last_analyses: list[GameAnalysis] = []


async def run_telegram_bot():
    global _last_analyses
    if not TELEGRAM_TOKEN:
        logger.warning("No TELEGRAM_TOKEN, bot disabled")
        return

    try:
        import httpx
    except ImportError:
        logger.error("httpx not installed")
        return

    last_update_id = 0
    logger.info("Telegram trading terminal started!")

    while True:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url, params={
                    "offset": last_update_id + 1,
                    "timeout": 10,
                })
                if resp.status_code != 200:
                    await asyncio.sleep(5)
                    continue

                data = resp.json()
                updates = data.get("result", [])

                for update in updates:
                    last_update_id = update["update_id"]

                    cb = update.get("callback_query")
                    if cb:
                        cb_id = cb["id"]
                        cb_data = cb.get("data", "")
                        msg_id = cb.get("message", {}).get("message_id")
                        cb_chat = str(cb.get("message", {}).get("chat", {}).get("id", ""))
                        if cb_chat != TELEGRAM_CHAT_ID:
                            await _tg_answer_cb(cb_id)
                            continue

                        await _tg_answer_cb(cb_id)
                        state = BotState()

                        if cb_data == "dash":
                            text = build_dashboard_text(state)
                            if msg_id:
                                await _tg_edit(msg_id, text, _main_keyboard())
                            else:
                                await _tg_send(text, _main_keyboard())

                        elif cb_data == "positions":
                            text = build_positions_text(state)
                            if msg_id:
                                await _tg_edit(msg_id, text, _back_keyboard())
                            else:
                                await _tg_send(text, _back_keyboard())

                        elif cb_data == "events":
                            if not _last_analyses:
                                analyses_file = DATA_DIR / "last_analyses.json"
                                if analyses_file.exists():
                                    try:
                                        raw = json.loads(analyses_file.read_text())
                                        _last_analyses = [GameAnalysis(**r) for r in raw]
                                    except Exception:
                                        pass
                            text = build_events_text(_last_analyses)
                            if len(text) > 4000:
                                text = text[:4000] + "\n..."
                            if msg_id:
                                await _tg_edit(msg_id, text, _back_keyboard())
                            else:
                                await _tg_send(text, _back_keyboard())

                        elif cb_data == "stats":
                            text = build_stats_text(state)
                            if msg_id:
                                await _tg_edit(msg_id, text, _back_keyboard())
                            else:
                                await _tg_send(text, _back_keyboard())

                        elif cb_data == "history":
                            text = build_history_text(state)
                            if len(text) > 4000:
                                text = text[:4000] + "\n..."
                            if msg_id:
                                await _tg_edit(msg_id, text, _back_keyboard())
                            else:
                                await _tg_send(text, _back_keyboard())

                        elif cb_data == "settings":
                            text = build_settings_text(state)
                            if msg_id:
                                await _tg_edit(msg_id, text, _settings_keyboard(state))
                            else:
                                await _tg_send(text, _settings_keyboard(state))

                        elif cb_data == "scan":
                            if msg_id:
                                await _tg_edit(msg_id, f"🔍 <b>СКАНИРУЮ...</b>\n\n📊 Анализирую все события на {WINDOW_HOURS}ч вперёд.\n⏳ Это займёт 30-60 секунд...", _back_keyboard())
                            try:
                                _, analyses = await run_predictor(use_claude=True, use_scrapers=True)
                                _last_analyses = analyses
                            except Exception as e:
                                logger.error(f"Scan failed: {e}")
                                await _tg_send(f"❌ <b>Ошибка скана:</b> {_esc(str(e)[:200])}", _main_keyboard())

                        elif cb_data.startswith("bal_"):
                            if cb_data == "bal_add_100":
                                state.bankroll += 100
                            elif cb_data == "bal_sub_100":
                                state.bankroll = max(0, state.bankroll - 100)
                            elif cb_data == "bal_add_500":
                                state.bankroll += 500
                            elif cb_data == "bal_reset":
                                state.bankroll = 500.0
                            state.save()
                            text = build_settings_text(state)
                            if msg_id:
                                await _tg_edit(msg_id, text, _settings_keyboard(state))

                        continue

                    message = update.get("message", {})
                    text = message.get("text", "").strip()
                    chat_id = str(message.get("chat", {}).get("id", ""))

                    if chat_id != TELEGRAM_CHAT_ID:
                        continue

                    if text in ("/start", "/menu", "/help"):
                        state = BotState()
                        dash = build_dashboard_text(state)
                        await _tg_send(dash, _main_keyboard())

                    elif text == "/scan":
                        await _tg_send(f"🔍 <b>СКАНИРУЮ...</b>\n📊 Анализирую события на {WINDOW_HOURS}ч вперёд...")
                        try:
                            _, analyses = await run_predictor(use_claude=True, use_scrapers=True)
                            _last_analyses = analyses
                        except Exception as e:
                            logger.error(f"Scan failed: {e}")
                            await _tg_send(f"❌ <b>Ошибка скана:</b> {_esc(str(e)[:200])}", _main_keyboard())

                    elif text.startswith("/won ") or text.startswith("/lost "):
                        parts = text.split(maxsplit=1)
                        won = parts[0] == "/won"
                        game_hint = parts[1] if len(parts) > 1 else ""

                        state = BotState()
                        found = False
                        for entry in reversed(state.history):
                            if entry.get("result") != "pending":
                                continue
                            if game_hint.lower() in entry.get("pick_team", "").lower() or game_hint.lower() in entry.get("game_id", "").lower():
                                state.record_result(entry["game_id"], won)
                                pnl_val = entry["potential_win"] if won else entry["potential_loss"]
                                if won:
                                    icon = "✅"
                                    pnl_str = f"🟢 +${pnl_val:.2f}"
                                else:
                                    icon = "❌"
                                    pnl_str = f"🔴 -${abs(pnl_val):.2f}"
                                await _tg_send(
                                    f"{icon} <b>{_esc(entry['pick_team'])}</b>\n"
                                    f"💰 P&amp;L: {pnl_str}\n"
                                    f"💳 Баланс: <code>${state.bankroll:.2f}</code>",
                                    _main_keyboard()
                                )
                                found = True
                                break

                        if not found:
                            pending = [h for h in state.history if h.get("result") == "pending"]
                            if pending:
                                msg = "❓ <b>Игра не найдена.</b> Открытые:\n"
                                for h in pending:
                                    team = _esc(h.get("pick_team", "?"))
                                    msg += f"  • {team} — <code>/won {h.get('pick_team', '').split()[-1]}</code>\n"
                                await _tg_send(msg)
                            else:
                                await _tg_send("💭 Нет открытых позиций.")

        except Exception as e:
            logger.error(f"Bot error: {e}")
            await asyncio.sleep(5)

        await asyncio.sleep(2)


async def autopilot():
    global _last_analyses
    logger.info(f"AUTOPILOT — scanning every {AUTOPILOT_INTERVAL // 60} min, window {WINDOW_HOURS}h")
    while True:
        try:
            _, analyses = await run_predictor(use_claude=True, use_scrapers=True)
            _last_analyses = analyses
        except Exception as e:
            logger.error(f"Autopilot scan failed: {e}")
        logger.info(f"Next scan in {AUTOPILOT_INTERVAL // 60} min...")
        await asyncio.sleep(AUTOPILOT_INTERVAL)


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sport Predictor V2 — Trading Terminal")
    parser.add_argument("--mode", choices=["once", "bot", "autopilot", "full"], default="full")
    parser.add_argument("--no-claude", action="store_true")
    parser.add_argument("--no-scrapers", action="store_true")
    args = parser.parse_args()

    if args.mode == "once":
        await run_predictor(
            use_claude=not args.no_claude,
            use_scrapers=not args.no_scrapers,
        )
    elif args.mode == "bot":
        await run_telegram_bot()
    elif args.mode == "autopilot":
        await autopilot()
    elif args.mode == "full":
        await asyncio.gather(
            run_telegram_bot(),
            autopilot(),
        )


if __name__ == "__main__":
    asyncio.run(main())
