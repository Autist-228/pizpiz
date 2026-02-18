import logging
from datetime import datetime, timezone
from dataclasses import dataclass

import aiohttp

from config import config

logger = logging.getLogger(__name__)

API_FOOTBALL_URL = "https://v3.football.api-sports.io"
FOOTBALL_DATA_URL = "https://api.football-data.org/v4"


@dataclass
class TeamStats:
    team_name: str
    sport: str
    wins: int
    losses: int
    draws: int
    win_rate: float
    recent_form: list[str]
    recent_win_rate: float
    avg_points_for: float
    avg_points_against: float
    home_win_rate: float
    away_win_rate: float
    streak: int
    streak_type: str
    elo: float
    raw: dict


class SportsStatsClient:
    def __init__(self):
        self._session: aiohttp.ClientSession | None = None
        self._team_cache: dict[str, TeamStats] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_team_stats(self, team_name: str, sport: str) -> TeamStats | None:
        cache_key = f"{team_name}:{sport}"
        if cache_key in self._team_cache:
            return self._team_cache[cache_key]

        stats = None
        sport_lower = sport.lower()
        is_soccer = any(
            kw in sport_lower
            for kw in ["soccer", "epl", "liga", "serie", "ligue", "bundesliga", "champs", "football"]
        )

        if is_soccer:
            if config.api_football_key:
                stats = await self._get_api_football_stats(team_name, sport)
            if stats is None:
                stats = await self._get_football_data_stats(team_name, sport)

        if stats is None:
            stats = self._default_stats(team_name, sport)

        self._team_cache[cache_key] = stats
        return stats

    async def _get_api_football_stats(self, team_name: str, sport: str) -> TeamStats | None:
        session = await self._get_session()
        headers = {"x-apisports-key": config.api_football_key}

        try:
            url = f"{API_FOOTBALL_URL}/teams"
            params = {"search": team_name}
            async with session.get(url, headers=headers, params=params) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                teams = data.get("response", [])
                if not teams:
                    return None

            team_id = teams[0]["team"]["id"]
            team_full_name = teams[0]["team"]["name"]

            league_id = self._sport_to_league_id(sport)
            season = datetime.now().year if datetime.now().month > 7 else datetime.now().year - 1

            stats_url = f"{API_FOOTBALL_URL}/teams/statistics"
            stats_params = {"team": team_id, "season": season, "league": league_id}
            async with session.get(stats_url, headers=headers, params=stats_params) as resp:
                if resp.status != 200:
                    return None
                stats_data = await resp.json()
                stats_resp = stats_data.get("response", {})

            if not stats_resp:
                return None

            fixtures = stats_resp.get("fixtures", {})
            wins_data = fixtures.get("wins", {})
            draws_data = fixtures.get("draws", {})
            loses_data = fixtures.get("loses", {})

            wins = wins_data.get("total", 0) or 0
            draws = draws_data.get("total", 0) or 0
            losses = loses_data.get("total", 0) or 0
            total = wins + draws + losses

            home_wins = wins_data.get("home", 0) or 0
            home_total = (fixtures.get("played", {}).get("home", 0)) or 1
            away_wins = wins_data.get("away", 0) or 0
            away_total = (fixtures.get("played", {}).get("away", 0)) or 1

            goals_for = stats_resp.get("goals", {}).get("for", {}).get("total", {}).get("total", 0) or 0
            goals_against = stats_resp.get("goals", {}).get("against", {}).get("total", {}).get("total", 0) or 0

            form_str = stats_resp.get("form", "") or ""
            recent_form = list(form_str[-10:]) if form_str else []
            last_5 = list(form_str[-5:]) if form_str else []
            recent_wr = last_5.count("W") / max(len(last_5), 1)

            streak_count = 0
            streak_type = "N"
            if form_str:
                streak_type = form_str[-1]
                for ch in reversed(form_str):
                    if ch == streak_type:
                        streak_count += 1
                    else:
                        break

            elo = 1500 + (wins - losses) * 15 + draws * 3

            return TeamStats(
                team_name=team_full_name,
                sport=sport,
                wins=wins,
                losses=losses,
                draws=draws,
                win_rate=wins / max(total, 1),
                recent_form=recent_form,
                recent_win_rate=recent_wr,
                avg_points_for=goals_for / max(total, 1),
                avg_points_against=goals_against / max(total, 1),
                home_win_rate=home_wins / max(home_total, 1),
                away_win_rate=away_wins / max(away_total, 1),
                streak=streak_count,
                streak_type=streak_type,
                elo=elo,
                raw={
                    "source": "api-football",
                    "team_id": team_id,
                    "league_id": league_id,
                    "season": season,
                    "clean_sheets": stats_resp.get("clean_sheet", {}).get("total", 0),
                    "failed_to_score": stats_resp.get("failed_to_score", {}).get("total", 0),
                    "penalty_scored": stats_resp.get("penalty", {}).get("scored", {}).get("total", 0),
                    "penalty_missed": stats_resp.get("penalty", {}).get("missed", {}).get("total", 0),
                    "biggest_win_streak": stats_resp.get("biggest", {}).get("streak", {}).get("wins", 0),
                    "biggest_loss_streak": stats_resp.get("biggest", {}).get("streak", {}).get("loses", 0),
                },
            )

        except Exception as e:
            logger.debug(f"API-Football error for {team_name}: {e}")
            return None

    async def _get_football_data_stats(self, team_name: str, sport: str) -> TeamStats | None:
        session = await self._get_session()
        league_code = self._sport_to_fd_code(sport)
        if not league_code:
            return None

        try:
            url = f"{FOOTBALL_DATA_URL}/competitions/{league_code}/standings"
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()

            standings = data.get("standings", [])
            if not standings:
                return None

            table = standings[0].get("table", [])
            target = None
            for entry in table:
                t = entry.get("team", {})
                name = t.get("name", "").lower()
                short = t.get("shortName", "").lower()
                tla = t.get("tla", "").lower()
                search = team_name.lower()
                if search in name or name in search or search in short or short in search or search == tla:
                    target = entry
                    break

            if not target:
                return None

            wins = target.get("won", 0)
            draws = target.get("draw", 0)
            losses = target.get("lost", 0)
            total = target.get("playedGames", 0)
            goals_for = target.get("goalsFor", 0)
            goals_against = target.get("goalsAgainst", 0)
            form_str = target.get("form", "") or ""
            recent_form = form_str.split(",") if "," in form_str else list(form_str)

            last_5 = recent_form[-5:]
            recent_wr = last_5.count("W") / max(len(last_5), 1)

            streak_count = 0
            streak_type = "N"
            if recent_form:
                streak_type = recent_form[-1]
                for ch in reversed(recent_form):
                    if ch == streak_type:
                        streak_count += 1
                    else:
                        break

            elo = 1500 + (wins - losses) * 15 + draws * 3

            return TeamStats(
                team_name=team_name,
                sport=sport,
                wins=wins,
                losses=losses,
                draws=draws,
                win_rate=wins / max(total, 1),
                recent_form=recent_form[-10:],
                recent_win_rate=recent_wr,
                avg_points_for=goals_for / max(total, 1),
                avg_points_against=goals_against / max(total, 1),
                home_win_rate=0.5,
                away_win_rate=0.5,
                streak=streak_count,
                streak_type=streak_type,
                elo=elo,
                raw={
                    "source": "football-data.org",
                    "position": target.get("position", 0),
                    "points": target.get("points", 0),
                    "goal_difference": target.get("goalDifference", 0),
                },
            )

        except Exception as e:
            logger.debug(f"football-data.org error for {team_name}: {e}")
            return None

    @staticmethod
    def _sport_to_league_id(sport: str) -> int:
        mapping = {
            "soccer_epl": 39,
            "soccer_spain_la_liga": 140,
            "soccer_italy_serie_a": 135,
            "soccer_germany_bundesliga": 78,
            "soccer_france_ligue_one": 61,
            "soccer_usa_mls": 253,
            "soccer_uefa_champs_league": 2,
        }
        for key, lid in mapping.items():
            if key in sport.lower():
                return lid
        if "epl" in sport.lower() or "premier" in sport.lower():
            return 39
        return 39

    @staticmethod
    def _sport_to_fd_code(sport: str) -> str:
        mapping = {
            "soccer_epl": "PL",
            "soccer_spain_la_liga": "PD",
            "soccer_italy_serie_a": "SA",
            "soccer_germany_bundesliga": "BL1",
            "soccer_france_ligue_one": "FL1",
            "soccer_uefa_champs_league": "CL",
        }
        for key, code in mapping.items():
            if key in sport.lower():
                return code
        if "epl" in sport.lower() or "premier" in sport.lower():
            return "PL"
        return ""

    @staticmethod
    def _default_stats(team_name: str, sport: str) -> TeamStats:
        return TeamStats(
            team_name=team_name,
            sport=sport,
            wins=0,
            losses=0,
            draws=0,
            win_rate=0.5,
            recent_form=[],
            recent_win_rate=0.5,
            avg_points_for=0,
            avg_points_against=0,
            home_win_rate=0.5,
            away_win_rate=0.5,
            streak=0,
            streak_type="N",
            elo=1500,
            raw={},
        )
