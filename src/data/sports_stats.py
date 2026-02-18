import logging
from datetime import datetime, timezone
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger(__name__)

NBA_API_URL = "https://www.balldontlie.io/api/v1"
FOOTBALL_API_URL = "https://api.football-data.org/v4"


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

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_team_stats(self, team_name: str, sport: str) -> TeamStats | None:
        if "nba" in sport.lower() or "basketball" in sport.lower():
            return await self._get_nba_stats(team_name)
        return self._default_stats(team_name, sport)

    async def _get_nba_stats(self, team_name: str) -> TeamStats | None:
        session = await self._get_session()
        try:
            url = f"{NBA_API_URL}/teams"
            async with session.get(url) as resp:
                if resp.status != 200:
                    return self._default_stats(team_name, "nba")
                data = await resp.json()
                teams = data.get("data", [])

            target_team = None
            for team in teams:
                full_name = f"{team.get('city', '')} {team.get('name', '')}".strip()
                if (team_name.lower() in full_name.lower() or
                        full_name.lower() in team_name.lower()):
                    target_team = team
                    break

            if not target_team:
                return self._default_stats(team_name, "nba")

            team_id = target_team["id"]
            games_url = f"{NBA_API_URL}/games"
            params = {
                "team_ids[]": team_id,
                "seasons[]": datetime.now().year if datetime.now().month > 9 else datetime.now().year - 1,
                "per_page": 30,
            }
            async with session.get(games_url, params=params) as resp:
                if resp.status != 200:
                    return self._default_stats(team_name, "nba")
                games_data = await resp.json()
                games = games_data.get("data", [])

            if not games:
                return self._default_stats(team_name, "nba")

            wins, losses = 0, 0
            home_wins, home_games = 0, 0
            away_wins, away_games = 0, 0
            points_for, points_against = [], []
            recent_form: list[str] = []
            streak_count = 0
            streak_type = "W"

            for game in sorted(games, key=lambda g: g.get("date", ""), reverse=True):
                home_id = game.get("home_team", {}).get("id")
                home_score = game.get("home_team_score", 0)
                away_score = game.get("visitor_team_score", 0)

                if home_score == 0 and away_score == 0:
                    continue

                is_home = home_id == team_id
                if is_home:
                    pf, pa = home_score, away_score
                    won = home_score > away_score
                    home_games += 1
                    if won:
                        home_wins += 1
                else:
                    pf, pa = away_score, home_score
                    won = away_score > home_score
                    away_games += 1
                    if won:
                        away_wins += 1

                points_for.append(pf)
                points_against.append(pa)

                if won:
                    wins += 1
                    recent_form.append("W")
                else:
                    losses += 1
                    recent_form.append("L")

            last_5 = recent_form[:5]
            recent_wr = last_5.count("W") / max(len(last_5), 1)

            for r in recent_form:
                if r == recent_form[0]:
                    streak_count += 1
                else:
                    break
            streak_type = recent_form[0] if recent_form else "W"

            total = wins + losses
            elo = 1500 + (wins - losses) * 10

            return TeamStats(
                team_name=team_name,
                sport="nba",
                wins=wins,
                losses=losses,
                draws=0,
                win_rate=wins / max(total, 1),
                recent_form=recent_form[:10],
                recent_win_rate=recent_wr,
                avg_points_for=sum(points_for) / max(len(points_for), 1),
                avg_points_against=sum(points_against) / max(len(points_against), 1),
                home_win_rate=home_wins / max(home_games, 1),
                away_win_rate=away_wins / max(away_games, 1),
                streak=streak_count,
                streak_type=streak_type,
                elo=elo,
                raw={"games_analyzed": total},
            )
        except Exception as e:
            logger.error(f"Error fetching NBA stats for {team_name}: {e}")
            return self._default_stats(team_name, "nba")

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
