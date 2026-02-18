import logging
from datetime import datetime, timezone
from dataclasses import dataclass

import aiohttp

from config import config

logger = logging.getLogger(__name__)


@dataclass
class BookmakerOdds:
    sport: str
    home_team: str
    away_team: str
    commence_time: datetime
    bookmaker: str
    home_win_prob: float
    away_win_prob: float
    draw_prob: float
    raw: dict


class OddsAPIClient:
    def __init__(self):
        self.base_url = config.odds_api_url
        self.api_key = config.odds_api_key
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

    async def fetch_all_sports_odds(self) -> list[BookmakerOdds]:
        all_odds: list[BookmakerOdds] = []
        for sport_key in config.sports:
            odds = await self.fetch_sport_odds(sport_key)
            all_odds.extend(odds)
        logger.info(f"OddsAPI: fetched {len(all_odds)} odds entries across {len(config.sports)} sports")
        return all_odds

    async def fetch_sport_odds(self, sport_key: str) -> list[BookmakerOdds]:
        if not self.api_key:
            logger.warning("No ODDS_API_KEY set, skipping odds fetch")
            return []

        session = await self._get_session()
        url = f"{self.base_url}/sports/{sport_key}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "us,eu,uk",
            "markets": "h2h",
            "oddsFormat": "decimal",
        }

        odds_list: list[BookmakerOdds] = []
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 401:
                    logger.error("OddsAPI: invalid API key")
                    return []
                if resp.status == 429:
                    logger.warning("OddsAPI: rate limited")
                    return []
                if resp.status != 200:
                    logger.warning(f"OddsAPI: {sport_key} returned {resp.status}")
                    return []

                data = await resp.json()
                for event in data:
                    parsed = self._parse_event(event, sport_key)
                    odds_list.extend(parsed)
        except Exception as e:
            logger.error(f"Error fetching odds for {sport_key}: {e}")

        return odds_list

    def _parse_event(self, event: dict, sport_key: str) -> list[BookmakerOdds]:
        results: list[BookmakerOdds] = []
        try:
            home_team = event.get("home_team", "")
            away_team = event.get("away_team", "")
            commence_str = event.get("commence_time", "")
            if commence_str:
                commence_str = commence_str.replace("Z", "+00:00")
                commence_time = datetime.fromisoformat(commence_str)
            else:
                commence_time = datetime.now(timezone.utc)

            for bookmaker in event.get("bookmakers", []):
                bk_name = bookmaker.get("key", "")
                for market in bookmaker.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}

                    home_odds = outcomes.get(home_team, 2.0)
                    away_odds = outcomes.get(away_team, 2.0)
                    draw_odds = outcomes.get("Draw", 0)

                    home_prob = 1.0 / home_odds if home_odds > 0 else 0
                    away_prob = 1.0 / away_odds if away_odds > 0 else 0
                    draw_prob = 1.0 / draw_odds if draw_odds > 0 else 0

                    total = home_prob + away_prob + draw_prob
                    if total > 0:
                        home_prob /= total
                        away_prob /= total
                        draw_prob /= total

                    results.append(BookmakerOdds(
                        sport=sport_key,
                        home_team=home_team,
                        away_team=away_team,
                        commence_time=commence_time,
                        bookmaker=bk_name,
                        home_win_prob=home_prob,
                        away_win_prob=away_prob,
                        draw_prob=draw_prob,
                        raw=event,
                    ))
        except Exception as e:
            logger.debug(f"Failed to parse odds event: {e}")

        return results

    def get_consensus_odds(self, odds_list: list[BookmakerOdds], home_team: str, away_team: str) -> dict:
        relevant = [
            o for o in odds_list
            if self._teams_match(o.home_team, o.away_team, home_team, away_team)
        ]
        if not relevant:
            return {"home_prob": 0.5, "away_prob": 0.5, "draw_prob": 0.0, "n_bookmakers": 0}

        sharp_bookmakers = ["pinnacle", "betfair", "matchbook", "smarkets"]
        sharp = [o for o in relevant if o.bookmaker in sharp_bookmakers]
        source = sharp if sharp else relevant

        avg_home = sum(o.home_win_prob for o in source) / len(source)
        avg_away = sum(o.away_win_prob for o in source) / len(source)
        avg_draw = sum(o.draw_prob for o in source) / len(source)

        total = avg_home + avg_away + avg_draw
        if total > 0:
            avg_home /= total
            avg_away /= total
            avg_draw /= total

        return {
            "home_prob": avg_home,
            "away_prob": avg_away,
            "draw_prob": avg_draw,
            "n_bookmakers": len(relevant),
            "has_sharp": len(sharp) > 0,
        }

    @staticmethod
    def _teams_match(bk_home: str, bk_away: str, q_home: str, q_away: str) -> bool:
        bk_h = bk_home.lower().strip()
        bk_a = bk_away.lower().strip()
        q_h = q_home.lower().strip()
        q_a = q_away.lower().strip()

        if bk_h == q_h and bk_a == q_a:
            return True
        if bk_h in q_h or q_h in bk_h:
            if bk_a in q_a or q_a in bk_a:
                return True
        return False
