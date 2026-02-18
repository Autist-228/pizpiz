import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import aiohttp

from config import config

logger = logging.getLogger(__name__)

HISTORY_DB = "data/historical.db"


class HistoricalDataCollector:
    def __init__(self):
        self.gamma_url = config.polymarket_gamma_url
        self.kalshi_url = config.kalshi_api_url
        self.odds_url = config.odds_api_url
        self._session: aiohttp.ClientSession | None = None
        self._init_db()

    def _init_db(self):
        Path(HISTORY_DB).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(HISTORY_DB)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS resolved_markets (
                market_id TEXT PRIMARY KEY,
                platform TEXT,
                question TEXT,
                category TEXT,
                tags TEXT,
                outcome_yes_price REAL,
                volume REAL,
                end_date TEXT,
                result INTEGER,
                home_team TEXT,
                away_team TEXT,
                sport TEXT,
                bookmaker_home_prob REAL,
                bookmaker_away_prob REAL,
                n_bookmakers INTEGER,
                collected_at TEXT,
                raw TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS historical_odds (
                event_id TEXT,
                sport TEXT,
                home_team TEXT,
                away_team TEXT,
                commence_time TEXT,
                bookmaker TEXT,
                home_prob REAL,
                away_prob REAL,
                draw_prob REAL,
                collected_at TEXT,
                PRIMARY KEY (event_id, bookmaker)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS game_results (
                game_id TEXT PRIMARY KEY,
                sport TEXT,
                home_team TEXT,
                away_team TEXT,
                home_score INTEGER,
                away_score INTEGER,
                game_date TEXT,
                season TEXT,
                collected_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS collection_log (
                collection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                records_collected INTEGER,
                timestamp TEXT,
                status TEXT,
                error TEXT
            )
        """)
        conn.commit()
        conn.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def collect_all(self, days_back: int = 180):
        logger.info(f"Starting historical data collection ({days_back} days back)...")
        results = await asyncio.gather(
            self._collect_polymarket_resolved(days_back),
            self._collect_kalshi_resolved(days_back),
            self._collect_historical_odds(),
            self._collect_nba_games(days_back),
            return_exceptions=True,
        )

        for i, r in enumerate(results):
            source = ["polymarket", "kalshi", "odds_api", "nba_games"][i]
            if isinstance(r, Exception):
                logger.error(f"Collection error for {source}: {r}")
                self._log_collection(source, 0, "error", str(r))
            else:
                logger.info(f"Collected {r} records from {source}")
                self._log_collection(source, r, "success", "")

        total = self.get_total_records()
        logger.info(f"Historical data collection complete. Total records: {total}")
        return total

    async def _collect_polymarket_resolved(self, days_back: int) -> int:
        session = await self._get_session()
        collected = 0
        offset = 0
        limit = 100
        max_pages = 30
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)

        while True:
            url = f"{self.gamma_url}/markets"
            params = {
                "limit": limit,
                "offset": offset,
                "closed": "true",
                "order": "volume",
                "ascending": "false",
            }

            try:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        break
                    data = await resp.json()
                    if not data:
                        break

                    batch_count = 0
                    reached_cutoff = False
                    for market in data:
                        end_str = market.get("endDate") or market.get("end_date_iso")
                        closed_str = market.get("closedTime", "")
                        date_str = closed_str or end_str
                        if not date_str:
                            continue

                        try:
                            end_date = datetime.fromisoformat(
                                date_str.replace("Z", "+00:00")
                            )
                        except (ValueError, TypeError):
                            continue

                        if end_date < cutoff:
                            reached_cutoff = True
                            continue

                        prices_raw = market.get("outcomePrices", "")
                        if isinstance(prices_raw, str) and prices_raw:
                            try:
                                prices = json.loads(prices_raw)
                            except (json.JSONDecodeError, ValueError):
                                continue
                        elif isinstance(prices_raw, list):
                            prices = prices_raw
                        else:
                            continue

                        if not prices or len(prices) < 2:
                            continue

                        try:
                            yes_final = float(prices[0])
                            no_final = float(prices[1])
                        except (ValueError, TypeError):
                            continue

                        if yes_final >= 0.95:
                            result = 1
                        elif no_final >= 0.95:
                            result = 0
                        else:
                            continue

                        question = market.get("question", "")
                        slug = market.get("slug", "")
                        search_text = (question + " " + slug).lower()
                        category = self._detect_sport_category(search_text)

                        cid = market.get(
                            "conditionId", market.get("condition_id", "")
                        )
                        self._store_resolved_market(
                            market_id=cid,
                            platform="polymarket",
                            question=question,
                            category=category,
                            tags=json.dumps([category]) if category else "[]",
                            yes_price=yes_final,
                            volume=float(market.get("volume", 0) or 0),
                            end_date=date_str,
                            result=result,
                            raw=json.dumps(market),
                        )
                        batch_count += 1

                    collected += batch_count
                    if len(data) < limit or reached_cutoff:
                        break
                    offset += limit
                    if offset >= limit * max_pages:
                        logger.info(f"Polymarket: reached page limit ({max_pages})")
                        break
                    await asyncio.sleep(0.3)

            except Exception as e:
                logger.error(f"Error collecting Polymarket history page {offset}: {e}")
                break

        return collected

    @staticmethod
    def _detect_sport_category(text: str) -> str:
        sport_map = {
            "nba": "basketball_nba",
            "basketball": "basketball_nba",
            "nfl": "americanfootball_nfl",
            "nhl": "icehockey_nhl",
            "hockey": "icehockey_nhl",
            "mlb": "baseball_mlb",
            "baseball": "baseball_mlb",
            "ufc": "mma_mixed_martial_arts",
            "mma": "mma_mixed_martial_arts",
            "fight": "mma_mixed_martial_arts",
            "premier league": "soccer_epl",
            "epl": "soccer_epl",
            "la liga": "soccer_spain_la_liga",
            "serie a": "soccer_italy_serie_a",
            "champions league": "soccer_uefa_champs_league",
            "soccer": "soccer",
            "tennis": "tennis",
            "counter-strike": "esports_csgo",
            "csgo": "esports_csgo",
            "cs2": "esports_csgo",
            "dota": "esports_dota2",
            "league of legends": "esports_lol",
        }
        for keyword, cat in sport_map.items():
            if keyword in text:
                return cat
        return ""

    async def _collect_kalshi_resolved(self, days_back: int) -> int:
        session = await self._get_session()
        collected = 0
        cursor: str | None = None
        max_pages = 30
        page_count = 0
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)

        sport_ticker_prefixes = [
            "KXNBA", "KXNFL", "KXNHL", "KXMLB", "KXUFC", "KXMMA",
            "KXEPL", "KXLALIGA", "KXSERIEA", "KXUCL", "KXMLS",
            "KXSOCCER", "KXTENNIS", "KXCRICKET",
            "KXMVE",
        ]

        while page_count < max_pages:
            page_count += 1
            url = f"{self.kalshi_url}/markets"
            params: dict = {"limit": 200, "status": "settled"}
            if cursor:
                params["cursor"] = cursor

            try:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        break
                    data = await resp.json()
                    markets = data.get("markets", [])
                    if not markets:
                        break

                    reached_cutoff = False
                    for market in markets:
                        end_str = (
                            market.get("close_time")
                            or market.get("expiration_time")
                        )
                        if not end_str:
                            continue

                        try:
                            end_date = datetime.fromisoformat(
                                end_str.replace("Z", "+00:00")
                            )
                        except (ValueError, TypeError):
                            continue

                        if end_date < cutoff:
                            reached_cutoff = True
                            continue

                        ticker = market.get("ticker", "")
                        event_ticker = market.get("event_ticker", "")
                        title = market.get("title", "")
                        search_key = (ticker + " " + event_ticker).upper()

                        is_sport = any(
                            search_key.startswith(p) or p in search_key
                            for p in sport_ticker_prefixes
                        )
                        if not is_sport:
                            search_text = (
                                title.lower()
                                + " " + market.get("category", "").lower()
                                + " " + market.get("sub_category", "").lower()
                            )
                            is_sport = self._detect_sport_category(search_text) != ""

                        result_str = market.get("result", "")
                        if result_str == "yes":
                            result = 1
                        elif result_str == "no":
                            result = 0
                        else:
                            exp_val = market.get("expiration_value", "")
                            if exp_val == "":
                                continue
                            result = 1 if exp_val == "yes" else 0

                        yes_price = (
                            float(market.get("last_price", 50) or 50) / 100
                        )

                        category = self._detect_sport_category(
                            (title + " " + event_ticker).lower()
                        )

                        self._store_resolved_market(
                            market_id=ticker,
                            platform="kalshi",
                            question=title,
                            category=category or market.get("category", ""),
                            tags=json.dumps(
                                [market.get("sub_category", "").lower()]
                            ),
                            yes_price=yes_price,
                            volume=float(market.get("volume", 0) or 0),
                            end_date=end_str,
                            result=result,
                            raw=json.dumps(market),
                        )
                        collected += 1

                    cursor = data.get("cursor")
                    if not cursor or reached_cutoff:
                        break
                    await asyncio.sleep(0.3)

            except Exception as e:
                logger.error(f"Error collecting Kalshi history: {e}")
                break

        return collected

    async def _collect_historical_odds(self) -> int:
        if not config.odds_api_key:
            logger.warning("No ODDS_API_KEY, skipping historical odds collection")
            return 0

        session = await self._get_session()
        collected = 0

        for sport_key in config.sports:
            url = f"{self.odds_url}/sports/{sport_key}/scores"
            params = {
                "apiKey": config.odds_api_key,
                "daysFrom": 3,
            }

            try:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        continue
                    data = await resp.json()
                    for event in data:
                        scores = event.get("scores", [])
                        if not scores:
                            continue

                        home = event.get("home_team", "")
                        away = event.get("away_team", "")
                        commence = event.get("commence_time", "")

                        home_score = 0
                        away_score = 0
                        for s in scores:
                            if s.get("name") == home:
                                home_score = int(s.get("score", 0) or 0)
                            elif s.get("name") == away:
                                away_score = int(s.get("score", 0) or 0)

                        if home_score == 0 and away_score == 0:
                            continue

                        self._store_game_result(
                            game_id=event.get("id", ""),
                            sport=sport_key,
                            home_team=home,
                            away_team=away,
                            home_score=home_score,
                            away_score=away_score,
                            game_date=commence,
                        )
                        collected += 1

            except Exception as e:
                logger.error(f"Error collecting scores for {sport_key}: {e}")

            await asyncio.sleep(0.3)

        return collected

    async def _collect_nba_games(self, days_back: int) -> int:
        session = await self._get_session()
        collected = 0
        nba_url = "https://api.balldontlie.io/v1"

        now = datetime.now(timezone.utc)
        year = now.year if now.month > 9 else now.year - 1
        page = 1
        per_page = 100

        while page <= 10:
            url = f"{nba_url}/games"
            params = {
                "seasons[]": year,
                "per_page": per_page,
                "page": page,
            }

            try:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        break
                    data = await resp.json()
                    games = data.get("data", [])
                    if not games:
                        break

                    for game in games:
                        home_score = game.get("home_team_score", 0)
                        away_score = game.get("visitor_team_score", 0)
                        if home_score == 0 and away_score == 0:
                            continue

                        home_name = (
                            f"{game.get('home_team', {}).get('city', '')} "
                            f"{game.get('home_team', {}).get('name', '')}"
                        ).strip()
                        away_name = (
                            f"{game.get('visitor_team', {}).get('city', '')} "
                            f"{game.get('visitor_team', {}).get('name', '')}"
                        ).strip()

                        self._store_game_result(
                            game_id=f"nba_{game.get('id', '')}",
                            sport="basketball_nba",
                            home_team=home_name,
                            away_team=away_name,
                            home_score=home_score,
                            away_score=away_score,
                            game_date=game.get("date", ""),
                            season=str(year),
                        )
                        collected += 1

                    meta = data.get("meta", {})
                    if page >= meta.get("total_pages", 1):
                        break
                    page += 1
                    await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error collecting NBA games page {page}: {e}")
                break

        return collected

    def _store_resolved_market(
        self,
        market_id: str,
        platform: str,
        question: str,
        category: str,
        tags: str,
        yes_price: float,
        volume: float,
        end_date: str,
        result: int,
        raw: str,
    ):
        conn = sqlite3.connect(HISTORY_DB)
        try:
            conn.execute("""
                INSERT OR REPLACE INTO resolved_markets
                (market_id, platform, question, category, tags,
                 outcome_yes_price, volume, end_date, result,
                 home_team, away_team, sport,
                 bookmaker_home_prob, bookmaker_away_prob, n_bookmakers,
                 collected_at, raw)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '', '', '', 0, 0, 0, ?, ?)
            """, (
                market_id, platform, question, category, tags,
                yes_price, volume, end_date, result,
                datetime.now(timezone.utc).isoformat(), raw,
            ))
            conn.commit()
        except Exception as e:
            logger.debug(f"Store resolved market error: {e}")
        finally:
            conn.close()

    def _store_game_result(
        self,
        game_id: str,
        sport: str,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        game_date: str,
        season: str = "",
    ):
        conn = sqlite3.connect(HISTORY_DB)
        try:
            conn.execute("""
                INSERT OR REPLACE INTO game_results
                (game_id, sport, home_team, away_team,
                 home_score, away_score, game_date, season, collected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                game_id, sport, home_team, away_team,
                home_score, away_score, game_date, season,
                datetime.now(timezone.utc).isoformat(),
            ))
            conn.commit()
        except Exception as e:
            logger.debug(f"Store game result error: {e}")
        finally:
            conn.close()

    def _log_collection(self, source: str, count: int, status: str, error: str):
        conn = sqlite3.connect(HISTORY_DB)
        conn.execute("""
            INSERT INTO collection_log (source, records_collected, timestamp, status, error)
            VALUES (?, ?, ?, ?, ?)
        """, (source, count, datetime.now(timezone.utc).isoformat(), status, error))
        conn.commit()
        conn.close()

    def get_total_records(self) -> int:
        conn = sqlite3.connect(HISTORY_DB)
        markets = conn.execute("SELECT COUNT(*) FROM resolved_markets").fetchone()[0]
        games = conn.execute("SELECT COUNT(*) FROM game_results").fetchone()[0]
        conn.close()
        return markets + games

    def get_training_data(self) -> list[dict]:
        conn = sqlite3.connect(HISTORY_DB)
        rows = conn.execute("""
            SELECT market_id, platform, question, category, tags,
                   outcome_yes_price, volume, end_date, result,
                   home_team, away_team, sport,
                   bookmaker_home_prob, bookmaker_away_prob, n_bookmakers
            FROM resolved_markets
            ORDER BY end_date DESC
        """).fetchall()
        conn.close()

        data = []
        for r in rows:
            data.append({
                "market_id": r[0],
                "platform": r[1],
                "question": r[2],
                "category": r[3],
                "tags": r[4],
                "yes_price": r[5],
                "volume": r[6],
                "end_date": r[7],
                "result": r[8],
                "home_team": r[9],
                "away_team": r[10],
                "sport": r[11],
                "bookmaker_home_prob": r[12],
                "bookmaker_away_prob": r[13],
                "n_bookmakers": r[14],
            })
        return data

    def get_game_results(self, sport: str = "") -> list[dict]:
        conn = sqlite3.connect(HISTORY_DB)
        if sport:
            rows = conn.execute(
                "SELECT * FROM game_results WHERE sport = ? ORDER BY game_date DESC",
                (sport,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM game_results ORDER BY game_date DESC"
            ).fetchall()
        conn.close()

        return [
            {
                "game_id": r[0],
                "sport": r[1],
                "home_team": r[2],
                "away_team": r[3],
                "home_score": r[4],
                "away_score": r[5],
                "game_date": r[6],
                "season": r[7],
            }
            for r in rows
        ]
