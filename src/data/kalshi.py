import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field

import aiohttp

from config import config

logger = logging.getLogger(__name__)

SPORT_CATEGORIES = [
    "sports", "nba", "nfl", "nhl", "mlb", "mls", "ufc", "mma", "soccer",
    "football", "basketball", "baseball", "tennis", "boxing", "hockey",
    "premier_league", "champions_league",
]


@dataclass
class KalshiMarket:
    ticker: str
    title: str
    subtitle: str
    yes_price: float
    no_price: float
    volume: int
    open_interest: int
    end_date: datetime | None
    category: str
    sub_category: str
    status: str
    result: str
    platform: str = "kalshi"
    raw: dict = field(default_factory=dict)


class KalshiClient:
    def __init__(self):
        self.base_url = config.kalshi_api_url
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

    async def fetch_sports_markets(self, horizon_hours: int | None = None) -> list[KalshiMarket]:
        horizon = horizon_hours or config.event_horizon_hours
        now = datetime.now(timezone.utc)
        deadline = now + timedelta(hours=horizon)
        markets: list[KalshiMarket] = []

        try:
            raw_markets = await self._fetch_all_events()
            for raw in raw_markets:
                market = self._parse_market(raw)
                if market is None:
                    continue
                if not self._is_sport_market(market):
                    continue
                if market.status != "open":
                    continue
                if market.end_date and market.end_date > deadline:
                    continue
                if market.end_date and market.end_date < now:
                    continue
                markets.append(market)
        except Exception as e:
            logger.error(f"Error fetching Kalshi sports markets: {e}")

        logger.info(f"Kalshi: found {len(markets)} sport markets within {horizon}h")
        return markets

    async def _fetch_all_events(self) -> list[dict]:
        session = await self._get_session()
        all_markets: list[dict] = []
        cursor: str | None = None

        while True:
            url = f"{self.base_url}/markets"
            params: dict = {"limit": 200, "status": "open"}
            if cursor:
                params["cursor"] = cursor

            try:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        logger.warning(f"Kalshi API returned {resp.status}")
                        break
                    data = await resp.json()
                    markets_data = data.get("markets", [])
                    if not markets_data:
                        break
                    all_markets.extend(markets_data)
                    cursor = data.get("cursor")
                    if not cursor:
                        break
            except Exception as e:
                logger.error(f"Error fetching Kalshi markets: {e}")
                break

        return all_markets

    async def fetch_market_orderbook(self, ticker: str) -> dict:
        session = await self._get_session()
        url = f"{self.base_url}/markets/{ticker}/orderbook"
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            logger.error(f"Error fetching Kalshi orderbook for {ticker}: {e}")
        return {"yes": [], "no": []}

    async def fetch_market_history(self, ticker: str) -> list[dict]:
        session = await self._get_session()
        url = f"{self.base_url}/markets/{ticker}/history"
        params = {"limit": 1000}
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("history", [])
        except Exception as e:
            logger.error(f"Error fetching Kalshi history for {ticker}: {e}")
        return []

    def _parse_market(self, raw: dict) -> KalshiMarket | None:
        try:
            end_date_str = raw.get("close_time") or raw.get("expiration_time")
            end_date = None
            if end_date_str:
                end_date_str = end_date_str.replace("Z", "+00:00")
                end_date = datetime.fromisoformat(end_date_str)

            yes_price = float(raw.get("yes_ask", 0) or raw.get("last_price", 50)) / 100
            no_price = 1.0 - yes_price

            return KalshiMarket(
                ticker=raw.get("ticker", ""),
                title=raw.get("title", ""),
                subtitle=raw.get("subtitle", ""),
                yes_price=yes_price,
                no_price=no_price,
                volume=int(raw.get("volume", 0) or 0),
                open_interest=int(raw.get("open_interest", 0) or 0),
                end_date=end_date,
                category=raw.get("category", ""),
                sub_category=raw.get("sub_category", ""),
                status=raw.get("status", ""),
                result=raw.get("result", ""),
                raw=raw,
            )
        except Exception as e:
            logger.debug(f"Failed to parse Kalshi market: {e}")
            return None

    def _is_sport_market(self, market: KalshiMarket) -> bool:
        search_text = (
            market.title.lower() + " " +
            market.subtitle.lower() + " " +
            market.category.lower() + " " +
            market.sub_category.lower()
        )
        return any(tag in search_text for tag in SPORT_CATEGORIES)
