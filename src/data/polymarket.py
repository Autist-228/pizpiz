import asyncio
import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field

import aiohttp

from config import config

logger = logging.getLogger(__name__)

SPORT_TAGS = [
    "sports", "nba", "nfl", "nhl", "mls", "ufc", "mma", "soccer", "football",
    "basketball", "baseball", "tennis", "boxing", "cricket", "hockey",
    "premier-league", "champions-league", "la-liga", "serie-a", "bundesliga",
]


@dataclass
class PolymarketMarket:
    condition_id: str
    question: str
    slug: str
    outcome_yes_price: float
    outcome_no_price: float
    volume: float
    liquidity: float
    end_date: datetime | None
    category: str
    tags: list[str]
    token_id_yes: str
    token_id_no: str
    active: bool
    closed: bool
    platform: str = "polymarket"
    raw: dict = field(default_factory=dict)


class PolymarketClient:
    def __init__(self):
        self.gamma_url = config.polymarket_gamma_url
        self.clob_url = config.polymarket_api_url
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

    async def fetch_sports_markets(self, horizon_hours: int | None = None) -> list[PolymarketMarket]:
        horizon = horizon_hours or config.event_horizon_hours
        now = datetime.now(timezone.utc)
        deadline = now + timedelta(hours=horizon)
        markets: list[PolymarketMarket] = []

        try:
            raw_markets = await self._fetch_gamma_markets()
            for raw in raw_markets:
                market = self._parse_market(raw)
                if market is None:
                    continue
                if not self._is_sport_market(market):
                    continue
                if market.closed or not market.active:
                    continue
                if market.end_date and market.end_date > deadline:
                    continue
                if market.end_date and market.end_date < now:
                    continue
                markets.append(market)
        except Exception as e:
            logger.error(f"Error fetching Polymarket sports markets: {e}")

        logger.info(f"Polymarket: found {len(markets)} sport markets within {horizon}h")
        return markets

    async def _fetch_gamma_markets(self) -> list[dict]:
        session = await self._get_session()
        all_markets: list[dict] = []
        offset = 0
        limit = 100

        while True:
            url = f"{self.gamma_url}/markets"
            params = {
                "limit": limit,
                "offset": offset,
                "active": "true",
                "closed": "false",
                "order": "volume",
                "ascending": "false",
            }
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        logger.warning(f"Polymarket gamma API returned {resp.status}")
                        break
                    data = await resp.json()
                    if not data:
                        break
                    all_markets.extend(data)
                    if len(data) < limit:
                        break
                    offset += limit
            except Exception as e:
                logger.error(f"Error fetching gamma markets page {offset}: {e}")
                break

        return all_markets

    async def fetch_market_orderbook(self, token_id: str) -> dict:
        session = await self._get_session()
        url = f"{self.clob_url}/book"
        params = {"token_id": token_id}
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            logger.error(f"Error fetching orderbook for {token_id}: {e}")
        return {"bids": [], "asks": []}

    async def fetch_price_history(self, condition_id: str, fidelity: int = 60) -> list[dict]:
        session = await self._get_session()
        url = f"{self.clob_url}/prices-history"
        params = {
            "market": condition_id,
            "interval": "max",
            "fidelity": fidelity,
        }
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("history", [])
        except Exception as e:
            logger.error(f"Error fetching price history for {condition_id}: {e}")
        return []

    def _parse_market(self, raw: dict) -> PolymarketMarket | None:
        try:
            end_date_str = raw.get("endDate") or raw.get("end_date_iso")
            end_date = None
            if end_date_str:
                end_date_str = end_date_str.replace("Z", "+00:00")
                end_date = datetime.fromisoformat(end_date_str)

            outcomes_prices = raw.get("outcomePrices", "")
            if isinstance(outcomes_prices, str) and outcomes_prices:
                import json
                prices = json.loads(outcomes_prices)
                yes_price = float(prices[0]) if len(prices) > 0 else 0.5
                no_price = float(prices[1]) if len(prices) > 1 else 1 - yes_price
            elif isinstance(outcomes_prices, list):
                yes_price = float(outcomes_prices[0]) if len(outcomes_prices) > 0 else 0.5
                no_price = float(outcomes_prices[1]) if len(outcomes_prices) > 1 else 1 - yes_price
            else:
                yes_price = 0.5
                no_price = 0.5

            tokens = raw.get("clobTokenIds", "")
            if isinstance(tokens, str) and tokens:
                import json
                token_ids = json.loads(tokens)
            elif isinstance(tokens, list):
                token_ids = tokens
            else:
                token_ids = ["", ""]

            tags_raw = raw.get("tags", [])
            if isinstance(tags_raw, str):
                import json
                try:
                    tags = json.loads(tags_raw)
                except (json.JSONDecodeError, ValueError):
                    tags = [tags_raw]
            else:
                tags = tags_raw or []

            return PolymarketMarket(
                condition_id=raw.get("conditionId", raw.get("condition_id", "")),
                question=raw.get("question", ""),
                slug=raw.get("slug", ""),
                outcome_yes_price=yes_price,
                outcome_no_price=no_price,
                volume=float(raw.get("volume", 0) or 0),
                liquidity=float(raw.get("liquidity", 0) or 0),
                end_date=end_date,
                category=raw.get("category", ""),
                tags=[t.lower() if isinstance(t, str) else str(t) for t in tags],
                token_id_yes=token_ids[0] if len(token_ids) > 0 else "",
                token_id_no=token_ids[1] if len(token_ids) > 1 else "",
                active=raw.get("active", True),
                closed=raw.get("closed", False),
                raw=raw,
            )
        except Exception as e:
            logger.debug(f"Failed to parse Polymarket market: {e}")
            return None

    def _is_sport_market(self, market: PolymarketMarket) -> bool:
        search_text = (
            market.question.lower() + " " +
            market.category.lower() + " " +
            " ".join(market.tags)
        )
        return any(tag in search_text for tag in SPORT_TAGS)
