import logging
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger(__name__)

NEWSAPI_URL = "https://newsapi.org/v2/everything"


@dataclass
class NewsItem:
    title: str
    description: str
    source: str
    url: str
    published_at: str
    relevance_score: float


class NewsClient:
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

    async def search_team_news(self, team_name: str, sport: str = "") -> list[NewsItem]:
        session = await self._get_session()
        query = f"{team_name} {sport}".strip()
        results: list[NewsItem] = []

        try:
            url = f"https://www.google.com/search?q={query}+news&tbm=nws"
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            }
            async with session.get(url, headers=headers, allow_redirects=True) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    results = self._parse_google_news(text, team_name)
        except Exception as e:
            logger.debug(f"News search error for {team_name}: {e}")

        return results

    async def search_matchup_news(self, team_a: str, team_b: str) -> list[NewsItem]:
        results_a = await self.search_team_news(team_a)
        results_b = await self.search_team_news(team_b)
        combined = results_a + results_b
        combined.sort(key=lambda x: x.relevance_score, reverse=True)
        return combined[:10]

    def _parse_google_news(self, html: str, team_name: str) -> list[NewsItem]:
        items: list[NewsItem] = []
        team_lower = team_name.lower()

        injury_keywords = [
            "injury", "injured", "out", "doubtful", "questionable", "ruled out",
            "day-to-day", "concussion", "sprain", "strain", "torn", "surgery",
            "suspension", "suspended", "rest", "load management", "dnp",
        ]
        positive_keywords = [
            "return", "cleared", "healthy", "back", "available", "upgrade",
            "winning streak", "dominant", "blowout", "career high",
        ]
        negative_keywords = [
            "loss", "losing streak", "struggling", "worst", "collapse",
            "fired", "trade", "controversy",
        ]

        snippets = html.split("<div")
        for snippet in snippets:
            clean = snippet.replace("<b>", "").replace("</b>", "")
            lower = clean.lower()

            if team_lower not in lower:
                continue

            relevance = 0.5
            title = clean[:200].strip()

            for kw in injury_keywords:
                if kw in lower:
                    relevance = 0.9
                    break
            for kw in positive_keywords:
                if kw in lower:
                    relevance = max(relevance, 0.7)
            for kw in negative_keywords:
                if kw in lower:
                    relevance = max(relevance, 0.7)

            if relevance > 0.5:
                items.append(NewsItem(
                    title=title[:200],
                    description="",
                    source="google_news",
                    url="",
                    published_at="",
                    relevance_score=relevance,
                ))

        return items[:5]

    def extract_sentiment_keywords(self, news_items: list[NewsItem]) -> dict:
        injury_count = 0
        positive_count = 0
        negative_count = 0

        injury_words = {"injury", "injured", "out", "doubtful", "questionable", "ruled out", "surgery", "torn"}
        positive_words = {"return", "cleared", "healthy", "available", "winning", "dominant", "career high"}
        negative_words = {"loss", "losing", "struggling", "worst", "collapse", "fired", "suspension"}

        for item in news_items:
            text = (item.title + " " + item.description).lower()
            for w in injury_words:
                if w in text:
                    injury_count += 1
            for w in positive_words:
                if w in text:
                    positive_count += 1
            for w in negative_words:
                if w in text:
                    negative_count += 1

        total = max(injury_count + positive_count + negative_count, 1)
        return {
            "injury_signals": injury_count,
            "positive_signals": positive_count,
            "negative_signals": negative_count,
            "net_sentiment": (positive_count - negative_count - injury_count * 0.5) / total,
            "has_injury_news": injury_count > 0,
        }
