import logging

import numpy as np

from config import config

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self._init_llm()

    def _init_llm(self):
        anthropic_key = getattr(config, "anthropic_api_key", "")
        if anthropic_key:
            try:
                import anthropic
                self.anthropic_client = anthropic.AsyncAnthropic(api_key=anthropic_key)
                logger.info("Anthropic Claude client initialized for NLP layer")
            except Exception as e:
                logger.warning(f"Could not init Anthropic: {e}")
        elif config.openai_api_key:
            try:
                from openai import AsyncOpenAI
                self.openai_client = AsyncOpenAI(api_key=config.openai_api_key)
                logger.info("OpenAI client initialized for NLP layer")
            except Exception as e:
                logger.warning(f"Could not init OpenAI: {e}")

    async def analyze_matchup(
        self,
        home_team: str,
        away_team: str,
        sport: str,
        news_items: list,
        market_price: float,
    ) -> dict:
        if self.anthropic_client:
            return await self._claude_analysis(
                home_team, away_team, sport, news_items, market_price
            )
        if self.openai_client:
            return await self._gpt_analysis(
                home_team, away_team, sport, news_items, market_price
            )
        return self._rule_based_analysis(news_items, market_price)

    async def _claude_analysis(
        self,
        home_team: str,
        away_team: str,
        sport: str,
        news_items: list,
        market_price: float,
    ) -> dict:
        news_text = ""
        for item in news_items[:5]:
            news_text += f"- {item.title}\n"

        prompt = f"""Analyze this sports matchup for betting value:

Sport: {sport}
Home: {home_team}
Away: {away_team}
Current market probability (home win): {market_price:.1%}

Recent news:
{news_text if news_text else "No recent news available."}

Rate from -1.0 (strong away) to +1.0 (strong home):
1. injury_impact: How do injuries affect this matchup?
2. momentum: Which team has better recent momentum?
3. matchup_quality: How good is this matchup for betting?
4. news_sentiment: What does news suggest about outcome?
5. overall_edge: Is market price fair, too high, or too low for home team?

Respond ONLY with these 5 numbers separated by commas. Example: 0.3,-0.1,0.5,0.2,0.15"""

        try:
            response = await self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            values = [float(v.strip()) for v in text.split(",")]

            if len(values) >= 5:
                return {
                    "injury_impact": float(np.clip(values[0], -1, 1)),
                    "momentum_signal": float(np.clip(values[1], -1, 1)),
                    "matchup_quality": float(np.clip(values[2], -1, 1)),
                    "news_sentiment": float(np.clip(values[3], -1, 1)),
                    "overall_edge": float(np.clip(values[4], -1, 1)),
                    "combined_nlp_signal": float(np.mean(values[:5])),
                    "source": "claude",
                    "confidence": 0.75,
                }
        except Exception as e:
            logger.warning(f"Claude analysis error: {e}")

        return self._rule_based_analysis(news_items, market_price)

    async def _gpt_analysis(
        self,
        home_team: str,
        away_team: str,
        sport: str,
        news_items: list,
        market_price: float,
    ) -> dict:
        news_text = ""
        for item in news_items[:5]:
            news_text += f"- {item.title}\n"

        prompt = f"""Analyze this sports matchup for betting value:

Sport: {sport}
Home: {home_team}
Away: {away_team}
Current market probability (home win): {market_price:.1%}

Recent news:
{news_text if news_text else "No recent news available."}

Rate from -1.0 (strong away) to +1.0 (strong home):
1. injury_impact: How do injuries affect this matchup?
2. momentum: Which team has better recent momentum?
3. matchup_quality: How good is this matchup for betting?
4. news_sentiment: What does news suggest about outcome?
5. overall_edge: Is market price fair, too high, or too low for home team?

Respond ONLY with these 5 numbers separated by commas. Example: 0.3,-0.1,0.5,0.2,0.15"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.3,
            )
            text = response.choices[0].message.content.strip()
            values = [float(v.strip()) for v in text.split(",")]

            if len(values) >= 5:
                return {
                    "injury_impact": np.clip(values[0], -1, 1),
                    "momentum_signal": np.clip(values[1], -1, 1),
                    "matchup_quality": np.clip(values[2], -1, 1),
                    "news_sentiment": np.clip(values[3], -1, 1),
                    "overall_edge": np.clip(values[4], -1, 1),
                    "combined_nlp_signal": float(np.mean(values[:5])),
                    "source": "gpt",
                    "confidence": 0.7,
                }
        except Exception as e:
            logger.warning(f"GPT analysis error: {e}")

        return self._rule_based_analysis(news_items, market_price)

    def _rule_based_analysis(self, news_items: list, market_price: float) -> dict:
        injury_score = 0.0
        momentum_score = 0.0
        sentiment_score = 0.0

        injury_keywords = {
            "injury": -0.3, "injured": -0.3, "out": -0.4, "ruled out": -0.5,
            "doubtful": -0.3, "questionable": -0.2, "surgery": -0.5,
            "torn": -0.5, "concussion": -0.4, "day-to-day": -0.15,
            "return": 0.3, "cleared": 0.3, "healthy": 0.2, "available": 0.2,
        }
        momentum_keywords = {
            "winning streak": 0.3, "dominant": 0.2, "blowout": 0.15,
            "career high": 0.1, "losing streak": -0.3, "struggling": -0.2,
            "worst": -0.15, "collapse": -0.25,
        }

        for item in news_items:
            text = (item.title + " " + item.description).lower()
            for keyword, score in injury_keywords.items():
                if keyword in text:
                    injury_score += score
            for keyword, score in momentum_keywords.items():
                if keyword in text:
                    momentum_score += score
            if any(w in text for w in ["upset", "underdog", "surprise"]):
                sentiment_score -= 0.1
            if any(w in text for w in ["favorite", "expected", "should win"]):
                sentiment_score += 0.1

        injury_score = float(np.clip(injury_score, -1, 1))
        momentum_score = float(np.clip(momentum_score, -1, 1))
        sentiment_score = float(np.clip(sentiment_score, -1, 1))

        combined = (injury_score * 0.4 + momentum_score * 0.35 + sentiment_score * 0.25)

        return {
            "injury_impact": injury_score,
            "momentum_signal": momentum_score,
            "matchup_quality": 0.0,
            "news_sentiment": sentiment_score,
            "overall_edge": combined,
            "combined_nlp_signal": combined,
            "source": "rule_based",
            "confidence": 0.4,
        }
