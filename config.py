import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    odds_api_key: str = os.getenv("ODDS_API_KEY", "")
    api_football_key: str = os.getenv("API_FOOTBALL_KEY", "")

    initial_bankroll: float = float(os.getenv("INITIAL_BANKROLL", "500"))
    min_edge: float = float(os.getenv("MIN_EDGE", "0.08"))
    max_bet_fraction: float = float(os.getenv("MAX_BET_FRACTION", "0.10"))
    scan_interval_minutes: int = int(os.getenv("SCAN_INTERVAL_MINUTES", "30"))
    event_horizon_hours: int = int(os.getenv("EVENT_HORIZON_HOURS", "12"))

    min_volume: float = 5000.0
    min_model_agreement: float = 0.6
    max_model_spread: float = 0.15
    min_minutes_before_event: int = 30
    max_hours_before_event: int = 48

    db_path: str = "data/paper_trades.db"

    polymarket_api_url: str = "https://clob.polymarket.com"
    polymarket_gamma_url: str = "https://gamma-api.polymarket.com"
    kalshi_api_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    odds_api_url: str = "https://api.the-odds-api.com/v4"

    sports: list[str] = field(default_factory=lambda: [
        "basketball_nba",
        "americanfootball_nfl",
        "soccer_epl",
        "soccer_spain_la_liga",
        "soccer_italy_serie_a",
        "soccer_germany_bundesliga",
        "soccer_france_ligue_one",
        "soccer_usa_mls",
        "soccer_uefa_champs_league",
        "mma_mixed_martial_arts",
        "icehockey_nhl",
    ])


config = Config()
