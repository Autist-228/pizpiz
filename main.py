import asyncio
import logging
import sys
from pathlib import Path

from config import config
from src.scanner.orchestrator import Orchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/logs/bot.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


async def main():
    Path("data/logs").mkdir(parents=True, exist_ok=True)
    Path("data/models").mkdir(parents=True, exist_ok=True)
    Path("data/cache").mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ML Sports Prediction Bot — Starting")
    logger.info(f"Platforms: Polymarket + Kalshi")
    logger.info(f"Mode: Paper Trading (${config.initial_bankroll} bankroll)")
    logger.info(f"Event horizon: {config.event_horizon_hours}h")
    logger.info(f"Scan interval: {config.scan_interval_minutes} min")
    logger.info(f"Min edge: {config.min_edge:.0%}")
    logger.info(f"Telegram: {'enabled' if config.telegram_bot_token else 'disabled'}")
    logger.info(f"OpenAI NLP: {'enabled' if config.openai_api_key else 'rule-based'}")
    logger.info("=" * 60)

    orchestrator = Orchestrator()

    try:
        await orchestrator.initialize()
        logger.info("Running first scan...")
        await orchestrator.run_single_scan()
        logger.info("First scan complete. Starting continuous loop...")
        await orchestrator.run_loop()
    except KeyboardInterrupt:
        logger.info("Shutting down (keyboard interrupt)...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        await orchestrator.shutdown()
        logger.info("Bot stopped.")


if __name__ == "__main__":
    asyncio.run(main())
