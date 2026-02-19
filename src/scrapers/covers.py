"""
Covers.com Scraper — Public betting percentages + line movement.
When 70%+ of public is on one side, smart money is usually opposite.
"""

import logging
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logger = logging.getLogger("scraper.covers")


@dataclass
class PublicBetting:
    home_team: str
    away_team: str
    home_pct: float
    away_pct: float
    home_spread: str
    total: str
    line_movement: str


def _get_driver():
    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    return webdriver.Chrome(options=opts)


def scrape_covers_nba() -> list[PublicBetting]:
    logger.info("Scraping Covers.com NBA public betting...")
    driver = _get_driver()
    results: list[PublicBetting] = []

    try:
        driver.get("https://www.covers.com/sport/basketball/nba/consensus")
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        import time
        time.sleep(3)

        soup = BeautifulSoup(driver.page_source, "lxml")

        rows = soup.find_all("div", class_=re.compile(r"covers-CoversConsensus|consensus", re.I))
        if not rows:
            rows = soup.find_all("tr")

        for row in rows:
            cells = row.find_all(["td", "div", "span"])
            texts = [c.get_text(strip=True) for c in cells if c.get_text(strip=True)]

            if len(texts) < 4:
                continue

            pcts = [t for t in texts if "%" in t]
            teams = [t for t in texts if "%" not in t and len(t) > 2 and not t.startswith("+") and not t.startswith("-")]

            if len(pcts) >= 2 and len(teams) >= 2:
                try:
                    home_pct = float(pcts[0].replace("%", "")) / 100
                    away_pct = float(pcts[1].replace("%", "")) / 100
                except (ValueError, IndexError):
                    home_pct, away_pct = 0.5, 0.5

                results.append(PublicBetting(
                    home_team=teams[0],
                    away_team=teams[1] if len(teams) > 1 else "",
                    home_pct=home_pct,
                    away_pct=away_pct,
                    home_spread="",
                    total="",
                    line_movement="",
                ))

        logger.info(f"Covers: {len(results)} games with public betting data")

    except Exception as e:
        logger.error(f"Covers scrape failed: {e}")
    finally:
        driver.quit()

    return results


def get_public_betting(
    public_data: list[PublicBetting], home_team: str, away_team: str
) -> dict:
    ht = home_team.lower()
    at = away_team.lower()

    for pb in public_data:
        ph = pb.home_team.lower()
        pa = pb.away_team.lower()
        if (ph in ht or ht in ph) and (pa in at or at in pa):
            return {
                "home_pct": pb.home_pct,
                "away_pct": pb.away_pct,
                "contrarian_signal": "HOME" if pb.away_pct > 0.70 else "AWAY" if pb.home_pct > 0.70 else None,
                "strength": abs(pb.home_pct - 0.5) * 2,
            }
        parts_h = ht.split()
        parts_a = at.split()
        if any(p in ph for p in parts_h if len(p) > 3) or any(p in pa for p in parts_a if len(p) > 3):
            return {
                "home_pct": pb.home_pct,
                "away_pct": pb.away_pct,
                "contrarian_signal": "HOME" if pb.away_pct > 0.70 else "AWAY" if pb.home_pct > 0.70 else None,
                "strength": abs(pb.home_pct - 0.5) * 2,
            }

    return {"home_pct": 0.5, "away_pct": 0.5, "contrarian_signal": None, "strength": 0}
