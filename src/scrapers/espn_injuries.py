"""
ESPN NBA Injury Scraper — Headless Selenium scraper.
Gets current injuries for all NBA teams with impact scoring.
"""

import logging
from dataclasses import dataclass

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logger = logging.getLogger("scraper.espn")

STAR_PLAYERS = {
    "L. James": 0.12, "S. Curry": 0.11, "G. Antetokounmpo": 0.13,
    "L. Doncic": 0.12, "N. Jokic": 0.13, "J. Tatum": 0.10,
    "K. Durant": 0.10, "J. Embiid": 0.12, "A. Edwards": 0.09,
    "D. Mitchell": 0.08, "J. Brunson": 0.09, "T. Young": 0.08,
    "D. Booker": 0.09, "S. Gilgeous-Alexander": 0.11, "J. Morant": 0.09,
    "D. Fox": 0.08, "B. Beal": 0.07, "P. George": 0.08,
    "K. Towns": 0.08, "B. Ingram": 0.07, "C. Cunningham": 0.08,
    "V. Wembanyama": 0.10, "P. Banchero": 0.08, "L. Ball": 0.07,
    "T. Halliburton": 0.09, "D. Murray": 0.07, "J. Harden": 0.08,
    "A. Davis": 0.10, "K. Leonard": 0.09, "J. Butler": 0.08,
    "Z. LaVine": 0.07, "D. Lillard": 0.08, "K. Irving": 0.08,
}

DEFAULT_PLAYER_IMPACT = 0.03

TEAM_NAME_MAP = {
    "Atlanta": "Atlanta Hawks", "Boston": "Boston Celtics",
    "Brooklyn": "Brooklyn Nets", "Charlotte": "Charlotte Hornets",
    "Chicago": "Chicago Bulls", "Cleveland": "Cleveland Cavaliers",
    "Dallas": "Dallas Mavericks", "Denver": "Denver Nuggets",
    "Detroit": "Detroit Pistons", "Golden State": "Golden State Warriors",
    "Houston": "Houston Rockets", "Indiana": "Indiana Pacers",
    "LA Clippers": "LA Clippers", "Los Angeles Lakers": "Los Angeles Lakers",
    "Memphis": "Memphis Grizzlies", "Miami": "Miami Heat",
    "Milwaukee": "Milwaukee Bucks", "Minnesota": "Minnesota Timberwolves",
    "New Orleans": "New Orleans Pelicans", "New York": "New York Knicks",
    "Oklahoma City": "Oklahoma City Thunder", "Orlando": "Orlando Magic",
    "Philadelphia": "Philadelphia 76ers", "Phoenix": "Phoenix Suns",
    "Portland": "Portland Trail Blazers", "Sacramento": "Sacramento Kings",
    "San Antonio": "San Antonio Spurs", "Toronto": "Toronto Raptors",
    "Utah": "Utah Jazz", "Washington": "Washington Wizards",
}


@dataclass
class InjuryEntry:
    team: str
    player: str
    position: str
    status: str
    description: str
    impact: float


def _get_driver():
    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    return webdriver.Chrome(options=opts)


def _resolve_team(raw_name: str) -> str:
    raw = raw_name.strip()
    for key, full in TEAM_NAME_MAP.items():
        if key.lower() in raw.lower() or raw.lower() in full.lower():
            return full
    return raw


def _player_impact(player_name: str, status: str) -> float:
    base = DEFAULT_PLAYER_IMPACT
    for star, imp in STAR_PLAYERS.items():
        if star.lower() in player_name.lower() or player_name.lower() in star.lower():
            base = imp
            break
    status_lower = status.lower()
    if "out" in status_lower:
        return base
    elif "doubtful" in status_lower:
        return base * 0.8
    elif "questionable" in status_lower:
        return base * 0.4
    elif "probable" in status_lower or "day-to-day" in status_lower:
        return base * 0.15
    return base * 0.2


def scrape_espn_injuries() -> dict[str, list[InjuryEntry]]:
    logger.info("Scraping ESPN NBA injuries...")
    driver = _get_driver()
    injuries: dict[str, list[InjuryEntry]] = {}

    try:
        driver.get("https://www.espn.com/nba/injuries")
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "table"))
        )
        soup = BeautifulSoup(driver.page_source, "lxml")

        team_sections = soup.find_all("div", class_="ResponsiveTable")
        if not team_sections:
            team_sections = soup.find_all("section")

        current_team = ""
        tables = soup.find_all("table")

        headings = soup.find_all(["h2", "h3", "div"])
        team_heading_map = {}
        for h in headings:
            text = h.get_text(strip=True)
            resolved = _resolve_team(text)
            if resolved != text or any(city in text for city in TEAM_NAME_MAP):
                team_heading_map[text] = resolved

        for section in soup.find_all("div", class_="ResponsiveTable"):
            header_el = section.find_previous(["h2", "h3", "div", "a"])
            if header_el:
                team_text = header_el.get_text(strip=True)
                current_team = _resolve_team(team_text)

            table = section.find("table")
            if not table:
                continue

            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all("td")
                if len(cells) < 3:
                    continue

                player = cells[0].get_text(strip=True)
                if not player or player.lower() in ("name", "player"):
                    continue

                pos = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                status_text = cells[-1].get_text(strip=True) if cells else ""
                desc = cells[2].get_text(strip=True) if len(cells) > 2 else ""

                impact = _player_impact(player, status_text)

                entry = InjuryEntry(
                    team=current_team,
                    player=player,
                    position=pos,
                    status=status_text,
                    description=desc,
                    impact=impact,
                )

                if current_team not in injuries:
                    injuries[current_team] = []
                injuries[current_team].append(entry)

        total = sum(len(v) for v in injuries.values())
        logger.info(f"ESPN: {total} injuries across {len(injuries)} teams")

    except Exception as e:
        logger.error(f"ESPN scrape failed: {e}")
    finally:
        driver.quit()

    return injuries


def get_team_injury_impact(injuries: dict[str, list[InjuryEntry]], team_name: str) -> float:
    team_injuries = injuries.get(team_name, [])
    if not team_injuries:
        for key, entries in injuries.items():
            if team_name.lower() in key.lower() or key.lower() in team_name.lower():
                team_injuries = entries
                break

    total_impact = sum(e.impact for e in team_injuries)
    return min(total_impact, 0.25)
