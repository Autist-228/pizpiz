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
    "lebron james": 0.12, "stephen curry": 0.11, "giannis antetokounmpo": 0.13,
    "luka doncic": 0.12, "nikola jokic": 0.13, "jayson tatum": 0.10,
    "kevin durant": 0.10, "joel embiid": 0.12, "anthony edwards": 0.09,
    "donovan mitchell": 0.08, "jalen brunson": 0.09, "trae young": 0.08,
    "devin booker": 0.09, "shai gilgeous-alexander": 0.11, "ja morant": 0.09,
    "de'aaron fox": 0.08, "bradley beal": 0.07, "paul george": 0.08,
    "karl-anthony towns": 0.08, "brandon ingram": 0.07, "cade cunningham": 0.08,
    "victor wembanyama": 0.10, "paolo banchero": 0.08, "lamelo ball": 0.07,
    "tyrese haliburton": 0.09, "dejounte murray": 0.07, "james harden": 0.08,
    "anthony davis": 0.10, "kawhi leonard": 0.09, "jimmy butler": 0.08,
    "zach lavine": 0.07, "damian lillard": 0.08, "kyrie irving": 0.08,
    "darius garland": 0.07, "fred vanvleet": 0.07, "tyler herro": 0.07,
    "jalen williams": 0.07, "franz wagner": 0.07, "kristaps porzingis": 0.07,
    "andrew wiggins": 0.06, "og anunoby": 0.06,
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


ALL_TEAM_NAMES = set(TEAM_NAME_MAP.values())


def _resolve_team(raw_name: str) -> str:
    raw = raw_name.strip()
    if raw in ALL_TEAM_NAMES:
        return raw
    for key, full in TEAM_NAME_MAP.items():
        if key.lower() in raw.lower() or raw.lower() in full.lower():
            return full
    return ""


def _player_impact(player_name: str, status: str) -> float:
    base = DEFAULT_PLAYER_IMPACT
    name_lower = player_name.lower().strip()
    for star, imp in STAR_PLAYERS.items():
        if star in name_lower or name_lower in star:
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

        sections = soup.find_all("div", class_="ResponsiveTable")
        if not sections:
            sections = soup.find_all("section")

        for section in sections:
            current_team = ""

            team_span = section.find("span", class_="injuries__teamName")
            if team_span:
                current_team = _resolve_team(team_span.get_text(strip=True))

            if not current_team:
                for span in section.find_all("span"):
                    text = span.get_text(strip=True)
                    resolved = _resolve_team(text)
                    if resolved:
                        current_team = resolved
                        break

            if not current_team:
                parent = section.parent
                if parent:
                    team_span = parent.find("span", class_="injuries__teamName")
                    if team_span:
                        current_team = _resolve_team(team_span.get_text(strip=True))

            if not current_team:
                continue

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
                status_text = ""
                desc = ""
                if len(cells) >= 5:
                    status_text = cells[3].get_text(strip=True)
                    desc = cells[4].get_text(strip=True)
                elif len(cells) >= 4:
                    status_text = cells[3].get_text(strip=True)
                    desc = cells[2].get_text(strip=True)
                else:
                    status_text = cells[-1].get_text(strip=True)

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
