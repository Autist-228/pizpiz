"""
Basketball Reference Scraper — H2H history + advanced stats.
"""

import logging
import re
import time
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("scraper.bball_ref")

TIMEZONE_MAP = {
    "ATL": "ET", "BOS": "ET", "BKN": "ET", "CHA": "ET", "CLE": "ET",
    "DET": "ET", "IND": "ET", "MIA": "ET", "NYK": "ET", "ORL": "ET",
    "PHI": "ET", "TOR": "ET", "WAS": "ET",
    "CHI": "CT", "MIL": "CT", "MIN": "CT", "NOP": "CT", "MEM": "CT",
    "HOU": "CT", "DAL": "CT", "SAS": "CT", "OKC": "CT",
    "DEN": "MT", "UTA": "MT", "PHX": "MT",
    "LAC": "PT", "LAL": "PT", "GSW": "PT", "SAC": "PT", "POR": "PT",
}

TEAM_ABBREV = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

CONFERENCE_EAST = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND", "MIA", "MIL", "NYK", "ORL", "PHI", "TOR", "WAS"}
CONFERENCE_WEST = {"DAL", "DEN", "GSW", "HOU", "LAC", "LAL", "MEM", "MIN", "NOP", "OKC", "PHX", "POR", "SAC", "SAS", "UTA"}


@dataclass
class TeamAdvancedStats:
    team: str
    abbrev: str
    off_rating: float = 0.0
    def_rating: float = 0.0
    net_rating: float = 0.0
    pace: float = 0.0
    sos: float = 0.0


def scrape_team_ratings() -> dict[str, TeamAdvancedStats]:
    logger.info("Scraping Basketball Reference team ratings...")
    stats: dict[str, TeamAdvancedStats] = {}

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.google.com/",
    }

    try:
        resp = requests.get(
            "https://www.basketball-reference.com/leagues/NBA_2026.html",
            headers=headers,
            timeout=15,
        )
        if resp.status_code != 200:
            logger.warning(f"BBall Ref returned {resp.status_code}, skipping")
            return stats

        soup = BeautifulSoup(resp.text, "lxml")

        misc_table = soup.find("table", id="advanced-team")
        if not misc_table:
            for comment in soup.find_all(string=lambda t: t and "advanced-team" in str(t)):
                comment_soup = BeautifulSoup(str(comment), "lxml")
                misc_table = comment_soup.find("table", id="advanced-team")
                if misc_table:
                    break

        if not misc_table:
            tables = soup.find_all("table")
            for t in tables:
                header_text = " ".join(th.get_text(strip=True) for th in t.find_all("th"))
                if "ORtg" in header_text or "Pace" in header_text:
                    misc_table = t
                    break

        if misc_table:
            rows = misc_table.find_all("tr")
            all_headers = []
            for row in rows[:3]:
                ths = row.find_all("th")
                h = [th.get_text(strip=True) for th in ths]
                if len(h) > len(all_headers):
                    all_headers = h

            for row in rows[1:]:
                cells = row.find_all(["td", "th"])
                if len(cells) < 5:
                    continue

                values = {}
                for i, cell in enumerate(cells):
                    key = all_headers[i] if i < len(all_headers) else f"col{i}"
                    values[key] = cell.get_text(strip=True)

                team_text = values.get("Team", values.get("col0", ""))
                team_text = re.sub(r"\*", "", team_text).strip()

                abbrev = ""
                for full_name, ab in TEAM_ABBREV.items():
                    if team_text.lower() in full_name.lower() or full_name.lower() in team_text.lower():
                        abbrev = ab
                        team_text = full_name
                        break

                if not abbrev:
                    continue

                def _float(key: str) -> float:
                    try:
                        return float(values.get(key, "0"))
                    except (ValueError, TypeError):
                        return 0.0

                stats[team_text] = TeamAdvancedStats(
                    team=team_text,
                    abbrev=abbrev,
                    off_rating=_float("ORtg"),
                    def_rating=_float("DRtg"),
                    net_rating=_float("NRtg") or _float("ORtg") - _float("DRtg"),
                    pace=_float("Pace"),
                    sos=_float("SOS"),
                )

        logger.info(f"BBall Ref: {len(stats)} teams with advanced stats")

    except requests.Timeout:
        logger.warning("BBall Ref timed out, skipping advanced stats")
    except Exception as e:
        logger.error(f"BBall Ref scrape failed: {e}")

    return stats


def get_travel_factor(home_team: str, away_team: str) -> float:
    home_abbrev = TEAM_ABBREV.get(home_team, "")
    away_abbrev = TEAM_ABBREV.get(away_team, "")

    if not home_abbrev or not away_abbrev:
        return 0.0

    home_tz = TIMEZONE_MAP.get(home_abbrev, "")
    away_tz = TIMEZONE_MAP.get(away_abbrev, "")
    tz_order = {"ET": 0, "CT": 1, "MT": 2, "PT": 3}
    home_idx = tz_order.get(home_tz, -1)
    away_idx = tz_order.get(away_tz, -1)

    if home_idx < 0 or away_idx < 0:
        return 0.0

    tz_diff = abs(home_idx - away_idx)
    if tz_diff >= 2:
        return 0.02 if away_idx > home_idx else -0.02
    elif tz_diff == 1:
        return 0.008 if away_idx > home_idx else -0.008

    return 0.0
