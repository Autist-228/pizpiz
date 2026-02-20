#!/usr/bin/env python3
"""
Crypto Arbitrage Scanner: Polymarket vs Exchange Prices
Paper trading mode - no wallets needed.

Monitors BTC/ETH/SOL price prediction markets on Polymarket,
compares with real-time exchange prices, finds mispriced contracts,
detects speed plays, and ranks all opportunities.
"""

import json
import logging
import math
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("arb")

DB_PATH = "data/crypto_arb.db"
SCAN_INTERVAL = 10
POLYMARKET_GAMMA = "https://gamma-api.polymarket.com"
KRAKEN_API = "https://api.kraken.com/0/public"
COINGECKO_API = "https://api.coingecko.com/api/v3"

CRYPTO_SLUGS = {
    "btc_2026": {
        "slug": "what-price-will-bitcoin-hit-before-2027",
        "coin": "BTC",
    },
    "btc_150k": {
        "slug": "when-will-bitcoin-hit-150k",
        "coin": "BTC",
    },
    "eth_2026": {
        "slug": "what-price-will-ethereum-hit-before-2027",
        "coin": "ETH",
    },
    "sol_2026": {
        "slug": "what-price-will-solana-hit-before-2027",
        "coin": "SOL",
    },
    "eth_gas": {
        "slug": "what-will-the-average-monthly-ethereum-gas-price-hit-before-2027",
        "coin": "ETH",
    },
    "ethena_2026": {
        "slug": "what-price-will-ethena-hit-before-2027",
        "coin": "ENA",
    },
}

EXCHANGE_PAIRS = {
    "BTC": {"kraken": "XXBTZUSD", "coingecko": "bitcoin"},
    "ETH": {"kraken": "XETHZUSD", "coingecko": "ethereum"},
    "SOL": {"kraken": "SOLUSD", "coingecko": "solana"},
}

HISTORICAL_VOL = {"BTC": 0.65, "ETH": 0.75, "SOL": 0.85, "ENA": 1.20}


@dataclass
class ExchangePrice:
    symbol: str
    price: float
    bid: float
    ask: float
    high_24h: float
    low_24h: float
    volume_24h: float
    change_1h: float = 0.0
    change_5m: float = 0.0
    timestamp: float = 0.0


@dataclass
class PolyContract:
    condition_id: str
    question: str
    coin: str
    direction: str
    target_price: float
    yes_price: float
    no_price: float
    volume: float
    end_date: str
    days_left: int
    slug: str
    event_key: str = ""


@dataclass
class Opportunity:
    contract: PolyContract
    exchange_price: float
    fair_value: float
    poly_price: float
    edge: float
    edge_pct: float
    action: str
    strategy: str
    expected_profit_cents: float
    confidence: float
    speed_flag: bool = False


class Database:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                bid REAL, ask REAL,
                high_24h REAL, low_24h REAL,
                volume_24h REAL
            );
            CREATE TABLE IF NOT EXISTS poly_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                condition_id TEXT NOT NULL,
                question TEXT,
                coin TEXT,
                direction TEXT,
                target_price REAL,
                yes_price REAL,
                no_price REAL,
                volume REAL,
                fair_value REAL,
                edge REAL
            );
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                question TEXT NOT NULL,
                coin TEXT NOT NULL,
                action TEXT NOT NULL,
                entry_price REAL NOT NULL,
                fair_value REAL NOT NULL,
                edge REAL NOT NULL,
                exchange_price REAL NOT NULL,
                status TEXT DEFAULT 'open',
                exit_price REAL DEFAULT 0,
                pnl REAL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS scan_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_contracts INTEGER,
                opportunities INTEGER,
                best_edge REAL,
                speed_plays INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_price_ts ON price_history(timestamp, symbol);
            CREATE INDEX IF NOT EXISTS idx_poly_ts ON poly_snapshots(timestamp, condition_id);
        """)
        self.conn.commit()

    def save_price(self, p: ExchangePrice):
        self.conn.execute(
            "INSERT INTO price_history (timestamp, symbol, price, bid, ask, high_24h, low_24h, volume_24h) VALUES (?,?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), p.symbol, p.price, p.bid, p.ask, p.high_24h, p.low_24h, p.volume_24h),
        )
        self.conn.commit()

    def save_poly_snapshot(self, c: PolyContract, fair_value: float, edge: float):
        self.conn.execute(
            "INSERT INTO poly_snapshots (timestamp, condition_id, question, coin, direction, target_price, yes_price, no_price, volume, fair_value, edge) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), c.condition_id, c.question, c.coin, c.direction, c.target_price, c.yes_price, c.no_price, c.volume, fair_value, edge),
        )
        self.conn.commit()

    def save_paper_trade(self, question: str, coin: str, action: str, entry: float, fair: float, edge: float, exch_price: float) -> int:
        cur = self.conn.execute(
            "INSERT INTO paper_trades (timestamp, question, coin, action, entry_price, fair_value, edge, exchange_price) VALUES (?,?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), question, coin, action, entry, fair, edge, exch_price),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_open_trades(self) -> list[dict]:
        cur = self.conn.execute("SELECT * FROM paper_trades WHERE status='open'")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_total_pnl(self) -> float:
        cur = self.conn.execute("SELECT COALESCE(SUM(pnl), 0) FROM paper_trades WHERE status='closed'")
        return cur.fetchone()[0]

    def close_trade(self, trade_id: int, exit_price: float, pnl: float):
        self.conn.execute(
            "UPDATE paper_trades SET status='closed', exit_price=?, pnl=? WHERE id=?",
            (exit_price, pnl, trade_id),
        )
        self.conn.commit()

    def save_scan_log(self, total: int, opps: int, best_edge: float, speed: int):
        self.conn.execute(
            "INSERT INTO scan_log (timestamp, total_contracts, opportunities, best_edge, speed_plays) VALUES (?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), total, opps, best_edge, speed),
        )
        self.conn.commit()


def norm_cdf(x: float) -> float:
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422804 * math.exp(-x * x / 2.0)
    p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.8212560 + t * 1.330274))))
    return 1.0 - p if x > 0 else p


def barrier_touch_prob(spot: float, barrier: float, time_years: float, vol: float, rate: float = 0.05) -> float:
    if spot <= 0 or barrier <= 0 or time_years <= 0 or vol <= 0:
        return 0.0
    if abs(spot - barrier) / spot < 0.005:
        return 1.0

    mu = rate - 0.5 * vol * vol
    sigma = vol
    sqrt_t = math.sqrt(time_years)

    if barrier > spot:
        ln_ratio = math.log(barrier / spot)
        d1 = (ln_ratio - mu * time_years) / (sigma * sqrt_t)
        d2 = (ln_ratio + mu * time_years) / (sigma * sqrt_t)
        exponent = 2.0 * mu * ln_ratio / (sigma * sigma)
    else:
        ln_ratio = math.log(spot / barrier)
        d1 = (ln_ratio + mu * time_years) / (sigma * sqrt_t)
        d2 = (ln_ratio - mu * time_years) / (sigma * sqrt_t)
        exponent = -2.0 * mu * ln_ratio / (sigma * sigma)

    exponent = max(min(exponent, 50), -50)
    p = norm_cdf(-d1) + math.exp(exponent) * norm_cdf(-d2)
    return max(0.0, min(1.0, p))


def calc_fair_value(coin: str, spot: float, target: float, direction: str, days_left: int) -> float:
    vol = HISTORICAL_VOL.get(coin, 0.70)
    time_years = max(days_left, 1) / 365.0
    if direction == "reach" and target <= spot:
        return 1.0
    if direction == "dip" and target >= spot:
        return 1.0
    return barrier_touch_prob(spot, target, time_years, vol)


def parse_target_price(question: str) -> tuple[str, float]:
    q = question.lower()
    direction = "reach"
    if "dip" in q or "below" in q or "drop" in q or "fall" in q:
        direction = "dip"

    dollar_match = re.search(r'\$\s*([\d,]+(?:\.\d+)?)\s*(k|K|m|M)?', question)
    if dollar_match:
        raw = dollar_match.group(1).replace(",", "")
        target = float(raw)
        suffix = dollar_match.group(2)
        if suffix and suffix.lower() == "k":
            target *= 1000
        elif suffix and suffix.lower() == "m":
            target *= 1_000_000
        return direction, target

    pct_match = re.search(r'(\d+(?:\.\d+)?)\s*%', question)
    if pct_match:
        return direction, 0.0

    number_matches = re.findall(r'(?:^|\s)([\d,]+(?:\.\d+)?)\s*(k|K)?', question)
    for raw, suffix in number_matches:
        val = float(raw.replace(",", ""))
        if 2020 <= val <= 2035:
            continue
        if val >= 50:
            if suffix and suffix.lower() == "k":
                val *= 1000
            return direction, val

    return direction, 0.0


def detect_coin(question: str, fallback: str = "") -> str:
    q = question.lower()
    if "bitcoin" in q or "btc" in q:
        return "BTC"
    if "ethereum" in q or " eth " in q or q.startswith("eth "):
        return "ETH"
    if "solana" in q or " sol " in q:
        return "SOL"
    if "ethena" in q or " ena " in q:
        return "ENA"
    return fallback


class ExchangeFeed:
    def __init__(self):
        self.prices: dict[str, ExchangePrice] = {}
        self.price_history: dict[str, list[tuple[float, float]]] = {
            "BTC": [], "ETH": [], "SOL": [],
        }

    def fetch_kraken(self) -> dict[str, ExchangePrice]:
        results = {}
        pairs = ",".join(v["kraken"] for v in EXCHANGE_PAIRS.values())
        try:
            resp = requests.get(
                f"{KRAKEN_API}/Ticker", params={"pair": pairs}, timeout=10
            )
            if resp.status_code != 200:
                return results
            data = resp.json().get("result", {})
            for coin, info in EXCHANGE_PAIRS.items():
                pair_key = info["kraken"]
                if pair_key not in data:
                    for k in data:
                        if pair_key.lower() in k.lower():
                            pair_key = k
                            break
                if pair_key not in data:
                    continue
                d = data[pair_key]
                price = float(d["c"][0])
                now = time.time()
                change_5m = 0.0
                change_1h = 0.0
                hist = self.price_history.get(coin, [])
                for ts, p in reversed(hist):
                    if now - ts >= 280 and change_5m == 0:
                        change_5m = (price - p) / p * 100
                    if now - ts >= 3500 and change_1h == 0:
                        change_1h = (price - p) / p * 100
                        break

                results[coin] = ExchangePrice(
                    symbol=coin,
                    price=price,
                    bid=float(d["b"][0]),
                    ask=float(d["a"][0]),
                    high_24h=float(d["h"][1]),
                    low_24h=float(d["l"][1]),
                    volume_24h=float(d["v"][1]),
                    change_5m=change_5m,
                    change_1h=change_1h,
                    timestamp=now,
                )
                hist.append((now, price))
                if len(hist) > 7200:
                    hist = hist[-3600:]
                self.price_history[coin] = hist
        except Exception as e:
            log.warning(f"Kraken error: {e}")
        return results

    def fetch_coingecko(self) -> dict[str, ExchangePrice]:
        results = {}
        try:
            ids = ",".join(v["coingecko"] for v in EXCHANGE_PAIRS.values())
            resp = requests.get(
                f"{COINGECKO_API}/simple/price",
                params={
                    "ids": ids,
                    "vs_currencies": "usd",
                    "include_24hr_vol": "true",
                    "include_24hr_change": "true",
                },
                timeout=10,
            )
            if resp.status_code != 200:
                return results
            data = resp.json()
            for coin, info in EXCHANGE_PAIRS.items():
                cg_id = info["coingecko"]
                if cg_id in data:
                    d = data[cg_id]
                    results[coin] = ExchangePrice(
                        symbol=coin,
                        price=d.get("usd", 0),
                        bid=d.get("usd", 0),
                        ask=d.get("usd", 0),
                        high_24h=0,
                        low_24h=0,
                        volume_24h=d.get("usd_24h_vol", 0),
                        timestamp=time.time(),
                    )
        except Exception as e:
            log.warning(f"CoinGecko error: {e}")
        return results

    def update(self) -> dict[str, ExchangePrice]:
        prices = self.fetch_kraken()
        if len(prices) < 2:
            cg = self.fetch_coingecko()
            for k, v in cg.items():
                if k not in prices:
                    prices[k] = v
        self.prices.update(prices)
        return self.prices


class PolymarketFeed:
    def __init__(self):
        self.contracts: list[PolyContract] = []
        self.last_full_fetch = 0.0

    def fetch_all(self) -> list[PolyContract]:
        contracts = []
        now = datetime.now(timezone.utc)

        for event_key, info in CRYPTO_SLUGS.items():
            slug = info["slug"]
            default_coin = info["coin"]
            try:
                resp = requests.get(
                    f"{POLYMARKET_GAMMA}/events/slug/{slug}", timeout=15
                )
                if resp.status_code != 200:
                    log.debug(f"Slug {slug}: HTTP {resp.status_code}")
                    continue
                ev_data = resp.json()
                if isinstance(ev_data, list):
                    ev_data = ev_data[0] if ev_data else {}

                for m in ev_data.get("markets", []):
                    question = m.get("question", "")
                    coin = detect_coin(question, default_coin)

                    prices_raw = m.get("outcomePrices", "[]")
                    if isinstance(prices_raw, str):
                        prices_list = json.loads(prices_raw)
                    else:
                        prices_list = prices_raw
                    if not prices_list:
                        continue

                    yes_price = float(prices_list[0])
                    no_price = float(prices_list[1]) if len(prices_list) > 1 else 1.0 - yes_price

                    if yes_price < 0.005 and no_price > 0.99:
                        continue
                    if yes_price > 0.995:
                        continue

                    end_str = m.get("endDate", "")
                    days_left = 365
                    if end_str:
                        try:
                            end_dt = datetime.fromisoformat(
                                end_str.replace("Z", "+00:00")
                            )
                            days_left = max((end_dt - now).days, 1)
                        except (ValueError, TypeError):
                            pass

                    direction, target = parse_target_price(question)

                    contracts.append(
                        PolyContract(
                            condition_id=m.get("conditionId", m.get("id", "")),
                            question=question,
                            coin=coin,
                            direction=direction,
                            target_price=target,
                            yes_price=yes_price,
                            no_price=no_price,
                            volume=float(m.get("volume", 0)),
                            end_date=end_str[:10] if end_str else "",
                            days_left=days_left,
                            slug=slug,
                            event_key=event_key,
                        )
                    )
            except Exception as e:
                log.warning(f"Polymarket error ({slug}): {e}")

        self.contracts = contracts
        self.last_full_fetch = time.time()
        return contracts


class SpeedDetector:
    def __init__(self):
        self.alerts: list[dict] = []

    def check(self, prices: dict[str, ExchangePrice]) -> list[dict]:
        self.alerts = []
        for coin, ep in prices.items():
            if abs(ep.change_5m) >= 1.5:
                tag = "UP" if ep.change_5m > 0 else "DOWN"
                self.alerts.append({
                    "coin": coin,
                    "type": "5m_move",
                    "change": ep.change_5m,
                    "msg": f"{coin} {tag} {abs(ep.change_5m):.2f}% in 5min @ ${ep.price:,.0f}",
                })
            if abs(ep.change_1h) >= 3.0:
                tag = "UP" if ep.change_1h > 0 else "DOWN"
                self.alerts.append({
                    "coin": coin,
                    "type": "1h_move",
                    "change": ep.change_1h,
                    "msg": f"{coin} {tag} {abs(ep.change_1h):.2f}% in 1hr @ ${ep.price:,.0f}",
                })
        return self.alerts


class OpportunityRanker:
    def __init__(self, min_edge: float = 0.03):
        self.min_edge = min_edge

    def rank(
        self,
        contracts: list[PolyContract],
        prices: dict[str, ExchangePrice],
        speed_alerts: list[dict],
    ) -> list[Opportunity]:
        opportunities = []
        speed_coins = {a["coin"] for a in speed_alerts}

        for c in contracts:
            if c.coin not in prices:
                continue
            ep = prices[c.coin]
            if c.target_price <= 0:
                continue

            fair = calc_fair_value(c.coin, ep.price, c.target_price, c.direction, c.days_left)
            is_speed = c.coin in speed_coins

            edge_yes = fair - c.yes_price
            if edge_yes > self.min_edge:
                pct = edge_yes / c.yes_price * 100 if c.yes_price > 0.001 else 0
                conf = min(edge_yes / 0.20, 1.0)
                if is_speed:
                    conf = min(conf * 1.3, 1.0)
                opportunities.append(Opportunity(
                    contract=c,
                    exchange_price=ep.price,
                    fair_value=fair,
                    poly_price=c.yes_price,
                    edge=edge_yes,
                    edge_pct=pct,
                    action=f"BUY YES @ {c.yes_price*100:.1f}c",
                    strategy="SPEED" if is_speed else "VALUE",
                    expected_profit_cents=edge_yes * 100,
                    confidence=conf,
                    speed_flag=is_speed,
                ))

            edge_no = (1.0 - fair) - c.no_price
            if edge_no > self.min_edge:
                pct = edge_no / c.no_price * 100 if c.no_price > 0.001 else 0
                conf = min(edge_no / 0.20, 1.0)
                if is_speed:
                    conf = min(conf * 1.3, 1.0)
                opportunities.append(Opportunity(
                    contract=c,
                    exchange_price=ep.price,
                    fair_value=fair,
                    poly_price=c.no_price,
                    edge=edge_no,
                    edge_pct=pct,
                    action=f"BUY NO @ {c.no_price*100:.1f}c",
                    strategy="SPEED" if is_speed else "VALUE",
                    expected_profit_cents=edge_no * 100,
                    confidence=conf,
                    speed_flag=is_speed,
                ))

        opportunities.sort(key=lambda o: o.edge, reverse=True)
        return opportunities


class PaperTrader:
    def __init__(self, db: Database, bankroll: float = 1000.0):
        self.db = db
        self.bankroll = bankroll
        self.max_positions = 20
        self.min_edge_to_trade = 0.05

    def should_trade(self, opp: Opportunity) -> bool:
        if opp.edge < self.min_edge_to_trade:
            return False
        if opp.contract.volume < 500:
            return False
        open_trades = self.db.get_open_trades()
        for t in open_trades:
            if t["question"] == opp.contract.question:
                return False
        if len(open_trades) >= self.max_positions:
            return False
        return True

    def execute(self, opp: Opportunity) -> int:
        return self.db.save_paper_trade(
            question=opp.contract.question,
            coin=opp.contract.coin,
            action=opp.action,
            entry=opp.poly_price,
            fair=opp.fair_value,
            edge=opp.edge,
            exch_price=opp.exchange_price,
        )

    def check_exits(self, contracts: list[PolyContract]):
        for trade in self.db.get_open_trades():
            for c in contracts:
                if c.question == trade["question"]:
                    current = c.yes_price if "YES" in trade["action"] else c.no_price
                    entry = trade["entry_price"]
                    pnl_pct = (current - entry) / entry * 100 if entry > 0 else 0
                    if pnl_pct > 25 or pnl_pct < -40:
                        self.db.close_trade(trade["id"], current, current - entry)
                        tag = "WIN" if current > entry else "LOSS"
                        log.info(
                            f"CLOSED #{trade['id']} [{tag}] {trade['question'][:40]} "
                            f"entry={entry*100:.1f}c exit={current*100:.1f}c pnl={pnl_pct:+.1f}%"
                        )
                    break


def format_dashboard(
    prices: dict[str, ExchangePrice],
    contracts: list[PolyContract],
    opportunities: list[Opportunity],
    speed_alerts: list[dict],
    open_trades: list[dict],
    total_pnl: float,
    scan_num: int,
    elapsed: float,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    w = 120
    lines = [
        "",
        "=" * w,
        f"  CRYPTO ARB SCANNER  |  Scan #{scan_num}  |  {now}  |  {elapsed:.1f}s  |  {len(contracts)} contracts",
        "=" * w,
        "",
        "  LIVE PRICES:",
    ]
    for coin in ["BTC", "ETH", "SOL"]:
        if coin in prices:
            ep = prices[coin]
            lines.append(
                f"    {coin}:  ${ep.price:>10,.2f}  "
                f"5m={ep.change_5m:>+6.2f}%  1h={ep.change_1h:>+6.2f}%  "
                f"24h: ${ep.low_24h:,.0f} - ${ep.high_24h:,.0f}"
            )

    if speed_alerts:
        lines.append("")
        lines.append("  >>> SPEED ALERTS <<<")
        for a in speed_alerts:
            lines.append(f"    !!! {a['msg']}")

    lines.append("")
    if not opportunities:
        lines.append("  NO OPPORTUNITIES (edge < 3%) — market efficiently priced right now")
    else:
        lines.append(f"  TOP OPPORTUNITIES ({len(opportunities)} found):")
        lines.append(
            f"  {'#':>3}  {'Coin':<4} {'Dir':<5} {'Target':>10} "
            f"{'Poly':>6} {'Fair':>6} {'Edge':>6} {'Edge%':>7} "
            f"{'Action':<22} {'Type':<7} {'Days':>5} {'Volume':>12}"
        )
        lines.append("  " + "-" * (w - 4))

        for i, opp in enumerate(opportunities[:40], 1):
            c = opp.contract
            target_str = f"${c.target_price:,.0f}" if c.target_price >= 1 else f"${c.target_price:.2f}"
            spd = " !!!" if opp.speed_flag else ""
            lines.append(
                f"  {i:>3}  {c.coin:<4} {c.direction:<5} {target_str:>10} "
                f"{opp.poly_price*100:>5.1f}c {opp.fair_value*100:>5.1f}c "
                f"{opp.edge*100:>5.1f}c {opp.edge_pct:>6.1f}% "
                f"{opp.action:<22} {opp.strategy:<7} "
                f"{c.days_left:>4}d ${c.volume:>11,.0f}{spd}"
            )

    if open_trades:
        lines.append("")
        lines.append(f"  PAPER PORTFOLIO ({len(open_trades)} open | closed PnL: {total_pnl*100:+.1f}c):")
        for t in open_trades:
            lines.append(
                f"    #{t['id']:<3} {t['coin']:<4} {t['action']:<22} "
                f"entry={t['entry_price']*100:.1f}c  edge={t['edge']*100:.1f}c  "
                f"@${t['exchange_price']:,.0f}"
            )

    lines.append("")
    lines.append("=" * w)
    return "\n".join(lines)


def run_scanner():
    log.info("Crypto Arb Scanner starting (paper mode)...")
    log.info(f"Monitoring {len(CRYPTO_SLUGS)} event groups across BTC/ETH/SOL")

    db = Database(DB_PATH)
    exchange = ExchangeFeed()
    poly = PolymarketFeed()
    speed = SpeedDetector()
    ranker = OpportunityRanker(min_edge=0.03)
    trader = PaperTrader(db)

    scan_num = 0
    poly_interval = 30
    last_poly = 0.0

    while True:
        try:
            scan_num += 1
            t0 = time.time()

            prices = exchange.update()
            if not prices:
                log.warning("No prices, retrying in 5s...")
                time.sleep(5)
                continue

            for ep in prices.values():
                db.save_price(ep)

            now = time.time()
            if now - last_poly >= poly_interval:
                contracts = poly.fetch_all()
                last_poly = now
                log.info(f"Polymarket: {len(contracts)} active contracts loaded")
            else:
                contracts = poly.contracts

            if not contracts:
                log.warning("No contracts yet, waiting...")
                time.sleep(3)
                continue

            speed_alerts = speed.check(prices)
            for a in speed_alerts:
                log.warning(f"SPEED: {a['msg']}")

            opportunities = ranker.rank(contracts, prices, speed_alerts)

            for c in contracts:
                if c.coin in prices and c.target_price > 0:
                    fair = calc_fair_value(
                        c.coin, prices[c.coin].price,
                        c.target_price, c.direction, c.days_left,
                    )
                    db.save_poly_snapshot(c, fair, fair - c.yes_price)

            new_trades = 0
            for opp in opportunities[:5]:
                if trader.should_trade(opp):
                    tid = trader.execute(opp)
                    new_trades += 1
                    log.info(
                        f"PAPER TRADE #{tid}: {opp.action} | "
                        f"{opp.contract.question[:50]} | edge={opp.edge*100:.1f}c"
                    )

            trader.check_exits(contracts)

            best_edge = opportunities[0].edge if opportunities else 0
            speed_count = sum(1 for o in opportunities if o.speed_flag)
            db.save_scan_log(len(contracts), len(opportunities), best_edge, speed_count)

            elapsed = time.time() - t0
            dashboard = format_dashboard(
                prices, contracts, opportunities, speed_alerts,
                db.get_open_trades(), db.get_total_pnl(),
                scan_num, elapsed,
            )
            print(dashboard, flush=True)

            sleep_time = max(SCAN_INTERVAL - elapsed, 1)
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            log.info("Scanner stopped.")
            break
        except Exception as e:
            log.error(f"Scan error: {e}", exc_info=True)
            time.sleep(10)


if __name__ == "__main__":
    run_scanner()
