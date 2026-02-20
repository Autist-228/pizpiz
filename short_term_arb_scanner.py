#!/usr/bin/env python3
"""
Short-term Crypto Arbitrage Scanner: Polymarket vs Exchange Prices
Scans daily (1-7 day), weekly, and monthly crypto prediction markets.
Uses barrier option pricing to find mispriced contracts.
Shows leverage hedging scenarios with Bybit/Kraken futures.
"""

import json
import math
import re
import sys
import time
from datetime import datetime, timezone

import requests

BASE = "https://gamma-api.polymarket.com"
KRAKEN_SPOT = "https://api.kraken.com/0/public/Ticker"
KRAKEN_FUTURES = "https://futures.kraken.com/derivatives/api/v3/tickers"

VOL = {"BTC": 0.65, "ETH": 0.75, "SOL": 0.85, "XRP": 0.90, "DOGE": 1.0}

SLUG_TEMPLATES = [
    "bitcoin-above-on-february-{d}",
    "bitcoin-price-on-february-{d}",
    "ethereum-above-on-february-{d}",
    "ethereum-price-on-february-{d}",
    "solana-above-on-february-{d}",
    "solana-price-on-february-{d}",
    "xrp-above-on-february-{d}",
    "xrp-price-on-february-{d}",
]

WEEKLY_SLUGS = [
    "what-price-will-bitcoin-hit-february-16-22",
    "what-price-will-ethereum-hit-february-16-22",
    "what-price-will-solana-hit-february-16-22",
    "what-price-will-xrp-hit-february-16-22",
]

MONTHLY_SLUGS = [
    "what-price-will-bitcoin-hit-in-february-2026",
    "what-price-will-ethereum-hit-in-february-2026",
    "what-price-will-solana-hit-in-february-2026",
    "what-price-will-xrp-hit-in-february-2026",
    "what-price-will-dogecoin-hit-in-february-2026",
]


def build_slugs():
    slugs = []
    for day in range(20, 27):
        for tmpl in SLUG_TEMPLATES:
            slugs.append(tmpl.format(d=day))
    slugs.extend(WEEKLY_SLUGS)
    slugs.extend(MONTHLY_SLUGS)
    return slugs


def fetch_markets(slugs):
    all_markets = []
    for slug in slugs:
        try:
            resp = requests.get(f"{BASE}/events", params={"slug": slug}, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    ev = data[0]
                    for m in ev.get("markets", []):
                        op = m.get("outcomePrices", "[]")
                        if isinstance(op, str):
                            try:
                                op = json.loads(op)
                            except ValueError:
                                op = []
                        all_markets.append({
                            "slug": slug,
                            "event": ev.get("title", ""),
                            "end": ev.get("endDate", ""),
                            "q": m.get("question", ""),
                            "yes": float(op[0]) if len(op) > 0 else 0,
                            "no": float(op[1]) if len(op) > 1 else 0,
                            "vol": float(m.get("volumeNum", 0) or 0),
                            "liq": float(m.get("liquidityNum", 0) or 0),
                        })
            time.sleep(0.1)
        except Exception:
            pass
    return all_markets


def fetch_spots():
    spot = {}
    r = requests.get(KRAKEN_SPOT, params={"pair": "XXBTZUSD,XETHZUSD,SOLUSD"}, timeout=10)
    for k, v in r.json().get("result", {}).items():
        if "XBT" in k:
            spot["BTC"] = float(v["c"][0])
        elif "ETH" in k:
            spot["ETH"] = float(v["c"][0])
        elif "SOL" in k:
            spot["SOL"] = float(v["c"][0])
    for pair, coin in [("XRPUSD", "XRP"), ("DOGEUSD", "DOGE")]:
        try:
            r2 = requests.get(KRAKEN_SPOT, params={"pair": pair}, timeout=10)
            for _k, v in r2.json().get("result", {}).items():
                spot[coin] = float(v["c"][0])
        except Exception:
            pass
    return spot


def fetch_futures():
    futures = {}
    r = requests.get(KRAKEN_FUTURES, timeout=10)
    for t in r.json().get("tickers", []):
        sym = t.get("symbol", "")
        for coin, fsym in [("BTC", "PF_XBTUSD"), ("ETH", "PF_ETHUSD"), ("SOL", "PF_SOLUSD")]:
            if sym == fsym:
                perp = float(t.get("last", 0) or 0)
                fr = t.get("fundingRate", 0)
                if not isinstance(fr, (int, float)):
                    fr = 0
                fr = float(fr)
                if abs(fr) > 0.01:
                    fr /= 100
                futures[coin] = {"perp": perp, "fr": fr}
    return futures


def detect_coin(t):
    t = t.lower()
    if "bitcoin" in t or "btc" in t:
        return "BTC"
    if "ethereum" in t or " eth " in t:
        return "ETH"
    if "solana" in t or " sol " in t:
        return "SOL"
    if "xrp" in t:
        return "XRP"
    if "doge" in t:
        return "DOGE"
    return None


def parse_target(q):
    q = q.lower()
    for pat, d in [
        (r'above\s+\$?([\d,]+(?:\.\d+)?)', "above"),
        (r'(?:hit|reach)\s+\$?([\d,]+(?:\.\d+)?)', "reach"),
        (r'dip\s+to\s+\$?([\d,]+(?:\.\d+)?)', "dip"),
        (r'below\s+\$?([\d,]+(?:\.\d+)?)', "below"),
        (r'<\s*\$?([\d,]+(?:\.\d+)?)', "below"),
        (r'>\s*\$?([\d,]+(?:\.\d+)?)', "above"),
    ]:
        m = re.search(pat, q)
        if m:
            return float(m.group(1).replace(",", "")), d
    m = re.search(r'\$([\d,]+)\s*[-\u2013]\s*\$([\d,]+)', q)
    if m:
        lo = float(m.group(1).replace(",", ""))
        hi = float(m.group(2).replace(",", ""))
        return (lo + hi) / 2, "range"
    return None, None


def ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def fair_above(s, t, d, v):
    if d <= 0:
        return 1.0 if s >= t else 0.0
    if t <= 0:
        return 1.0
    tt = d / 365.0
    dd = (math.log(s / t) + (-0.5 * v**2) * tt) / (v * math.sqrt(tt))
    return ncdf(dd)


def fair_touch(s, t, d, v):
    if s >= t:
        return 1.0
    if t <= 0:
        return 0.0
    tt = d / 365.0
    if tt <= 0:
        return 1.0 if s >= t else 0.0
    ratio = math.log(t / s)
    sst = v * math.sqrt(tt)
    if sst == 0:
        return 0.0
    d1 = (-ratio + 0.5 * v**2 * tt) / sst
    d2 = (-ratio - 0.5 * v**2 * tt) / sst
    return min(max(ncdf(d1) + (t / s) * ncdf(d2), 0), 1)


def fair_dip(s, k, d, v):
    if s <= k:
        return 1.0
    if k <= 0:
        return 0.0
    tt = d / 365.0
    if tt <= 0:
        return 0.0
    sst = v * math.sqrt(tt)
    if sst == 0:
        return 0.0
    lnks = math.log(k / s)
    mu = -0.5 * v**2
    d1 = (lnks + mu * tt) / sst
    d2 = (lnks - mu * tt) / sst
    return min(max(ncdf(d1) + (s / k) * ncdf(d2), 0), 1)


def analyze(all_markets, spot, now):
    results = []
    for m in all_markets:
        coin = detect_coin(m["event"] + " " + m["q"])
        if not coin or coin not in spot:
            continue
        target, direction = parse_target(m["q"])
        if target is None:
            continue
        s = spot[coin]
        try:
            end = datetime.fromisoformat(m["end"].replace("Z", "+00:00"))
            d = max((end - now).total_seconds() / 86400, 0.001)
        except Exception:
            d = 1
        y, n = m["yes"], m["no"]
        if y < 0.003 and n < 0.003:
            continue
        if y >= 0.99 or n >= 0.99:
            continue
        if y <= 0.01 or n <= 0.01:
            continue
        if m["liq"] < 100:
            continue

        v = VOL.get(coin, 0.75)
        if direction == "above":
            fair = fair_above(s, target, d, v)
        elif direction == "reach":
            fair = fair_touch(s, target, d, v)
        elif direction == "dip":
            fair = fair_dip(s, target, d, v)
        elif direction == "below":
            fair = 1 - fair_above(s, target, d, v)
        elif direction == "range":
            fair = 0.10
        else:
            fair = 0.5

        ey = fair - y
        en = (1 - fair) - n
        if abs(ey) > abs(en):
            be, bs, bp = ey, "BUY YES", y
        else:
            be, bs, bp = en, "BUY NO", n

        dist = (target - s) / s * 100 if s > 0 else 0
        et = m["event"]
        if "16-22" in et:
            tp = "WEEKLY"
        elif "above" in et.lower() or "price on" in et.lower():
            tp = "DAILY"
        else:
            tp = "MONTHLY"

        results.append({
            "coin": coin, "q": m["q"], "event": et, "target": target,
            "dir": direction, "spot": s, "dist": dist,
            "yes": y, "no": n, "days": d, "vol": m["vol"], "liq": m["liq"],
            "type": tp, "end": m["end"], "fair": fair,
            "edge_y": ey, "edge_n": en,
            "best_edge": be, "best_side": bs, "best_price": bp,
        })

    results.sort(key=lambda x: abs(x["best_edge"]), reverse=True)
    return results


def format_report(results, spot, futures, now, n_events, n_markets):
    L = []

    def w(s=""):
        L.append(s)

    w("=" * 130)
    w("POLYMARKET CRYPTO ARB SCANNER — SHORT-TERM")
    w(f"Generated: {now.strftime('%Y-%m-%d %H:%M UTC')}")
    w("=" * 130)

    w("\nLIVE SPOT PRICES:")
    for c in ["BTC", "ETH", "SOL", "XRP", "DOGE"]:
        if c in spot:
            p = spot[c]
            w(f"  {c}: ${p:,.2f}" if p > 10 else f"  {c}: ${p:,.4f}")

    w("\nFUTURES DATA:")
    for c in ["BTC", "ETH", "SOL"]:
        if c in futures:
            f_ = futures[c]
            basis = (f_["perp"] - spot.get(c, 0)) / spot.get(c, 1) * 100 if spot.get(c, 0) else 0
            ann = f_["fr"] * 3 * 365 * 100
            w(f"  {c}: Perp=${f_['perp']:,.2f}  Basis={basis:+.3f}%  FR/8h={f_['fr']*100:.4f}%  Ann={ann:+.1f}%")

    daily = [r for r in results if r["type"] == "DAILY"]
    weekly = [r for r in results if r["type"] == "WEEKLY"]
    monthly = [r for r in results if r["type"] == "MONTHLY"]
    w(f"\nSCANNED: {n_events} events, {n_markets} markets, {len(results)} opportunities")
    w(f"  DAILY: {len(daily)} | WEEKLY: {len(weekly)} | MONTHLY: {len(monthly)}")

    w("\n" + "=" * 130)
    w("TOP 50 ARBITRAGE OPPORTUNITIES BY EDGE")
    w("=" * 130)
    hdr = f"{'#':>3} {'Coin':>4} {'Type':>7} {'Question':<52} {'Days':>5} {'Spot':>10} {'Target':>10} {'Dist%':>7} {'YES':>6} {'NO':>6} {'Fair':>6} {'EDGE':>7} {'Side':>8}"
    w(hdr)
    w("-" * 130)
    for i, a in enumerate(results[:50]):
        qs = a["q"][:50]
        sp = f"${a['spot']:,.0f}" if a["spot"] > 10 else f"${a['spot']:,.3f}"
        tg = f"${a['target']:,.0f}" if a["target"] > 10 else f"${a['target']:,.3f}"
        w(f"{i+1:>3} {a['coin']:>4} {a['type']:>7} {qs:<52} {a['days']:>5.1f} {sp:>10} {tg:>10} {a['dist']:>+7.1f}% {a['yes']:>5.1%} {a['no']:>5.1%} {a['fair']:>5.1%} {a['best_edge']:>+6.1%} {a['best_side']:>8}")

    w("\n" + "=" * 130)
    w("EXPIRING TODAY — URGENT")
    w("=" * 130)
    today = sorted(
        [a for a in results if a["days"] < 1 and abs(a["best_edge"]) > 0.005],
        key=lambda x: abs(x["best_edge"]), reverse=True,
    )
    w(f"{'#':>3} {'Coin':>4} {'Question':<52} {'Hrs':>5} {'Spot':>10} {'Target':>10} {'YES':>6} {'NO':>6} {'Fair':>6} {'EDGE':>7} {'Side':>8}")
    w("-" * 130)
    for i, a in enumerate(today[:30]):
        qs = a["q"][:50]
        hrs = a["days"] * 24
        sp = f"${a['spot']:,.0f}" if a["spot"] > 10 else f"${a['spot']:,.3f}"
        tg = f"${a['target']:,.0f}" if a["target"] > 10 else f"${a['target']:,.3f}"
        w(f"{i+1:>3} {a['coin']:>4} {qs:<52} {hrs:>4.0f}h {sp:>10} {tg:>10} {a['yes']:>5.1%} {a['no']:>5.1%} {a['fair']:>5.1%} {a['best_edge']:>+6.1%} {a['best_side']:>8}")

    w("\n" + "=" * 130)
    w("EXPIRING IN 1-3 DAYS")
    w("=" * 130)
    near = sorted(
        [a for a in results if 0.5 < a["days"] <= 4 and abs(a["best_edge"]) > 0.005],
        key=lambda x: abs(x["best_edge"]), reverse=True,
    )
    w(f"{'#':>3} {'Coin':>4} {'Type':>7} {'Question':<50} {'Days':>5} {'Spot':>10} {'Target':>10} {'YES':>6} {'NO':>6} {'Fair':>6} {'EDGE':>7} {'Side':>8}")
    w("-" * 130)
    for i, a in enumerate(near[:40]):
        qs = a["q"][:48]
        sp = f"${a['spot']:,.0f}" if a["spot"] > 10 else f"${a['spot']:,.3f}"
        tg = f"${a['target']:,.0f}" if a["target"] > 10 else f"${a['target']:,.3f}"
        w(f"{i+1:>3} {a['coin']:>4} {a['type']:>7} {qs:<50} {a['days']:>5.1f} {sp:>10} {tg:>10} {a['yes']:>5.1%} {a['no']:>5.1%} {a['fair']:>5.1%} {a['best_edge']:>+6.1%} {a['best_side']:>8}")

    w("\n" + "=" * 130)
    w("LEVERAGE HEDGED PLAYS (Polymarket + Exchange Futures)")
    w("Capital: $1,000 = $600 Polymarket + $400 Exchange hedge")
    w("=" * 130)
    good = [a for a in results if abs(a["best_edge"]) > 0.02]
    for lev in [5, 10, 25, 50]:
        liq_d = 100.0 / lev
        notional = 400 * lev
        w(f"\n--- {lev}x LEVERAGE (Liq dist: {liq_d:.1f}%, Notional: ${notional:,}) ---")
        w(f"  {'#':>3} {'Coin':>4} {'Type':>7} {'Question':<40} {'Days':>5} {'Edge':>6} {'Side':>8} {'Hedge':>6} {'Win$':>8} {'Lose$':>8} {'EV$':>8} {'ROI%':>6}")
        w(f"  {'-' * 112}")
        for i, a in enumerate(good[:20]):
            price = a["best_price"]
            fair_v = a["fair"]
            days_v = a["days"]
            coin_a = a["coin"]
            if a["dir"] in ("above", "reach"):
                hedge = "LONG" if a["best_side"] == "BUY YES" else "SHORT"
            else:
                hedge = "SHORT" if a["best_side"] == "BUY YES" else "LONG"
            contracts = 600 / price if price > 0 else 0
            fr_daily = abs(futures.get(coin_a, {}).get("fr", 0.0001)) * 3
            fund = notional * fr_daily * days_v
            win = contracts * (1 - price) - fund
            lose = -600 - fund
            if a["best_side"] == "BUY YES":
                ev = win * fair_v + lose * (1 - fair_v)
            else:
                ev = win * (1 - fair_v) + lose * fair_v
            roi = ev / 1000 * 100
            qs = a["q"][:38]
            w(f"  {i+1:>3} {coin_a:>4} {a['type']:>7} {qs:<40} {days_v:>5.1f} {a['best_edge']:>+5.1%} {a['best_side']:>8} {hedge:>6} {win:>+8.0f} {lose:>+8.0f} {ev:>+8.0f} {roi:>+5.1f}%")

    w("\n" + "=" * 130)
    w("FUNDING RATE ARBITRAGE (Cash & Carry)")
    w("=" * 130)
    for coin in ["BTC", "ETH", "SOL"]:
        if coin in futures and coin in spot:
            f_ = futures[coin]
            s_ = spot[coin]
            fr = f_["fr"]
            ann = fr * 3 * 365 * 100
            w(f"\n  {coin}:")
            w(f"    Spot: ${s_:,.2f}  Perp: ${f_['perp']:,.2f}")
            w(f"    Funding rate: {fr*100:+.4f}%/8h  = {ann:+.1f}%/yr")
            if fr > 0:
                w(f"    Strategy: SHORT perp + BUY spot = earn {ann:.1f}%/yr")
                w(f"    Per $10k: earn ${abs(fr)*3*10000:.2f}/day = ${abs(fr)*3*10000*30:.2f}/month")
            else:
                w(f"    Strategy: LONG perp + SHORT/SELL spot = earn {abs(ann):.1f}%/yr")
                w(f"    Per $10k: earn ${abs(fr)*3*10000:.2f}/day = ${abs(fr)*3*10000*30:.2f}/month")

    w("\n" + "=" * 130)
    w("SUMMARY")
    w("=" * 130)
    big = len([a for a in results if abs(a["best_edge"]) > 0.05])
    med = len([a for a in results if 0.02 < abs(a["best_edge"]) <= 0.05])
    sm = len([a for a in results if 0.01 < abs(a["best_edge"]) <= 0.02])
    tiny = len([a for a in results if abs(a["best_edge"]) <= 0.01])
    w(f"  Big edge (>5%):    {big} opportunities")
    w(f"  Med edge (2-5%):   {med} opportunities")
    w(f"  Small edge (1-2%): {sm} opportunities")
    w(f"  Tiny edge (<1%):   {tiny} opportunities")
    w("")
    w(f"  Expiring <24h:     {len([a for a in results if a['days'] < 1])}")
    w(f"  Expiring 1-3 days: {len([a for a in results if 1 <= a['days'] <= 3])}")
    w(f"  Weekly:            {len(weekly)}")
    w(f"  Monthly:           {len(monthly)}")
    w("")
    w(f"  BTC opportunities: {len([a for a in results if a['coin'] == 'BTC'])}")
    w(f"  ETH opportunities: {len([a for a in results if a['coin'] == 'ETH'])}")
    w(f"  SOL opportunities: {len([a for a in results if a['coin'] == 'SOL'])}")
    w(f"  XRP opportunities: {len([a for a in results if a['coin'] == 'XRP'])}")
    w(f"  DOGE opportunities: {len([a for a in results if a['coin'] == 'DOGE'])}")

    return "\n".join(L)


def main():
    now = datetime.now(timezone.utc)
    print("Building slug list...", flush=True)
    slugs = build_slugs()
    print(f"Fetching {len(slugs)} events from Polymarket...", flush=True)
    all_markets = fetch_markets(slugs)
    n_events = len(set(m["slug"] for m in all_markets))
    print(f"Got {len(all_markets)} markets from {n_events} events", flush=True)

    print("Fetching spot prices...", flush=True)
    spot = fetch_spots()
    print(f"Spots: {spot}", flush=True)

    print("Fetching futures data...", flush=True)
    futures = fetch_futures()
    print(f"Futures: {futures}", flush=True)

    print("Analyzing...", flush=True)
    results = analyze(all_markets, spot, now)
    print(f"Found {len(results)} opportunities", flush=True)

    report = format_report(results, spot, futures, now, n_events, len(all_markets))

    out_path = "short_term_arb_results.txt"
    if len(sys.argv) > 1:
        out_path = sys.argv[1]
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\nReport written to {out_path}", flush=True)
    print(report)


if __name__ == "__main__":
    main()
