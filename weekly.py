"""Weekly report: core signals, what the rules said on each of the last 7 days, and altcoin status.

    python weekly.py

Alert only. Logs core signals and altcoin SELL / WATCH / DEAD statuses to journal.csv.
"""
import pandas as pd

import daily
from alts import scan
from data import fetch_fng, fetch_prices
from indicators import build_table
from strategy import first_match, load_rules


def week_events(cfg):
    """Days in the last week when a dip-buy or sell rule was true for a core coin."""
    fng = fetch_fng(False)
    lines = []
    for coin in cfg["coins"]:
        df = build_table(fetch_prices(coin, False), fng).iloc[-7:]
        for label, rules in (("dip-buy rule", cfg["dip_buy"]["rules"]), ("sell rule", cfg["sell"]["rules"])):
            days = [d for d, r in df.iterrows() if first_match(rules, r)[0]]
            if days:
                lines.append(f"   {coin}: {label} true on {len(days)} of the last 7 days "
                             f"({days[0].date()} to {days[-1].date()})")
    return lines or ["   no dip-buy or sell rule was true for BTC, ETH or SOL"]


def main():
    cfg = load_rules()
    print("=== BTC / ETH / SOL: today's signal ===")
    daily.signals()
    print("\n=== Last 7 days ===")
    print("\n".join(week_events(cfg)))
    print("\n=== Altcoins ===")
    rows = scan()
    journal = daily.read_journal()
    for r in rows:
        extra = "" if "price_usd" not in r else f"  ${r['price_usd']:.6g}, RSI {r['rsi14']}, 7 days {r['gain_7d_pct']:+d}%"
        print(f"   {r['status']:8}{r['coin']:10}{r['reason']}{extra}")
        logged = journal[(journal["type"] == "SIGNAL") & (journal["coin"] == r["coin"])
                         & (journal["date"] == str(r.get("date", "")))]
        if r["status"] in ("SELL", "WATCH", "DEAD") and logged.empty:
            daily.append({"type": "SIGNAL", "date": r.get("date", ""), "coin": r["coin"], "action": r["status"],
                          "kind": "alt", "price_usd": r.get("price_usd", ""), "rsi14": r.get("rsi14", ""),
                          "pct_vs_ma200": r.get("pct_vs_ma200", ""), "reasons": r["reason"]})
    counts = pd.Series([r["status"] for r in rows]).value_counts()
    print("\n   " + ", ".join(f"{n} {s}" for s, n in counts.items()))
    print("\nAlert only: no orders were placed.")


if __name__ == "__main__":
    main()
