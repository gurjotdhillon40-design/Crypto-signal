"""Altcoin tier: weekly status for each altcoin, plus a backtest of the sell rules.

    python alts.py              # status table: SELL / WATCH / HOLD / WEAK / DEAD / TOO NEW
    python alts.py --backtest   # selling into strength vs just holding, per coin

No buy signals. Alert only.
"""
import argparse

import pandas as pd
import requests

from data import fetch_prices
from indicators import build_table
from strategy import first_match, load_rules


def alt_table(coin, refresh=True):
    df = build_table(fetch_prices(coin, refresh))
    usd_volume = df["close"] * df["volume"]
    df["gain_7d_pct"] = (df["close"] / df["close"].shift(7) - 1) * 100
    df["volume_ratio"] = usd_volume.rolling(7).mean() / usd_volume.rolling(90).mean()
    df["volume_usd_30d"] = usd_volume.rolling(30).mean()
    return df


def status(df, a, lookback=7):
    """(status, reason) for the latest day. A sell rule hit in the last `lookback` days counts."""
    row = df.iloc[-1]
    if row["volume_usd_30d"] < a["dead_volume_usd_per_day"]:
        return "DEAD", f"only ${row['volume_usd_30d']:,.0f} traded a day over the last 30 days"
    if pd.isna(row["ma200"]):
        return "TOO NEW", "under 200 days of price history"
    for date, r in df.iloc[-lookback:][::-1].iterrows():
        rule, why = first_match(a["sell_rules"], r)
        if rule:
            return "SELL", f"sell {rule['tranche_pct']}% ({rule['name']}, {date.date()}): " + "; ".join(why)
    w = a["watch"]
    if row["gain_7d_pct"] >= w["gain_7d_pct_min"] and row["volume_ratio"] >= w["volume_ratio_min"]:
        return "WATCH", (f"up {row['gain_7d_pct']:.0f}% in 7 days on {row['volume_ratio']:.1f}x normal volume; "
                         "sell rule not reached yet")
    if row["pct_vs_ma200"] < 0:
        return "WEAK", f"{row['pct_vs_ma200']:.0f}% below its 200-day average"
    return "HOLD", f"{row['pct_vs_ma200']:+.0f}% above its 200-day average, no sell rule reached"


def scan(refresh=True):
    """Status rows for every altcoin in rules.yaml."""
    a = load_rules()["alts"]
    rows = []
    for coin in a["coins"]:
        try:
            df = alt_table(coin, refresh)
        except requests.HTTPError:
            rows.append({"coin": coin, "status": "DEAD", "reason": "no longer listed on Coinbase"})
            continue
        s, reason = status(df, a)
        r = df.iloc[-1]
        rows.append({"coin": coin, "status": s, "date": r.name.date(), "price_usd": r["close"],
                     "rsi14": round(r["rsi14"]), "pct_vs_ma200": None if pd.isna(r["ma200"]) else round(r["pct_vs_ma200"]),
                     "gain_7d_pct": round(r["gain_7d_pct"]), "from_high_pct": round((r["close"] / df["high"].max() - 1) * 100),
                     "reason": reason})
    order = ["SELL", "WATCH", "HOLD", "WEAK", "TOO NEW", "DEAD"]
    return sorted(rows, key=lambda r: order.index(r["status"]))


def backtest_coin(df, a, fee_pct):
    """Start with 1000 of the coin. Sell tranches of what is left when a sell rule fires;
    fill at the next day's open. Returns (final value, number of sells, final value if held)."""
    df = df.loc[df["ma200"].dropna().index[0]:]
    fee = fee_pct / 100
    units = start_units = 1000 * (1 - fee) / df["open"].iloc[0]
    cash, sells, last, pending = 0.0, 0, None, None
    for i, (_, row) in enumerate(df.iterrows()):
        if pending:
            sold = units * pending / 100
            units -= sold
            cash += sold * row["open"] * (1 - fee)
            sells += 1
            pending = None
        rule, _ = first_match(a["sell_rules"], row)
        if rule and (last is None or i - last >= a["sell_cooldown_days"]):
            last, pending = i, rule["tranche_pct"]
    price = df["close"].iloc[-1]
    return cash + units * price, sells, start_units * price, df.index[0].date()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--no-refresh", action="store_true")
    args = ap.parse_args()
    cfg = load_rules()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    if not args.backtest:
        print(pd.DataFrame(scan(not args.no_refresh)).to_string(index=False))
        return
    rows = []
    for coin in cfg["alts"]["coins"]:
        try:
            df = alt_table(coin, not args.no_refresh)
        except requests.HTTPError:
            continue
        if df["ma200"].notna().sum() < 30:
            continue
        final, sells, held, start = backtest_coin(df, cfg["alts"], cfg["fee_pct"])
        rows.append({"coin": coin, "from": start, "sells": sells, "sell_rules_final": round(final),
                     "just_hold_final": round(held)})
    t = pd.DataFrame(rows)
    print("Value today of 1000 put into each coin (from its first day with a 200-day average):")
    print(t.to_string(index=False))
    print(f"\nTotal: sell rules {t['sell_rules_final'].sum():,} vs just holding {t['just_hold_final'].sum():,} "
          f"(from {1000 * len(t):,} put in)")


if __name__ == "__main__":
    main()
