"""Daily run: print BUY / HOLD / SELL with reasons and log it to journal.csv.

    python daily.py                                        # today's signals
    python daily.py trade SOL buy 100 --units 0.95         # record a buy: money spent, coins received
    python daily.py trade SOL sell 150 --units 0.9 --notes "took profit"
    python daily.py skip SOL                               # record that you skipped this month's buy

Alert only. This script never places orders.
Your reserve, position and average cost are worked out from the trades you record here.
"""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data import _get, fetch_fng, fetch_prices
from indicators import build_table
from strategy import State, decide, load_rules

JOURNAL = Path(__file__).parent / "journal.csv"
FIELDS = ["logged_at", "type", "date", "coin", "action", "kind", "price_usd", "fng", "rsi14", "ma200",
          "pct_vs_ma200", "reasons", "suggested_amount", "amount", "units", "followed_signal", "notes"]


def read_journal():
    if not JOURNAL.exists():
        return pd.DataFrame(columns=FIELDS)
    return pd.read_csv(JOURNAL, dtype=str).fillna("")


def append(row):
    new = not JOURNAL.exists()
    with open(JOURNAL, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({"logged_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"), **row})


def usd_to_local(currency):
    if currency == "USD":
        return 1.0
    return float(_get(f"https://api.exchange.coinbase.com/products/USDC-{currency}/ticker")["price"])


def state_from_journal(journal, coin, coin_cfg, cfg, today):
    """Rebuild reserve, position and average cost from recorded trades."""
    start = pd.Period(str(cfg["start_month"]), "M")
    months = max((pd.Period(today, "M") - start).n + 1, 0)
    st = State(reserve=months * coin_cfg["monthly_deposit"])
    trades = journal[(journal["type"] == "TRADE") & (journal["coin"] == coin)]
    for _, t in trades.iterrows():
        amount, units = float(t["amount"] or 0), float(t["units"] or 0)
        date = pd.Timestamp(t["date"])
        if t["action"] == "BUY":
            st.reserve -= amount
            st.units += units
            st.cost += amount
            if t["kind"] == "dip":
                st.days_since_dip = (today - date).days
        elif t["action"] == "SELL":
            st.cost -= st.cost * min(units / st.units, 1) if st.units else 0
            st.units = max(st.units - units, 0)
            st.reserve += amount
            st.days_since_sell = (today - date).days
        st.peak_units = max(st.peak_units, st.units)
        if t["kind"] in ("monthly", "monthly-skip") and date.to_period("M") == pd.Period(today, "M"):
            st.monthly_done = True
    return st


def signals():
    cfg = load_rules()
    cur = cfg["currency"]
    fx = usd_to_local(cur)
    fng = fetch_fng()
    journal = read_journal()
    for coin, coin_cfg in cfg["coins"].items():
        row = build_table(fetch_prices(coin), fng).iloc[-1]
        date = row.name
        today = pd.Timestamp(datetime.now(timezone.utc).date())
        st = state_from_journal(journal, coin, coin_cfg, cfg, today)
        actions = decide(row, st, coin_cfg, cfg, price=row["close"] * fx)
        headline = next((a["action"] for a in actions if a["action"] != "HOLD"), "HOLD")
        print(f"\n{coin}  {headline}   (close {date.date()}: ${row['close']:,.2f} USD)")
        print(f"   Fear & Greed {row['fng']:.0f} | RSI {row['rsi14']:.0f} | "
              f"200-day average ${row['ma200']:,.2f} ({row['pct_vs_ma200']:+.0f}%)")
        held = f"{st.units:.6g} {coin}, average cost {st.cost / st.units:,.2f} {cur}" if st.units else "nothing yet"
        print(f"   You hold: {held} | reserve {st.reserve:,.2f} {cur}")
        suggestions = []
        for a in actions:
            print(f"   - {a['reasons'][0]}")
            if a["action"] == "BUY":
                suggestions.append(f"buy {a['amount']:.2f} {cur} ({a['kind']})")
            elif a["action"] == "SELL":
                suggestions.append(f"sell {a['units']:.6g} {coin}, about {a['amount']:.2f} {cur}")
        for s in suggestions:
            print(f"   -> {s}")
        done = journal[(journal["type"] == "SIGNAL") & (journal["coin"] == coin)
                       & (journal["date"] == str(date.date()))]
        if done.empty:      # one signal row per coin per day, even if run twice
            append({"type": "SIGNAL", "date": date.date(), "coin": coin, "action": headline,
                    "kind": "+".join(a["kind"] for a in actions),
                    "price_usd": round(row["close"], 2), "fng": row["fng"], "rsi14": round(row["rsi14"], 1),
                    "ma200": round(row["ma200"], 2), "pct_vs_ma200": round(row["pct_vs_ma200"], 1),
                    "reasons": " | ".join(a["reasons"][0] for a in actions),
                    "suggested_amount": "; ".join(suggestions)})
    print(f"\nLogged to {JOURNAL}. Alert only: no orders were placed.")


def trade(args):
    journal = read_journal()
    coin, action = args.coin.upper(), args.side.upper()
    today = datetime.now(timezone.utc).date()
    sig = journal[(journal["type"] == "SIGNAL") & (journal["coin"] == coin)]
    last = sig.iloc[-1] if not sig.empty else None
    last_action = last["action"] if last is not None else ""
    kind = args.kind
    if not kind:
        if action == "SELL":
            kind = "sell"
        else:   # first buy of the month counts as the monthly buy, later ones as dip buys
            month = journal[(journal["type"] == "TRADE") & (journal["coin"] == coin)
                            & journal["kind"].isin(["monthly", "monthly-skip"])
                            & (journal["date"].str[:7] == str(today)[:7])]
            kind = "monthly" if month.empty else "dip"
    append({"type": "TRADE", "date": today, "coin": coin, "action": action, "kind": kind,
            "amount": args.amount, "units": args.units,
            "followed_signal": "yes" if last_action == action else "no", "notes": args.notes})
    print(f"Logged {action} of {coin} ({kind}): {args.amount} for {args.units} {coin}. Latest signal was "
          f"{last_action or 'none'}: {'followed' if last_action == action else 'NOT following the system'}.")


def skip(args):
    append({"type": "TRADE", "date": datetime.now(timezone.utc).date(), "coin": args.coin.upper(),
            "action": "SKIP", "kind": "monthly-skip", "notes": args.notes})
    print(f"Logged: skipped this month's {args.coin.upper()} buy.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    t = sub.add_parser("trade")
    t.add_argument("coin")
    t.add_argument("side", choices=["buy", "sell"])
    t.add_argument("amount", type=float, help="money spent (buy) or received (sell), in your currency")
    t.add_argument("--units", type=float, required=True, help="coins received (buy) or sold (sell)")
    t.add_argument("--kind", choices=["monthly", "dip", "sell"])
    t.add_argument("--notes", default="")
    s = sub.add_parser("skip")
    s.add_argument("coin")
    s.add_argument("--notes", default="")
    a = ap.parse_args()
    {"trade": trade, "skip": skip}.get(a.cmd, lambda _: signals())(a)
