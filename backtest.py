"""Backtest the monthly plan in rules.yaml against buying the full deposit every month.

    python backtest.py                    # from when Fear & Greed data exists (2018)
    python backtest.py --start 2023-01-01 # any sub-period

Both approaches receive the same deposit on the first day of each month.
No lookahead: a signal from day D's close is filled at day D+1's open, minus fees.
Amounts are treated as being in the price currency (USD); exchange-rate moves are ignored.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data import fetch_fng, fetch_prices
from indicators import build_table
from strategy import State, decide, load_rules

OUT = Path(__file__).parent / "output"


def run_plan(df, coin_cfg, cfg):
    """Returns a daily table (value, deposited, reserve) and the list of trades."""
    fee = cfg["fee_pct"] / 100
    st = State()
    deposited, month, pending = 0.0, None, []
    last_dip = last_sell = None
    days, trades = [], []
    for i, (date, row) in enumerate(df.iterrows()):
        if (date.year, date.month) != month:
            month = (date.year, date.month)
            st.reserve += coin_cfg["monthly_deposit"]
            deposited += coin_cfg["monthly_deposit"]
            st.monthly_done = False
        for a in pending:                       # fill yesterday's signals at today's open
            if a["action"] == "BUY":
                amount = min(a["amount"], st.reserve)
                st.units += amount * (1 - fee) / row["open"]
                st.reserve -= amount
                st.cost += amount
            else:
                units = min(a["units"], st.units)
                amount = units * row["open"] * (1 - fee)
                st.cost -= st.cost * units / st.units
                st.units -= units
                st.reserve += amount
            st.peak_units = max(st.peak_units, st.units)
            trades.append({"date": date, "action": a["action"], "kind": a["kind"],
                           "price": row["open"], "amount": round(amount, 2)})
        st.days_since_dip = None if last_dip is None else i - last_dip
        st.days_since_sell = None if last_sell is None else i - last_sell
        actions = decide(row, st, coin_cfg, cfg)
        if any(a["kind"] in ("monthly", "monthly-skip") for a in actions):
            st.monthly_done = True
        if any(a["kind"] == "dip" for a in actions):
            last_dip = i
        if any(a["kind"] == "sell" for a in actions):
            last_sell = i
        pending = [a for a in actions if a["action"] != "HOLD"]
        days.append({"date": date, "value": st.reserve + st.units * row["close"],
                     "deposited": deposited, "reserve": st.reserve})
    return pd.DataFrame(days).set_index("date"), trades


def run_buy_everything(df, coin_cfg, cfg):
    """Benchmark: spend the whole deposit on the first day of every month, never sell."""
    fee = cfg["fee_pct"] / 100
    units, deposited, month, days = 0.0, 0.0, None, []
    for date, row in df.iterrows():
        if (date.year, date.month) != month:
            month = (date.year, date.month)
            units += coin_cfg["monthly_deposit"] * (1 - fee) / row["open"]
            deposited += coin_cfg["monthly_deposit"]
        days.append({"date": date, "value": units * row["close"], "deposited": deposited, "reserve": 0.0})
    return pd.DataFrame(days).set_index("date")


def stats(t):
    ratio = t["value"] / t["deposited"]
    return {"deposited": t["deposited"].iloc[-1], "final_value": t["value"].iloc[-1],
            "profit_pct": (ratio.iloc[-1] - 1) * 100,
            "worst_vs_deposited_pct": (ratio.min() - 1) * 100,
            "best_vs_deposited_pct": (ratio.max() - 1) * 100,
            "days_underwater_pct": (ratio < 1).mean() * 100,
            "cash_at_end": t["reserve"].iloc[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--no-refresh", action="store_true")
    ap.add_argument("--rules")
    args = ap.parse_args()

    cfg = load_rules(args.rules)
    fng = fetch_fng(not args.no_refresh)
    label = "plan" + (f"_{args.start or 'start'}_{args.end or 'now'}" if args.start or args.end else "")
    OUT.mkdir(exist_ok=True)

    rows, all_trades = [], []
    fig, axes = plt.subplots(len(cfg["coins"]), 1, figsize=(11, 4 * len(cfg["coins"])))
    for ax, (coin, coin_cfg) in zip(axes, cfg["coins"].items()):
        df = build_table(fetch_prices(coin, not args.no_refresh), fng)
        df = df.loc[df[["ma200", "rsi14", "fng"]].dropna().index[0]:].loc[args.start:args.end]
        plan, trades = run_plan(df, coin_cfg, cfg)
        bench = run_buy_everything(df, coin_cfg, cfg)
        all_trades += [{"coin": coin, **t} for t in trades]
        for name, t in (("Plan (rules)", plan), ("Buy full deposit monthly", bench)):
            rows.append({"coin": coin, "approach": name, "from": df.index[0].date(), "to": df.index[-1].date(),
                         **{k: round(v, 1) for k, v in stats(t).items()},
                         "buys": sum(x["action"] == "BUY" for x in trades) if t is plan else "",
                         "sells": sum(x["action"] == "SELL" for x in trades) if t is plan else ""})
        ax.plot(plan.index, plan["value"], label="Plan (rules)", linewidth=1.8)
        ax.plot(bench.index, bench["value"], label="Buy full deposit monthly", linewidth=1.1)
        ax.plot(plan.index, plan["deposited"], label="Money deposited", color="gray", linestyle="--", linewidth=1)
        for kind, marker, color, name in (("dip", "^", "green", "Dip buy"), ("sell", "v", "red", "Sell tranche")):
            d = [x["date"] for x in trades if x["kind"] == kind]
            ax.scatter(d, plan["value"].loc[d], marker=marker, color=color, s=24, zorder=3, label=name)
        ax.set_title(f"{coin}: depositing {coin_cfg['monthly_deposit']} a month")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"backtest_{label}.png", dpi=130)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / f"backtest_{label}.csv", index=False)
    pd.DataFrame(all_trades).to_csv(OUT / f"backtest_{label}_trades.csv", index=False)
    pd.set_option("display.width", 250)
    print(result.to_string(index=False))
    print(f"\nChart: {OUT / f'backtest_{label}.png'}")


if __name__ == "__main__":
    main()
