"""Turn one day's indicators plus your position into BUY / SELL / HOLD actions."""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

CHECKS = {  # condition name -> (indicator column, comparison, wording)
    "fng_max": ("fng", "<=", "Fear & Greed {v:.0f} is at or below {t}"),
    "fng_min": ("fng", ">=", "Fear & Greed {v:.0f} is at or above {t}"),
    "rsi_max": ("rsi14", "<=", "RSI {v:.0f} is at or below {t}"),
    "rsi_min": ("rsi14", ">=", "RSI {v:.0f} is at or above {t}"),
    "pct_vs_ma200_max": ("pct_vs_ma200", "<=", "price is {v:+.0f}% vs 200-day average (limit {t}%)"),
    "pct_vs_ma200_min": ("pct_vs_ma200", ">=", "price is {v:+.0f}% vs 200-day average (threshold {t}%)"),
}


def load_rules(path=None):
    return yaml.safe_load(open(path or Path(__file__).parent / "rules.yaml"))


@dataclass
class State:
    """Your position in one coin. Money and avg_cost are in the same currency as `price`."""
    reserve: float = 0.0           # deposited money not yet spent, plus sale proceeds kept in the plan
    taken_out: float = 0.0         # profit taken out of the plan on big wins
    units: float = 0.0
    cost: float = 0.0              # money paid for the units still held
    peak_units: float = 0.0
    monthly_done: bool = False     # this month's regular buy already made or skipped
    days_since_dip: int | None = None
    days_since_sell: int | None = None


def match(rule, row):
    """Reasons if every condition in the rule holds, else None. Missing data never matches."""
    reasons = []
    for key, threshold in rule.items():
        if key not in CHECKS:
            continue
        col, op, text = CHECKS[key]
        v = row[col]
        if pd.isna(v) or not (v <= threshold if op == "<=" else v >= threshold):
            return None
        reasons.append(text.format(v=v, t=threshold))
    return reasons or None


def first_match(rules, row):
    for rule in rules:
        reasons = match(rule, row)
        if reasons:
            return rule, reasons
    return None, None


def decide(row, st, coin_cfg, cfg, price=None):
    """List of actions for today. Each is a dict: action (BUY/SELL/HOLD), kind, amount, reasons.
    `price` is the coin's price in your currency (defaults to the close)."""
    price = row["close"] if price is None else price
    actions = []
    high, high_why = first_match(cfg["high_rules"], row)
    low, low_why = first_match(cfg["low_rules"], row)

    if not st.monthly_done:
        if high:
            actions.append({"action": "HOLD", "kind": "monthly-skip", "amount": 0,
                            "reasons": ["monthly buy skipped, market is high: " + "; ".join(high_why)]})
        else:
            want = coin_cfg["low_buy"] if low else coin_cfg["normal_buy"]
            why = ("monthly buy, larger because market is low: " + "; ".join(low_why)) if low \
                else "regular monthly buy (market is neither high nor low)"
            if min(want, st.reserve) > 0:
                actions.append({"action": "BUY", "kind": "monthly", "amount": min(want, st.reserve), "reasons": [why]})

    dip = cfg["dip_buy"]
    rule, why = first_match(dip["rules"], row)
    if rule and st.reserve > 0:
        if st.days_since_dip is not None and st.days_since_dip < dip["cooldown_days"]:
            actions.append({"action": "HOLD", "kind": "dip-cooldown", "amount": 0,
                            "reasons": [f"dip rule '{rule['name']}' is true but the last dip buy was "
                                        f"{st.days_since_dip} days ago (cooldown {dip['cooldown_days']} days)"]})
        else:
            spent = sum(a["amount"] for a in actions if a["action"] == "BUY")
            amount = (st.reserve - spent) * dip["reserve_pct"] / 100
            if amount >= cfg.get("min_trade", 0) and amount > 0:
                actions.append({"action": "BUY", "kind": "dip", "amount": amount,
                                "reasons": [f"extra dip buy, {dip['reserve_pct']}% of reserve: " + "; ".join(why)]})

    sell = cfg["sell"]
    rule, why = first_match(sell["rules"], row)
    sellable = max(st.units - st.peak_units * sell["core_hold_pct"] / 100, 0)
    if rule and sellable > 0:
        avg = st.cost / st.units
        gain = (price / avg - 1) * 100
        if gain < sell["min_profit_pct"]:
            actions.append({"action": "HOLD", "kind": "sell-no-profit", "amount": 0,
                            "reasons": [f"sell rule '{rule['name']}' is true but you are {gain:+.0f}% vs your "
                                        f"average cost (needs +{sell['min_profit_pct']}%)"]})
        elif st.days_since_sell is not None and st.days_since_sell < sell["cooldown_days"]:
            actions.append({"action": "HOLD", "kind": "sell-cooldown", "amount": 0,
                            "reasons": [f"sell rule '{rule['name']}' is true but the last sell was "
                                        f"{st.days_since_sell} days ago (cooldown {sell['cooldown_days']} days)"]})
        else:
            units = sellable * rule["tranche_pct"] / 100
            big = gain >= sell.get("take_out_min_gain_pct", float("inf"))
            take_out = units * (price - avg) if big else 0.0
            where = (f"big win: take out the {take_out:.2f} profit, put the rest back in the reserve" if big
                     else "smaller win: all of it goes back into the reserve")
            actions.append({"action": "SELL", "kind": "sell", "units": units, "amount": units * price,
                            "take_out": take_out,
                            "reasons": [f"sell {rule['tranche_pct']}% of sellable position, you are {gain:+.0f}% "
                                        f"vs average cost ({where}): " + "; ".join(why)]})
    if not actions:
        actions.append({"action": "HOLD", "kind": "none", "amount": 0,
                        "reasons": ["this month's buy is done and no dip or sell rule is true"]})
    return actions


def take_out_on_sale(units, sale_amount, avg_cost, price, cfg):
    """Part of a sale's proceeds taken out as profit (big wins only)."""
    gain = (price / avg_cost - 1) * 100 if avg_cost else 0
    if gain < cfg["sell"].get("take_out_min_gain_pct", float("inf")):
        return 0.0
    return max(sale_amount - units * avg_cost, 0.0)
