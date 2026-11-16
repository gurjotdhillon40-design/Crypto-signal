import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest import run_buy_everything, run_plan
from indicators import build_table, rsi
from strategy import State, decide

COIN = {"monthly_deposit": 100, "normal_buy": 60, "low_buy": 90}
CFG = {"fee_pct": 0.6,
       "high_rules": [{"name": "greed", "fng_min": 75}],
       "low_rules": [{"name": "fear", "fng_max": 25}],
       "dip_buy": {"rules": [{"name": "extreme fear", "fng_max": 15}], "reserve_pct": 25, "cooldown_days": 14},
       "sell": {"rules": [{"name": "greed", "fng_min": 75, "pct_vs_ma200_min": 40, "tranche_pct": 15}],
                "min_profit_pct": 20, "core_hold_pct": 25, "cooldown_days": 7}}


def prices(n=400, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    d = pd.date_range("2020-01-01", periods=n)
    return pd.DataFrame({"date": d, "open": close * 0.99, "high": close, "low": close, "close": close, "volume": 1})


def row(fng=50, pct=0, close=100):
    return pd.Series({"fng": fng, "rsi14": 50, "pct_vs_ma200": pct, "close": close})


def kinds(actions):
    return [a["kind"] for a in actions]


def test_rsi_bounds_and_extremes():
    assert rsi(pd.Series(np.arange(1, 60, dtype=float))).iloc[-1] == 100
    assert rsi(prices()["close"]).dropna().between(0, 100).all()


def test_indicators_do_not_look_ahead():
    p = prices()
    pd.testing.assert_frame_equal(build_table(p).iloc[:300], build_table(p.iloc[:300]))


def test_monthly_buy_sizes_by_zone():
    st = State(reserve=100)
    assert decide(row(50), st, COIN, CFG)[0]["amount"] == 60
    assert decide(row(20), st, COIN, CFG)[0]["amount"] == 90
    assert kinds(decide(row(80), st, COIN, CFG)) == ["monthly-skip"]


def test_missing_data_gives_only_the_normal_monthly_buy():
    a = decide(row(np.nan), State(reserve=100), COIN, CFG)
    assert kinds(a) == ["monthly"] and a[0]["amount"] == 60


def test_dip_buy_uses_reserve_left_after_monthly_and_respects_cooldown():
    a = decide(row(10), State(reserve=490), COIN, CFG)
    assert kinds(a) == ["monthly", "dip"] and a[1]["amount"] == (490 - 90) * 0.25
    a = decide(row(10), State(reserve=400, monthly_done=True, days_since_dip=3), COIN, CFG)
    assert kinds(a) == ["dip-cooldown"]


def test_never_sells_at_a_loss_or_below_core():
    st = State(units=10, cost=1000, peak_units=10, monthly_done=True)          # average cost 100
    assert kinds(decide(row(90, 60, close=110), st, COIN, CFG)) == ["sell-no-profit"]
    a = decide(row(90, 60, close=150), st, COIN, CFG)
    assert kinds(a) == ["sell"] and abs(a[0]["units"] - (10 - 2.5) * 0.15) < 1e-9
    st = State(units=2.5, cost=250, peak_units=10, monthly_done=True)          # only the core is left
    assert kinds(decide(row(90, 60, close=150), st, COIN, CFG)) == ["none"]


def test_backtest_fills_next_day_open_and_tracks_deposits():
    df = build_table(prices()).iloc[200:290].copy()
    df["fng"] = 50
    plan, trades = run_plan(df, COIN, CFG)
    months = df.index.to_period("M").nunique()
    assert plan["deposited"].iloc[-1] == 100 * months
    assert trades[0]["date"] == df.index[1] and trades[0]["price"] == df["open"].iloc[1]
    assert abs(plan["reserve"].iloc[-1] - 40 * months) < 1e-9          # 60 of each 100 is spent
    assert all(t["action"] == "BUY" for t in trades)


def test_benchmark_spends_everything():
    df = build_table(prices()).iloc[200:290]
    b = run_buy_everything(df, COIN, CFG)
    assert b["reserve"].eq(0).all() and b["deposited"].iloc[-1] == 100 * df.index.to_period("M").nunique()
