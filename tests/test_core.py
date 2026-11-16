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


# --- altcoin tier ---
from alts import backtest_coin, status

ALTS = {"sell_rules": [{"name": "run", "pct_vs_ma200_min": 50, "rsi_min": 70, "tranche_pct": 20}],
        "sell_cooldown_days": 10, "watch": {"gain_7d_pct_min": 30, "volume_ratio_min": 2},
        "dead_volume_usd_per_day": 50000}


def alt_df(n=30, **last):
    d = pd.DataFrame({"open": 1.0, "close": 1.0, "ma200": 1.0, "rsi14": 50.0, "pct_vs_ma200": 10.0, "fng": np.nan,
                      "gain_7d_pct": 0.0, "volume_ratio": 1.0, "volume_usd_30d": 1e6},
                     index=pd.date_range("2024-01-01", periods=n))
    for k, v in last.items():
        d.iloc[-1, d.columns.get_loc(k)] = v
    return d


def test_alt_statuses():
    assert status(alt_df(), ALTS)[0] == "HOLD"
    assert status(alt_df(pct_vs_ma200=-20), ALTS)[0] == "WEAK"
    assert status(alt_df(pct_vs_ma200=80, rsi14=80), ALTS)[0] == "SELL"
    assert status(alt_df(pct_vs_ma200=80, rsi14=60), ALTS)[0] == "HOLD"       # both conditions needed
    assert status(alt_df(gain_7d_pct=40, volume_ratio=3), ALTS)[0] == "WATCH"
    assert status(alt_df(volume_usd_30d=1000, pct_vs_ma200=80, rsi14=80), ALTS)[0] == "DEAD"
    assert status(alt_df(ma200=np.nan), ALTS)[0] == "TOO NEW"


def test_alt_sell_seen_for_a_week_then_forgotten():
    d = alt_df(30)
    d.iloc[-5, [d.columns.get_loc("pct_vs_ma200"), d.columns.get_loc("rsi14")]] = [80, 80]
    assert status(d, ALTS)[0] == "SELL"
    assert status(d, ALTS, lookback=3)[0] == "HOLD"


def test_alt_backtest_sells_next_open_with_cooldown():
    d = alt_df(40)
    d["pct_vs_ma200"], d["rsi14"] = 80.0, 80.0          # sell rule true every day
    d["open"] = d["close"] = 2.0
    final, sells, held, _ = backtest_coin(d, ALTS, 0)
    assert sells == 4                                    # signals on days 0, 10, 20, 30
    assert abs(final - held) < 1e-9                      # flat price, no fee: selling changes nothing
