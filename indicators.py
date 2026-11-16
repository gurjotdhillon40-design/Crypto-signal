"""Indicators. Every value on a row uses only that day's close and earlier."""
import pandas as pd


def rsi(close, period=14):
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def build_table(prices, fng=None):
    """Prices plus ma200, rsi14, pct_vs_ma200 and fng, indexed by date."""
    df = prices.set_index("date").sort_index().copy()
    df["ma200"] = df["close"].rolling(200).mean()
    df["rsi14"] = rsi(df["close"])
    df["pct_vs_ma200"] = (df["close"] / df["ma200"] - 1) * 100
    if fng is not None:
        # the index has a few missing days; carry the last value forward at most 3 days
        df["fng"] = fng.set_index("date")["fng"].reindex(df.index).ffill(limit=3)
    else:
        df["fng"] = float("nan")
    return df
