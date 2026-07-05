"""
momentum.py — Price-momentum-based alpha factors.

References
----------
- Jegadeesh & Titman (1993), "Returns to Buying Winners and Selling Losers:
  Implications for Stock Market Efficiency", Journal of Finance.
- Asness, Moskowitz & Pedersen (2013), "Value and Momentum Everywhere".
"""
from __future__ import annotations

import pandas as pd


def compute_12_1_momentum(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Classic 12-1 month momentum factor: the trailing 12-month return,
    excluding the most recent month (to avoid the short-term reversal
    effect). Positive values indicate stronger past performance.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide frame of adjusted close prices, index=date, columns=tickers.

    Returns
    -------
    pd.DataFrame
        Same shape as `prices`; each cell is the 12-1 momentum score as of
        that date.
    """
    return prices.pct_change(252).shift(21)


def compute_1m_reversal(prices: pd.DataFrame) -> pd.DataFrame:
    """
    1-month reversal factor — a contrarian signal. Stocks that fell sharply
    over the past month tend to bounce, and vice versa, so the raw 1-month
    return is negated.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide frame of adjusted close prices, index=date, columns=tickers.

    Returns
    -------
    pd.DataFrame
        Negated 1-month (21 trading day) return.
    """
    return -prices.pct_change(21)


def compute_3m_momentum(prices: pd.DataFrame) -> pd.DataFrame:
    """Medium-term (63 trading day / ~3 month) momentum, no skip-month."""
    return prices.pct_change(63)


def compute_52w_high_proximity(prices: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """
    Proximity to the 52-week high: price / rolling_max(price, 252d).
    Based on George & Hwang (2004) "The 52-Week High and Momentum Investing".
    Values close to 1.0 indicate the stock is trading near its 52-week high.
    """
    rolling_max = prices.rolling(window=window, min_periods=window // 2).max()
    return prices / rolling_max
