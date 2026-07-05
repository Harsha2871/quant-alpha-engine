"""
ic_analysis.py — Information Coefficient (IC) research tools.

The Information Coefficient measures the cross-sectional (Spearman rank)
correlation between a factor's values at time T and the forward return over
the subsequent N periods. It is the standard measure of a factor's raw
predictive power in cross-sectional equity research (see Grinold & Kahn,
"Active Portfolio Management").

ICIR (Information Coefficient Information Ratio) = mean(IC) / std(IC),
capturing not just average predictive power but its *consistency* over time
— a factor with IC=0.03 every period is far more useful than one that
averages 0.03 by swinging between +0.15 and -0.09.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def compute_forward_returns(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Forward return over `horizon` trading days, aligned so that
    forward_returns.loc[t] = (price[t+horizon] / price[t]) - 1.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide price frame, index=date, columns=tickers.
    horizon : int
        Number of trading days to look forward.

    Returns
    -------
    pd.DataFrame
        Forward returns, same shape as `prices` (NaN in the last `horizon`
        rows since no future price exists yet).
    """
    return prices.shift(-horizon) / prices - 1.0


def compute_rank_ic(factor: pd.DataFrame, forward_returns: pd.DataFrame) -> pd.Series:
    """
    Cross-sectional Spearman rank IC at each date: the rank correlation
    between factor values and forward returns across the universe.

    Parameters
    ----------
    factor : pd.DataFrame
        Factor scores, index=date, columns=tickers.
    forward_returns : pd.DataFrame
        Forward returns, same shape/alignment as `factor`.

    Returns
    -------
    pd.Series
        Rank IC time series, index=date. NaN on dates with fewer than 3
        valid (factor, return) pairs.
    """
    common_dates = factor.index.intersection(forward_returns.index)
    common_cols = factor.columns.intersection(forward_returns.columns)
    factor = factor.loc[common_dates, common_cols]
    fwd = forward_returns.loc[common_dates, common_cols]

    ic_values = {}
    for date in common_dates:
        f_row = factor.loc[date]
        r_row = fwd.loc[date]
        valid = f_row.notna() & r_row.notna()
        if valid.sum() < 3:
            ic_values[date] = np.nan
            continue
        corr, _ = stats.spearmanr(f_row[valid], r_row[valid])
        ic_values[date] = corr

    return pd.Series(ic_values, name="rank_ic")


def compute_icir(ic_series: pd.Series) -> float:
    """
    ICIR = mean(IC) / std(IC). Measures the consistency of a factor's
    predictive power, analogous to a Sharpe ratio for the IC series.

    Parameters
    ----------
    ic_series : pd.Series
        Time series of IC values (e.g. from `compute_rank_ic`).

    Returns
    -------
    float
        ICIR, or NaN if the IC series has zero variance / too few points.
    """
    clean = ic_series.dropna()
    if len(clean) < 2 or clean.std(ddof=0) == 0:
        return float("nan")
    return float(clean.mean() / clean.std(ddof=0))


@dataclass
class ICSummary:
    """Summary statistics for a factor's IC at one forward-return horizon."""

    horizon: int
    mean_ic: float
    std_ic: float
    icir: float
    hit_rate: float  # fraction of periods with IC > 0
    n_periods: int


def summarize_ic(factor: pd.DataFrame, prices: pd.DataFrame, horizon: int) -> ICSummary:
    """
    Computes a full IC summary (mean IC, std, ICIR, hit rate) for a single
    factor at a single forward-return horizon.

    Parameters
    ----------
    factor : pd.DataFrame
        Factor scores, index=date, columns=tickers.
    prices : pd.DataFrame
        Price frame used to compute forward returns.
    horizon : int
        Forward-return horizon in trading days.

    Returns
    -------
    ICSummary
    """
    fwd_returns = compute_forward_returns(prices, horizon)
    ic_series = compute_rank_ic(factor, fwd_returns)
    clean = ic_series.dropna()

    if clean.empty:
        return ICSummary(horizon=horizon, mean_ic=float("nan"), std_ic=float("nan"),
                          icir=float("nan"), hit_rate=float("nan"), n_periods=0)

    return ICSummary(
        horizon=horizon,
        mean_ic=float(clean.mean()),
        std_ic=float(clean.std(ddof=0)),
        icir=compute_icir(clean),
        hit_rate=float((clean > 0).mean()),
        n_periods=int(len(clean)),
    )


def multi_horizon_ic(
    factor: pd.DataFrame,
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 10, 21),
) -> pd.DataFrame:
    """
    Computes IC summary statistics across multiple forward-return horizons
    (standard practice: 1-day, 1-week, 2-week, 1-month).

    Parameters
    ----------
    factor : pd.DataFrame
        Factor scores, index=date, columns=tickers.
    prices : pd.DataFrame
        Price frame used to compute forward returns.
    horizons : tuple[int, ...]
        Forward-return horizons in trading days.

    Returns
    -------
    pd.DataFrame
        One row per horizon with columns: mean_ic, std_ic, icir, hit_rate, n_periods.
    """
    rows = []
    for h in horizons:
        summary = summarize_ic(factor, prices, h)
        rows.append({
            "horizon": summary.horizon,
            "mean_ic": summary.mean_ic,
            "std_ic": summary.std_ic,
            "icir": summary.icir,
            "hit_rate": summary.hit_rate,
            "n_periods": summary.n_periods,
        })
    return pd.DataFrame(rows).set_index("horizon")
