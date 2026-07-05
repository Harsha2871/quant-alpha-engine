"""
factor_decay.py — Studies how a factor's predictive power (IC) decays as the
forward-return horizon lengthens. Fast-decaying factors (e.g. short-term
reversal) need frequent rebalancing; slow-decaying factors (e.g. value) can
be held longer, which matters directly for transaction-cost-aware portfolio
construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_engine.research.ic_analysis import compute_forward_returns, compute_rank_ic


def compute_ic_decay(
    factor: pd.DataFrame,
    prices: pd.DataFrame,
    max_horizon: int = 42,
    step: int = 1,
) -> pd.Series:
    """
    Computes mean rank IC for forward-return horizons 1..max_horizon (in
    steps of `step`), tracing out the factor's "decay curve".

    Parameters
    ----------
    factor : pd.DataFrame
        Factor scores, index=date, columns=tickers.
    prices : pd.DataFrame
        Price frame used to compute forward returns.
    max_horizon : int
        Longest forward horizon (trading days) to evaluate.
    step : int
        Step size between evaluated horizons.

    Returns
    -------
    pd.Series
        Index = horizon (trading days), values = mean rank IC at that horizon.
    """
    horizons = range(1, max_horizon + 1, step)
    decay = {}
    for h in horizons:
        fwd = compute_forward_returns(prices, h)
        ic_series = compute_rank_ic(factor, fwd)
        decay[h] = ic_series.mean()
    return pd.Series(decay, name="mean_ic")


def half_life(decay_curve: pd.Series) -> float:
    """
    Estimates the "half-life" of a factor's predictive power: the horizon at
    which mean IC first drops to half (or less) of its horizon=1 value.

    Parameters
    ----------
    decay_curve : pd.Series
        Output of `compute_ic_decay`, index=horizon, values=mean IC.

    Returns
    -------
    float
        Horizon (in the same units as decay_curve.index) at which IC decays
        to half its initial magnitude, or NaN if it never does within the
        provided range.
    """
    clean = decay_curve.dropna()
    if clean.empty:
        return float("nan")

    initial = clean.iloc[0]
    if initial == 0 or np.isnan(initial):
        return float("nan")

    target = abs(initial) / 2.0
    for horizon, ic in clean.items():
        if abs(ic) <= target:
            return float(horizon)

    return float("nan")  # never decayed to half within the tested range


def decay_summary(factor: pd.DataFrame, prices: pd.DataFrame, max_horizon: int = 42) -> dict:
    """
    Convenience wrapper returning the decay curve plus its estimated
    half-life in a single dict, suitable for JSON serialization.
    """
    curve = compute_ic_decay(factor, prices, max_horizon=max_horizon)
    return {
        "decay_curve": {int(k): (float(v) if pd.notna(v) else None) for k, v in curve.items()},
        "half_life_days": half_life(curve),
    }
