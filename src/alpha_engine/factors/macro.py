"""
macro.py — Macro-conditioned alpha factors.

Combines individual-stock sensitivities with macro regime state (interest
rates, volatility) to produce cross-sectional signals. The idea follows the
"macro factor timing" literature: stocks' exposure to systematic macro
factors (rates, vol) can be used to tilt portfolios ahead of regime shifts.

References
----------
- Ang, Chen & Xing (2006), "Downside Risk".
- Fama & French (1989), "Business Conditions and Expected Returns on Stocks
  and Bonds" — macro state variables predict cross-sectional returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rate_sensitivity(
    prices: pd.DataFrame,
    fed_funds_rate: pd.Series,
    window: int = 63,
) -> pd.DataFrame:
    """
    Rolling beta of each stock's daily returns to changes in the fed funds
    rate — a measure of interest-rate sensitivity. Negative beta stocks
    (rate-sensitive, e.g. growth/duration-like equities) can be shorted when
    rates are rising and vice versa.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide adjusted-close price frame, index=date, columns=tickers.
    fed_funds_rate : pd.Series
        Monthly (or any frequency) fed funds rate series, will be
        forward-filled and aligned to `prices.index`.
    window : int
        Rolling window (trading days) for the beta calculation.

    Returns
    -------
    pd.DataFrame
        Rolling rate-sensitivity beta per ticker, same shape as `prices`.
    """
    rate_daily = fed_funds_rate.reindex(prices.index, method="ffill")
    rate_changes = rate_daily.diff()
    stock_returns = prices.pct_change()

    betas = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    rate_var = rate_changes.rolling(window).var()

    for ticker in prices.columns:
        cov = stock_returns[ticker].rolling(window).cov(rate_changes)
        betas[ticker] = cov / rate_var.replace(0, np.nan)

    return betas


def compute_vix_regime_factor(
    prices: pd.DataFrame,
    vix: pd.Series,
    high_vix_threshold: float = 25.0,
    low_vix_threshold: float = 15.0,
) -> pd.DataFrame:
    """
    VIX regime factor: tilts toward low-beta / defensive behavior (measured
    via trailing realized volatility) when VIX signals a high-fear regime,
    and toward high-beta / cyclical behavior in low-VIX, risk-on regimes.

    The factor score for a stock is its NEGATIVE trailing realized volatility
    in high-VIX regimes (rewarding low-vol / defensive names) and its
    POSITIVE trailing realized volatility in low-VIX regimes (rewarding
    higher-beta cyclicals), and 0 in the neutral band.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide adjusted-close price frame, index=date, columns=tickers.
    vix : pd.Series
        VIX level series, aligned/ffilled to `prices.index`.
    high_vix_threshold, low_vix_threshold : float
        Regime cutoffs.

    Returns
    -------
    pd.DataFrame
        VIX-regime-conditioned factor score, same shape as `prices`.
    """
    vix_daily = vix.reindex(prices.index, method="ffill")
    realized_vol = prices.pct_change().rolling(21).std() * np.sqrt(252)

    regime = pd.Series(0, index=prices.index)
    regime[vix_daily > high_vix_threshold] = -1  # defensive regime
    regime[vix_daily < low_vix_threshold] = 1    # risk-on regime

    factor = realized_vol.mul(regime, axis=0)
    return factor


def compute_macro_composite(
    prices: pd.DataFrame,
    fed_funds_rate: pd.Series,
    vix: pd.Series,
) -> pd.DataFrame:
    """
    Combines rate sensitivity and VIX regime signals into a single
    cross-sectionally z-scored composite macro factor.

    Returns
    -------
    pd.DataFrame
        Composite macro factor score, same shape as `prices`.
    """
    rate_sens = compute_rate_sensitivity(prices, fed_funds_rate)
    vix_regime = compute_vix_regime_factor(prices, vix)

    def _row_zscore(row: pd.Series) -> pd.Series:
        std = row.std(ddof=0)
        return (row - row.mean()) / std if std and std > 0 else row * 0

    rate_z = rate_sens.apply(_row_zscore, axis=1)
    vix_z = vix_regime.apply(_row_zscore, axis=1)

    return (rate_z + vix_z) / 2.0
