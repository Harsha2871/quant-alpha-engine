"""
portfolio_constructor.py — Cross-sectional long/short portfolio construction
from factor scores.

Standard quintile-sort methodology: rank the universe by factor score at
each rebalance date, go long the top quantile (e.g. top 20%) and short the
bottom quantile (bottom 20%), equal-weighted within each leg. This is the
canonical academic construction used in Fama-French factor portfolios.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PortfolioWeights:
    """Target weights for one rebalance date: positive = long, negative = short."""

    date: pd.Timestamp
    weights: pd.Series  # index=ticker, sums to 0 (dollar-neutral) by construction


def quantile_portfolio(
    factor_scores: pd.Series,
    top_quantile: float = 0.20,
    bottom_quantile: float = 0.20,
) -> pd.Series:
    """
    Builds a single-period long-top/short-bottom quantile portfolio from a
    cross-section of factor scores.

    Parameters
    ----------
    factor_scores : pd.Series
        Index=ticker, values=factor score at this rebalance date. NaNs are
        dropped before ranking.
    top_quantile : float
        Fraction of the universe (by rank) to go long, e.g. 0.20 = top 20%.
    bottom_quantile : float
        Fraction of the universe (by rank) to go short, e.g. 0.20 = bottom 20%.

    Returns
    -------
    pd.Series
        Index=ticker (only names with a nonzero position), values=weight.
        Long leg sums to +1.0, short leg sums to -1.0 (dollar-neutral,
        equal-weighted within each leg).
    """
    scores = factor_scores.dropna()
    n = len(scores)
    if n < 5:
        # Too few names to form a meaningful quintile split.
        return pd.Series(dtype=float)

    ranked = scores.sort_values(ascending=False)
    n_long = max(1, int(np.floor(n * top_quantile)))
    n_short = max(1, int(np.floor(n * bottom_quantile)))

    long_names = ranked.index[:n_long]
    short_names = ranked.index[-n_short:]

    weights = pd.Series(0.0, index=scores.index)
    weights.loc[long_names] = 1.0 / n_long
    weights.loc[short_names] = -1.0 / n_short

    return weights[weights != 0.0]


def build_rebalance_schedule(dates: pd.DatetimeIndex, freq: str = "ME") -> pd.DatetimeIndex:
    """
    Given a daily trading calendar, returns the subset of dates corresponding
    to period-end rebalances (default: month-end).

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Full trading calendar (e.g. from a price DataFrame's index).
    freq : str
        Pandas offset alias for the rebalance frequency ('ME' = month-end,
        'W' = weekly, 'QE' = quarter-end).

    Returns
    -------
    pd.DatetimeIndex
        Subset of `dates` marking each rebalance date (last trading day of
        each period).
    """
    series = pd.Series(dates, index=dates)
    period_ends = series.resample(freq).last().dropna()
    return pd.DatetimeIndex(period_ends.values)


def construct_portfolio_series(
    factor_panel: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    top_quantile: float = 0.20,
    bottom_quantile: float = 0.20,
) -> dict[pd.Timestamp, pd.Series]:
    """
    Applies `quantile_portfolio` at every rebalance date to produce a full
    history of target weights.

    Parameters
    ----------
    factor_panel : pd.DataFrame
        Index=date, columns=tickers, values=factor score.
    rebalance_dates : pd.DatetimeIndex
        Dates on which to re-form the portfolio.
    top_quantile, bottom_quantile : float
        See `quantile_portfolio`.

    Returns
    -------
    dict[pd.Timestamp, pd.Series]
        Mapping from rebalance date to target weight Series (ticker -> weight).
    """
    portfolios = {}
    for date in rebalance_dates:
        if date not in factor_panel.index:
            # Use the nearest prior available date (handles holidays / gaps).
            valid_dates = factor_panel.index[factor_panel.index <= date]
            if len(valid_dates) == 0:
                continue
            date_to_use = valid_dates[-1]
        else:
            date_to_use = date

        scores = factor_panel.loc[date_to_use]
        weights = quantile_portfolio(scores, top_quantile, bottom_quantile)
        if not weights.empty:
            portfolios[date] = weights

    return portfolios
