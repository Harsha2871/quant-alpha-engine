"""
performance.py — Standard portfolio performance/risk metrics: Sharpe,
Sortino, max drawdown, and alpha vs a benchmark (e.g. SPY).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Geometric annualized return from a periodic return series."""
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    cumulative = (1 + clean).prod()
    n_periods = len(clean)
    if n_periods == 0 or cumulative <= 0:
        return float("nan")
    return float(cumulative ** (periods_per_year / n_periods) - 1)


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualized standard deviation of returns."""
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    return float(clean.std(ddof=0) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate_annual: float = 0.02,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Annualized Sharpe ratio: (mean excess return / std of returns) * sqrt(periods_per_year).
    """
    clean = returns.dropna()
    if clean.empty or clean.std(ddof=0) == 0:
        return float("nan")
    rf_periodic = (1 + risk_free_rate_annual) ** (1 / periods_per_year) - 1
    excess = clean - rf_periodic
    return float(excess.mean() / clean.std(ddof=0) * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate_annual: float = 0.02,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Annualized Sortino ratio: like Sharpe, but the denominator only penalizes
    downside deviation (returns below the risk-free rate).
    """
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    rf_periodic = (1 + risk_free_rate_annual) ** (1 / periods_per_year) - 1
    excess = clean - rf_periodic
    downside = excess[excess < 0]
    downside_std = np.sqrt((downside ** 2).mean()) if len(downside) > 0 else 0.0
    if downside_std == 0:
        return float("nan")
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown of an equity/cumulative-return curve,
    expressed as a negative fraction (e.g. -0.184 = -18.4%).
    """
    clean = equity_curve.dropna()
    if clean.empty:
        return float("nan")
    running_max = clean.cummax()
    drawdown = clean / running_max - 1.0
    return float(drawdown.min())


def equity_curve_from_returns(returns: pd.Series, starting_value: float = 1.0) -> pd.Series:
    """Converts a periodic return series into a cumulative equity curve."""
    clean = returns.fillna(0.0)
    return starting_value * (1 + clean).cumprod()


def alpha_vs_benchmark(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate_annual: float = 0.02,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Jensen's alpha: annualized excess return of the strategy over what would
    be predicted by its beta exposure to the benchmark (e.g. SPY), following
    the single-factor CAPM regression: R_p - Rf = alpha + beta * (R_b - Rf) + e.
    """
    common_index = returns.index.intersection(benchmark_returns.index)
    r = returns.loc[common_index].dropna()
    b = benchmark_returns.loc[common_index].dropna()
    common_index = r.index.intersection(b.index)
    r, b = r.loc[common_index], b.loc[common_index]

    if len(r) < 3:
        return float("nan")

    rf_periodic = (1 + risk_free_rate_annual) ** (1 / periods_per_year) - 1
    excess_r = r - rf_periodic
    excess_b = b - rf_periodic

    beta_denom = np.var(excess_b)
    if beta_denom == 0:
        return float("nan")

    beta = np.cov(excess_r, excess_b)[0, 1] / beta_denom
    alpha_periodic = excess_r.mean() - beta * excess_b.mean()
    return float(alpha_periodic * periods_per_year)


@dataclass
class PerformanceReport:
    """Full performance summary for a strategy return series."""

    annualized_return: float
    annualized_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    alpha_vs_benchmark: float
    beta_vs_benchmark: float
    benchmark_annualized_return: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """CAPM beta of `returns` relative to `benchmark_returns`."""
    common_index = returns.index.intersection(benchmark_returns.index)
    r, b = returns.loc[common_index].dropna(), benchmark_returns.loc[common_index].dropna()
    common_index = r.index.intersection(b.index)
    r, b = r.loc[common_index], b.loc[common_index]
    if len(r) < 3 or np.var(b) == 0:
        return float("nan")
    return float(np.cov(r, b)[0, 1] / np.var(b))


def build_performance_report(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate_annual: float = 0.02,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> PerformanceReport:
    """Computes the full suite of performance metrics in one call."""
    equity = equity_curve_from_returns(returns)
    return PerformanceReport(
        annualized_return=annualized_return(returns, periods_per_year),
        annualized_volatility=annualized_volatility(returns, periods_per_year),
        sharpe=sharpe_ratio(returns, risk_free_rate_annual, periods_per_year),
        sortino=sortino_ratio(returns, risk_free_rate_annual, periods_per_year),
        max_drawdown=max_drawdown(equity),
        alpha_vs_benchmark=alpha_vs_benchmark(returns, benchmark_returns, risk_free_rate_annual, periods_per_year),
        beta_vs_benchmark=compute_beta(returns, benchmark_returns),
        benchmark_annualized_return=annualized_return(benchmark_returns, periods_per_year),
    )
