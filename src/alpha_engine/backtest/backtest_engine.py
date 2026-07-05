"""
backtest_engine.py — Walk-forward, cross-sectional long/short backtest
engine with transaction cost modeling.

At each monthly rebalance date, target weights are computed from the
current factor cross-section (see `portfolio_constructor.quantile_portfolio`).
Between rebalances, the portfolio drifts with daily returns. Transaction
costs are charged on turnover (change in weights) at each rebalance,
following the standard bps-per-trade convention.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alpha_engine.backtest.portfolio_constructor import (
    build_rebalance_schedule,
    quantile_portfolio,
)
from alpha_engine.backtest.performance import build_performance_report, PerformanceReport


@dataclass
class BacktestResult:
    """Full output of a walk-forward backtest run."""

    daily_returns: pd.Series
    equity_curve: pd.Series
    weights_history: dict = field(default_factory=dict)
    turnover_history: pd.Series = field(default_factory=pd.Series)
    performance: PerformanceReport | None = None

    def to_summary_dict(self) -> dict:
        return {
            "performance": self.performance.to_dict() if self.performance else {},
            "equity_curve": {
                str(d.date()): float(v) for d, v in self.equity_curve.items()
            },
            "avg_turnover": float(self.turnover_history.mean()) if len(self.turnover_history) else 0.0,
            "n_rebalances": len(self.weights_history),
        }


def _compute_turnover(prev_weights: pd.Series, new_weights: pd.Series) -> float:
    """Sum of absolute weight changes across the union of held tickers."""
    all_tickers = prev_weights.index.union(new_weights.index)
    prev = prev_weights.reindex(all_tickers, fill_value=0.0)
    new = new_weights.reindex(all_tickers, fill_value=0.0)
    return float((new - prev).abs().sum())


def run_walk_forward_backtest(
    prices: pd.DataFrame,
    factor_panel: pd.DataFrame,
    top_quantile: float = 0.20,
    bottom_quantile: float = 0.20,
    transaction_cost_bps: float = 10.0,
    rebalance_freq: str = "ME",
    benchmark_prices: pd.Series | None = None,
    risk_free_rate_annual: float = 0.02,
) -> BacktestResult:
    """
    Runs a full walk-forward cross-sectional long/short backtest.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide adjusted-close price frame, index=date, columns=tickers.
    factor_panel : pd.DataFrame
        Factor scores aligned to `prices`, index=date, columns=tickers.
        Must not use any information not available as-of that date (i.e.
        factors should already be point-in-time / lagged appropriately).
    top_quantile, bottom_quantile : float
        Long/short quantile cutoffs, see `quantile_portfolio`.
    transaction_cost_bps : float
        One-way transaction cost in basis points, applied to turnover at
        each rebalance.
    rebalance_freq : str
        Pandas resample alias, default 'ME' (month-end).
    benchmark_prices : pd.Series, optional
        Benchmark price series (e.g. SPY) used for the performance report.
        If not provided, alpha/beta will be NaN in the report.
    risk_free_rate_annual : float
        Annual risk-free rate used in Sharpe/Sortino/alpha calculations.

    Returns
    -------
    BacktestResult
    """
    daily_returns_matrix = prices.pct_change()
    rebalance_dates = build_rebalance_schedule(prices.index, freq=rebalance_freq)

    current_weights = pd.Series(dtype=float)
    strategy_returns = pd.Series(0.0, index=prices.index)
    turnover_history = {}
    weights_history = {}

    trading_dates = prices.index
    cost_rate = transaction_cost_bps / 10_000.0

    for i, date in enumerate(trading_dates):
        # Rebalance at (or just after) scheduled dates using that day's factor scores.
        if date in rebalance_dates and date in factor_panel.index:
            scores = factor_panel.loc[date]
            new_weights = quantile_portfolio(scores, top_quantile, bottom_quantile)

            if not new_weights.empty:
                turnover = _compute_turnover(current_weights, new_weights)
                turnover_history[date] = turnover
                # Transaction cost drag applied on the rebalance day itself.
                strategy_returns.loc[date] -= turnover * cost_rate
                current_weights = new_weights
                weights_history[date] = new_weights

        # Apply today's return using yesterday's (post-rebalance) weights.
        if not current_weights.empty and i > 0:
            today_stock_returns = daily_returns_matrix.loc[date].reindex(current_weights.index)
            valid = today_stock_returns.notna()
            if valid.any():
                period_return = float((current_weights[valid] * today_stock_returns[valid]).sum())
                strategy_returns.loc[date] += period_return

    equity_curve = (1 + strategy_returns.fillna(0.0)).cumprod()

    benchmark_returns = (
        benchmark_prices.pct_change().reindex(prices.index)
        if benchmark_prices is not None
        else pd.Series(0.0, index=prices.index)
    )

    performance = build_performance_report(
        strategy_returns, benchmark_returns, risk_free_rate_annual=risk_free_rate_annual
    )

    return BacktestResult(
        daily_returns=strategy_returns,
        equity_curve=equity_curve,
        weights_history=weights_history,
        turnover_history=pd.Series(turnover_history, name="turnover"),
        performance=performance,
    )
