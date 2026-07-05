"""Tests for alpha_engine.backtest.backtest_engine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_engine.backtest.backtest_engine import run_walk_forward_backtest, BacktestResult
from alpha_engine.factors.momentum import compute_12_1_momentum


@pytest.fixture
def synthetic_universe():
    """20 tickers over 3 years of business days with varied deterministic drifts."""
    dates = pd.bdate_range("2019-01-01", periods=756)  # ~3 trading years
    tickers = [f"T{i}" for i in range(20)]
    rng = np.random.default_rng(7)

    data = {}
    for i, ticker in enumerate(tickers):
        drift = -0.0003 + 0.00006 * i  # spread of drifts so factor has signal
        vol = 0.012
        returns = rng.normal(drift, vol, len(dates))
        data[ticker] = 100 * np.exp(np.cumsum(returns))

    prices = pd.DataFrame(data, index=dates)
    benchmark = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(dates)))), index=dates)
    return prices, benchmark


def test_backtest_returns_result_object(synthetic_universe):
    prices, benchmark = synthetic_universe
    factor_panel = compute_12_1_momentum(prices)

    result = run_walk_forward_backtest(
        prices=prices,
        factor_panel=factor_panel,
        top_quantile=0.2,
        bottom_quantile=0.2,
        transaction_cost_bps=10.0,
        benchmark_prices=benchmark,
    )

    assert isinstance(result, BacktestResult)
    assert len(result.equity_curve) == len(prices)
    assert len(result.weights_history) > 0


def test_backtest_equity_curve_starts_near_one(synthetic_universe):
    prices, benchmark = synthetic_universe
    factor_panel = compute_12_1_momentum(prices)
    result = run_walk_forward_backtest(prices, factor_panel, benchmark_prices=benchmark)
    assert result.equity_curve.iloc[0] == pytest.approx(1.0, abs=0.05)


def test_backtest_performance_report_has_expected_fields(synthetic_universe):
    prices, benchmark = synthetic_universe
    factor_panel = compute_12_1_momentum(prices)
    result = run_walk_forward_backtest(prices, factor_panel, benchmark_prices=benchmark)

    perf_dict = result.performance.to_dict()
    for key in ["annualized_return", "annualized_volatility", "sharpe", "sortino",
                "max_drawdown", "alpha_vs_benchmark", "beta_vs_benchmark"]:
        assert key in perf_dict


def test_higher_transaction_costs_reduce_or_equal_return(synthetic_universe):
    prices, benchmark = synthetic_universe
    factor_panel = compute_12_1_momentum(prices)

    low_cost = run_walk_forward_backtest(prices, factor_panel, transaction_cost_bps=0.0, benchmark_prices=benchmark)
    high_cost = run_walk_forward_backtest(prices, factor_panel, transaction_cost_bps=500.0, benchmark_prices=benchmark)

    # Higher transaction costs should never produce a strictly better final equity value.
    assert high_cost.equity_curve.iloc[-1] <= low_cost.equity_curve.iloc[-1] + 1e-9


def test_backtest_with_no_factor_signal_produces_flat_or_valid_curve():
    dates = pd.bdate_range("2020-01-01", periods=300)
    tickers = [f"T{i}" for i in range(10)]
    prices = pd.DataFrame(100.0, index=dates, columns=tickers)  # flat prices, no signal
    factor_panel = pd.DataFrame(0.0, index=dates, columns=tickers)

    result = run_walk_forward_backtest(prices, factor_panel)
    # With flat prices, equity should stay very close to 1.0 (no returns to capture).
    assert result.equity_curve.iloc[-1] == pytest.approx(1.0, abs=0.05)


def test_backtest_turnover_history_is_non_negative(synthetic_universe):
    prices, benchmark = synthetic_universe
    factor_panel = compute_12_1_momentum(prices)
    result = run_walk_forward_backtest(prices, factor_panel, benchmark_prices=benchmark)
    assert (result.turnover_history >= 0).all()
