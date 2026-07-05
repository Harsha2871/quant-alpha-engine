"""Tests for alpha_engine.backtest.portfolio_constructor."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_engine.backtest.portfolio_constructor import (
    quantile_portfolio,
    build_rebalance_schedule,
    construct_portfolio_series,
)


def test_quantile_portfolio_basic_long_short_split():
    scores = pd.Series(
        {f"T{i}": float(i) for i in range(10)}  # T0=0 (worst) ... T9=9 (best)
    )
    weights = quantile_portfolio(scores, top_quantile=0.2, bottom_quantile=0.2)

    # Top 2 (T8, T9) should be long, bottom 2 (T0, T1) should be short.
    assert set(weights[weights > 0].index) == {"T8", "T9"}
    assert set(weights[weights < 0].index) == {"T0", "T1"}


def test_quantile_portfolio_weights_sum_to_zero_dollar_neutral():
    scores = pd.Series({f"T{i}": float(i) for i in range(20)})
    weights = quantile_portfolio(scores, top_quantile=0.25, bottom_quantile=0.25)
    assert weights[weights > 0].sum() == pytest.approx(1.0)
    assert weights[weights < 0].sum() == pytest.approx(-1.0)
    assert weights.sum() == pytest.approx(0.0, abs=1e-9)


def test_quantile_portfolio_equal_weighted_within_leg():
    scores = pd.Series({f"T{i}": float(i) for i in range(10)})
    weights = quantile_portfolio(scores, top_quantile=0.2, bottom_quantile=0.2)
    long_weights = weights[weights > 0]
    short_weights = weights[weights < 0]
    assert long_weights.nunique() == 1  # all long positions equal-weighted
    assert short_weights.nunique() == 1  # all short positions equal-weighted


def test_quantile_portfolio_too_few_names_returns_empty():
    scores = pd.Series({"A": 1.0, "B": 2.0})
    weights = quantile_portfolio(scores, top_quantile=0.2, bottom_quantile=0.2)
    assert weights.empty


def test_quantile_portfolio_drops_nans():
    scores = pd.Series({f"T{i}": float(i) for i in range(8)} | {"BAD": np.nan})
    weights = quantile_portfolio(scores, top_quantile=0.25, bottom_quantile=0.25)
    assert "BAD" not in weights.index


def test_build_rebalance_schedule_monthly():
    dates = pd.bdate_range("2022-01-01", "2022-06-30")
    schedule = build_rebalance_schedule(dates, freq="ME")
    # Should produce roughly one date per month (6 months of data).
    assert 5 <= len(schedule) <= 6
    # Each rebalance date must actually be in the original calendar.
    assert all(d in dates for d in schedule)


def test_construct_portfolio_series_builds_weights_for_each_rebalance():
    dates = pd.bdate_range("2022-01-01", periods=100)
    tickers = [f"T{i}" for i in range(10)]
    rng = np.random.default_rng(1)
    factor_panel = pd.DataFrame(rng.normal(size=(100, 10)), index=dates, columns=tickers)

    rebalance_dates = build_rebalance_schedule(dates, freq="ME")
    portfolios = construct_portfolio_series(factor_panel, rebalance_dates, top_quantile=0.2, bottom_quantile=0.2)

    assert len(portfolios) > 0
    for date, weights in portfolios.items():
        assert isinstance(weights, pd.Series)
        assert not weights.empty
