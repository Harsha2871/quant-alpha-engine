"""Tests for alpha_engine.research.ic_analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_engine.research.ic_analysis import (
    compute_forward_returns,
    compute_rank_ic,
    compute_icir,
    summarize_ic,
    multi_horizon_ic,
    ICSummary,
)


@pytest.fixture
def perfect_signal_setup():
    """
    Builds a factor panel that perfectly (positively) ranks forward returns
    at every date, so the resulting IC should be exactly 1.0 everywhere.
    """
    dates = pd.bdate_range("2021-01-01", periods=60)
    tickers = [f"T{i}" for i in range(10)]

    rng = np.random.default_rng(0)
    prices = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    prices.iloc[0] = 100.0

    factor = pd.DataFrame(index=dates, columns=tickers, dtype=float)

    for t in range(1, len(dates)):
        # Assign each ticker a fixed rank-based daily return so today's factor
        # value (= yesterday's price rank) perfectly predicts tomorrow's return rank.
        ranks = rng.permutation(len(tickers))
        base_returns = np.linspace(-0.02, 0.02, len(tickers))
        returns = base_returns[ranks]
        prices.iloc[t] = prices.iloc[t - 1].values * (1 + returns)
        factor.iloc[t - 1] = ranks  # factor known the day before the return happens

    return prices.astype(float), factor.astype(float)


def test_compute_forward_returns_basic():
    dates = pd.bdate_range("2022-01-01", periods=5)
    prices = pd.DataFrame({"A": [100, 110, 121, 133.1, 146.41]}, index=dates)
    fwd = compute_forward_returns(prices, horizon=1)
    # (110/100)-1 = 0.10 for day 0
    assert fwd["A"].iloc[0] == pytest.approx(0.10, abs=1e-9)
    # Last row should be NaN (no future price)
    assert pd.isna(fwd["A"].iloc[-1])


def test_compute_rank_ic_perfect_positive_correlation(perfect_signal_setup):
    prices, factor = perfect_signal_setup
    fwd_returns = compute_forward_returns(prices, horizon=1)
    ic_series = compute_rank_ic(factor, fwd_returns)
    clean = ic_series.dropna()
    assert len(clean) > 0
    # Every date should show perfect rank correlation (constructed by design).
    assert np.allclose(clean.values, 1.0, atol=1e-9)


def test_compute_icir_perfect_consistency(perfect_signal_setup):
    prices, factor = perfect_signal_setup
    fwd_returns = compute_forward_returns(prices, horizon=1)
    ic_series = compute_rank_ic(factor, fwd_returns)
    icir = compute_icir(ic_series)
    # Zero variance in IC -> ICIR is undefined (NaN) by our guard.
    assert np.isnan(icir)


def test_compute_icir_with_variable_ic():
    ic_series = pd.Series([0.05, 0.03, -0.01, 0.06, 0.02])
    icir = compute_icir(ic_series)
    expected = ic_series.mean() / ic_series.std(ddof=0)
    assert icir == pytest.approx(expected)


def test_compute_icir_insufficient_data_returns_nan():
    assert np.isnan(compute_icir(pd.Series([0.05])))
    assert np.isnan(compute_icir(pd.Series(dtype=float)))


def test_summarize_ic_returns_valid_summary(perfect_signal_setup):
    prices, factor = perfect_signal_setup
    summary = summarize_ic(factor, prices, horizon=1)
    assert isinstance(summary, ICSummary)
    assert summary.n_periods > 0
    assert summary.mean_ic == pytest.approx(1.0, abs=1e-9)
    assert summary.hit_rate == pytest.approx(1.0)


def test_multi_horizon_ic_has_expected_index(perfect_signal_setup):
    prices, factor = perfect_signal_setup
    table = multi_horizon_ic(factor, prices, horizons=(1, 5, 10))
    assert list(table.index) == [1, 5, 10]
    assert "mean_ic" in table.columns
    assert "icir" in table.columns


def test_rank_ic_handles_insufficient_names():
    dates = pd.bdate_range("2022-01-01", periods=3)
    factor = pd.DataFrame({"A": [1, 2, 3], "B": [3, 2, 1]}, index=dates)
    fwd = pd.DataFrame({"A": [0.1, 0.2, np.nan], "B": [0.2, 0.1, np.nan]}, index=dates)
    ic = compute_rank_ic(factor, fwd)
    # Only 2 valid names per row < 3 required -> NaN everywhere.
    assert ic.isna().all()
