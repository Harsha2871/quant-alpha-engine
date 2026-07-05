"""Tests for alpha_engine.factors.momentum."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_engine.factors.momentum import (
    compute_12_1_momentum,
    compute_1m_reversal,
    compute_3m_momentum,
    compute_52w_high_proximity,
)


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Deterministic synthetic price panel: one steadily-rising, one flat, one falling."""
    dates = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(42)

    rising = 100 * np.exp(np.cumsum(np.full(len(dates), 0.001)))
    flat = np.full(len(dates), 50.0) + rng.normal(0, 0.01, len(dates))
    falling = 100 * np.exp(np.cumsum(np.full(len(dates), -0.001)))

    return pd.DataFrame({"RISER": rising, "FLAT": flat, "FALLER": falling}, index=dates)


def test_12_1_momentum_shape(sample_prices):
    mom = compute_12_1_momentum(sample_prices)
    assert mom.shape == sample_prices.shape
    assert list(mom.columns) == list(sample_prices.columns)


def test_12_1_momentum_ranks_riser_above_faller(sample_prices):
    mom = compute_12_1_momentum(sample_prices)
    last_valid = mom.dropna().iloc[-1]
    assert last_valid["RISER"] > last_valid["FLAT"] > last_valid["FALLER"]


def test_12_1_momentum_uses_correct_windows(sample_prices):
    # pct_change(252).shift(21) — verify against manual computation for one column.
    mom = compute_12_1_momentum(sample_prices)
    manual = sample_prices["RISER"].pct_change(252).shift(21)
    pd.testing.assert_series_equal(mom["RISER"], manual, check_names=False)


def test_1m_reversal_is_negative_of_1m_return(sample_prices):
    reversal = compute_1m_reversal(sample_prices)
    raw_1m = sample_prices.pct_change(21)
    pd.testing.assert_frame_equal(reversal, -raw_1m)


def test_1m_reversal_contrarian_direction(sample_prices):
    reversal = compute_1m_reversal(sample_prices)
    last_valid = reversal.dropna().iloc[-1]
    # RISER has positive momentum -> reversal score should be negative (contrarian).
    assert last_valid["RISER"] < 0
    # FALLER has negative momentum -> reversal score should be positive.
    assert last_valid["FALLER"] > 0


def test_3m_momentum_no_skip(sample_prices):
    mom3m = compute_3m_momentum(sample_prices)
    manual = sample_prices.pct_change(63)
    pd.testing.assert_frame_equal(mom3m, manual)


def test_52w_high_proximity_bounded_and_correct_for_riser(sample_prices):
    proximity = compute_52w_high_proximity(sample_prices)
    last_valid = proximity.dropna().iloc[-1]
    # A monotonically rising series should be at/near its 52-week high (proximity ~= 1).
    assert last_valid["RISER"] == pytest.approx(1.0, abs=1e-6)
    # A monotonically falling series should be well below its 52-week high.
    assert last_valid["FALLER"] < 0.9


def test_momentum_handles_empty_dataframe():
    empty = pd.DataFrame()
    result = compute_12_1_momentum(empty)
    assert result.empty
