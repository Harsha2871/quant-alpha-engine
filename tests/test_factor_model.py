"""Tests for alpha_engine.ml.factor_model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_engine.ml.factor_model import walk_forward_train_test


def test_walk_forward_train_test_rejects_empty_chronological_test_window():
    """An ML backtest must not silently replace an empty test window with random rows."""
    dates = pd.bdate_range("2020-01-01", periods=40)
    tickers = [f"T{i}" for i in range(6)]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])

    rng = np.random.default_rng(42)
    panel = pd.DataFrame(
        {
            "momentum_12_1": rng.normal(size=len(index)),
            "quality_composite": rng.normal(size=len(index)),
            "forward_return": rng.normal(size=len(index)),
            "target_quintile": np.tile([0, 1, 2, 3, 4, 0], len(dates)),
        },
        index=index,
    )

    with pytest.raises(ValueError, match="chronological test partition is empty"):
        walk_forward_train_test(
            panel,
            feature_cols=["momentum_12_1", "quality_composite"],
            train_years=3,
            test_years=1,
        )
