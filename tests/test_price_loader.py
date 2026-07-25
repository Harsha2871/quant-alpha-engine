"""Tests for alpha_engine.data.price_loader."""
from __future__ import annotations

import pandas as pd
import pytest

from alpha_engine.data import price_loader
from alpha_engine.data.price_loader import PriceLoader


def test_price_loader_requires_explicit_synthetic_mode_when_yfinance_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(price_loader, "_HAS_YFINANCE", False)

    loader = PriceLoader(cache_dir=tmp_path, use_cache=False, allow_synthetic=False)

    with pytest.raises(RuntimeError, match="Synthetic price data is disabled"):
        loader.get_ohlcv(["AAPL"], start="2023-01-01", end="2023-01-10")


def test_price_loader_marks_synthetic_output_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setattr(price_loader, "_HAS_YFINANCE", False)

    loader = PriceLoader(cache_dir=tmp_path, use_cache=False, allow_synthetic=True)
    data = loader.get_ohlcv(["AAPL"], start="2023-01-01", end="2023-01-10")

    assert isinstance(data, pd.DataFrame)
    assert data.attrs["data_source"] == "synthetic"
    assert data.attrs["synthetic"] is True
