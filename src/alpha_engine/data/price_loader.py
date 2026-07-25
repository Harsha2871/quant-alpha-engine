"""
price_loader.py — OHLCV price data fetcher backed by yfinance, with a
pickle-based on-disk cache to avoid re-hitting the network on every run.

The cache key is a hash of (tickers, start, end, interval). Cached frames are
stored as pickles under `config.CACHE_DIR`. Synthetic data is available for
offline demos and tests, but it must be enabled explicitly so empirical
research runs do not accidentally report mock-data results.
"""
from __future__ import annotations

import hashlib
import pickle
import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from alpha_engine.config import CACHE_DIR

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    _HAS_YFINANCE = True
except ImportError:  # pragma: no cover
    _HAS_YFINANCE = False


def _cache_key(tickers: Iterable[str], start: str, end: str, interval: str) -> str:
    raw = f"{sorted(tickers)}|{start}|{end}|{interval}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"prices_{key}.pkl"


def _synthetic_ohlcv(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Deterministic synthetic OHLCV generator used as an offline fallback.

    Uses a per-ticker seeded geometric brownian motion so results are
    reproducible across runs, which is important for tests and demos.
    """
    dates = pd.bdate_range(start=start, end=end)
    frames = {}
    for i, ticker in enumerate(tickers):
        rng = np.random.default_rng(seed=abs(hash(ticker)) % (2**32))
        n = len(dates)
        drift = 0.0004 + 0.0001 * (i % 5)
        vol = 0.015 + 0.002 * (i % 3)
        returns = rng.normal(loc=drift, scale=vol, size=n)
        price = 100 * np.exp(np.cumsum(returns))
        high = price * (1 + np.abs(rng.normal(0, 0.004, n)))
        low = price * (1 - np.abs(rng.normal(0, 0.004, n)))
        open_ = price * (1 + rng.normal(0, 0.002, n))
        volume = rng.integers(1_000_000, 20_000_000, n)
        frames[ticker] = pd.DataFrame(
            {
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": price,
                "Adj Close": price,
                "Volume": volume,
            },
            index=dates,
        )
    return pd.concat(frames, axis=1)


class PriceLoader:
    """Fetches and caches OHLCV data for a universe of tickers."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
        allow_synthetic: bool = False,
    ):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache
        self.allow_synthetic = allow_synthetic

    def get_ohlcv(
        self,
        tickers: list[str],
        start: str = "2019-01-01",
        end: str = "2024-01-01",
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Returns a wide DataFrame with a MultiIndex column of (ticker, field),
        fields = Open/High/Low/Close/Adj Close/Volume.
        """
        key = _cache_key(tickers, start, end, interval)
        cache_file = _cache_path(key)

        if self.use_cache and not force_refresh and cache_file.exists():
            logger.info("Loading prices from cache: %s", cache_file)
            with open(cache_file, "rb") as fh:
                return pickle.load(fh)

        df = self._fetch(tickers, start, end, interval)

        if self.use_cache:
            with open(cache_file, "wb") as fh:
                pickle.dump(df, fh)

        return df

    def _synthetic_or_raise(
        self,
        tickers: list[str],
        start: str,
        end: str,
        reason: str,
    ) -> pd.DataFrame:
        if not self.allow_synthetic:
            raise RuntimeError(
                "Synthetic price data is disabled. Enable allow_synthetic=True only for "
                f"offline demos/tests. Original failure: {reason}"
            )

        logger.warning("Using synthetic price data: %s", reason)
        data = _synthetic_ohlcv(tickers, start, end)
        data.attrs["data_source"] = "synthetic"
        data.attrs["synthetic"] = True
        data.attrs["synthetic_reason"] = reason
        return data

    def _fetch(self, tickers: list[str], start: str, end: str, interval: str) -> pd.DataFrame:
        if not _HAS_YFINANCE:
            return self._synthetic_or_raise(tickers, start, end, "yfinance is not installed")

        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                interval=interval,
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            if raw is None or raw.empty:
                raise ValueError("yfinance returned an empty frame")

            # yfinance returns single-level columns for a single ticker
            if len(tickers) == 1 and not isinstance(raw.columns, pd.MultiIndex):
                raw = pd.concat({tickers[0]: raw}, axis=1)

            raw.attrs["data_source"] = "yfinance"
            raw.attrs["synthetic"] = False
            return raw
        except Exception as exc:  # pragma: no cover - network dependent
            return self._synthetic_or_raise(tickers, start, end, f"yfinance fetch failed: {exc}")

    def get_close_prices(
        self,
        tickers: list[str],
        start: str = "2019-01-01",
        end: str = "2024-01-01",
    ) -> pd.DataFrame:
        """Convenience method returning only the (Adj) Close price matrix, columns=tickers."""
        ohlcv = self.get_ohlcv(tickers, start=start, end=end)
        closes = {}
        for ticker in tickers:
            if ticker not in ohlcv.columns.get_level_values(0):
                continue
            sub = ohlcv[ticker]
            field = "Adj Close" if "Adj Close" in sub.columns else "Close"
            closes[ticker] = sub[field]
        close_prices = pd.DataFrame(closes)
        close_prices.attrs.update(ohlcv.attrs)
        return close_prices
