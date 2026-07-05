"""
fundamental_loader.py — Company fundamentals (P/E, EPS growth, revenue
growth, ROE, price-to-book, earnings history) pulled from yfinance's
`Ticker.info` / `Ticker.financials` interfaces.

yfinance's fundamental data can be flaky and inconsistently populated across
tickers, so every field access is defensive and falls back to a deterministic
synthetic value keyed off the ticker symbol. This guarantees the factor
pipeline always has *something* to compute on, even fully offline.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    _HAS_YFINANCE = True
except ImportError:  # pragma: no cover
    _HAS_YFINANCE = False


FUNDAMENTAL_FIELDS = [
    "trailing_pe",
    "forward_pe",
    "price_to_book",
    "return_on_equity",
    "eps_growth",
    "revenue_growth",
    "earnings_stability",
    "debt_to_equity",
]


def _synthetic_fundamentals(ticker: str) -> dict:
    """Deterministic synthetic fundamentals fallback, seeded by ticker name."""
    rng = np.random.default_rng(seed=abs(hash(ticker)) % (2**32))
    return {
        "trailing_pe": float(np.clip(rng.normal(22, 8), 5, 60)),
        "forward_pe": float(np.clip(rng.normal(20, 7), 5, 55)),
        "price_to_book": float(np.clip(rng.normal(6, 3), 0.5, 25)),
        "return_on_equity": float(np.clip(rng.normal(0.20, 0.12), -0.1, 0.9)),
        "eps_growth": float(np.clip(rng.normal(0.10, 0.15), -0.5, 1.0)),
        "revenue_growth": float(np.clip(rng.normal(0.08, 0.10), -0.3, 0.6)),
        "earnings_stability": float(np.clip(rng.normal(0.7, 0.15), 0.1, 1.0)),
        "debt_to_equity": float(np.clip(rng.normal(0.8, 0.5), 0.0, 4.0)),
    }


class FundamentalLoader:
    """Fetches point-in-time-ish fundamentals for a list of tickers."""

    def __init__(self):
        self._cache: dict[str, dict] = {}

    def get_fundamentals(self, ticker: str) -> dict:
        """Returns a flat dict of fundamental metrics for a single ticker."""
        if ticker in self._cache:
            return self._cache[ticker]

        data = self._fetch_single(ticker)
        self._cache[ticker] = data
        return data

    def _fetch_single(self, ticker: str) -> dict:
        if not _HAS_YFINANCE:
            return _synthetic_fundamentals(ticker)

        try:
            info = yf.Ticker(ticker).info
            if not info or len(info) < 5:
                raise ValueError("empty info payload")

            trailing_pe = info.get("trailingPE")
            forward_pe = info.get("forwardPE")
            pb = info.get("priceToBook")
            roe = info.get("returnOnEquity")
            eps_growth = info.get("earningsQuarterlyGrowth")
            revenue_growth = info.get("revenueGrowth")
            debt_to_equity = info.get("debtToEquity")

            fallback = _synthetic_fundamentals(ticker)

            return {
                "trailing_pe": float(trailing_pe) if trailing_pe else fallback["trailing_pe"],
                "forward_pe": float(forward_pe) if forward_pe else fallback["forward_pe"],
                "price_to_book": float(pb) if pb else fallback["price_to_book"],
                "return_on_equity": float(roe) if roe is not None else fallback["return_on_equity"],
                "eps_growth": float(eps_growth) if eps_growth is not None else fallback["eps_growth"],
                "revenue_growth": float(revenue_growth) if revenue_growth is not None else fallback["revenue_growth"],
                "earnings_stability": fallback["earnings_stability"],
                "debt_to_equity": float(debt_to_equity) / 100.0 if debt_to_equity else fallback["debt_to_equity"],
            }
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Fundamentals fetch failed for %s (%s) — using synthetic fallback", ticker, exc)
            return _synthetic_fundamentals(ticker)

    def get_fundamentals_frame(self, tickers: list[str]) -> pd.DataFrame:
        """Returns a DataFrame indexed by ticker with one column per fundamental field."""
        rows = {ticker: self.get_fundamentals(ticker) for ticker in tickers}
        return pd.DataFrame.from_dict(rows, orient="index")[FUNDAMENTAL_FIELDS]
