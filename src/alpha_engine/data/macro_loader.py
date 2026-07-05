"""
macro_loader.py — Macroeconomic data loader using the FRED API (via
`fredapi`) for series such as the effective federal funds rate and CPI, plus
VIX pulled from yfinance (^VIX).

If no FRED API key is configured (or the request fails), realistic mock
series are generated deterministically so the rest of the pipeline keeps
working end-to-end.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from alpha_engine.config import FRED_API_KEY

logger = logging.getLogger(__name__)

try:
    from fredapi import Fred
    _HAS_FREDAPI = True
except ImportError:  # pragma: no cover
    _HAS_FREDAPI = False

try:
    import yfinance as yf
    _HAS_YFINANCE = True
except ImportError:  # pragma: no cover
    _HAS_YFINANCE = False


# FRED series IDs
FRED_SERIES = {
    "fed_funds_rate": "FEDFUNDS",   # Effective Federal Funds Rate
    "cpi": "CPIAUCSL",              # CPI for All Urban Consumers
}


def _mock_series(name: str, start: str, end: str) -> pd.Series:
    """Deterministic mock macro series used as an offline fallback."""
    dates = pd.date_range(start=start, end=end, freq="MS")
    rng = np.random.default_rng(seed=abs(hash(name)) % (2**32))
    n = len(dates)

    if name == "fed_funds_rate":
        base = np.linspace(0.25, 5.33, n) if n > 1 else np.array([5.33])
        noise = rng.normal(0, 0.05, n)
        values = np.clip(base + noise, 0, None)
    elif name == "cpi":
        base = 255 * np.exp(np.linspace(0, 0.22, n)) if n > 1 else np.array([255.0])
        noise = rng.normal(0, 0.3, n)
        values = base + noise
    elif name == "vix":
        values = 16 + rng.normal(0, 4, n).cumsum() * 0.05
        values = np.clip(values, 9, 80)
    else:
        values = rng.normal(0, 1, n)

    return pd.Series(values, index=dates, name=name)


class MacroLoader:
    """Loads macroeconomic series: Fed funds rate, CPI, and VIX."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or FRED_API_KEY
        self._fred = None
        if self.api_key and _HAS_FREDAPI:
            try:
                self._fred = Fred(api_key=self.api_key)
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not initialize FRED client: %s", exc)
                self._fred = None

    def _get_fred_series(self, series_id: str, name: str, start: str, end: str) -> pd.Series:
        if self._fred is None:
            logger.info("No FRED client available — using mock series for %s", name)
            return _mock_series(name, start, end)
        try:
            data = self._fred.get_series(series_id, observation_start=start, observation_end=end)
            data.name = name
            return data
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("FRED fetch failed for %s (%s) — using mock fallback", name, exc)
            return _mock_series(name, start, end)

    def get_fed_funds_rate(self, start: str = "2019-01-01", end: str = "2024-01-01") -> pd.Series:
        """Effective federal funds rate, monthly."""
        return self._get_fred_series(FRED_SERIES["fed_funds_rate"], "fed_funds_rate", start, end)

    def get_cpi(self, start: str = "2019-01-01", end: str = "2024-01-01") -> pd.Series:
        """CPI for All Urban Consumers, monthly, seasonally adjusted."""
        return self._get_fred_series(FRED_SERIES["cpi"], "cpi", start, end)

    def get_cpi_yoy(self, start: str = "2019-01-01", end: str = "2024-01-01") -> pd.Series:
        """Year-over-year CPI inflation rate, derived from the CPI level series."""
        # Pull extra lookback so the YoY calc has enough history at `start`.
        lookback_start = (pd.Timestamp(start) - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
        cpi = self.get_cpi(lookback_start, end)
        yoy = cpi.pct_change(12) * 100
        return yoy.loc[start:end].rename("cpi_yoy")

    def get_vix(self, start: str = "2019-01-01", end: str = "2024-01-01") -> pd.Series:
        """CBOE Volatility Index (VIX), daily close."""
        if _HAS_YFINANCE:
            try:
                raw = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=False)
                if raw is not None and not raw.empty:
                    col = "Adj Close" if "Adj Close" in raw.columns else "Close"
                    series = raw[col]
                    series.name = "vix"
                    return series
            except Exception as exc:  # pragma: no cover
                logger.warning("VIX fetch failed (%s) — using mock fallback", exc)
        return _mock_series("vix", start, end)

    def get_macro_frame(self, start: str = "2019-01-01", end: str = "2024-01-01") -> pd.DataFrame:
        """Combined monthly macro frame: fed_funds_rate, cpi_yoy, vix (resampled to month-end)."""
        fed = self.get_fed_funds_rate(start, end)
        cpi_yoy = self.get_cpi_yoy(start, end)
        vix = self.get_vix(start, end)
        vix_monthly = vix.resample("MS").mean()
        vix_monthly.name = "vix"

        df = pd.concat([fed.rename("fed_funds_rate"), cpi_yoy, vix_monthly], axis=1)
        return df.ffill().dropna(how="all")
