"""
Global configuration for the alpha engine.

Centralizes universe definitions, backtest parameters, cache paths, and
environment-driven settings (e.g. FRED API key). Keeping configuration in one
module makes it trivial to swap universes or tune backtest assumptions
without touching business logic.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / ".cache"
RESULTS_DIR = PROJECT_ROOT / "results"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Universe — a small, liquid S&P 500 subset used across examples/tests.
# ---------------------------------------------------------------------------
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JPM", "V", "UNH",
    "HD", "PG", "MA", "XOM", "AVGO",
    "COST", "MRK", "ABBV", "PEP", "KO",
]

BENCHMARK_TICKER = "SPY"

# ---------------------------------------------------------------------------
# Backtest defaults
# ---------------------------------------------------------------------------
TRANSACTION_COST_BPS = 10  # 10 basis points per trade (one-way)
TOP_QUANTILE = 0.20        # long top 20%
BOTTOM_QUANTILE = 0.20     # short bottom 20%
REBALANCE_FREQ = "ME"      # monthly rebalance (pandas month-end offset alias)
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE_ANNUAL = 0.02

# ---------------------------------------------------------------------------
# External APIs
# ---------------------------------------------------------------------------
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")


@dataclass
class BacktestConfig:
    """Container for parameters controlling a single backtest run."""

    universe: list[str] = field(default_factory=lambda: list(DEFAULT_UNIVERSE))
    start: str = "2019-01-01"
    end: str = "2024-01-01"
    top_quantile: float = TOP_QUANTILE
    bottom_quantile: float = BOTTOM_QUANTILE
    transaction_cost_bps: float = TRANSACTION_COST_BPS
    rebalance_freq: str = REBALANCE_FREQ
    benchmark: str = BENCHMARK_TICKER
