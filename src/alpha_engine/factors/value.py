"""
value.py — Fundamentals-based value factors.

References
----------
- Fama & French (1992, 1993), the "HML" (High-Minus-Low book-to-market)
  value factor.
- Basu (1977) on the P/E anomaly.

Convention: value factors are oriented so that HIGHER scores = cheaper /
more attractive to a value investor. Since P/E and P/B are "cheaper is
lower", we invert them (1/PE, 1/PB) so all factors point the same direction
and can be compared/combined consistently.
"""
from __future__ import annotations

import pandas as pd


def compute_pe_factor(pe_ratios: pd.Series) -> pd.Series:
    """
    Earnings yield factor = 1 / trailing P/E. Higher earnings yield (cheaper
    stock relative to earnings) scores higher.

    Parameters
    ----------
    pe_ratios : pd.Series
        Index = ticker, values = trailing P/E ratio.

    Returns
    -------
    pd.Series
        Earnings yield per ticker. Non-positive or missing P/E is treated as
        NaN (cannot be meaningfully ranked as "cheap").
    """
    pe = pe_ratios.copy().astype(float)
    pe = pe.where(pe > 0)
    return (1.0 / pe).rename("pe_factor")


def compute_pb_factor(pb_ratios: pd.Series) -> pd.Series:
    """
    Book-to-market-style factor = 1 / price-to-book. Higher score = cheaper
    relative to book value (classic Fama-French HML orientation).

    Parameters
    ----------
    pb_ratios : pd.Series
        Index = ticker, values = price-to-book ratio.

    Returns
    -------
    pd.Series
        Inverse P/B (book-to-price) per ticker.
    """
    pb = pb_ratios.copy().astype(float)
    pb = pb.where(pb > 0)
    return (1.0 / pb).rename("pb_factor")


def compute_composite_value(fundamentals: pd.DataFrame) -> pd.Series:
    """
    Combines the P/E and P/B value signals into a single composite value
    score via cross-sectional z-scoring and averaging — a standard technique
    to blend heterogeneous factor units (Asness, Frazzini & Pedersen 2019,
    "Quality Minus Junk" methodology uses a similar z-score-and-average
    approach for combining sub-signals).

    Parameters
    ----------
    fundamentals : pd.DataFrame
        Must contain columns 'trailing_pe' and 'price_to_book', index=ticker.

    Returns
    -------
    pd.Series
        Composite value score per ticker (higher = more attractively valued).
    """
    pe_factor = compute_pe_factor(fundamentals["trailing_pe"])
    pb_factor = compute_pb_factor(fundamentals["price_to_book"])

    def _zscore(s: pd.Series) -> pd.Series:
        return (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else s * 0

    composite = pd.concat([_zscore(pe_factor), _zscore(pb_factor)], axis=1).mean(axis=1)
    return composite.rename("composite_value")
