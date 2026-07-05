"""
quality.py — Profitability / quality factors.

References
----------
- Novy-Marx (2013), "The Other Side of Value: The Gross Profitability
  Premium".
- Asness, Frazzini & Pedersen (2019), "Quality Minus Junk", which combines
  profitability, growth, safety, and payout signals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_roe_factor(fundamentals: pd.DataFrame) -> pd.Series:
    """
    Return-on-equity quality factor. Higher ROE indicates the firm generates
    more profit per dollar of shareholder equity — a core "quality" trait.

    Parameters
    ----------
    fundamentals : pd.DataFrame
        Must contain a 'return_on_equity' column, index=ticker.

    Returns
    -------
    pd.Series
        ROE per ticker, renamed 'roe_factor'.
    """
    return fundamentals["return_on_equity"].astype(float).rename("roe_factor")


def compute_earnings_stability(fundamentals: pd.DataFrame) -> pd.Series:
    """
    Earnings stability factor — a proxy for the "safety" leg of quality
    investing. Higher values indicate more consistent, less volatile
    earnings (firms with stable earnings tend to command a quality premium
    and lower discount rates).

    Parameters
    ----------
    fundamentals : pd.DataFrame
        Must contain an 'earnings_stability' column, index=ticker, valued in
        [0, 1] where 1.0 = perfectly stable earnings history.

    Returns
    -------
    pd.Series
        Earnings stability score per ticker.
    """
    return fundamentals["earnings_stability"].astype(float).rename("earnings_stability_factor")


def compute_leverage_penalty(fundamentals: pd.DataFrame) -> pd.Series:
    """
    Safety/leverage factor — penalizes high debt-to-equity firms, in the
    spirit of the "safety" pillar of Asness et al.'s Quality Minus Junk.
    Higher score = safer (lower leverage).

    Parameters
    ----------
    fundamentals : pd.DataFrame
        Must contain a 'debt_to_equity' column, index=ticker.

    Returns
    -------
    pd.Series
        1 / (1 + debt_to_equity), so more leverage -> lower score.
    """
    dte = fundamentals["debt_to_equity"].astype(float).clip(lower=0)
    return (1.0 / (1.0 + dte)).rename("leverage_safety_factor")


def compute_composite_quality(fundamentals: pd.DataFrame) -> pd.Series:
    """
    Composite quality score blending profitability (ROE), earnings
    stability, and leverage safety via cross-sectional z-scores.

    Parameters
    ----------
    fundamentals : pd.DataFrame
        Must contain 'return_on_equity', 'earnings_stability',
        'debt_to_equity' columns, index=ticker.

    Returns
    -------
    pd.Series
        Composite quality score per ticker.
    """
    roe = compute_roe_factor(fundamentals)
    stability = compute_earnings_stability(fundamentals)
    leverage = compute_leverage_penalty(fundamentals)

    def _zscore(s: pd.Series) -> pd.Series:
        std = s.std(ddof=0)
        return (s - s.mean()) / std if std and not np.isnan(std) and std > 0 else s * 0

    composite = pd.concat([_zscore(roe), _zscore(stability), _zscore(leverage)], axis=1).mean(axis=1)
    return composite.rename("composite_quality")
