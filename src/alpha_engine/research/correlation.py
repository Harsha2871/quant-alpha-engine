"""
correlation.py — Factor correlation and multicollinearity diagnostics.

Highly correlated factors add redundancy (and estimation risk) rather than
diversification when combined in a multi-factor model. This module provides
a correlation matrix across factors plus Variance Inflation Factor (VIF)
scores to flag problematic collinearity before factors are fed into the ML
combination model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def factor_correlation_matrix(factor_scores: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Cross-sectional Pearson correlation matrix between factor scores at a
    single point in time (or pooled across time if the inputs are stacked
    panel series).

    Parameters
    ----------
    factor_scores : dict[str, pd.Series]
        Mapping of factor_name -> Series (index=ticker or (date,ticker)).

    Returns
    -------
    pd.DataFrame
        Square correlation matrix, index/columns = factor names.
    """
    df = pd.DataFrame(factor_scores)
    return df.corr(method="pearson")


def compute_vif(factor_scores: dict[str, pd.Series]) -> pd.Series:
    """
    Variance Inflation Factor for each factor: VIF_i = 1 / (1 - R_i^2), where
    R_i^2 is the R-squared from regressing factor i on all other factors.
    VIF > 5 (some practitioners use >10) is a common rule-of-thumb threshold
    for problematic multicollinearity.

    Parameters
    ----------
    factor_scores : dict[str, pd.Series]
        Mapping of factor_name -> Series (aligned index across factors).

    Returns
    -------
    pd.Series
        VIF per factor name. NaN if a factor has zero variance or too few
        overlapping observations with the others.
    """
    df = pd.DataFrame(factor_scores).dropna()
    if df.shape[0] < df.shape[1] + 2 or df.shape[1] < 2:
        return pd.Series({name: np.nan for name in df.columns})

    vif = {}
    for col in df.columns:
        y = df[col].values
        X = df.drop(columns=[col]).values
        X_design = np.column_stack([np.ones(len(X)), X])

        try:
            coefs, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
            y_pred = X_design @ coefs
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            r_squared = min(max(r_squared, 0.0), 0.999999)  # guard against division blowup
            vif[col] = 1.0 / (1.0 - r_squared)
        except np.linalg.LinAlgError:
            vif[col] = np.nan

    return pd.Series(vif, name="vif")


def flag_high_collinearity(vif_series: pd.Series, threshold: float = 5.0) -> list[str]:
    """Returns factor names whose VIF exceeds `threshold`."""
    return sorted(vif_series[vif_series > threshold].index.tolist())
