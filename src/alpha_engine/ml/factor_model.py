"""
factor_model.py — ML model that combines multiple alpha factor scores into a
single predicted alpha score per stock per period.

Approach: frame the problem as a classification task — predict which
forward-21-day-return quintile (0=worst, 4=best) a stock will fall into,
given its current cross-section of factor scores as features. This is a
common QLib-style formulation: turning the noisy regression problem (predict
exact return) into a more robust ranking/classification problem.

Uses LightGBM if available (faster, typically stronger on tabular factor
data), otherwise falls back to sklearn's RandomForestClassifier so the
pipeline works with only the core dependencies installed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

logger = logging.getLogger(__name__)

try:
    from lightgbm import LGBMClassifier
    _HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover
    _HAS_LIGHTGBM = False


def build_training_panel(
    factor_frames: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    n_quintiles: int = 5,
) -> pd.DataFrame:
    """
    Stacks multiple wide factor frames (date x ticker) plus forward returns
    into a single long/tidy panel suitable for scikit-learn:
    one row per (date, ticker), one column per factor, plus the target label.

    Parameters
    ----------
    factor_frames : dict[str, pd.DataFrame]
        Mapping factor_name -> wide DataFrame (index=date, columns=tickers).
    forward_returns : pd.DataFrame
        Forward returns aligned to the same date/ticker grid, used to build
        the quintile target label.
    n_quintiles : int
        Number of quantile buckets for the classification target.

    Returns
    -------
    pd.DataFrame
        Long panel with a MultiIndex (date, ticker), one column per factor
        plus 'target_quintile' and 'forward_return'.
    """
    def _stack_all(frame: pd.DataFrame) -> pd.Series:
        # `future_stack=True` (pandas >= 2.1) preserves NaN rows instead of
        # dropping them, matching the legacy `stack(dropna=False)` behavior
        # that this pipeline relies on. Fall back gracefully on older pandas.
        try:
            return frame.stack(future_stack=True)
        except TypeError:
            return frame.stack(dropna=False)

    stacked = {}
    for name, frame in factor_frames.items():
        stacked[name] = _stack_all(frame)

    panel = pd.DataFrame(stacked)
    fwd_stacked = _stack_all(forward_returns)
    panel["forward_return"] = fwd_stacked

    panel = panel.dropna(subset=["forward_return"])

    def _quintile(group: pd.Series) -> pd.Series:
        try:
            return pd.qcut(group, n_quintiles, labels=False, duplicates="drop")
        except ValueError:
            return pd.Series(np.nan, index=group.index)

    panel["target_quintile"] = (
        panel["forward_return"].groupby(level=0).transform(_quintile)
    )
    panel = panel.dropna(subset=["target_quintile"])
    panel["target_quintile"] = panel["target_quintile"].astype(int)

    return panel


@dataclass
class FactorModelResult:
    """Holds a trained model plus its walk-forward test performance."""

    model: object
    feature_names: list[str]
    train_accuracy: float
    test_accuracy: float
    classification_report_text: str


class FactorCombinationModel:
    """
    Wraps a classifier (LightGBM if available, else RandomForest) that maps
    a stock's factor scores to a predicted forward-return quintile, which is
    then used as a combined "alpha score" (higher predicted quintile = more
    attractive).
    """

    def __init__(self, use_lightgbm: bool = True, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.feature_names: list[str] = []

        if use_lightgbm and _HAS_LIGHTGBM:
            self.model = LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=random_state,
                verbosity=-1,
            )
            self._backend = "lightgbm"
        else:
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=10,
                random_state=random_state,
                n_jobs=-1,
            )
            self._backend = "random_forest"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FactorCombinationModel":
        """Fits the classifier on factor features X and quintile labels y."""
        self.feature_names = list(X.columns)
        self.model.fit(X.values, y.values)
        return self

    def predict_alpha_score(self, X: pd.DataFrame) -> pd.Series:
        """
        Returns a continuous alpha score = expected quintile (probability-
        weighted), which is smoother and more useful for ranking than the
        raw discrete class prediction.
        """
        X_aligned = X[self.feature_names]
        proba = self.model.predict_proba(X_aligned.values)
        classes = self.model.classes_
        expected_quintile = (proba * classes).sum(axis=1)
        return pd.Series(expected_quintile, index=X.index, name="ml_alpha_score")

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[float, str]:
        """Returns (accuracy, sklearn classification_report string) on held-out data."""
        preds = self.model.predict(X_test[self.feature_names].values)
        acc = accuracy_score(y_test.values, preds)
        report = classification_report(y_test.values, preds, zero_division=0)
        return float(acc), report


def walk_forward_train_test(
    panel: pd.DataFrame,
    feature_cols: list[str],
    train_years: int = 3,
    test_years: int = 1,
) -> FactorModelResult:
    """
    Splits the panel chronologically (train on the first `train_years`,
    test on the following `test_years`) and trains a FactorCombinationModel.

    Parameters
    ----------
    panel : pd.DataFrame
        Long panel with MultiIndex (date, ticker), from `build_training_panel`.
    feature_cols : list[str]
        Column names to use as model features (factor names).
    train_years, test_years : int
        Chronological split sizes.

    Returns
    -------
    FactorModelResult
    """
    dates = panel.index.get_level_values(0)
    unique_dates = pd.Index(sorted(dates.unique()))

    if len(unique_dates) < 10:
        raise ValueError("Not enough dates in panel to perform a walk-forward split")

    start_date = unique_dates.min()
    train_end = start_date + pd.DateOffset(years=train_years)
    test_end = train_end + pd.DateOffset(years=test_years)

    train_mask = (dates >= start_date) & (dates < train_end)
    test_mask = (dates >= train_end) & (dates < test_end)

    train_panel = panel.loc[train_mask]
    test_panel = panel.loc[test_mask]

    if train_panel.empty:
        raise ValueError("Training partition is empty — check date range vs train_years")
    if test_panel.empty:
        raise ValueError(
            "chronological test partition is empty — reduce train_years/test_years "
            "or provide a longer date range. Refusing to use a random split because "
            "that would leak time information in a walk-forward evaluation."
        )

    X_train, y_train = train_panel[feature_cols].fillna(0.0), train_panel["target_quintile"]
    X_test, y_test = test_panel[feature_cols].fillna(0.0), test_panel["target_quintile"]

    model = FactorCombinationModel()
    model.fit(X_train, y_train)

    train_preds = model.model.predict(X_train.values)
    train_acc = accuracy_score(y_train.values, train_preds)
    test_acc, report = model.evaluate(X_test, y_test)

    return FactorModelResult(
        model=model,
        feature_names=feature_cols,
        train_accuracy=float(train_acc),
        test_accuracy=test_acc,
        classification_report_text=report,
    )
