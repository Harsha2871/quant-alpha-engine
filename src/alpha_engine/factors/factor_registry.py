"""
factor_registry.py — Registry pattern for alpha factors.

Allows factors to be looked up by name (string) so that scripts, the API,
and the ML factor-combination model can dynamically enumerate and compute
"all factors" without hardcoding imports everywhere. New factors are added
by registering them with `@register_factor(...)` or via `register()`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from alpha_engine.factors import momentum, value, quality, macro


@dataclass
class FactorSpec:
    """Metadata + callable for a single registered factor."""

    name: str
    category: str  # 'momentum' | 'value' | 'quality' | 'macro'
    func: Callable[..., pd.DataFrame | pd.Series]
    description: str
    requires: tuple[str, ...] = ("prices",)  # inputs the func needs, e.g. 'prices', 'fundamentals', 'macro'


class FactorRegistry:
    """Central registry mapping factor name -> FactorSpec."""

    def __init__(self):
        self._factors: dict[str, FactorSpec] = {}

    def register(self, spec: FactorSpec) -> None:
        if spec.name in self._factors:
            raise ValueError(f"Factor '{spec.name}' is already registered")
        self._factors[spec.name] = spec

    def get(self, name: str) -> FactorSpec:
        if name not in self._factors:
            raise KeyError(f"Unknown factor '{name}'. Available: {self.list_names()}")
        return self._factors[name]

    def list_names(self) -> list[str]:
        return sorted(self._factors.keys())

    def list_by_category(self, category: str) -> list[str]:
        return sorted(name for name, spec in self._factors.items() if spec.category == category)

    def all(self) -> dict[str, FactorSpec]:
        return dict(self._factors)

    def compute(
        self,
        name: str,
        prices: Optional[pd.DataFrame] = None,
        fundamentals: Optional[pd.DataFrame] = None,
        fed_funds_rate: Optional[pd.Series] = None,
        vix: Optional[pd.Series] = None,
    ):
        """Dispatches to the correct factor function based on its `requires` inputs."""
        spec = self.get(name)
        kwargs = {}
        if "prices" in spec.requires:
            if prices is None:
                raise ValueError(f"Factor '{name}' requires `prices`")
            kwargs["prices"] = prices
        if "fundamentals" in spec.requires:
            if fundamentals is None:
                raise ValueError(f"Factor '{name}' requires `fundamentals`")
            kwargs["fundamentals"] = fundamentals
        if "fed_funds_rate" in spec.requires:
            if fed_funds_rate is None:
                raise ValueError(f"Factor '{name}' requires `fed_funds_rate`")
            kwargs["fed_funds_rate"] = fed_funds_rate
        if "vix" in spec.requires:
            if vix is None:
                raise ValueError(f"Factor '{name}' requires `vix`")
            kwargs["vix"] = vix
        return spec.func(**kwargs)


def _build_default_registry() -> FactorRegistry:
    registry = FactorRegistry()

    registry.register(FactorSpec(
        name="momentum_12_1",
        category="momentum",
        func=momentum.compute_12_1_momentum,
        description="12-month return excluding most recent month (Jegadeesh & Titman, 1993).",
        requires=("prices",),
    ))
    registry.register(FactorSpec(
        name="reversal_1m",
        category="momentum",
        func=momentum.compute_1m_reversal,
        description="Negative 1-month return; short-term contrarian reversal signal.",
        requires=("prices",),
    ))
    registry.register(FactorSpec(
        name="momentum_3m",
        category="momentum",
        func=momentum.compute_3m_momentum,
        description="3-month (63 trading day) raw momentum, no skip-month.",
        requires=("prices",),
    ))
    registry.register(FactorSpec(
        name="high_52w_proximity",
        category="momentum",
        func=momentum.compute_52w_high_proximity,
        description="Price proximity to trailing 52-week high (George & Hwang, 2004).",
        requires=("prices",),
    ))
    registry.register(FactorSpec(
        name="value_pe",
        category="value",
        func=value.compute_pe_factor,
        description="Earnings yield (1 / trailing P/E). Higher = cheaper.",
        requires=("fundamentals",),
    ))
    registry.register(FactorSpec(
        name="value_pb",
        category="value",
        func=value.compute_pb_factor,
        description="Book-to-price (1 / P/B). Fama-French HML style value signal.",
        requires=("fundamentals",),
    ))
    registry.register(FactorSpec(
        name="value_composite",
        category="value",
        func=value.compute_composite_value,
        description="Z-scored blend of earnings yield and book-to-price.",
        requires=("fundamentals",),
    ))
    registry.register(FactorSpec(
        name="quality_roe",
        category="quality",
        func=quality.compute_roe_factor,
        description="Return on equity (Novy-Marx profitability premium).",
        requires=("fundamentals",),
    ))
    registry.register(FactorSpec(
        name="quality_earnings_stability",
        category="quality",
        func=quality.compute_earnings_stability,
        description="Earnings stability score — the 'safety' pillar of quality investing.",
        requires=("fundamentals",),
    ))
    registry.register(FactorSpec(
        name="quality_composite",
        category="quality",
        func=quality.compute_composite_quality,
        description="Z-scored blend of ROE, earnings stability, and leverage safety (QMJ-style).",
        requires=("fundamentals",),
    ))
    registry.register(FactorSpec(
        name="macro_rate_sensitivity",
        category="macro",
        func=macro.compute_rate_sensitivity,
        description="Rolling beta of stock returns to fed funds rate changes.",
        requires=("prices", "fed_funds_rate"),
    ))
    registry.register(FactorSpec(
        name="macro_vix_regime",
        category="macro",
        func=macro.compute_vix_regime_factor,
        description="Regime-conditioned realized-volatility tilt based on VIX level.",
        requires=("prices", "vix"),
    ))

    return registry


# Module-level singleton, mirroring the common "registry" pattern used across
# the codebase (scripts/API import this directly).
DEFAULT_REGISTRY = _build_default_registry()


def list_factors() -> list[str]:
    """Returns all registered factor names."""
    return DEFAULT_REGISTRY.list_names()


def load_factor(name: str) -> FactorSpec:
    """Returns the FactorSpec for a given factor name."""
    return DEFAULT_REGISTRY.get(name)


def compute_factor(name: str, **kwargs):
    """Computes a named factor, dispatching required inputs from kwargs."""
    return DEFAULT_REGISTRY.compute(name, **kwargs)
