"""
main.py — FastAPI application exposing the alpha engine over HTTP.

Endpoints
---------
POST /compute-factors  body: {tickers, factors} -> {factor_scores}
POST /backtest          body: {tickers, start, end} -> {metrics, equity_curve}
GET  /factor-stats      params: factor_name -> {IC, ICIR, turnover, correlation_to_others}
GET  /health            -> {status: "ok"}

Run locally with:
    uvicorn alpha_engine.api.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from alpha_engine.config import DEFAULT_UNIVERSE, BENCHMARK_TICKER, RISK_FREE_RATE_ANNUAL
from alpha_engine.data.price_loader import PriceLoader
from alpha_engine.data.fundamental_loader import FundamentalLoader
from alpha_engine.data.macro_loader import MacroLoader
from alpha_engine.factors.factor_registry import DEFAULT_REGISTRY, list_factors
from alpha_engine.research.ic_analysis import multi_horizon_ic
from alpha_engine.research.correlation import factor_correlation_matrix
from alpha_engine.backtest.backtest_engine import run_walk_forward_backtest

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Quant Alpha Engine API",
    description="Factor research, IC analysis, and walk-forward backtesting over HTTP.",
    version="0.1.0",
)

_price_loader = PriceLoader()
_fundamental_loader = FundamentalLoader()
_macro_loader = MacroLoader()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ComputeFactorsRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=1, description="Universe of ticker symbols")
    factors: list[str] = Field(..., min_length=1, description="Factor names, see /factor-stats or GET /factors")
    start: str = Field("2022-01-01", description="Start date, YYYY-MM-DD")
    end: str = Field("2024-01-01", description="End date, YYYY-MM-DD")


class ComputeFactorsResponse(BaseModel):
    factor_scores: dict


class BacktestRequest(BaseModel):
    tickers: list[str] = Field(default_factory=lambda: list(DEFAULT_UNIVERSE))
    start: str = Field("2019-01-01", description="Start date, YYYY-MM-DD")
    end: str = Field("2024-01-01", description="End date, YYYY-MM-DD")
    factor: str = Field("momentum_12_1", description="Factor to use for portfolio construction")
    top_quantile: float = Field(0.20, gt=0, lt=1)
    bottom_quantile: float = Field(0.20, gt=0, lt=1)
    transaction_cost_bps: float = Field(10.0, ge=0)


class BacktestResponse(BaseModel):
    metrics: dict
    equity_curve: list[dict]


class FactorStatsResponse(BaseModel):
    factor_name: str
    ic: dict
    turnover: Optional[float]
    correlation_to_others: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    """Simple liveness check."""
    return {"status": "ok"}


@app.get("/factors")
def factors() -> dict:
    """Lists all registered factor names and their metadata."""
    return {
        name: {"category": spec.category, "description": spec.description}
        for name, spec in DEFAULT_REGISTRY.all().items()
    }


@app.post("/compute-factors", response_model=ComputeFactorsResponse)
def compute_factors(req: ComputeFactorsRequest) -> ComputeFactorsResponse:
    """Computes the requested factors for the given ticker universe."""
    unknown = [f for f in req.factors if f not in list_factors()]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown factors: {unknown}. Available: {list_factors()}")

    prices = _price_loader.get_close_prices(req.tickers, start=req.start, end=req.end)
    fundamentals = _fundamental_loader.get_fundamentals_frame(req.tickers)
    macro_frame = _macro_loader.get_macro_frame(start=req.start, end=req.end)

    fed_daily = macro_frame["fed_funds_rate"].reindex(prices.index, method="ffill") if "fed_funds_rate" in macro_frame else None
    vix_daily = _macro_loader.get_vix(start=req.start, end=req.end).reindex(prices.index, method="ffill")

    results = {}
    for factor_name in req.factors:
        spec = DEFAULT_REGISTRY.get(factor_name)
        try:
            computed = DEFAULT_REGISTRY.compute(
                factor_name,
                prices=prices if "prices" in spec.requires else None,
                fundamentals=fundamentals if "fundamentals" in spec.requires else None,
                fed_funds_rate=fed_daily if "fed_funds_rate" in spec.requires else None,
                vix=vix_daily if "vix" in spec.requires else None,
            )
        except Exception as exc:
            logger.exception("Failed computing factor %s", factor_name)
            raise HTTPException(status_code=500, detail=f"Failed computing factor '{factor_name}': {exc}")

        # Reduce to latest cross-section for API response (per-ticker snapshot).
        if hasattr(computed, "columns"):  # DataFrame: date x ticker
            latest = computed.iloc[-1].dropna()
            results[factor_name] = latest.to_dict()
        else:  # Series: ticker -> value
            results[factor_name] = computed.dropna().to_dict()

    return ComputeFactorsResponse(factor_scores=results)


@app.post("/backtest", response_model=BacktestResponse)
def backtest(req: BacktestRequest) -> BacktestResponse:
    """Runs a walk-forward long/short backtest using the requested factor."""
    if req.factor not in list_factors():
        raise HTTPException(status_code=400, detail=f"Unknown factor '{req.factor}'. Available: {list_factors()}")

    prices = _price_loader.get_close_prices(req.tickers, start=req.start, end=req.end)
    benchmark_prices = _price_loader.get_close_prices([BENCHMARK_TICKER], start=req.start, end=req.end)[BENCHMARK_TICKER]

    spec = DEFAULT_REGISTRY.get(req.factor)
    if spec.requires != ("prices",):
        raise HTTPException(status_code=400, detail=f"Factor '{req.factor}' needs extra inputs not supported via /backtest yet")

    factor_panel = DEFAULT_REGISTRY.compute(req.factor, prices=prices)

    result = run_walk_forward_backtest(
        prices=prices,
        factor_panel=factor_panel,
        top_quantile=req.top_quantile,
        bottom_quantile=req.bottom_quantile,
        transaction_cost_bps=req.transaction_cost_bps,
        benchmark_prices=benchmark_prices,
        risk_free_rate_annual=RISK_FREE_RATE_ANNUAL,
    )

    equity_curve = [
        {"date": str(date.date()), "value": float(value)}
        for date, value in result.equity_curve.items()
    ]

    return BacktestResponse(metrics=result.performance.to_dict(), equity_curve=equity_curve)


@app.get("/factor-stats", response_model=FactorStatsResponse)
def factor_stats(
    factor_name: str = Query(..., description="Registered factor name"),
    tickers: str = Query(",".join(DEFAULT_UNIVERSE), description="Comma-separated tickers"),
    start: str = Query("2019-01-01"),
    end: str = Query("2024-01-01"),
) -> FactorStatsResponse:
    """Returns IC/ICIR, average turnover, and correlation-to-other-factors for one factor."""
    if factor_name not in list_factors():
        raise HTTPException(status_code=400, detail=f"Unknown factor '{factor_name}'. Available: {list_factors()}")

    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    prices = _price_loader.get_close_prices(ticker_list, start=start, end=end)

    spec = DEFAULT_REGISTRY.get(factor_name)
    if spec.requires != ("prices",):
        raise HTTPException(status_code=400, detail=f"Factor '{factor_name}' needs extra inputs not supported via /factor-stats yet")

    factor_panel = DEFAULT_REGISTRY.compute(factor_name, prices=prices)
    ic_table = multi_horizon_ic(factor_panel, prices, horizons=(1, 5, 10, 21))

    # Correlation to a couple of other price-only factors for context.
    other_price_factors = [n for n in DEFAULT_REGISTRY.list_names()
                            if DEFAULT_REGISTRY.get(n).requires == ("prices",) and n != factor_name]
    corr_scores = {}
    if other_price_factors:
        latest_date = factor_panel.dropna(how="all").index[-1] if not factor_panel.dropna(how="all").empty else None
        if latest_date is not None:
            base_scores = {factor_name: factor_panel.loc[latest_date]}
            for other in other_price_factors[:3]:
                other_panel = DEFAULT_REGISTRY.compute(other, prices=prices)
                if latest_date in other_panel.index:
                    base_scores[other] = other_panel.loc[latest_date]
            if len(base_scores) > 1:
                corr_matrix = factor_correlation_matrix(base_scores)
                corr_scores = corr_matrix[factor_name].drop(factor_name).dropna().to_dict()

    return FactorStatsResponse(
        factor_name=factor_name,
        ic=ic_table.to_dict(orient="index"),
        turnover=None,  # computed only in a full backtest run; see /backtest
        correlation_to_others=corr_scores,
    )

