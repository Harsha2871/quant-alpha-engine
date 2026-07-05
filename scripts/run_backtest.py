#!/usr/bin/env python
"""
run_backtest.py — CLI to run a walk-forward, cross-sectional long/short
backtest for a chosen factor against a benchmark.

Usage
-----
    python scripts/run_backtest.py --factor momentum_12_1 --start 2019-01-01 --end 2024-01-01
    python scripts/run_backtest.py --factor value_composite --tickers AAPL MSFT JPM --top-q 0.3 --bottom-q 0.3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alpha_engine.config import DEFAULT_UNIVERSE, BENCHMARK_TICKER, RESULTS_DIR
from alpha_engine.data.price_loader import PriceLoader
from alpha_engine.factors.factor_registry import DEFAULT_REGISTRY
from alpha_engine.backtest.backtest_engine import run_walk_forward_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a walk-forward cross-sectional backtest.")
    parser.add_argument("--factor", default="momentum_12_1", help="Registered factor name to trade on")
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--top-q", type=float, default=0.20)
    parser.add_argument("--bottom-q", type=float, default=0.20)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--output", default=str(RESULTS_DIR / "latest_backtest_results.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = args.tickers if args.tickers else list(DEFAULT_UNIVERSE)

    spec = DEFAULT_REGISTRY.get(args.factor)
    if spec.requires != ("prices",):
        raise SystemExit(
            f"Factor '{args.factor}' requires extra inputs {spec.requires}; "
            f"this CLI currently only supports price-only factors."
        )

    print(f"Loading prices for {len(tickers)} tickers + benchmark {BENCHMARK_TICKER}...")
    loader = PriceLoader()
    prices = loader.get_close_prices(tickers, start=args.start, end=args.end)
    benchmark_prices = loader.get_close_prices([BENCHMARK_TICKER], start=args.start, end=args.end)[BENCHMARK_TICKER]

    print(f"Computing factor '{args.factor}'...")
    factor_panel = DEFAULT_REGISTRY.compute(args.factor, prices=prices)

    print("Running walk-forward backtest...")
    result = run_walk_forward_backtest(
        prices=prices,
        factor_panel=factor_panel,
        top_quantile=args.top_q,
        bottom_quantile=args.bottom_q,
        transaction_cost_bps=args.cost_bps,
        benchmark_prices=benchmark_prices,
    )

    perf = result.performance.to_dict()
    print("\n" + "=" * 50)
    print(f"Backtest results for factor: {args.factor}")
    print("=" * 50)
    for key, value in perf.items():
        print(f"{key:28s}: {value:.4f}" if isinstance(value, float) else f"{key:28s}: {value}")
    print(f"Average monthly turnover     : {result.turnover_history.mean():.4f}")
    print(f"Number of rebalances         : {len(result.weights_history)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_summary_dict(), indent=2, default=str))
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
