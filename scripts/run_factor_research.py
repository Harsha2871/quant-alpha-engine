#!/usr/bin/env python
"""
run_factor_research.py — CLI to analyze all (or selected) price-based factors
for a universe of tickers: computes IC, ICIR, and decay statistics and
prints a research summary table.

Usage
-----
    python scripts/run_factor_research.py --tickers AAPL MSFT GOOGL --start 2019-01-01 --end 2024-01-01
    python scripts/run_factor_research.py --universe-default --horizons 1 5 10 21
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from alpha_engine.config import DEFAULT_UNIVERSE
from alpha_engine.data.price_loader import PriceLoader
from alpha_engine.factors.factor_registry import DEFAULT_REGISTRY
from alpha_engine.research.ic_analysis import multi_horizon_ic
from alpha_engine.research.factor_decay import decay_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run factor IC research across a ticker universe.")
    parser.add_argument("--tickers", nargs="+", default=None, help="Explicit ticker list")
    parser.add_argument("--universe-default", action="store_true", help="Use the built-in default universe")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10, 21])
    parser.add_argument("--output", default=None, help="Optional path to write JSON results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = args.tickers if args.tickers else list(DEFAULT_UNIVERSE)
    if not args.tickers and not args.universe_default:
        print(f"No --tickers given; defaulting to built-in universe: {tickers}")

    print(f"Loading prices for {len(tickers)} tickers from {args.start} to {args.end}...")
    loader = PriceLoader()
    prices = loader.get_close_prices(tickers, start=args.start, end=args.end)
    print(f"Loaded price matrix: {prices.shape[0]} dates x {prices.shape[1]} tickers")

    price_only_factors = [
        name for name in DEFAULT_REGISTRY.list_names()
        if DEFAULT_REGISTRY.get(name).requires == ("prices",)
    ]

    all_results = {}
    print("\n" + "=" * 78)
    print(f"{'Factor':28s} {'Horizon':>8s} {'Mean IC':>10s} {'ICIR':>10s} {'Hit Rate':>10s}")
    print("=" * 78)

    for factor_name in price_only_factors:
        factor_panel = DEFAULT_REGISTRY.compute(factor_name, prices=prices)
        ic_table = multi_horizon_ic(factor_panel, prices, horizons=tuple(args.horizons))
        decay = decay_summary(factor_panel, prices, max_horizon=max(args.horizons))

        for horizon, row in ic_table.iterrows():
            print(f"{factor_name:28s} {horizon:>8d} {row['mean_ic']:>10.4f} {row['icir']:>10.4f} {row['hit_rate']:>10.2%}")

        all_results[factor_name] = {
            "ic_table": ic_table.to_dict(orient="index"),
            "decay": decay,
        }
        print("-" * 78)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(all_results, indent=2, default=str))
        print(f"\nSaved full research results to {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
