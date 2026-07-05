#!/usr/bin/env python
"""
plot_performance.py — CLI to plot the equity curve and factor exposures from
a saved backtest results JSON (as produced by `run_backtest.py`), using both
matplotlib (static PNG) and plotly (interactive HTML).

Usage
-----
    python scripts/plot_performance.py --input results/sample_backtest_results.json
    python scripts/plot_performance.py --input results/latest_backtest_results.json --output-dir results/plots
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:  # pragma: no cover
    _HAS_PLOTLY = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot equity curve from backtest results JSON.")
    parser.add_argument("--input", required=True, help="Path to backtest results JSON")
    parser.add_argument("--output-dir", default="results/plots")
    return parser.parse_args()


def load_equity_curve(results: dict) -> pd.Series:
    curve = results.get("equity_curve", {})
    series = pd.Series({pd.Timestamp(k): v for k, v in curve.items()}).sort_index()
    return series


def plot_matplotlib(equity_curve: pd.Series, metrics: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity_curve.index, equity_curve.values, label="Strategy", color="#1f77b4", linewidth=1.5)
    ax.set_title("Alpha Engine — Cumulative Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Growth of $1")
    ax.grid(alpha=0.3)

    subtitle_parts = []
    if "sharpe" in metrics:
        subtitle_parts.append(f"Sharpe: {metrics['sharpe']:.2f}")
    if "max_drawdown" in metrics:
        subtitle_parts.append(f"Max DD: {metrics['max_drawdown']:.1%}")
    if "annualized_return" in metrics:
        subtitle_parts.append(f"Ann. Return: {metrics['annualized_return']:.1%}")
    if subtitle_parts:
        ax.text(0.01, 0.98, " | ".join(subtitle_parts), transform=ax.transAxes,
                verticalalignment="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved matplotlib chart to {output_path}")


def plot_plotly(equity_curve: pd.Series, output_path: Path) -> None:
    if not _HAS_PLOTLY:
        print("plotly not installed — skipping interactive HTML chart")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, mode="lines", name="Strategy"))
    fig.update_layout(
        title="Alpha Engine — Cumulative Equity Curve (Interactive)",
        xaxis_title="Date",
        yaxis_title="Cumulative Growth of $1",
        template="plotly_white",
    )
    fig.write_html(str(output_path))
    print(f"Saved interactive plotly chart to {output_path}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = json.loads(input_path.read_text())
    metrics = results.get("performance", results.get("metrics", {}))
    equity_curve = load_equity_curve(results)

    if equity_curve.empty:
        raise SystemExit(f"No equity_curve data found in {input_path}")

    plot_matplotlib(equity_curve, metrics, output_dir / "equity_curve.png")
    plot_plotly(equity_curve, output_dir / "equity_curve.html")


if __name__ == "__main__":
    main()
