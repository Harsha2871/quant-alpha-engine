# Reproducibility Notes

This project separates empirical runs from demo runs.

## Empirical Run

Use yfinance-backed price data:

```bash
python scripts/run_factor_research.py --universe-default --start 2019-01-01 --end 2024-01-01 --output results/latest_research_results.json
python scripts/run_backtest.py --factor momentum_12_1 --start 2019-01-01 --end 2024-01-01 --cost-bps 10 --output results/latest_backtest_results.json
```

The output JSON includes a `data_source` block. For empirical runs, `data_source.synthetic` should be `false`.

## Demo Run

Use this only to exercise the pipeline without network access:

```bash
python scripts/run_factor_research.py --universe-default --allow-synthetic --output results/latest_research_results.json
python scripts/run_backtest.py --factor momentum_12_1 --allow-synthetic --output results/latest_backtest_results.json
```

Demo output is labeled with `data_source.synthetic: true`. Do not use demo-mode metrics in a resume bullet or performance claim.

## Default Configuration

- Universe: see `DEFAULT_UNIVERSE` in `src/alpha_engine/config.py`.
- Benchmark: `SPY`.
- Rebalance frequency: month end.
- Portfolio: long top quintile, short bottom quintile, equal-weighted by leg.
- Transaction cost: 10 bps one-way by default.
- Factor example: `momentum_12_1`.

## Interpretation

Backtest metrics are sensitive to universe choice, rebalance timing, missing data, transaction costs, and data vendor adjustments. Treat the results as a reproducible research exercise rather than evidence of a deployable strategy.
