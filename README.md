# Quant Alpha Engine

Research-oriented Python toolkit for testing simple cross-sectional equity factors. The project includes price/fundamental/macro data loaders, a small factor registry, IC analysis, long/short portfolio construction, transaction-cost modeling, and a FastAPI surface for experimenting with factor scores and backtests.

This is a learning/research project, not an investment system. The sample outputs are meant to demonstrate the workflow and should be treated as reproducible examples, not trading recommendations.

## What It Does

- Computes momentum, value, quality, and macro-style factor scores.
- Runs rank IC / ICIR analysis over forward-return horizons.
- Builds equal-weighted long/short quantile portfolios.
- Applies turnover-based transaction costs at rebalance time.
- Exposes factor scoring and backtest endpoints through FastAPI.
- Keeps synthetic price data opt-in so mock runs are not confused with empirical results.

## Architecture

```text
data loaders -> factor registry -> research/backtest modules -> scripts/API
```

Important modules:

- `src/alpha_engine/data/price_loader.py`: yfinance-backed OHLCV loader with explicit synthetic demo mode.
- `src/alpha_engine/factors/`: factor implementations and registry.
- `src/alpha_engine/research/`: IC, decay, and correlation analysis.
- `src/alpha_engine/backtest/`: quantile portfolio construction, walk-forward backtest, performance metrics.
- `src/alpha_engine/ml/factor_model.py`: chronological factor-combination experiment using RandomForest or LightGBM.
- `src/alpha_engine/api/main.py`: FastAPI endpoints for factors, backtests, and factor stats.

## Quickstart

```bash
make install
make test
make research
make backtest
make plot
make api
```

By default, research and backtest scripts expect real price data from `yfinance`. If yfinance is unavailable and you only want to exercise the pipeline offline, use the demo targets:

```bash
make research-demo
make backtest-demo
```

Demo mode enables deterministic synthetic prices and labels the generated JSON output with `data_source.synthetic: true`.

## Reproducing The Sample Run

The checked-in sample in `results/sample_backtest_results.json` summarizes a 20-stock example using 12-1 momentum from 2019-01-01 to 2024-01-01 with monthly rebalancing and 10 bps one-way transaction costs.

To regenerate comparable outputs with live yfinance data:

```bash
python scripts/run_factor_research.py \
  --universe-default \
  --start 2019-01-01 \
  --end 2024-01-01 \
  --output results/latest_research_results.json

python scripts/run_backtest.py \
  --factor momentum_12_1 \
  --start 2019-01-01 \
  --end 2024-01-01 \
  --cost-bps 10 \
  --output results/latest_backtest_results.json
```

See `docs/reproducibility.md` for the data assumptions and caveats behind those numbers.

## Factor Set

| Factor | Category | Notes |
|---|---|---|
| `momentum_12_1` | Momentum | 12-month return shifted by 21 trading days to skip the most recent month. |
| `reversal_1m` | Momentum | Negative 21-day return. |
| `momentum_3m` | Momentum | 63-day return. |
| `high_52w_proximity` | Momentum | Price divided by rolling 52-week high. |
| `value_pe`, `value_pb`, `value_composite` | Value | Uses available fundamentals, with synthetic fallback only for demo continuity. |
| `quality_roe`, `quality_earnings_stability`, `quality_composite` | Quality | Basic profitability/stability/leverage-style scores. |
| `macro_rate_sensitivity`, `macro_vix_regime` | Macro | Simple macro sensitivity/regime examples. |

## API

```bash
make api
```

Then open `http://localhost:8000/docs`.

Endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check. |
| `GET` | `/factors` | Registered factor metadata. |
| `POST` | `/compute-factors` | Latest factor scores for a ticker universe. |
| `POST` | `/backtest` | Walk-forward long/short factor backtest. |
| `GET` | `/factor-stats` | IC table and simple cross-factor correlation snapshot. |

## Testing

```bash
pytest -v
```

The tests cover factor math, IC analysis, portfolio construction, backtest accounting, explicit synthetic-data handling, and the chronological ML split guard.

## Known Limitations

- The default universe is a small hand-picked subset and does not address survivorship bias.
- yfinance is convenient but not point-in-time institutional market data.
- Fundamental data from `yfinance.info` is not a point-in-time historical fundamentals source.
- The transaction-cost model is a simple turnover cost, not a market-impact model.
- The ML factor-combination module is an experiment. It now refuses random fallback splits because random time splits would undermine walk-forward evaluation.

See `docs/known_limitations.md` for more detail.

## Resume Framing

Useful one-line framing:

> Built a Python factor-research engine for cross-sectional equity strategies, including factor registry, IC/ICIR analysis, long/short portfolio construction, transaction-cost modeling, and FastAPI endpoints for experimentation.

## License

MIT
