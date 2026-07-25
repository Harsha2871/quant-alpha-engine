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

## Usage

Run the default research workflow with real yfinance data:

```bash
make research
make backtest
make plot
```

Run the same workflow offline with deterministic demo data:

```bash
make research-demo
make backtest-demo
make plot
```

Run a smaller custom universe:

```bash
python scripts/run_factor_research.py \
  --tickers AAPL MSFT NVDA JPM XOM \
  --start 2020-01-01 \
  --end 2024-01-01 \
  --horizons 1 5 21 \
  --output results/custom_research_results.json

python scripts/run_backtest.py \
  --tickers AAPL MSFT NVDA JPM XOM \
  --factor momentum_12_1 \
  --start 2020-01-01 \
  --end 2024-01-01 \
  --cost-bps 10 \
  --output results/custom_backtest_results.json
```

Inspect the generated output:

```bash
python -m json.tool results/latest_backtest_results.json | head -80
python -m json.tool results/latest_research_results.json | head -80
```

Start the local API:

```bash
make api
```

Then call it:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/backtest \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "MSFT", "NVDA", "JPM", "XOM"],
    "factor": "momentum_12_1",
    "start": "2020-01-01",
    "end": "2024-01-01",
    "transaction_cost_bps": 10
  }'
```

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

## Developer Notes

The implementation follows a simple pipeline:

```text
data loaders -> factor registry -> research/backtest modules -> scripts/API
```

The most useful entry points are the scripts in `scripts/`, the FastAPI app, and the tests. The source tree is organized by responsibility: data loading, factor definitions, research metrics, backtesting, ML experiments, and API handlers.
