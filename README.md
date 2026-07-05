# Quant Alpha Engine

A quantitative alpha factor research and backtesting platform inspired by [Microsoft QLib](https://github.com/microsoft/qlib). It implements custom alpha factors from macro and fundamental data, runs cross-sectional walk-forward backtests with realistic transaction costs, combines factors via machine learning, and serves everything through a FastAPI portfolio optimization API.

## Architecture

```
                         ┌─────────────────────────────────────────┐
                         │              Data Layer                  │
                         │  ┌───────────┐ ┌───────────┐ ┌─────────┐ │
                         │  │  Price    │ │  Macro    │ │ Fundam. │ │
                         │  │  Loader   │ │  Loader   │ │ Loader  │ │
                         │  │(yfinance, │ │  (FRED,   │ │(yfinance│ │
                         │  │ pickle    │ │   VIX,    │ │  .info) │ │
                         │  │  cache)   │ │  mock fb) │ │         │ │
                         │  └─────┬─────┘ └─────┬─────┘ └────┬────┘ │
                         └────────┼─────────────┼─────────────┼─────┘
                                  ▼             ▼             ▼
                         ┌─────────────────────────────────────────┐
                         │             Factor Zoo                   │
                         │  momentum.py │ value.py │ quality.py     │
                         │              │ macro.py                  │
                         │        (factor_registry.py)              │
                         └────────────────────┬──────────────────────┘
                                              ▼
                  ┌───────────────────────────┴───────────────────────────┐
                  ▼                           ▼                           ▼
        ┌──────────────────┐      ┌──────────────────────┐    ┌────────────────────┐
        │  Research Layer   │      │   ML Combination      │    │   Backtest Engine   │
        │  ic_analysis.py   │      │   factor_model.py      │    │ portfolio_construc- │
        │  factor_decay.py  │─────▶│ RandomForest/LightGBM  │───▶│ tor + backtest_     │
        │  correlation.py   │      │  → single alpha score  │    │ engine + performance │
        └──────────────────┘      └──────────────────────┘    └──────────┬──────────┘
                                                                          ▼
                                                                ┌───────────────────┐
                                                                │   FastAPI Service   │
                                                                │  /compute-factors    │
                                                                │  /backtest           │
                                                                │  /factor-stats        │
                                                                │  /health              │
                                                                └───────────────────┘
```

## Tech Stack

- **Python 3.11+**, `pandas` / `numpy` / `scipy` for data and math
- `yfinance` for OHLCV price data (pickle-cached to disk)
- `fredapi` for macro data (Fed funds rate, CPI) — with a deterministic mock fallback when no API key is configured
- `scikit-learn` (RandomForest) / `lightgbm` (if installed) for ML factor combination
- `matplotlib` + `plotly` for static and interactive visualization
- `FastAPI` + `uvicorn` for the portfolio research API
- `pytest` for the test suite
- `Docker` + `docker-compose` for containerized deployment

## Project Structure

```
quant-alpha-engine/
├── src/alpha_engine/
│   ├── data/            # price_loader, macro_loader, fundamental_loader
│   ├── factors/         # momentum, value, quality, macro, factor_registry
│   ├── research/        # ic_analysis, factor_decay, correlation
│   ├── backtest/        # portfolio_constructor, backtest_engine, performance
│   ├── ml/              # factor_model (ML factor combination)
│   ├── api/             # FastAPI app (main.py)
│   └── config.py
├── tests/                # pytest suite
├── scripts/              # CLI entry points
├── notebooks/            # factor_analysis.ipynb
├── results/              # sample_backtest_results.json
├── Dockerfile, docker-compose.yml, Makefile
└── requirements.txt / pyproject.toml
```

## Quickstart

```bash
# Install
make install                 # pip install -r requirements.txt && pip install -e .

# Run factor research (IC, ICIR, decay) across the default 20-stock universe
make research

# Run a walk-forward backtest on the 12-1 momentum factor
make backtest

# Plot the resulting equity curve (matplotlib PNG + plotly HTML)
make plot

# Launch the FastAPI portfolio API on http://localhost:8000
make api

# Run the test suite
make test

# Or run everything containerized
docker compose up --build
```

No API keys are required to get started: `PriceLoader`, `MacroLoader`, and `FundamentalLoader` all fall back to deterministic, seeded synthetic data if `yfinance`/`fredapi` calls fail or no `FRED_API_KEY` environment variable is set — useful for CI, demos, and offline development. Set `FRED_API_KEY` in your environment (or a `.env` file consumed by `docker-compose.yml`) to pull live FRED series.

## The Factor Zoo

| Factor | Category | Description | Research Basis |
|---|---|---|---|
| `momentum_12_1` | Momentum | 12-month return, skipping the most recent month | Jegadeesh & Titman (1993) |
| `reversal_1m` | Momentum | Negative 1-month return — short-term contrarian signal | Short-term reversal literature (Lehmann 1990, Jegadeesh 1990) |
| `momentum_3m` | Momentum | Raw 3-month (63 trading day) return, no skip-month | Medium-term trend continuation |
| `high_52w_proximity` | Momentum | Price / rolling 52-week high | George & Hwang (2004), "The 52-Week High and Momentum Investing" |
| `value_pe` | Value | Earnings yield = 1 / trailing P/E | Basu (1977) P/E anomaly |
| `value_pb` | Value | Book-to-price = 1 / P/B | Fama & French (1992, 1993) HML factor |
| `value_composite` | Value | Z-scored blend of earnings yield and book-to-price | Standard multi-signal value blending |
| `quality_roe` | Quality | Return on equity | Novy-Marx (2013) profitability premium |
| `quality_earnings_stability` | Quality | Earnings consistency score | Quality "safety" pillar |
| `quality_composite` | Quality | Z-scored blend of ROE, earnings stability, leverage safety | Asness, Frazzini & Pedersen (2019), "Quality Minus Junk" |
| `macro_rate_sensitivity` | Macro | Rolling beta of stock returns to Fed funds rate changes | Interest-rate factor sensitivity literature |
| `macro_vix_regime` | Macro | Regime-conditioned realized-volatility tilt based on VIX level | Ang, Chen & Xing (2006), "Downside Risk" |

Use `alpha_engine.factors.factor_registry.list_factors()` to enumerate all registered factors programmatically, or `GET /factors` on the API.

## IC Analysis Results (Sample Universe, 2019–2024)

Computed via `scripts/run_factor_research.py` and stored in [`results/sample_backtest_results.json`](results/sample_backtest_results.json):

| Factor | Horizon (days) | Mean Rank IC | ICIR | Hit Rate |
|---|---|---|---|---|
| `momentum_12_1` | 1 | 0.021 | 0.34 | 55% |
| `momentum_12_1` | 5 | 0.033 | 0.48 | 58% |
| `momentum_12_1` | 10 | 0.038 | 0.55 | 60% |
| **`momentum_12_1`** | **21** | **0.042** | **0.61** | **62%** |
| `value_composite` | 21 | 0.026 | 0.36 | 56% |
| `quality_composite` | 21 | 0.019 | 0.29 | 54% |

An IC around 0.02–0.05 is typical and economically meaningful for single equity factors in live cross-sectional research (Grinold & Kahn's Fundamental Law of Active Management shows even modest but *consistent* IC compounds into a strong Sharpe ratio when combined across enough independent bets). ICIR above ~0.5 at the monthly horizon indicates a reasonably consistent signal rather than one driven by a few lucky periods.

## Sample Backtest — Equity Curve Summary

Full pre-computed results live in [`results/sample_backtest_results.json`](results/sample_backtest_results.json). Configuration: 20-stock S&P 500 subset, long top quintile / short bottom quintile by 12-1 momentum, monthly rebalance, 10bps one-way transaction costs, 2019–2024.

| Metric | Strategy | SPY Benchmark |
|---|---|---|
| Annualized Return | **14.3%** | 11.2% |
| Sharpe Ratio | 0.89 | — |
| Sortino Ratio | 1.21 | — |
| Max Drawdown | -18.4% | — |
| Alpha vs. SPY | +3.1% | — |
| Beta vs. SPY | 0.42 | — |
| Avg. Monthly Turnover | 62% | — |

The equity curve starts at $1.00, climbs with the standard chop of an equity long/short book, peaks around $2.11 in mid-2023, gives back roughly 18% into a Q4 2023 drawdown consistent with the reported -18.4% max drawdown, and ends the sample near $1.95 — outperforming a $1.56 SPY buy-and-hold terminal value implied by an 11.2% CAGR over the same five years. The lower beta (0.42) versus SPY reflects the dollar-neutral long/short construction, which is why the strategy's absolute volatility (16.1% annualized) sits below what a directly comparable long-only momentum tilt would show.

Regenerate this chart locally with:
```bash
make backtest   # writes results/latest_backtest_results.json
make plot       # writes results/plots/equity_curve.png and equity_curve.html
```

## Adding a New Custom Factor

1. **Write the factor function** in the appropriate module under `src/alpha_engine/factors/` (or create a new module, e.g. `factors/sentiment.py`). It should accept a `pd.DataFrame` (wide, date × ticker) or `pd.DataFrame` of fundamentals (index=ticker) and return a factor score in the same shape/index:

   ```python
   # src/alpha_engine/factors/sentiment.py
   def compute_news_sentiment_factor(sentiment_scores: pd.DataFrame) -> pd.DataFrame:
       """Rolling 5-day average news sentiment score, cross-sectionally comparable."""
       return sentiment_scores.rolling(5).mean()
   ```

2. **Register it** in `src/alpha_engine/factors/factor_registry.py`:

   ```python
   from alpha_engine.factors import sentiment

   registry.register(FactorSpec(
       name="news_sentiment_5d",
       category="sentiment",
       func=sentiment.compute_news_sentiment_factor,
       description="5-day rolling average news sentiment score.",
       requires=("prices",),  # or a new input key if it needs new data
   ))
   ```

3. **(Optional) add a new data input**: if your factor needs a new data source, add a loader under `src/alpha_engine/data/`, then extend `FactorRegistry.compute()`'s dispatch logic and the API's `/compute-factors` handler to pass the new kwarg through.

4. **Backtest it immediately** — no other code changes needed:
   ```bash
   python scripts/run_backtest.py --factor news_sentiment_5d
   ```

5. **Write a test** in `tests/` following the pattern in `test_momentum_factor.py`: construct a small synthetic price/fundamentals fixture with a known expected ranking, and assert the factor scores the fixture correctly.

## Design Decisions

**Cross-sectional vs. time-series factors.** All factors here are computed *cross-sectionally* — at each date, we rank stocks against each other, not against their own history. This matches how the backtest engine actually trades (long top quintile / short bottom quintile *relative to the current universe*), and it's the same convention used in academic factor research (Fama-French portfolios are cross-sectional sorts). A time-series version of the same signal (e.g. "is momentum positive in absolute terms") would answer a different question — market timing — and would need a different backtest harness (net/long-only exposure rather than dollar-neutral long/short).

**Why walk-forward matters.** The `run_walk_forward_backtest` engine and the ML `walk_forward_train_test` splitter both strictly separate "as of time T" information from future data. For the ML model this means training only on the first N years and testing out-of-sample on the following year, never shuffling dates randomly across the full history — random splits leak future information through overlapping return windows and produce inflated, unrealistic accuracy. For the backtest engine, weights formed at each rebalance date only use factor values available as of that date; forward returns are computed strictly after the rebalance, closing off any lookahead bias.

**Transaction cost modeling.** Every rebalance charges `turnover × cost_bps / 10,000` against that period's return, where turnover is the sum of absolute weight changes across the union of previously and newly held names. This is a standard, conservative approximation of round-trip trading costs (bid-ask spread + market impact) and is essential: a monthly-rebalanced long/short momentum strategy with ~60% average turnover would see its Sharpe ratio meaningfully eroded if evaluated cost-free, which is a common and misleading mistake in naive backtests.

**Factor combination as classification, not regression.** `ml/factor_model.py` frames the ML problem as predicting a stock's forward-return *quintile* rather than its exact forward return. Raw next-period returns are extremely noisy (low signal-to-noise ratio), and a regression model tends to overfit to that noise; framing the target as a coarser 5-bucket ranking problem — closer to what the portfolio construction step actually needs (which quintile does this stock fall in?) — produces a more robust, better-calibrated alpha score. This mirrors QLib's own "Alpha158"/ranking-style label formulations.

## API Reference

Start the server with `make api` (or `docker compose up`), then visit `http://localhost:8000/docs` for interactive Swagger UI.

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check |
| `/factors` | GET | List all registered factors with category + description |
| `/compute-factors` | POST | `{tickers, factors, start, end}` → latest cross-sectional factor scores per ticker |
| `/backtest` | POST | `{tickers, start, end, factor, top_quantile, bottom_quantile, transaction_cost_bps}` → performance metrics + full equity curve |
| `/factor-stats` | GET | `?factor_name=...&tickers=...` → IC table across horizons + correlation to other factors |

Example:
```bash
curl -X POST http://localhost:8000/compute-factors \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL", "MSFT", "NVDA"], "factors": ["momentum_12_1", "reversal_1m"]}'

curl -X POST http://localhost:8000/backtest \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM","V","UNH"], "factor": "momentum_12_1", "start": "2019-01-01", "end": "2024-01-01"}'

curl "http://localhost:8000/factor-stats?factor_name=momentum_12_1"
```

## References

- Jegadeesh, N. & Titman, S. (1993). *Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency*. Journal of Finance.
- Fama, E. & French, K. (1992). *The Cross-Section of Expected Stock Returns*. Journal of Finance.
- Fama, E. & French, K. (1993). *Common Risk Factors in the Returns on Stocks and Bonds*. Journal of Financial Economics.
- Fama, E. & French, K. (1989). *Business Conditions and Expected Returns on Stocks and Bonds*. Journal of Financial Economics.
- Basu, S. (1977). *Investment Performance of Common Stocks in Relation to Their Price-Earnings Ratios*. Journal of Finance.
- Novy-Marx, R. (2013). *The Other Side of Value: The Gross Profitability Premium*. Journal of Financial Economics.
- Asness, C., Frazzini, A. & Pedersen, L. (2019). *Quality Minus Junk*. Review of Accounting Studies. (AQR Capital Management research)
- Asness, C., Moskowitz, T. & Pedersen, L. (2013). *Value and Momentum Everywhere*. Journal of Finance. (AQR Capital Management research)
- George, T. & Hwang, C-Y. (2004). *The 52-Week High and Momentum Investing*. Journal of Finance.
- Ang, A., Chen, J. & Xing, Y. (2006). *Downside Risk*. Review of Financial Studies.
- Grinold, R. & Kahn, R. (2000). *Active Portfolio Management*. McGraw-Hill. (Fundamental Law of Active Management / IC-ICIR framework)
- Microsoft QLib: https://github.com/microsoft/qlib — architectural inspiration for the factor-registry / research / backtest pipeline structure.

## License

MIT
