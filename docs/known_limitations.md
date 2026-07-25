# Known Limitations

This project is intentionally scoped as a personal research engine.

- The default universe is a small, current large-cap subset and does not solve survivorship bias.
- yfinance adjusted-close data is convenient for examples but is not a point-in-time institutional data source.
- yfinance fundamentals are current snapshots, not historical point-in-time fundamentals. Value and quality factors should be treated as examples until backed by a proper fundamentals dataset.
- Synthetic data exists only for tests and offline demos. It is opt-in for price data and labeled in output metadata.
- The backtest does not model borrow fees, short constraints, taxes, financing, slippage curves, or capacity.
- The transaction-cost model is a simple turnover penalty.
- The ML module is experimental. It uses chronological splits and refuses random fallback splits to avoid time leakage.
- The FastAPI app is a local experimentation surface. It does not include auth, rate limiting, observability, or deployment hardening.
