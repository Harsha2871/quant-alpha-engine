.PHONY: install research backtest api test lint clean docker-build docker-up docker-down

PYTHON ?= python3
PIP ?= pip3

install: ## Install project + dependencies in editable mode
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

research: ## Run factor research (IC/ICIR/decay) across the default universe
	$(PYTHON) scripts/run_factor_research.py --universe-default --output results/latest_research_results.json

research-demo: ## Run factor research with deterministic synthetic data allowed
	$(PYTHON) scripts/run_factor_research.py --universe-default --allow-synthetic --output results/latest_research_results.json

backtest: ## Run the default walk-forward backtest (12-1 momentum factor)
	$(PYTHON) scripts/run_backtest.py --factor momentum_12_1 --output results/latest_backtest_results.json

backtest-demo: ## Run the default backtest with deterministic synthetic data allowed
	$(PYTHON) scripts/run_backtest.py --factor momentum_12_1 --allow-synthetic --output results/latest_backtest_results.json

plot: ## Plot equity curve from the latest backtest results
	$(PYTHON) scripts/plot_performance.py --input results/latest_backtest_results.json

api: ## Launch the FastAPI portfolio API with hot reload
	uvicorn alpha_engine.api.main:app --reload --host 0.0.0.0 --port 8000

test: ## Run the pytest suite
	pytest -v

lint: ## Basic syntax/style check via python -m py_compile
	find src tests scripts -name "*.py" | xargs -n1 $(PYTHON) -m py_compile

clean: ## Remove caches and bytecode
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .cache .pytest_cache build dist *.egg-info

docker-build: ## Build the Docker image
	docker compose build

docker-up: ## Start the API via docker-compose
	docker compose up

docker-down: ## Stop docker-compose services
	docker compose down
