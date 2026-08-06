.DEFAULT_GOAL := help
.PHONY: help install install-dev run dev ui ingest reindex stats eval eval-fast test test-cov lint format typecheck check vapi vapi-list clean clean-index

PYTHON ?= python

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies
	$(PYTHON) -m pip install -r requirements.txt

install-dev:  ## Install runtime + dev dependencies
	$(PYTHON) -m pip install -r requirements-dev.txt

run:  ## Start the API server
	$(PYTHON) -m app.main

dev:  ## Start the API server with autoreload
	$(PYTHON) -m uvicorn app.main:app --app-dir src --reload --port 8000

ui:  ## Start the React Web UI dev server
	npm run dev --prefix frontend

ingest:  ## Index everything in data/documents
	$(PYTHON) scripts/ingest.py

reindex:  ## Wipe the index and rebuild from scratch
	$(PYTHON) scripts/ingest.py --reset

stats:  ## Show what is currently indexed
	$(PYTHON) scripts/ingest.py --stats

eval:  ## Retrieval + grounding smoke evaluation
	$(PYTHON) scripts/evaluate.py

eval-fast:  ## Retrieval-only evaluation (no LLM cost)
	$(PYTHON) scripts/evaluate.py --retrieval-only

vapi:  ## Create or update the Vapi assistant (needs VAPI_SERVER_URL)
	$(PYTHON) scripts/setup_vapi_assistant.py

vapi-list:  ## List assistants on the configured Vapi account
	$(PYTHON) scripts/setup_vapi_assistant.py --list

test:  ## Run the test suite
	$(PYTHON) -m pytest

test-cov:  ## Run tests with a coverage report
	$(PYTHON) -m pytest --cov=src/app --cov-report=term-missing

lint:  ## Lint with ruff
	$(PYTHON) -m ruff check .

format:  ## Auto-fix lint issues and format
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

typecheck:  ## Type-check with mypy
	$(PYTHON) -m mypy src

check: lint typecheck test  ## Lint, type-check and test

clean:  ## Remove caches and build artefacts
	-rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov coverage.xml
	-find . -type d -name __pycache__ -prune -exec rm -rf {} +

clean-index:  ## Delete the vector store (documents are kept)
	-rm -rf data/vector_store
