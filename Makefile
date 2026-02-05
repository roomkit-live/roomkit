.PHONY: install lint format typecheck test coverage all clean docs publish deploy release

install:
	uv sync --extra dev

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

typecheck:
	uv run mypy src/roomkit/

test:
	uv run pytest

coverage:
	uv run pytest --cov=roomkit --cov-report=term-missing

all: lint typecheck test

clean:
	rm -rf dist/ build/ .mypy_cache/ .pytest_cache/ .ruff_cache/ htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true

# --- Documentation ---

docs:
	uv run mkdocs build --strict

docs-serve:
	uv run mkdocs serve

# --- Publishing ---

build-dist: clean
	python -m build

publish: build-dist
	twine upload dist/*

# --- Website deployment ---

deploy:
	./deploy.sh full

deploy-status:
	./deploy.sh status

# --- Full release (publish to PyPI + deploy website) ---

release: all build-dist
	@echo "Publishing to PyPI..."
	twine upload dist/*
	@echo "Deploying website..."
	./deploy.sh full
	@echo "Release complete!"
