.PHONY: install lint format typecheck security test coverage all clean docs publish deploy release

install:
	uv sync --extra dev

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

typecheck:
	uv run mypy src/roomkit/

security:
	uv run bandit -r src/ -c pyproject.toml

test:
	uv run pytest

coverage:
	uv run pytest --cov=roomkit --cov-report=term-missing

all: lint typecheck security test

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
	uv run python -m build

publish: build-dist
	uv run twine upload dist/*

# --- Website deployment ---

deploy:
	./deploy.sh full

deploy-status:
	./deploy.sh status

# --- Full release (publish to PyPI + deploy website) ---

release: all build-dist
	@echo "Publishing to PyPI..."
	uv run twine upload dist/*
	@echo "Deploying website..."
	./deploy.sh full
	@echo "Release complete!"
