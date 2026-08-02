.PHONY: audit install lint format typecheck security test coverage all clean docs deploy release

install:
	uv sync --extra dev

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .

typecheck:
	uv run ty check src/roomkit/

security:
	uv run bandit -r src/ -c pyproject.toml

# Same two passes CI runs, so a new advisory can be seen before pushing:
# the core gates, the extras only report.
audit:
	uv export --frozen --no-dev --no-emit-project --format requirements-txt -o /tmp/rk-core.txt
	uvx pip-audit --requirement /tmp/rk-core.txt
	-uv export --frozen --no-dev --no-emit-project --extra all --no-hashes --format requirements-txt -o /tmp/rk-all.txt
	-uvx pip-audit --requirement /tmp/rk-all.txt

test:
	uv run pytest

coverage:
	uv run pytest --cov=roomkit --cov-report=term-missing

stress:
	uv run pytest -m stress -v

llms-full:
	uv run python scripts/build_llms_full.py

all: lint typecheck security test

clean:
	rm -rf dist/ build/ .pytest_cache/ .ruff_cache/ htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true

# --- Documentation ---

docs:
	uv run mkdocs build --strict

docs-serve:
	uv run mkdocs serve

# --- Website deployment ---

deploy:
	./deploy.sh full

deploy-status:
	./deploy.sh status

# --- Release (bump, test, build, publish, tag, push) ---

release:
ifndef VERSION
	$(error VERSION is required. Usage: make release VERSION=0.4.1)
endif
	./scripts/release.sh $(VERSION)
