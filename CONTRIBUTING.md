# Contributing to RoomKit

Thank you for your interest in contributing to RoomKit! This guide will get you from zero to a passing test suite in under five minutes.

## Quick Start

```bash
# 1. Fork and clone
git clone https://github.com/<you>/roomkit.git
cd roomkit

# 2. Install (requires Python 3.12+ and uv)
uv sync --extra dev

# 3. Verify everything works
make all          # lint + typecheck + security scan + tests
```

That's it. If `make all` passes, you're ready to contribute.

## Finding Something to Work On

Look for issues tagged with these labels:

| Label | What it means |
|-------|---------------|
| **good first issue** | Small, well-scoped tasks ideal for first-time contributors. Usually a single file change with clear acceptance criteria. |
| **help wanted** | Larger tasks where we'd welcome community help. May span multiple files but the scope is well-defined. |
| **docs** | Documentation improvements — great if you want to contribute without touching core code. |
| **provider** | Adding or improving a provider integration (SMS, Email, Voice, etc.). Self-contained and easy to test in isolation. |

If you want to contribute but nothing in the tracker appeals to you, here are areas that always welcome help:

- **New provider integrations** — add support for a new SMS, email, or chat platform
- **Example scripts** — runnable demos in `examples/` showing real-world usage
- **Test coverage** — we target 80%+; run `uv run pytest --cov=roomkit --cov-report=term-missing` to find gaps
- **Documentation** — guides, tutorials, and API doc improvements in `../roomkit-docs/`

## Development Workflow

1. **Create a branch** from `main` for your work
2. **Make your changes** — keep commits focused and atomic
3. **Run `make all`** before pushing — this runs the same checks as CI
4. **Open a pull request** with a clear description of what and why

### What `make all` Runs

```bash
uv run ruff check src/ tests/       # Lint (E/F/I/N/UP/B/SIM rules)
uv run ruff format --check src/ tests/  # Format check
uv run mypy src/roomkit/             # Type check (strict)
uv run bandit -r src/ -c pyproject.toml  # Security scan
uv run pytest                        # Tests
```

You can run these individually while developing. Use `uv run ruff check --fix` to auto-fix lint issues.

## Code Style

- **Python 3.12+** — use `X | None` unions, not `Optional[X]`
- **`from __future__ import annotations`** — always the first import
- **Async-first** — never use synchronous I/O in async methods
- **All imports at the top** — no inline imports except lazy imports for optional deps behind `try/except ImportError`
- **Ruff** — 99-char line length, format with `ruff format`
- **Type hints** on all public methods
- **Logging** via `logging.getLogger("roomkit.xxx")` — no `print()`

## Testing

```bash
uv run pytest                              # Run all tests
uv run pytest tests/test_X.py -v           # Run a specific file
uv run pytest -k "keyword" -v              # Run by keyword
uv run pytest --cov=roomkit --cov-report=term-missing  # With coverage
```

- Framework: **pytest** with `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed
- Use mock providers (`MockVADProvider`, `MockSTTProvider`, etc.) for voice tests
- Use `make_event()` from `tests/conftest.py` for building test events
- Run with `uv run pytest` (not `python -m pytest`)

## Adding New Features

Every new feature, provider, or channel type **must** include:

1. **Tests** — unit tests covering the happy path and error cases
2. **Documentation** — update relevant pages in `../roomkit-docs/docs/`
3. **Example** — a runnable script in `examples/` demonstrating the feature

### New Provider Checklist

1. Config in `src/roomkit/providers/<name>/config.py`
2. Implementation in `src/roomkit/providers/<name>/<type>.py`
3. Export from `__init__.py` and `src/roomkit/__init__.py`
4. Tests in `tests/test_providers/test_<name>.py`
5. Example in `examples/`
6. Docs update in `../roomkit-docs/`

### New Pipeline Stage Checklist

1. ABC in `src/roomkit/voice/pipeline/<stage>/base.py`
2. Mock in `src/roomkit/voice/pipeline/<stage>/mock.py`
3. Export from subdirectory and `voice/pipeline/__init__.py`
4. Tests, example, and docs guide

## Pull Request Guidelines

- **Keep PRs focused** — one logical change per PR
- **Write a clear description** — explain what changed and why
- **All checks must pass** — `make all` is the gate
- **Add tests** — untested code won't be merged
- **Don't break public API** — if you need to change a public interface, discuss it in an issue first

## Getting Help

- Open an issue for bugs or feature proposals
- Tag your issue with `question` if you need guidance on approach
- Check existing issues and PRs before starting work to avoid duplication

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
