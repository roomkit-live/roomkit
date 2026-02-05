# Contributing to RoomKit

Thank you for your interest in contributing to RoomKit, a project by Tchat N Sign.

## Development Setup

```bash
uv sync --extra dev
```

## Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run `make all` to verify lint, typecheck, and tests pass
5. Submit a pull request

## Code Style

- Python 3.12+, pure async/await
- All files under 500 lines
- Format with `ruff format`, lint with `ruff check`
- Type-check with `mypy --strict`

## Testing

```bash
make test       # run tests
make coverage   # run with coverage
```

All new code must include tests. Aim for >90% coverage.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
