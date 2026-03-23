# Project Style Guide

## Naming
- Functions and variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`

## Type Hints
- Required on all public method signatures
- Use `X | None` instead of `Optional[X]`

## Formatting
- Max line length: 99 characters
- Use `from __future__ import annotations` as the first import

## Error Handling
- Never use bare `except:` -- always catch specific exceptions
- Log errors with `logging.getLogger()`, never `print()`
