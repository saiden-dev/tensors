# Default: run all checks, fixes, and tests
default: fix check test

# Run all linting and type checks
check:
    uv run ruff check .
    uv run mypy tensors/

# Run tests
test:
    uv run pytest

# Auto-fix linting issues and format code
fix:
    uv run ruff check --fix .
    uv run ruff format .

# Format code only
format:
    uv run ruff format .

# Lint only (no fixes)
lint:
    uv run ruff check .

# Type check only
types:
    uv run mypy tensors/
