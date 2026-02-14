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

# Run UI dev server with hot reload
ui-dev:
    cd tensors/server/ui && npm run dev

# Build UI for production
ui-build:
    cd tensors/server/ui && npm run build

# Deploy to junkpile (build, sync, restart, verify)
deploy:
    ./scripts/deploy.sh
