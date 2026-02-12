# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`tensors` is a Python CLI tool (`tsr`) for reading safetensor file metadata and interacting with the CivitAI API (search, fetch, download models). Built with Typer, Rich, and httpx.

## Commands

```bash
# Install dependencies
uv sync --group dev

# Run everything (fix, check, test)
just

# Individual tasks
just check          # ruff check + mypy
just test           # pytest with coverage
just fix            # auto-fix lint + format
just types          # mypy only

# Run a single test
uv run pytest tests/test_tensors.py::TestClassName::test_name -v
```

## Architecture

Five modules with clean separation:

- **`cli.py`** — Typer CLI with commands: `info`, `search`, `get`, `dl`, `config`. Legacy mode auto-converts bare `tsr file.safetensors` to `tsr info`.
- **`api.py`** — CivitAI REST API wrapper using httpx. Search, fetch model/version/by-hash, download with resume support and Rich progress bars.
- **`config.py`** — XDG-compliant paths (`~/.config/tensors/config.toml`, `~/.local/share/tensors/`). Enums with `to_api()` methods for CivitAI parameter mapping. API key resolution: env var → config file → legacy `~/.sftrc`.
- **`safetensor.py`** — Binary safetensor header parsing (8-byte u64 LE header size → JSON metadata). SHA256 streaming hash computation.
- **`display.py`** — Rich table formatting for all output types. All major commands support `--json` output.

Entry point: `tsr = "tensors:main"` (pyproject.toml `[project.scripts]`).

## Code Standards

- Python 3.12+, strict mypy, line length 130
- Ruff with extended rule set (E, W, F, I, B, C4, UP, ARG, SIM, TCH, PTH, PL, RUF)
- PLR0913 (too many arguments) is intentionally ignored for CLI commands
- Tests use respx for HTTP mocking and a `temp_safetensor` fixture from conftest.py

## Release

Tags trigger PyPI publish: `git tag v0.1.x && git push origin v0.1.x`. See RELEASE.md for binary builds (Nuitka).
