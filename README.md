[![Static Badge](https://img.shields.io/badge/pip_install-tensors-blue)](https://pypi.org/project/tensors)

# tensors

A CLI tool for working with safetensor files and CivitAI models.

## Features

https://github.com/user-attachments/assets/2e7629b4-34e7-4cbc-b50e-31d7fdd30239

- **Read safetensor metadata** - Parse headers, count tensors, extract embedded metadata
- **CivitAI integration** - Search models, fetch info, identify files by hash
- **Download models** - Resume support, type-based default paths
- **Hash verification** - SHA256 computation with progress display

## Installation

```bash
# Clone and install
git clone https://github.com/aladac/tensors.git
cd tensors
uv sync

# Or install directly
uv pip install git+https://github.com/aladac/tensors.git
```

## Usage

### Search CivitAI

```bash
# Search by query
tsr search "illustrious"

# Filter by type and base model
tsr search -t lora -b sdxl

# Sort by newest, limit results
tsr search -t checkpoint -s newest -n 10
```

### Get Model Info

```bash
# Get model info by ID (shows all versions)
tsr get 12345

# Get specific version info
tsr get -v 67890
```

### Download Models

```bash
# Download latest version of a model
tsr dl -m 12345

# Download specific version
tsr dl -v 67890

# Download by hash lookup
tsr dl -H ABC123...

# Custom output directory
tsr dl -m 12345 -o ./models
```

### Inspect Local Files

```bash
# Read safetensor file and lookup on CivitAI
tsr info model.safetensors

# Skip CivitAI lookup
tsr info model.safetensors --skip-civitai

# Output as JSON
tsr info model.safetensors -j

# Save metadata files
tsr info model.safetensors --save-to ./metadata
```

### Configuration

```bash
# Show current config
tsr config

# Set CivitAI API key
tsr config --set-key YOUR_API_KEY
```

## Configuration

Config file: `~/.config/tensors/config.toml`

```toml
[api]
civitai_key = "your-api-key"
```

Or set via environment variable:
```bash
export CIVITAI_API_KEY="your-api-key"
```

## Default Paths

Models are downloaded to XDG-compliant paths:

| Type | Path |
|------|------|
| Checkpoint | `~/.local/share/tensors/models/checkpoints/` |
| LoRA | `~/.local/share/tensors/models/loras/` |
| Metadata | `~/.local/share/tensors/metadata/` |

## Search Options

| Option | Values |
|--------|--------|
| `-t, --type` | checkpoint, lora, embedding, vae, controlnet, locon |
| `-b, --base` | sd15, sdxl, pony, flux, illustrious |
| `-s, --sort` | downloads, rating, newest |
| `-n, --limit` | Number of results (default: 20) |

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .

# Type check
uv run mypy tensors.py
```

## License

MIT
