[![Static Badge](https://img.shields.io/badge/pip_install-tensors-blue)](https://pypi.org/project/tensors)

# tensors

A CLI tool for working with safetensor files, CivitAI models, and stable-diffusion.cpp image generation.

## Features

https://github.com/user-attachments/assets/2e7629b4-34e7-4cbc-b50e-31d7fdd30239

- **Read safetensor metadata** - Parse headers, count tensors, extract embedded metadata
- **CivitAI integration** - Search models, fetch info, identify files by hash
- **Download models** - Resume support, type-based default paths
- **Hash verification** - SHA256 computation with progress display
- **Image generation** - txt2img/img2img via stable-diffusion.cpp server
- **Server wrapper** - FastAPI wrapper for sd-server process management

## Installation

```bash
# Clone and install
git clone https://github.com/aladac/tensors.git
cd tensors
uv sync

# Or install directly
uv pip install git+https://github.com/aladac/tensors.git

# With server wrapper support
pip install tensors[server]
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

### Generate Images

Requires a running [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) server.

```bash
# Generate an image
tsr generate "a cat sitting on a roof"

# Custom size, steps, and output
tsr generate "sunset over mountains" -W 768 -H 512 --steps 30 -o ./output

# Multiple images with seed
tsr generate "cyberpunk city" -b 4 -s 42

# With sampler and negative prompt
tsr generate "portrait" --sampler euler_a -n "blurry, low quality"
```

### Server Wrapper

Manage sd-server process via a REST API. Requires `pip install tensors[server]`.

```bash
# Start the wrapper API
tsr serve

# Custom host and port
tsr serve --host 0.0.0.0 --port 9000
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

## Generate Options

| Option | Description |
|--------|-------------|
| `-W` | Image width (default: 512) |
| `-H` | Image height (default: 512) |
| `--steps` | Sampling steps (default: 20) |
| `--cfg-scale` | CFG scale (default: 7.0) |
| `-s` | RNG seed, -1 for random (default: -1) |
| `-b` | Batch size / number of images (default: 1) |
| `-n` | Negative prompt |
| `-o` | Output directory (default: .) |
| `--sampler` | Sampler name |
| `--scheduler` | Scheduler name |
| `--host` | sd-server address (default: 127.0.0.1) |
| `--port` | sd-server port (default: 1234) |

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
uv run mypy tensors/
```

## License

MIT
