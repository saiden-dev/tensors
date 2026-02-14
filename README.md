[![Static Badge](https://img.shields.io/badge/pip_install-tensors-blue)](https://pypi.org/project/tensors)

# tensors

A CLI tool for working with safetensor files, CivitAI models, and stable-diffusion.cpp image generation. Supports both local and remote server modes.

## Features

https://github.com/user-attachments/assets/2e7629b4-34e7-4cbc-b50e-31d7fdd30239

- **Read safetensor metadata** - Parse headers, count tensors, extract embedded metadata
- **CivitAI integration** - Search models, fetch info, identify files by hash
- **Download models** - Resume support, type-based default paths
- **Hash verification** - SHA256 computation with progress display
- **Image generation** - txt2img/img2img via stable-diffusion.cpp server
- **Server wrapper** - FastAPI wrapper for sd-server with hot reload
- **Models database** - SQLite cache for local files and CivitAI metadata
- **Image gallery** - Manage generated images with metadata
- **Remote mode** - Control remote tsr servers via `--remote` flag

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
# Start the wrapper API with a model
tsr serve --model /path/to/model.safetensors

# Custom host and port
tsr serve --model /path/to/model.safetensors --host 0.0.0.0 --port 8080

# Check server status
tsr status

# Hot-reload with a new model
tsr reload --model /path/to/other_model.safetensors
```

### Models Database

Track local safetensor files and cache CivitAI metadata for offline access.

```bash
# Scan a directory for safetensor files
tsr db scan /path/to/models

# Link unscanned files to CivitAI by hash
tsr db link

# Cache full CivitAI model data
tsr db cache 12345

# List local files with CivitAI info
tsr db list

# Search cached models offline
tsr db search "pony"
tsr db search -t lora -b sdxl

# Get trigger words for a LoRA
tsr db triggers model.safetensors

# Show database statistics
tsr db stats
```

### Image Gallery

Manage generated images on a remote server.

```bash
# List images in gallery
tsr images list --remote junkpile

# Show image metadata
tsr images show IMAGE_ID --remote junkpile

# Download an image
tsr images download IMAGE_ID --remote junkpile -o ./downloads

# Delete an image
tsr images delete IMAGE_ID --remote junkpile
```

### Model Management

List and switch models on a remote server.

```bash
# List available models
tsr models list --remote junkpile

# Show active model
tsr models active --remote junkpile

# Switch to a different model
tsr models switch /path/to/model.safetensors --remote junkpile

# List available LoRAs
tsr models loras --remote junkpile
```

### Remote Mode

Control a remote tsr server instead of local operations.

```bash
# Configure a remote server
tsr remote add junkpile http://junkpile:8080

# Set default remote
tsr remote default junkpile

# List configured remotes
tsr remote list

# Generate on remote server
tsr generate "a cat" --remote junkpile

# Download model to remote server
tsr dl -m 12345 --remote junkpile

# All commands support --remote flag
tsr status --remote junkpile
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

[remotes]
junkpile = "http://junkpile:8080"
local = "http://localhost:8080"

# Optional: set default remote for all commands
default_remote = "junkpile"
```

Or set API key via environment variable:
```bash
export CIVITAI_API_KEY="your-api-key"
```

## Default Paths

Data is stored in XDG-compliant paths:

| Type | Path |
|------|------|
| Config | `~/.config/tensors/config.toml` |
| Database | `~/.local/share/tensors/models.db` |
| Checkpoints | `~/.local/share/tensors/models/checkpoints/` |
| LoRAs | `~/.local/share/tensors/models/loras/` |
| Gallery | `~/.local/share/tensors/gallery/` |
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

## Server API Endpoints

When running `tsr serve`, the following endpoints are available:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | Server status and active model |
| `/reload` | POST | Hot-reload with new model |
| `/api/images` | GET | List gallery images |
| `/api/images/{id}` | GET | Get image file |
| `/api/images/{id}/meta` | GET | Get image metadata |
| `/api/images/{id}/edit` | POST | Update image metadata |
| `/api/images/{id}` | DELETE | Delete image |
| `/api/models` | GET | List available models |
| `/api/models/active` | GET | Get active model |
| `/api/models/switch` | POST | Switch model |
| `/api/models/loras` | GET | List available LoRAs |
| `/api/generate` | POST | Generate images |
| `/api/download` | POST | Start CivitAI download |
| `/api/db/files` | GET | List local files |
| `/api/db/models` | GET | Search cached models |
| `/api/db/stats` | GET | Database statistics |

All sd-server endpoints (`/sdapi/v1/*`) are proxied through to the underlying process.

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=tensors

# Lint and format
uv run ruff check .
uv run ruff format .

# Type check
uv run mypy tensors/
```

## License

MIT
