<p align="center">
  <img src="icon.svg" alt="tensors" width="96" height="96">
</p>

<h1 align="center">tensors</h1>

<p align="center">
  <a href="https://pypi.org/project/tensors"><img src="https://img.shields.io/pypi/v/tensors?color=blue" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/coverage-50%25-yellow" alt="Coverage">
  <img src="https://img.shields.io/badge/python-3.12+-blue" alt="Python">
  <a href="https://github.com/saiden-dev/tensors/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-BSL--1.1-blue" alt="License"></a>
</p>

<p align="center">
  A CLI tool for working with safetensor files, CivitAI models, and image generation via ComfyUI or stable-diffusion.cpp.
</p>

## Features

- **Read safetensor metadata** - Parse headers, count tensors, extract embedded metadata
- **CivitAI integration** - Search models, fetch info, identify files by hash
- **Download models** - Resume support, type-based default paths
- **Hash verification** - SHA256 computation with progress display
- **ComfyUI integration** - Generate images via ComfyUI with CLI and API
- **Image generation** - txt2img/img2img via stable-diffusion.cpp server
- **Server wrapper** - FastAPI wrapper with ComfyUI proxy and hot reload
- **Models database** - SQLite cache for local files and CivitAI metadata
- **Image gallery** - Manage generated images with metadata
- **Remote mode** - Control remote tsr servers via `--remote` flag

## Installation

```bash
# Clone and install
git clone https://github.com/saiden-dev/tensors.git
cd tensors
uv sync

# Or install directly
uv pip install git+https://github.com/saiden-dev/tensors.git

# With server wrapper support
pip install tensors[server]
```

## Usage

### Search Models

Search across CivitAI and Hugging Face with a unified interface.

```bash
# Search both CivitAI and Hugging Face (default)
tsr search "flux lora"

# Search CivitAI only
tsr search "illustrious" -P civitai

# Search Hugging Face only
tsr search "flux" -P hf

# Filter by type and base model (CivitAI)
tsr search -t lora -b sdxl

# Sort by newest, limit results
tsr search -t checkpoint -s newest -n 10

# Filter by tag and period
tsr search --tag anime -p week -b illustrious

# By creator/author
tsr search -u "username"           # CivitAI
tsr search -a "stabilityai" -P hf  # Hugging Face

# SFW only with commercial use filter
tsr search --sfw --commercial sell

# Hugging Face pipeline filter
tsr search --pipeline text-to-image -P hf
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

### Hugging Face Integration

Get info and download safetensor files from Hugging Face Hub.

```bash
# Get model info and list safetensor files
tsr hf get black-forest-labs/FLUX.1-schnell

# List safetensor files only
tsr hf files black-forest-labs/FLUX.1-schnell

# Download a specific safetensor file
tsr hf dl black-forest-labs/FLUX.1-schnell -f ae.safetensors

# Download all safetensor files from a model
tsr hf dl author/model --all -o ./models
```

> **Note:** For searching Hugging Face, use `tsr search -P hf "query"` (see [Search Models](#search-models)).

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

### Generate Images (ComfyUI)

Generate images via ComfyUI backend.

```bash
# Check ComfyUI status
tsr comfy status

# List available models
tsr comfy models

# Generate an image
tsr comfy generate "a cat sitting on a windowsill"

# With model and options
tsr comfy generate "sunset over mountains" \
  -m dreamshaper_8.safetensors \
  -W 1024 -H 1024 \
  --steps 20 --cfg 7.0

# With negative prompt and seed
tsr comfy generate "portrait" \
  -n "blurry, low quality" \
  --seed 42

# Run arbitrary workflow
tsr comfy run workflow.json

# Queue management
tsr comfy queue           # View queue
tsr comfy queue --clear   # Clear queue

# View generation history
tsr comfy history
tsr comfy history PROMPT_ID
```

### Generate Images (sd.cpp)

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
tsr serve --model /path/to/model.safetensors --host 0.0.0.0 --port 51200

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
tsr remote add junkpile http://junkpile:51200

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
junkpile = "http://junkpile:51200"
local = "http://localhost:51200"

# Optional: set default remote for all commands
default_remote = "junkpile"

[comfyui]
url = "http://127.0.0.1:8188"
default_model = "dreamshaper_8.safetensors"
width = 1024
height = 1024
steps = 20
cfg = 7.0
sampler = "euler"
scheduler = "normal"

[paths]
checkpoints = "/path/to/models/checkpoints"
loras = "/path/to/models/loras"
```

Or set via environment variables:
```bash
export CIVITAI_API_KEY="your-api-key"      # For CivitAI API access
export TENSORS_API_KEY="your-server-key"   # For server authentication
export COMFYUI_URL="http://127.0.0.1:8188" # ComfyUI backend URL
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

| Option | Values | Provider |
|--------|--------|----------|
| `-P, --provider` | civitai, hf, all (default: all) | Both |
| `-t, --type` | checkpoint, lora, embedding, vae, controlnet, locon | CivitAI |
| `-b, --base` | sd14, sd15, sd2, sdxl, pony, flux, illustrious, noobai, auraflow | CivitAI |
| `-s, --sort` | downloads, rating, newest | Both |
| `-n, --limit` | Number of results (default: 25) | Both |
| `-p, --period` | all, year, month, week, day | CivitAI |
| `--tag` | Filter by tag (e.g., "anime") | Both |
| `-u, --user` | Filter by creator username | CivitAI |
| `-a, --author` | Filter by author/organization | HuggingFace |
| `--pipeline` | Pipeline tag (text-to-image, etc.) | HuggingFace |
| `--nsfw` | none, soft, mature, x | CivitAI |
| `--sfw` | Exclude NSFW content | CivitAI |
| `--commercial` | none, image, rent, sell | CivitAI |
| `--page` | Page number for pagination | CivitAI |

## Hugging Face Download Options

| Option | Description |
|--------|-------------|
| `-f, --file` | Specific file to download |
| `--all` | Download all safetensor files |
| `-o, --output` | Output directory |

## ComfyUI Generate Options

| Option | Description |
|--------|-------------|
| `-m, --model` | Checkpoint model name |
| `-W, --width` | Image width (default: 1024) |
| `-H, --height` | Image height (default: 1024) |
| `--steps` | Sampling steps (default: 20) |
| `--cfg` | CFG scale (default: 7.0) |
| `--seed` | RNG seed, -1 for random (default: -1) |
| `-n, --negative` | Negative prompt |
| `-o, --output` | Output path |
| `--sampler` | Sampler name (default: euler) |
| `--scheduler` | Scheduler name (default: normal) |

## sd.cpp Generate Options

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

**OpenAPI Documentation:** Visit `/docs` for interactive Scalar API documentation.

**Authentication:** If `TENSORS_API_KEY` is set, all endpoints except `/status` and `/docs` require authentication via:
- Header: `X-API-Key: your-key`
- Query param: `?api_key=your-key`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | Server status (public) |
| `/docs` | GET | OpenAPI documentation (public) |
| `/reload` | POST | Hot-reload with new model |
| `/api/search` | GET | Unified search (CivitAI + HuggingFace) |
| `/api/images` | GET | List gallery images |
| `/api/images/{id}` | GET | Get image file |
| `/api/images/{id}/meta` | GET | Get image metadata |
| `/api/images/{id}/edit` | POST | Update image metadata |
| `/api/images/{id}` | DELETE | Delete image |
| `/api/models` | GET | List available models |
| `/api/models/active` | GET | Get active model |
| `/api/models/switch` | POST | Switch model |
| `/api/models/loras` | GET | List available LoRAs |
| `/api/generate` | POST | Generate images (sd.cpp) |
| `/api/download` | POST | Start CivitAI download |
| `/api/comfyui/status` | GET | ComfyUI system stats |
| `/api/comfyui/queue` | GET | ComfyUI queue status |
| `/api/comfyui/queue` | DELETE | Clear ComfyUI queue |
| `/api/comfyui/models` | GET | List ComfyUI models |
| `/api/comfyui/history` | GET | ComfyUI generation history |
| `/api/comfyui/history/{id}` | GET | Get specific history entry |
| `/api/comfyui/generate` | POST | Generate image via ComfyUI |
| `/api/comfyui/workflow` | POST | Run arbitrary ComfyUI workflow |
| `/api/db/files` | GET | List local files |
| `/api/db/models` | GET | Search cached models |
| `/api/db/stats` | GET | Database statistics |

All sd-server endpoints (`/sdapi/v1/*`) are proxied through to the underlying process.

## Public API

A public API is available at `https://tensors-api.saiden.dev`. Authentication required via `X-API-Key` header.

## TypeScript Client

A TypeScript client is available for the server API:

```bash
npm install @saiden/tensors
```

```typescript
import { Configuration, SearchApi, CivitAIApi } from '@saiden/tensors'

const config = new Configuration({
  apiKey: 'your-api-key',
  // basePath defaults to https://tensors-api.saiden.dev
})

// Unified search
const search = new SearchApi(config)
const results = await search.searchModelsApiSearchGet({ query: 'flux' })

// CivitAI specific
const civitai = new CivitAIApi(config)
const models = await civitai.searchModelsApiCivitaiSearchGet({ types: 'LORA' })
```

Repository: [github.com/saiden-dev/tensors-typescript](https://github.com/saiden-dev/tensors-typescript)

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

[BSL-1.1](LICENSE)
