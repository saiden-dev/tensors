# Plan: tsr Server/Client Architecture with Models Database

Transform `tsr` into a unified server/client tool for remote image generation on junkpile (ROCm GPU server), with model-specific Docker images, a models database, and full image management capabilities.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         junkpile (server)                           │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────┐    ┌────────────────────────────────────┐ │
│  │ sd-server:pony       │◄───│  tsr serve (FastAPI)               │ │
│  │ sd-server:illustrious│    │  - POST /api/generate              │ │
│  │ sd-server:flux       │    │  - GET/POST/DELETE /api/images     │ │
│  │ (Docker/ROCm)        │    │  - GET /api/models, /api/loras     │ │
│  └──────────────────────┘    │  - POST /api/download (CivitAI)    │ │
│                              │  - GET/POST /api/db/* (models.db)  │ │
│  ┌──────────────────────┐    │                                    │ │
│  │     models.db        │◄───┤                                    │ │
│  │ (SQLite: CivitAI +   │    └────────────────────────────────────┘ │
│  │  local file cache)   │                    ▲                      │
│  └──────────────────────┘                    │ :8080                │
└──────────────────────────────────────────────│──────────────────────┘
                                               │ HTTP
┌──────────────────────────────────────────────│──────────────────────┐
│                      local machine           │                      │
├──────────────────────────────────────────────┼──────────────────────┤
│  tsr generate "prompt" --remote junkpile                            │
│  tsr images list --remote junkpile                                  │
│  tsr images delete <id> --remote junkpile                           │
│  tsr models list --remote junkpile                                  │
│  tsr models switch pony --remote junkpile                           │
│  tsr dl 999258 --remote junkpile                                    │
│  tsr db search "pony" --remote junkpile                             │
└─────────────────────────────────────────────────────────────────────┘
```

## Phase 1: Model-Specific Docker Images

### Description
Create parameterized Dockerfiles that produce model-family-specific images with optimal defaults baked in. Each image knows its best sampler, scheduler, resolution, CFG scale, and negative prompt.

### Steps

#### Step 1.1: Create model defaults configuration
- **Objective**: Define optimal generation parameters per model family
- **Files**: `rocm-docker/model-defaults.toml`
- **Dependencies**: None
- **Implementation**:
  - Create TOML with sections: `[sd15]`, `[sdxl]`, `[pony]`, `[illustrious]`, `[flux]`
  - Each section: `width`, `height`, `steps`, `cfg_scale`, `sampler`, `scheduler`, `negative_prompt`
  - Reference: `models.md` has the research already

#### Step 1.2: Parameterize Dockerfile for model families
- **Objective**: Single Dockerfile that builds model-specific images via build args
- **Files**: `rocm-docker/Dockerfile.sd-server`
- **Dependencies**: Step 1.1
- **Implementation**:
  - Add `ARG MODEL_FAMILY=sdxl` with validation
  - Inject defaults from model-defaults.toml as ENV vars
  - Keep entrypoint.sh flexible (env vars override baked defaults)
  - Build targets: `sd-server:pony`, `sd-server:illustrious`, `sd-server:flux`

#### Step 1.3: Add build script for all model variants
- **Objective**: Automated build of all model-specific images
- **Files**: `rocm-docker/build-all.sh`
- **Dependencies**: Step 1.2
- **Implementation**:
  - Loop through model families, build each with appropriate args
  - Tag pattern: `sd-server:{family}`
  - Push to registry (optional)

## Phase 2: Models Database in tensors

### Description
Move the SQLite database from rocm-docker into tensors as a proper module with full CRUD operations, exposed via CLI and API.

### Steps

#### Step 2.1: Create database module
- **Objective**: SQLite wrapper with schema management and CRUD operations
- **Files**: `tensors/db.py`, `tensors/schema.sql`
- **Dependencies**: None
- **Implementation**:
  - Move schema from `rocm-docker/import_models.py` to `tensors/schema.sql`
  - `Database` class with connection management, migrations
  - Methods: `scan_files()`, `link_civitai()`, `cache_model()`, `search_models()`, `get_triggers()`
  - Use existing `tensors/api.py` for CivitAI fetches
  - Config: `DATA_DIR / "models.db"`

#### Step 2.2: Add db CLI commands
- **Objective**: Expose database operations via `tsr db` subcommand group
- **Files**: `tensors/cli.py`
- **Dependencies**: Step 2.1
- **Implementation**:
  - `tsr db scan <directory>` — Scan safetensors, compute hashes, store metadata
  - `tsr db link` — Match unlinked files to CivitAI by hash
  - `tsr db cache <model_id>` — Fetch and cache full CivitAI model data
  - `tsr db list` — List local files with CivitAI info (uses view)
  - `tsr db search <query>` — Search cached models offline
  - `tsr db triggers <file>` — Show trigger words for a LoRA
  - All commands support `--json` output

#### Step 2.3: Add database API endpoints
- **Objective**: Expose database queries via HTTP API
- **Files**: `tensors/server/routes.py`
- **Dependencies**: Step 2.1
- **Implementation**:
  - `GET /api/db/files` — List local files
  - `GET /api/db/models` — Search cached models
  - `GET /api/db/models/{id}` — Get model details
  - `GET /api/db/triggers/{file_path}` — Get trigger words
  - `POST /api/db/scan` — Trigger directory scan
  - `POST /api/db/link` — Trigger CivitAI linking

## Phase 3: Enhanced Server API

### Description
Extend the existing FastAPI server with image gallery management, model switching, and CivitAI download capabilities.

### Steps

#### Step 3.1: Add image gallery endpoints
- **Objective**: CRUD for generated images with metadata
- **Files**: `tensors/server/routes.py`, `tensors/server/gallery.py`
- **Dependencies**: Phase 2
- **Implementation**:
  - `GET /api/images` — List images (paginated, newest first), metadata from sidecar JSON
  - `GET /api/images/{id}` — Get image file
  - `GET /api/images/{id}/meta` — Get generation metadata
  - `DELETE /api/images/{id}` — Delete image + sidecar
  - `POST /api/images/{id}/edit` — Update metadata (tags, notes)
  - Images stored in `DATA_DIR / "gallery/"` with `{timestamp}_{seed}.png` + `.json` sidecar
  - Gallery config: output directory, max storage, cleanup policy

#### Step 3.2: Add model management endpoints
- **Objective**: List available models, switch active model, hot-reload
- **Files**: `tensors/server/routes.py`
- **Dependencies**: None
- **Implementation**:
  - `GET /api/models` — List available checkpoints (scan models directory)
  - `GET /api/models/active` — Current loaded model info
  - `POST /api/models/switch` — Switch model (calls sd-server reload or container swap)
  - `GET /api/loras` — List available LoRAs
  - Container strategy: either reload sd-server with new model, or run multiple containers per model family

#### Step 3.3: Add CivitAI download proxy endpoint
- **Objective**: Download models directly to server via API
- **Files**: `tensors/server/routes.py`
- **Dependencies**: Step 2.1
- **Implementation**:
  - `POST /api/download` — Accept model/version ID or hash, download to appropriate directory
  - Stream progress via SSE or polling endpoint
  - Auto-scan and link after download
  - Use existing `tensors/api.py` download logic

#### Step 3.4: Enhance generation endpoint
- **Objective**: Full generation control with gallery integration
- **Files**: `tensors/server/routes.py`
- **Dependencies**: Step 3.1
- **Implementation**:
  - `POST /api/generate` — Forward to sd-server, save result to gallery
  - Accept all sd-server params: prompt, negative, width, height, steps, cfg, sampler, scheduler, seed, loras
  - Return image ID, metadata, and base64 (optional)
  - Support batch generation
  - Auto-increment seed for batches

## Phase 4: Client Mode for tsr CLI

### Description
Add `--remote` flag to existing commands to talk to a remote tsr server instead of local operations or direct CivitAI API.

### Steps

#### Step 4.1: Create remote client module
- **Objective**: HTTP client wrapper for tsr server API
- **Files**: `tensors/client.py`
- **Dependencies**: Phase 3
- **Implementation**:
  - `TsrClient` class wrapping httpx
  - Methods mirror server endpoints: `generate()`, `list_images()`, `delete_image()`, `list_models()`, `switch_model()`, `download()`, `db_search()`
  - Handle streaming responses for downloads
  - Auth: API key header (optional, future)

#### Step 4.2: Add remote configuration
- **Objective**: Configure remote server URL in config.toml
- **Files**: `tensors/config.py`
- **Dependencies**: None
- **Implementation**:
  - Add `[remotes]` section: `junkpile = "http://junkpile:8080"`
  - `--remote <name>` flag resolves to URL from config
  - `--remote <url>` accepts direct URL
  - Default remote configurable: `default_remote = "junkpile"`

#### Step 4.3: Update CLI commands with --remote support
- **Objective**: All relevant commands work against remote server
- **Files**: `tensors/cli.py`
- **Dependencies**: Step 4.1, Step 4.2
- **Implementation**:
  - `tsr generate` — Use remote if `--remote`, else local sd-server
  - `tsr images list/delete/show` — New subcommand group for gallery
  - `tsr models list/switch` — New subcommand group
  - `tsr dl` — Proxy through remote if `--remote`
  - `tsr db *` — All db commands support `--remote`
  - Consistent UX: same output format local vs remote

## Phase 5: Docker Deployment Automation

### Description
Scripts and configs for deploying and managing sd-server containers on junkpile.

### Steps

#### Step 5.1: Create docker-compose for multi-model setup
- **Objective**: Run multiple sd-server containers, one per model family
- **Files**: `rocm-docker/docker-compose.yml`
- **Dependencies**: Phase 1
- **Implementation**:
  - Service per model family: `sd-pony`, `sd-illustrious`, `sd-flux`
  - Shared volumes: `/models`, `/loras`, `/output`
  - Each on different port: 1234, 1235, 1236
  - tsr server routes to correct container based on active model
  - Health checks

#### Step 5.2: Create deployment script
- **Objective**: One-command deploy/update on junkpile
- **Files**: `rocm-docker/deploy.sh`
- **Dependencies**: Step 5.1
- **Implementation**:
  - Copy files to junkpile
  - Build images
  - Pull models if missing
  - Start containers
  - Start tsr server
  - Verify health

#### Step 5.3: Add systemd service for tsr server
- **Objective**: Auto-start tsr server on boot
- **Files**: `rocm-docker/tsr-server.service`
- **Dependencies**: Step 5.2
- **Implementation**:
  - systemd unit file
  - Depends on docker.service
  - Restart on failure
  - Install instructions

## Phase 6: Tests

### Steps

#### Step 6.1: Test database module
- **Files**: `tests/test_db.py`
- **Dependencies**: Phase 2
- **Implementation**:
  - Test schema creation, migrations
  - Test CRUD operations
  - Test CivitAI linking logic
  - Use temp database

#### Step 6.2: Test server API endpoints
- **Files**: `tests/test_server.py`
- **Dependencies**: Phase 3
- **Implementation**:
  - Use FastAPI TestClient
  - Mock sd-server responses
  - Test gallery CRUD
  - Test model listing/switching

#### Step 6.3: Test client module
- **Files**: `tests/test_client.py`
- **Dependencies**: Phase 4
- **Implementation**:
  - Mock HTTP responses with respx
  - Test all client methods
  - Test error handling
