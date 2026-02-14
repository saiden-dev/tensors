# TODO: tsr Server/Client Architecture

## Phase 1: Model-Specific Docker Images
- [ ] Step 1.1: Create `rocm-docker/model-defaults.toml` (optimal params per model family)
- [ ] Step 1.2: Parameterize `Dockerfile.sd-server` with `MODEL_FAMILY` build arg
- [ ] Step 1.3: Create `rocm-docker/build-all.sh` (build all model variants)

## Phase 2: Models Database in tensors
- [ ] Step 2.1: Create `tensors/db.py` + `tensors/schema.sql` (SQLite wrapper, schema, CRUD)
- [ ] Step 2.2: Add `tsr db` CLI commands (scan, link, cache, list, search, triggers)
- [ ] Step 2.3: Add `/api/db/*` endpoints (files, models, triggers, scan, link)

## Phase 3: Enhanced Server API
- [ ] Step 3.1: Add `/api/images` gallery endpoints (list, get, delete, edit)
- [ ] Step 3.2: Add `/api/models` endpoints (list, active, switch, loras)
- [ ] Step 3.3: Add `/api/download` endpoint (CivitAI proxy download)
- [ ] Step 3.4: Enhance `/api/generate` (gallery integration, full params)

## Phase 4: Client Mode for tsr CLI
- [ ] Step 4.1: Create `tensors/client.py` (TsrClient HTTP wrapper)
- [ ] Step 4.2: Add `[remotes]` config section + `--remote` flag support
- [ ] Step 4.3: Update CLI commands with `--remote` support (generate, images, models, dl, db)

## Phase 5: Docker Deployment Automation
- [ ] Step 5.1: Create `rocm-docker/docker-compose.yml` (multi-model setup)
- [ ] Step 5.2: Create `rocm-docker/deploy.sh` (one-command deploy)
- [ ] Step 5.3: Create `rocm-docker/tsr-server.service` (systemd unit)

## Phase 6: Tests
- [ ] Step 6.1: `tests/test_db.py` (database module tests)
- [ ] Step 6.2: `tests/test_server.py` (API endpoint tests)
- [ ] Step 6.3: `tests/test_client.py` (client module tests)

---

## Quick Reference

### ROCm Docker Run (Unrestricted)
```bash
docker run -d --name sd-server \
  --privileged \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size=8G \
  -v /path/to/models:/models \
  -p 1234:1234 \
  -e MODEL=/models/model.safetensors \
  sd-server:rocm
```

### sd-server API Endpoints
- `POST /sdapi/v1/txt2img` — Generate image (A1111 compatible)
- `POST /sdapi/v1/img2img` — Edit image
- `GET /sdapi/v1/loras` — List LoRAs
- `GET /sdapi/v1/samplers` — List samplers
- `GET /sdapi/v1/schedulers` — List schedulers

### Model Family Defaults (from models.md)
| Family | Resolution | Steps | CFG | Sampler | Scheduler |
|--------|------------|-------|-----|---------|-----------|
| SD 1.5 | 512×512 | 20-30 | 7-8 | DPM++ 2M | Karras |
| SDXL | 1024×1024 | 25-30 | 5-7 | DPM++ 2M | Karras |
| Pony | 1024×1024 | 25-30 | 5-7 | Euler a | simple |
| Illustrious | 1024×1024 | 25-30 | 5-7 | Euler a | simple |
| Flux | 1024×1024 | 20-30 | 1-3 | Euler | simple |
