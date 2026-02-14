# Tensors Refactoring Task

## Goal

Transform tensors from a "sd-server wrapper" to a **pure client** (CLI + UI) that talks to sd-server API directly.

- Default API URL: `https://sd-api.saiden.dev`
- API key auth via config: `sd_server_api_key`
- No wrapping/proxying/restarting of sd-server

## Current Architecture

### CLI Commands

| Command | What it does | Talks to |
|---------|-------------|----------|
| `tsr info <file>` | Read safetensor metadata, fetch CivitAI info | Local file + CivitAI API |
| `tsr search` | Search CivitAI models | CivitAI API |
| `tsr get <id>` | Fetch model info from CivitAI | CivitAI API |
| `tsr dl <id>` | Download model from CivitAI | CivitAI API |
| `tsr config` | Manage config (API keys, remotes) | Local config |
| `tsr generate` | Generate images | sd-server API (direct or via wrapper) |
| `tsr status` | Show wrapper status | tensors wrapper API |
| **`tsr reload`** | **Reload sd-server with new model** | **tensors wrapper API** |
| `tsr serve` | Start wrapper API (proxies to sd-server) | Starts FastAPI server |
| `tsr db` | Manage local models DB | Local SQLite |
| `tsr images` | Manage remote gallery | tensors wrapper API |
| `tsr models` | List models on remote | tensors wrapper API |
| `tsr remote` | Manage remote server config | Local config |

### Server (`tsr serve`)

Starts a FastAPI server that:
1. **Proxies** all requests to sd-server (catch-all route)
2. **Serves Vue UI** at `/`
3. **Adds features**: gallery, CivitAI search, model listing, downloads
4. **Has a `/reload` endpoint** that proxies to sd-server

## What to Remove

| Remove | Reason |
|--------|--------|
| `tsr reload` command | sd-server manages its own models |
| `/reload` route in server | Same |
| `switch_model` in client.py | Same |
| Proxy wrapper concept | tensors should call API directly, not proxy |

## What to Keep/Refactor

| Keep | Change |
|------|--------|
| `tsr serve` | Just serve Vue UI, no proxying |
| `tsr generate` | Call sd-server API directly with API key |
| Vue UI | Call sd-server API directly (already does via `/api/*`) |
| `tsr models` | List models from sd-server API directly |

## Already Done

- [x] Added `get_sd_server_api_key()` to config.py
- [x] Added `sd_server_api_key` to app state in server/__init__.py
- [x] Created `sd_client.py` with `get_sd_headers()` helper
- [x] Updated `generate_routes.py` to use API key headers
- [x] Updated `routes.py` to use API key headers
- [x] Updated `models_routes.py` to use API key headers
- [x] Created local config at `~/.xdg/tensors/config.toml` with sd-api.saiden.dev

## Still TODO

- [ ] Remove `tsr reload` command from cli.py
- [ ] Remove `/reload` route if it exists
- [ ] Remove `switch_model` from client.py
- [ ] Decide: keep `tsr serve` as UI server or remove entirely?
- [ ] Update Vue UI to call sd-server API directly (not via `/api/*` proxy)
- [ ] Clean up unused wrapper/proxy code
