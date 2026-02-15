# Deploy Tensors to Junkpile

Deploy tensors API server to junkpile.

Run the deploy script:

```bash
./scripts/deploy.sh
```

## What it does

1. **Sync code** - Rsyncs Python code to `/opt/tensors/app/`
2. **Fix permissions** - Sets ownership to `tensors:tensors`
3. **Restart tensors** - Runs `sudo systemctl restart tensors`
4. **Verify tensors** - Checks service is running

## Service Structure

| Item | Value |
|------|-------|
| User/Group | `tensors:tensors` |
| Install path | `/opt/tensors/app` |
| Venv | `/opt/tensors/venv` |
| Service | `tensors.service` |
| Port | 51200 |

## API Endpoints

- `GET /status` - Health check
- `GET /api/civitai/*` - CivitAI search and fetch
- `GET /api/db/*` - Database management
- `GET /api/images/*` - Image gallery
- `POST /api/download` - Download models
