# Deploy Tensors to Junkpile

Build, deploy, and restart tensors on junkpile with verification.

Run the deploy script:

```bash
./scripts/deploy.sh
```

## What it does

1. **Build UI** - Runs `npm run build` in `tensors/server/ui/`
2. **Sync code** - Rsyncs Python code and static files to junkpile
3. **Restart tensors** - Runs `sudo systemctl restart tensors`
4. **Verify tensors** - Checks `/api/models/status` responds
5. **Verify sd-server** - Checks sd-server is active
6. **Verify external** - Checks `sd-api.saiden.dev` responds

## Endpoints

- **UI**: https://tensors.saiden.dev
- **API**: https://sd-api.saiden.dev (requires `X-API-Key` header)
