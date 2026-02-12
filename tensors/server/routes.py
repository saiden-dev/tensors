"""FastAPI route handlers for the sd-server wrapper API."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from tensors.server.models import ReloadRequest, ServerConfig

if TYPE_CHECKING:
    from tensors.server.process import ProcessManager

logger = logging.getLogger(__name__)


def create_router(pm: ProcessManager) -> APIRouter:
    """Build a router with /status, /reload, and catch-all proxy."""
    router = APIRouter()

    @router.get("/status")
    def status() -> dict[str, Any]:
        return pm.status()

    @router.post("/reload")
    async def reload(req: ReloadRequest) -> Response:
        new_config = ServerConfig(
            model=req.model,
            port=pm.config.port if pm.config else 1234,
            args=pm.config.args if pm.config else [],
        )
        pm.stop()
        pm.start(new_config)
        ready = await pm.wait_ready()
        if not ready:
            return JSONResponse({"error": "sd-server failed to become ready", "model": req.model}, status_code=503)
        return JSONResponse({"ok": True, "model": req.model, "pid": pm.proc.pid if pm.proc else None})

    @router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
    async def proxy(request: Request, path: str) -> Response:
        if pm.proc is None or pm.proc.poll() is not None:
            return JSONResponse({"error": "sd-server is not running"}, status_code=503)
        assert pm.config is not None
        url = f"http://127.0.0.1:{pm.config.port}/{path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"
        body = await request.body()
        headers = dict(request.headers)
        headers.pop("host", None)
        client = request.app.state.client
        upstream = await client.request(
            method=request.method,
            url=url,
            headers=headers,
            content=body,
            timeout=300,
        )
        return StreamingResponse(
            content=upstream.iter_bytes(),
            status_code=upstream.status_code,
            headers=dict(upstream.headers),
        )

    return router
