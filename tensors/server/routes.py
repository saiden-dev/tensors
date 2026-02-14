"""FastAPI route handlers for the sd-server wrapper API."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from tensors.server.sd_client import get_sd_headers

logger = logging.getLogger(__name__)


def create_router() -> APIRouter:
    """Build a router with /status and catch-all proxy."""
    router = APIRouter()

    @router.get("/status")
    async def status(request: Request) -> dict[str, Any]:
        """Check if the external sd-server is reachable."""
        sd_server_url = request.app.state.sd_server_url
        try:
            headers = get_sd_headers(request)
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(sd_server_url, headers=headers)
                return {
                    "status": "ok",
                    "sd_server_url": sd_server_url,
                    "sd_server_status": r.status_code,
                }
        except httpx.HTTPError as e:
            return {
                "status": "error",
                "sd_server_url": sd_server_url,
                "error": str(e),
            }

    @router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
    async def proxy(request: Request, path: str) -> Response:
        """Proxy all requests to the external sd-server."""
        sd_server_url = request.app.state.sd_server_url
        url = f"{sd_server_url}/{path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"

        body = await request.body()
        headers = dict(request.headers)
        headers.pop("host", None)
        # Add API key if configured
        headers.update(get_sd_headers(request))
        client = request.app.state.client

        try:
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
        except httpx.ConnectError:
            return JSONResponse(
                {"error": f"Cannot connect to sd-server at {sd_server_url}"},
                status_code=503,
            )
        except httpx.TimeoutException:
            return JSONResponse(
                {"error": f"Timeout connecting to sd-server at {sd_server_url}"},
                status_code=504,
            )

    return router
