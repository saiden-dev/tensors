"""sd-server wrapper — FastAPI app for proxying to an external sd-server."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from tensors.config import get_sd_server_api_key, get_sd_server_url
from tensors.server.civitai_routes import create_civitai_router
from tensors.server.db_routes import create_db_router
from tensors.server.download_routes import create_download_router
from tensors.server.gallery_routes import create_gallery_router
from tensors.server.generate_routes import create_generate_router
from tensors.server.models_routes import create_models_router
from tensors.server.routes import create_router

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = ["app", "create_app"]

logger = logging.getLogger(__name__)


def create_app(sd_server_url: str | None = None) -> FastAPI:
    """Build the FastAPI application that proxies to an external sd-server.

    Args:
        sd_server_url: URL of the sd-server to proxy to. If None, uses
                       get_sd_server_url() to resolve from env/config.
    """
    backend_url = sd_server_url or get_sd_server_url()
    api_key = get_sd_server_api_key()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        _app.state.sd_server_url = backend_url
        _app.state.sd_server_api_key = api_key
        logger.info(f"Proxying to sd-server at: {backend_url}")
        if api_key:
            logger.info("Using API key authentication for sd-server")
        async with httpx.AsyncClient(timeout=300) as client:
            _app.state.client = client
            yield

    app = FastAPI(title="sd-server wrapper", lifespan=lifespan)

    # Serve Vue UI static files
    static_dir = Path(__file__).parent / "static"
    assets_dir = static_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/", include_in_schema=False)
    async def gallery_ui() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/vite.svg", include_in_schema=False)
    async def vite_icon() -> FileResponse:
        return FileResponse(static_dir / "vite.svg")

    app.include_router(create_civitai_router())  # Must be before catch-all proxy
    app.include_router(create_db_router())  # Must be before catch-all proxy
    app.include_router(create_gallery_router())  # Must be before catch-all proxy
    app.include_router(create_models_router())  # Must be before catch-all proxy
    app.include_router(create_download_router())  # Must be before catch-all proxy
    app.include_router(create_generate_router())  # Must be before catch-all proxy
    app.include_router(create_router())
    return app


# Module-level app instance for uvicorn
app = create_app()
