"""sd-server wrapper — FastAPI app for managing and proxying to sd-server."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from tensors.server.civitai_routes import create_civitai_router
from tensors.server.db_routes import create_db_router
from tensors.server.download_routes import create_download_router
from tensors.server.gallery_routes import create_gallery_router
from tensors.server.generate_routes import create_generate_router
from tensors.server.models import ServerConfig
from tensors.server.models_routes import create_models_router
from tensors.server.process import ProcessManager
from tensors.server.routes import create_router

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = ["ProcessManager", "ServerConfig", "app", "create_app"]

logger = logging.getLogger(__name__)


def create_app(config: ServerConfig | None = None) -> FastAPI:
    """Build the FastAPI application with process manager and proxy client."""
    pm = ProcessManager()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        async with httpx.AsyncClient(timeout=300) as client:
            _app.state.client = client
            if config is not None:
                pm.start(config)
                logger.info("waiting for sd-server to become ready...")
                ready = await pm.wait_ready()
                if ready:
                    logger.info("sd-server is ready")
                else:
                    logger.warning("sd-server did not become ready in time")
            yield
        pm.stop()

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
    app.include_router(create_models_router(pm))  # Must be before catch-all proxy
    app.include_router(create_download_router())  # Must be before catch-all proxy
    app.include_router(create_generate_router(pm))  # Must be before catch-all proxy
    app.include_router(create_router(pm))
    app.state.pm = pm
    return app


# Module-level app instance for uvicorn
app = create_app()
