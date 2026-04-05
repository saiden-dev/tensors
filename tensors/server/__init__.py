"""Tensors server — FastAPI app for gallery and CivitAI management."""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from scalar_fastapi import get_scalar_api_reference

from tensors.config import get_server_api_key
from tensors.server.auth_routes import create_auth_router
from tensors.server.civitai_routes import create_civitai_router
from tensors.server.comfyui_api_routes import create_comfyui_api_router
from tensors.server.comfyui_routes import create_comfyui_router
from tensors.server.db_routes import create_db_router
from tensors.server.download_routes import create_download_router
from tensors.server.gallery_routes import create_gallery_router
from tensors.server.search_routes import create_search_router

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi.responses import HTMLResponse

__all__ = ["app", "create_app"]

# Configure logging for tensors package
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     %(name)s - %(message)s",
    stream=sys.stdout,
)
# Set tensors loggers to INFO
logging.getLogger("tensors").setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Build the FastAPI application for gallery and model management."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        api_key = get_server_api_key()
        if api_key:
            logger.info("Tensors server starting (auth enabled)")
        else:
            logger.info("Tensors server starting (no auth)")
        yield

    app = FastAPI(
        title="tensors",
        description="API for CivitAI model management and image gallery",
        version="0.1.18",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )

    # CORS — configurable via CORS_ORIGINS env var (comma-separated, default: *)
    cors_raw = os.environ.get("CORS_ORIGINS", "*")
    cors_origins = ["*"] if cors_raw.strip() == "*" else [o.strip() for o in cors_raw.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Public endpoints (no auth)
    @app.get("/status")
    async def status() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/docs", include_in_schema=False)
    async def scalar_docs() -> HTMLResponse:
        return get_scalar_api_reference(
            openapi_url=app.openapi_url or "/openapi.json",
            title="tensors API",
        )

    # Shared OAuth auth (no API key required)
    app.include_router(create_auth_router())

    # ComfyUI proxy (handles its own session auth)
    app.include_router(create_comfyui_router())

    # Protected routers (API key auth)
    from tensors.server.auth import verify_api_key  # noqa: PLC0415

    app.include_router(create_search_router(), dependencies=[Depends(verify_api_key)])
    app.include_router(create_civitai_router(), dependencies=[Depends(verify_api_key)])
    app.include_router(create_db_router(), dependencies=[Depends(verify_api_key)])
    app.include_router(create_gallery_router(), dependencies=[Depends(verify_api_key)])
    app.include_router(create_download_router(), dependencies=[Depends(verify_api_key)])
    app.include_router(create_comfyui_api_router(), dependencies=[Depends(verify_api_key)])
    return app


# Module-level app instance for uvicorn
app = create_app()
