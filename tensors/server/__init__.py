"""Tensors server — FastAPI app for gallery and CivitAI management."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI

from tensors.server.civitai_routes import create_civitai_router
from tensors.server.db_routes import create_db_router
from tensors.server.download_routes import create_download_router
from tensors.server.gallery_routes import create_gallery_router

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = ["app", "create_app"]

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Build the FastAPI application for gallery and model management."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        logger.info("Tensors server starting")
        yield

    app = FastAPI(title="tensors", lifespan=lifespan)

    @app.get("/status")
    async def status() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(create_civitai_router())
    app.include_router(create_db_router())
    app.include_router(create_gallery_router())
    app.include_router(create_download_router())
    return app


# Module-level app instance for uvicorn
app = create_app()
