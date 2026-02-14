"""sd-server wrapper — FastAPI app for managing and proxying to sd-server."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import httpx
from fastapi import FastAPI

from tensors.server.db_routes import create_db_router
from tensors.server.models import ServerConfig
from tensors.server.process import ProcessManager
from tensors.server.routes import create_router

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = ["ProcessManager", "ServerConfig", "create_app"]

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
    app.include_router(create_db_router())  # Must be first to avoid catch-all conflict
    app.include_router(create_router(pm))
    app.state.pm = pm
    return app
