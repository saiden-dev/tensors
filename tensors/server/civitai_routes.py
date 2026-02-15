"""FastAPI route handlers for CivitAI API endpoints."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import APIRouter, Query, Response
from fastapi.responses import JSONResponse

from tensors.config import CIVITAI_API_BASE, load_api_key
from tensors.db import Database

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/civitai", tags=["CivitAI"])


def _get_headers(api_key: str | None) -> dict[str, str]:
    """Get headers for CivitAI API requests."""
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


@router.get("/search", response_model=None)
async def search_models(
    query: str | None = Query(default=None, description="Search query"),
    types: str | None = Query(default=None, description="Model type (Checkpoint, LORA, LoCon, etc.)"),
    base_models: str | None = Query(default=None, alias="baseModels", description="Base model (SD 1.5, SDXL 1.0, Pony, etc.)"),
    sort: str = Query(default="Most Downloaded", description="Sort order"),
    limit: int = Query(default=20, le=100, description="Max results"),
    nsfw: bool = Query(default=True, description="Include NSFW models"),
) -> dict[str, Any] | Response:
    """Search CivitAI models."""
    api_key = load_api_key()

    params: dict[str, Any] = {
        "limit": min(limit, 100),
        "nsfw": str(nsfw).lower(),
        "sort": sort,
    }

    if query:
        params["query"] = query
    if types:
        params["types"] = types
    if base_models:
        params["baseModels"] = base_models

    url = f"{CIVITAI_API_BASE}/models"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params, headers=_get_headers(api_key))
            response.raise_for_status()
            result: dict[str, Any] = response.json()

            # Cache all models from search results
            items = result.get("items", [])
            if items:
                try:
                    with Database() as db:
                        db.init_schema()
                        for model_data in items:
                            db.cache_model(model_data)
                except Exception as e:
                    logger.warning("Failed to cache search results: %s", e)

            return result
    except httpx.HTTPStatusError as e:
        logger.error("CivitAI API error: %s", e.response.status_code)
        return JSONResponse({"error": f"API error: {e.response.status_code}"}, status_code=e.response.status_code)
    except httpx.RequestError as e:
        logger.error("CivitAI request error: %s", e)
        return JSONResponse({"error": f"Request error: {e}"}, status_code=500)


@router.get("/model/{model_id}", response_model=None)
async def get_model(model_id: int) -> dict[str, Any] | Response:
    """Get model details from CivitAI and cache to database."""
    api_key = load_api_key()
    url = f"{CIVITAI_API_BASE}/models/{model_id}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=_get_headers(api_key))
            response.raise_for_status()
            result: dict[str, Any] = response.json()

            # Cache the model data to database
            try:
                with Database() as db:
                    db.init_schema()
                    db.cache_model(result)
            except Exception as e:
                logger.warning("Failed to cache model %d: %s", model_id, e)

            return result
    except httpx.HTTPStatusError:
        return JSONResponse({"error": "Model not found"}, status_code=404)
    except httpx.RequestError as e:
        return JSONResponse({"error": f"Request error: {e}"}, status_code=500)


def create_civitai_router() -> APIRouter:
    """Return the CivitAI API router."""
    return router
