"""FastAPI route handlers for database API endpoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel as PydanticBaseModel

from tensors.api import fetch_civitai_by_hash, fetch_civitai_model
from tensors.config import load_api_key
from tensors.db import Database

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/db", tags=["database"])


# =============================================================================
# Request/Response Models
# =============================================================================


class ScanRequest(PydanticBaseModel):
    """Request body for directory scan."""

    directory: str


class CacheRequest(PydanticBaseModel):
    """Request body for caching a model."""

    model_id: int


# =============================================================================
# File Endpoints
# =============================================================================


@router.get("/files")
def list_files() -> list[dict[str, Any]]:
    """List all local files with CivitAI info."""
    with Database() as db:
        db.init_schema()
        return db.list_local_files()


@router.get("/files/{file_id}")
def get_file(file_id: int) -> dict[str, Any]:
    """Get local file by ID."""
    with Database() as db:
        db.init_schema()
        files = db.list_local_files()
        for f in files:
            if f.get("id") == file_id:
                return f
        raise HTTPException(status_code=404, detail="File not found")


# =============================================================================
# Model Endpoints
# =============================================================================


@router.get("/models")
def search_models(
    query: str | None = Query(default=None, description="Search query"),
    model_type: str | None = Query(default=None, alias="type", description="Model type filter"),
    base_model: str | None = Query(default=None, alias="base", description="Base model filter"),
    limit: int = Query(default=20, le=100, description="Max results"),
) -> list[dict[str, Any]]:
    """Search cached models offline."""
    with Database() as db:
        db.init_schema()
        return db.search_models(
            query=query,
            model_type=model_type,
            base_model=base_model,
            limit=limit,
        )


@router.get("/models/{civitai_id}")
def get_model(civitai_id: int) -> dict[str, Any]:
    """Get cached model by CivitAI ID."""
    with Database() as db:
        db.init_schema()
        model = db.get_model(civitai_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found in cache")
        return model


# =============================================================================
# Trigger Endpoints
# =============================================================================


@router.get("/triggers")
def get_triggers_by_path(file_path: str = Query(..., description="Path to safetensor file")) -> list[str]:
    """Get trigger words for a local file by path."""
    with Database() as db:
        db.init_schema()
        return db.get_triggers(file_path)


@router.get("/triggers/{version_id}")
def get_triggers_by_version(version_id: int) -> list[str]:
    """Get trigger words for a version by CivitAI version ID."""
    with Database() as db:
        db.init_schema()
        return db.get_triggers_by_version(version_id)


# =============================================================================
# Stats Endpoint
# =============================================================================


@router.get("/stats")
def get_stats() -> dict[str, Any]:
    """Get database statistics."""
    with Database() as db:
        db.init_schema()
        return db.get_stats()


# =============================================================================
# Action Endpoints
# =============================================================================


@router.post("/scan")
def scan_directory(req: ScanRequest) -> dict[str, Any]:
    """Scan directory for safetensor files and add to database."""
    path = Path(req.directory).resolve()
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {path}")

    with Database() as db:
        db.init_schema()
        results = db.scan_directory(path)
        return {"scanned": len(results), "files": results}


@router.post("/link")
def link_files() -> dict[str, Any]:
    """Link unlinked local files to CivitAI by hash lookup."""
    api_key = load_api_key()
    linked: list[dict[str, Any]] = []

    with Database() as db:
        db.init_schema()
        unlinked = db.get_unlinked_files()

        for file_info in unlinked:
            sha256 = file_info["sha256"]
            civitai_data = fetch_civitai_by_hash(sha256, api_key)

            if civitai_data:
                version_id: int = civitai_data.get("id", 0)
                model_id: int = civitai_data.get("modelId", 0)
                if version_id and model_id:
                    db.link_file_to_civitai(file_info["id"], model_id, version_id)
                    linked.append(
                        {
                            "file_path": file_info["file_path"],
                            "model_id": model_id,
                            "version_id": version_id,
                            "name": civitai_data.get("name", ""),
                        }
                    )

    return {"linked": len(linked), "results": linked}


@router.post("/cache")
def cache_model(req: CacheRequest) -> dict[str, Any]:
    """Fetch and cache full CivitAI model data."""
    api_key = load_api_key()

    model_data = fetch_civitai_model(req.model_id, api_key)
    if not model_data:
        raise HTTPException(status_code=404, detail=f"Model {req.model_id} not found on CivitAI")

    with Database() as db:
        db.init_schema()
        internal_id = db.cache_model(model_data)

    return {
        "model_id": req.model_id,
        "internal_id": internal_id,
        "name": model_data.get("name"),
    }


def create_db_router() -> APIRouter:
    """Return the database API router."""
    return router
