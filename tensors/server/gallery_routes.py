"""FastAPI route handlers for image gallery endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel as PydanticBaseModel

from tensors.server.gallery import Gallery

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/images", tags=["Gallery"])

# Shared gallery instance
_gallery: Gallery | None = None


def get_gallery() -> Gallery:
    """Get or create the gallery instance."""
    global _gallery  # noqa: PLW0603
    if _gallery is None:
        _gallery = Gallery()
    return _gallery


# =============================================================================
# Request/Response Models
# =============================================================================


class MetadataUpdate(PydanticBaseModel):
    """Request body for updating image metadata."""

    tags: list[str] | None = None
    notes: str | None = None
    rating: int | None = None
    favorite: bool | None = None


# =============================================================================
# Gallery Endpoints
# =============================================================================


@router.get("")
def list_images(
    limit: int = Query(default=50, le=200, description="Max images to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    newest_first: bool = Query(default=True, description="Sort newest first"),
) -> dict[str, Any]:
    """List images in the gallery, paginated."""
    gallery = get_gallery()
    images = gallery.list_images(limit=limit, offset=offset, newest_first=newest_first)
    total = gallery.count()

    return {
        "images": [img.to_dict() for img in images],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{image_id}")
def get_image(image_id: str) -> FileResponse:
    """Get an image file by ID."""
    gallery = get_gallery()
    image = gallery.get_image(image_id)

    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(
        path=image.path,
        media_type="image/png",
        filename=image.path.name,
    )


@router.get("/{image_id}/meta")
def get_image_metadata(image_id: str) -> dict[str, Any]:
    """Get metadata for an image."""
    gallery = get_gallery()
    image = gallery.get_image(image_id)

    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    metadata = gallery.get_metadata(image_id) or {}
    return {
        "id": image_id,
        "path": str(image.path),
        "created_at": image.created_at,
        "metadata": metadata,
    }


@router.post("/{image_id}/edit")
def edit_image_metadata(image_id: str, updates: MetadataUpdate) -> dict[str, Any]:
    """Update metadata for an image."""
    gallery = get_gallery()

    # Build update dict from non-None values
    update_dict: dict[str, Any] = {}
    if updates.tags is not None:
        update_dict["tags"] = updates.tags
    if updates.notes is not None:
        update_dict["notes"] = updates.notes
    if updates.rating is not None:
        update_dict["rating"] = updates.rating
    if updates.favorite is not None:
        update_dict["favorite"] = updates.favorite

    result = gallery.update_metadata(image_id, update_dict)
    if result is None:
        raise HTTPException(status_code=404, detail="Image not found")

    return {"id": image_id, "metadata": result}


@router.delete("/{image_id}")
def delete_image(image_id: str) -> dict[str, Any]:
    """Delete an image and its metadata."""
    gallery = get_gallery()
    deleted = gallery.delete_image(image_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Image not found")

    return {"deleted": True, "id": image_id}


@router.get("/stats/summary")
def gallery_stats() -> dict[str, Any]:
    """Get gallery statistics."""
    gallery = get_gallery()
    return {
        "total_images": gallery.count(),
        "gallery_dir": str(gallery.gallery_dir),
    }


def create_gallery_router() -> APIRouter:
    """Return the gallery API router."""
    return router
