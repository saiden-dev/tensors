"""FastAPI route handlers for CivitAI download proxy."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel as PydanticBaseModel

from tensors.api import download_model_with_progress, fetch_civitai_by_hash, fetch_civitai_model, fetch_civitai_model_version
from tensors.config import MODELS_DIR, load_api_key
from tensors.db import Database

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/download", tags=["Download"])

# Track active downloads
_active_downloads: dict[str, dict[str, Any]] = {}


# =============================================================================
# Request/Response Models
# =============================================================================


class DownloadRequest(PydanticBaseModel):
    """Request body for downloading a model."""

    version_id: int | None = None
    model_id: int | None = None
    hash: str | None = None
    output_dir: str | None = None  # Override default path


# =============================================================================
# Helper Functions
# =============================================================================


def _resolve_version_id(
    version_id: int | None,
    model_id: int | None,
    hash_val: str | None,
    api_key: str | None,
) -> tuple[int | None, dict[str, Any] | None]:
    """Resolve version ID and get version info."""
    if version_id:
        info = fetch_civitai_model_version(version_id, api_key)
        return version_id, info

    if hash_val:
        info = fetch_civitai_by_hash(hash_val.upper(), api_key)
        if info:
            return info.get("id"), info
        return None, None

    if model_id:
        model_data = fetch_civitai_model(model_id, api_key)
        if model_data:
            versions = model_data.get("modelVersions", [])
            if versions:
                latest = versions[0]
                return latest.get("id"), latest
        return None, None

    return None, None


def _get_output_dir(version_info: dict[str, Any], override: str | None) -> Path:
    """Determine output directory based on model type."""
    if override:
        return Path(override)

    model_type = version_info.get("model", {}).get("type", "Checkpoint")

    # Map type to directory
    type_dirs = {
        "Checkpoint": MODELS_DIR / "checkpoints",
        "LORA": MODELS_DIR / "loras",
        "LoCon": MODELS_DIR / "loras",
        "TextualInversion": MODELS_DIR / "embeddings",
        "VAE": MODELS_DIR / "vae",
        "Controlnet": MODELS_DIR / "controlnet",
    }

    return type_dirs.get(model_type, MODELS_DIR / "other")


_KB = 1024
_MB = _KB * 1024
_GB = _MB * 1024


def _format_size(size_bytes: int) -> str:
    """Format bytes as human readable string."""
    if size_bytes >= _GB:
        return f"{size_bytes / _GB:.1f} GB"
    if size_bytes >= _MB:
        return f"{size_bytes / _MB:.1f} MB"
    if size_bytes >= _KB:
        return f"{size_bytes / _KB:.1f} KB"
    return f"{size_bytes} B"


def _do_download(
    version_id: int,
    dest_path: Path,
    api_key: str | None,
    download_id: str,
) -> None:
    """Background task to perform the download."""
    try:
        _active_downloads[download_id]["status"] = "downloading"
        _active_downloads[download_id]["downloaded"] = 0
        _active_downloads[download_id]["total"] = 0
        _active_downloads[download_id]["speed"] = 0
        _active_downloads[download_id]["progress"] = 0

        def on_progress(downloaded: int, total: int, speed: float) -> None:
            """Update progress in active downloads dict."""
            _active_downloads[download_id]["downloaded"] = downloaded
            _active_downloads[download_id]["total"] = total
            _active_downloads[download_id]["speed"] = speed
            _active_downloads[download_id]["downloaded_str"] = _format_size(downloaded)
            _active_downloads[download_id]["total_str"] = _format_size(total) if total > 0 else "Unknown"
            _active_downloads[download_id]["speed_str"] = f"{_format_size(int(speed))}/s"
            if total > 0:
                _active_downloads[download_id]["progress"] = round(100 * downloaded / total, 1)

        success = download_model_with_progress(version_id, dest_path, api_key, on_progress, resume=True)

        if success:
            _active_downloads[download_id]["status"] = "completed"
            _active_downloads[download_id]["progress"] = 100
            _active_downloads[download_id]["path"] = str(dest_path)

            # Auto-scan and link the downloaded file
            _auto_link_file(dest_path, api_key)
        else:
            _active_downloads[download_id]["status"] = "failed"
            _active_downloads[download_id]["error"] = "Download failed"

    except Exception as e:
        logger.exception("Download failed")
        _active_downloads[download_id]["status"] = "failed"
        _active_downloads[download_id]["error"] = str(e)


def _auto_link_file(file_path: Path, api_key: str | None) -> None:
    """Auto-scan and link the downloaded file to CivitAI."""
    try:
        with Database() as db:
            db.init_schema()
            # Scan the single file
            results = db.scan_directory(file_path.parent)

            # Find and link the new file
            for result in results:
                if result["file_path"] == str(file_path):
                    sha256 = result["sha256"]
                    civitai_data = fetch_civitai_by_hash(sha256, api_key)
                    if civitai_data:
                        version_id = civitai_data.get("id", 0)
                        model_id = civitai_data.get("modelId", 0)
                        if version_id and model_id:
                            db.link_file_to_civitai(result["id"], model_id, version_id)
    except Exception:
        logger.exception("Auto-link failed")


# =============================================================================
# Download Endpoints
# =============================================================================


@router.post("")
def start_download(req: DownloadRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    """Start a model download (async with progress tracking)."""
    api_key = load_api_key()

    # Resolve version ID
    version_id, version_info = _resolve_version_id(
        req.version_id,
        req.model_id,
        req.hash,
        api_key,
    )

    if not version_id or not version_info:
        raise HTTPException(status_code=404, detail="Model/version not found on CivitAI")

    # Get output directory
    output_dir = _get_output_dir(version_info, req.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get filename from version info
    files = version_info.get("files", [])
    primary_file = next((f for f in files if f.get("primary")), files[0] if files else None)

    if not primary_file:
        raise HTTPException(status_code=400, detail="No files found for this version")

    filename = primary_file.get("name", f"model-{version_id}.safetensors")
    dest_path = output_dir / filename

    # Create download tracking entry
    download_id = f"{version_id}_{int(__import__('time').time())}"
    _active_downloads[download_id] = {
        "id": download_id,
        "version_id": version_id,
        "status": "queued",
        "path": str(dest_path),
        "filename": filename,
        "model_name": version_info.get("model", {}).get("name", "Unknown"),
        "version_name": version_info.get("name", "Unknown"),
    }

    # Start background download
    background_tasks.add_task(_do_download, version_id, dest_path, api_key, download_id)

    return {
        "download_id": download_id,
        "status": "queued",
        "version_id": version_id,
        "destination": str(dest_path),
        "model_name": version_info.get("model", {}).get("name"),
        "version_name": version_info.get("name"),
    }


@router.get("/status/{download_id}")
def get_download_status(download_id: str) -> dict[str, Any]:
    """Get status of a download."""
    if download_id not in _active_downloads:
        raise HTTPException(status_code=404, detail="Download not found")

    return _active_downloads[download_id]


@router.get("/active")
def list_active_downloads() -> dict[str, Any]:
    """List all active/recent downloads."""
    return {
        "downloads": list(_active_downloads.values()),
        "total": len(_active_downloads),
    }


def create_download_router() -> APIRouter:
    """Return the download API router."""
    return router
