"""FastAPI route handlers for model management endpoints."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Request

from tensors.config import MODELS_DIR

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_HTTP_OK = 200

# Keywords for detecting base model category
_SD15_KEYWORDS = ("sd15", "sd1.5", "sd-1.5", "sd_1.5", "1.5", "sd-1-", "v1-5")
_LARGE_KEYWORDS = ("sdxl", "xl", "pony", "illustrious", "ilust", "noob", "animagine")


def _detect_model_category(name: str) -> str:
    """Detect model category from filename. Returns 'sd15' or 'large'."""
    name_lower = name.lower()

    # Check SD 1.5 keywords first
    for kw in _SD15_KEYWORDS:
        if kw in name_lower:
            return "sd15"

    # Check large model keywords
    for kw in _LARGE_KEYWORDS:
        if kw in name_lower:
            return "large"

    # Default to large (SDXL/Pony/Illustrious are more common now)
    return "large"


# =============================================================================
# Helper Functions
# =============================================================================


def scan_models(directory: Path, extensions: tuple[str, ...] = (".safetensors", ".gguf")) -> list[dict[str, Any]]:
    """Scan directory for model files."""
    models: list[dict[str, Any]] = []

    if not directory.exists():
        return models

    for ext in extensions:
        for path in directory.rglob(f"*{ext}"):
            stat = path.stat()
            name = path.stem
            models.append(
                {
                    "name": name,
                    "path": str(path),
                    "filename": path.name,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": stat.st_mtime,
                    "category": _detect_model_category(name),
                }
            )

    # Sort by name
    models.sort(key=lambda x: x["name"].lower())
    return models


def scan_loras(directory: Path | None = None) -> list[dict[str, Any]]:
    """Scan for LoRA files."""
    lora_dir = directory or MODELS_DIR / "loras"
    return scan_models(lora_dir, extensions=(".safetensors", ".gguf"))


def scan_checkpoints(directory: Path | None = None) -> list[dict[str, Any]]:
    """Scan for checkpoint files."""
    checkpoint_dir = directory or MODELS_DIR / "checkpoints"
    return scan_models(checkpoint_dir, extensions=(".safetensors", ".gguf"))


# =============================================================================
# Router Factory
# =============================================================================


def create_models_router() -> APIRouter:
    """Build a router with /api/models/* endpoints."""
    router = APIRouter(prefix="/api/models", tags=["models"])

    @router.get("")
    def list_models() -> dict[str, Any]:
        """List available checkpoint models."""
        checkpoints = scan_checkpoints()
        return {
            "models": checkpoints,
            "total": len(checkpoints),
        }

    @router.get("/active")
    async def get_active_model(request: Request) -> dict[str, Any]:
        """Get information about the currently loaded model from sd-server."""
        import httpx  # noqa: PLC0415

        sd_server_url = request.app.state.sd_server_url

        # Try to get current model from sd-server's options endpoint
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{sd_server_url}/sdapi/v1/options")
                if response.status_code == _HTTP_OK:
                    options = response.json()
                    model_name = options.get("sd_model_checkpoint")
                    return {
                        "loaded": True,
                        "model": model_name,
                        "sd_server_url": sd_server_url,
                    }
        except httpx.HTTPError:
            pass

        return {
            "loaded": False,
            "model": None,
            "sd_server_url": sd_server_url,
            "error": "Cannot connect to sd-server",
        }

    @router.get("/loras")
    def list_loras() -> dict[str, Any]:
        """List available LoRA files."""
        loras = scan_loras()
        return {
            "loras": loras,
            "total": len(loras),
        }

    @router.get("/scan")
    def scan_all_models() -> dict[str, Any]:
        """Scan all model directories."""
        checkpoints = scan_checkpoints()
        loras = scan_loras()

        return {
            "checkpoints": checkpoints,
            "loras": loras,
            "total_checkpoints": len(checkpoints),
            "total_loras": len(loras),
        }

    return router
