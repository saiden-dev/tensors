"""FastAPI route handlers for model management endpoints."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from tensors.config import MODELS_DIR
from tensors.server.sd_client import get_sd_headers

logger = logging.getLogger(__name__)

_HTTP_OK = 200
_SD_ENV_FILE = Path("/etc/default/sd-server")


class SwitchModelRequest(BaseModel):
    """Request body for switching models."""

    model: str  # Model filename or full path


async def _run_command(*args: str) -> tuple[int, str, str]:
    """Run a shell command and return (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode or 0, stdout.decode(), stderr.decode()


def _read_env_file() -> dict[str, str]:
    """Read the sd-server environment file."""
    env: dict[str, str] = {}
    if _SD_ENV_FILE.exists():
        for raw_line in _SD_ENV_FILE.read_text().splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
    return env


def _write_env_file(env: dict[str, str]) -> str:
    """Generate env file content."""
    lines = ["# sd-server configuration"]
    for key, value in env.items():
        lines.append(f"{key}={value}")
    return "\n".join(lines) + "\n"

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
            headers = get_sd_headers(request)
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{sd_server_url}/sdapi/v1/options", headers=headers)
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

    @router.post("/switch")
    async def switch_model(req: SwitchModelRequest) -> dict[str, Any]:
        """Switch sd-server to a different model by updating env and restarting."""
        # Find the model file
        checkpoints = scan_checkpoints()
        model_path: str | None = None

        for cp in checkpoints:
            if cp["filename"] == req.model or cp["path"] == req.model or cp["name"] == req.model:
                model_path = cp["path"]
                break

        if not model_path:
            raise HTTPException(status_code=404, detail=f"Model not found: {req.model}")

        # Read current env, update SD_MODEL
        env = _read_env_file()
        old_model = env.get("SD_MODEL", "")
        env["SD_MODEL"] = model_path

        # Write new env file via sudo tee
        new_content = _write_env_file(env)
        proc = await asyncio.create_subprocess_exec(
            "sudo", "tee", str(_SD_ENV_FILE),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate(new_content.encode())
        if proc.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Failed to write env file: {stderr.decode()}")

        # Restart sd-server
        returncode, _stdout, stderr = await _run_command("sudo", "systemctl", "restart", "sd-server")
        if returncode != 0:
            raise HTTPException(status_code=500, detail=f"Failed to restart sd-server: {stderr}")

        logger.info(f"Switched model from {old_model} to {model_path}")

        return {
            "ok": True,
            "old_model": old_model,
            "new_model": model_path,
            "message": "Model switched, sd-server restarting",
        }

    @router.get("/status")
    async def sd_server_status() -> dict[str, Any]:
        """Get sd-server systemd service status."""
        _returncode, stdout, _stderr = await _run_command("systemctl", "is-active", "sd-server")
        is_active = stdout.strip() == "active"

        env = _read_env_file()

        return {
            "service": "sd-server",
            "active": is_active,
            "status": stdout.strip(),
            "current_model": env.get("SD_MODEL"),
            "host": env.get("SD_HOST"),
            "port": env.get("SD_PORT"),
        }

    return router
