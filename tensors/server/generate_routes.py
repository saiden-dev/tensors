"""FastAPI route handlers for image generation with gallery integration."""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

from tensors.server.gallery import Gallery
from tensors.server.sd_client import get_sd_headers

logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================


class LoraConfig(PydanticBaseModel):
    """LoRA configuration for sd-server."""

    path: str
    multiplier: float = Field(default=1.0, ge=0.0, le=2.0)


class GenerateRequest(PydanticBaseModel):
    """Request body for image generation."""

    prompt: str
    negative_prompt: str = ""
    width: int = Field(default=512, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
    steps: int = Field(default=20, ge=1, le=150)
    cfg_scale: float = Field(default=7.0, ge=0, le=30)
    seed: int = -1
    sampler_name: str = ""
    scheduler: str = ""
    batch_size: int = Field(default=1, ge=1, le=16)
    save_to_gallery: bool = True
    return_base64: bool = False
    lora: LoraConfig | None = None


# =============================================================================
# Helper Functions
# =============================================================================


def _build_sd_request(req: GenerateRequest) -> dict[str, Any]:
    """Build request body for sd-server."""
    prompt = req.prompt

    # sd-server expects LoRA in prompt as <lora:name:weight> syntax
    if req.lora:
        lora_name = Path(req.lora.path).stem
        lora_tag = f"<lora:{lora_name}:{req.lora.multiplier}>"
        prompt = f"{prompt} {lora_tag}"

    body: dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": req.negative_prompt,
        "width": req.width,
        "height": req.height,
        "steps": req.steps,
        "cfg_scale": req.cfg_scale,
        "seed": req.seed,
        "batch_size": req.batch_size,
    }
    if req.sampler_name:
        body["sampler_name"] = req.sampler_name
    if req.scheduler:
        body["scheduler"] = req.scheduler
    return body


def _parse_info(info: Any) -> dict[str, Any]:
    """Parse info from sd-server response."""
    if isinstance(info, str):
        import json  # noqa: PLC0415

        try:
            return dict(json.loads(info))
        except json.JSONDecodeError:
            return {"raw": info}
    return info if isinstance(info, dict) else {}


def _process_image(
    img_b64: str,
    index: int,
    seed: int,
    req: GenerateRequest,
    gallery: Gallery,
    model: str | None,
) -> dict[str, Any]:
    """Process a single generated image."""
    image_bytes = base64.b64decode(img_b64)
    image_info: dict[str, Any] = {"index": index, "seed": seed}

    if req.save_to_gallery:
        metadata = {
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "width": req.width,
            "height": req.height,
            "steps": req.steps,
            "cfg_scale": req.cfg_scale,
            "sampler": req.sampler_name,
            "scheduler": req.scheduler,
            "model": model,
            "lora": req.lora.path if req.lora else None,
            "lora_weight": req.lora.multiplier if req.lora else None,
            "generated_at": time.time(),
        }
        gallery_img = gallery.save_image(image_bytes, metadata=metadata, seed=seed)
        image_info["id"] = gallery_img.id
        image_info["path"] = str(gallery_img.path)

    if req.return_base64:
        image_info["base64"] = img_b64

    return image_info


# =============================================================================
# Router Factory
# =============================================================================


def create_generate_router() -> APIRouter:  # noqa: PLR0915
    """Build a router with /api/generate endpoint."""
    router = APIRouter(prefix="/api", tags=["generate"])
    gallery = Gallery()

    @router.post("/generate")
    async def generate(request: Request, req: GenerateRequest) -> dict[str, Any]:
        """Generate images with gallery integration."""
        sd_server_url = request.app.state.sd_server_url
        body = _build_sd_request(req)
        url = f"{sd_server_url}/sdapi/v1/txt2img"

        try:
            headers = get_sd_headers(request)
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(url, json=body, headers=headers)
                response.raise_for_status()
                result = response.json()
        except httpx.ConnectError as e:
            raise HTTPException(status_code=503, detail=f"Cannot connect to sd-server: {e}") from e
        except httpx.HTTPError as e:
            logger.exception("Generation failed")
            raise HTTPException(status_code=502, detail=f"sd-server error: {e}") from e

        images_data = result.get("images", [])
        info = _parse_info(result.get("info", {}))
        all_seeds = info.get("all_seeds", [req.seed] * len(images_data))

        # Get model info from sd-server response if available
        model_name = info.get("sd_model_name") or info.get("model")

        output_images = [
            _process_image(img_b64, i, all_seeds[i] if i < len(all_seeds) else req.seed + i, req, gallery, model_name)
            for i, img_b64 in enumerate(images_data)
        ]

        return {
            "images": output_images,
            "parameters": result.get("parameters", body),
            "info": info,
            "saved_to_gallery": req.save_to_gallery,
            "total": len(output_images),
        }

    @router.get("/samplers")
    async def list_samplers(request: Request) -> dict[str, Any]:
        """List available samplers from sd-server."""
        sd_server_url = request.app.state.sd_server_url
        url = f"{sd_server_url}/sdapi/v1/samplers"

        try:
            headers = get_sd_headers(request)
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                return {"samplers": response.json()}
        except httpx.ConnectError as e:
            raise HTTPException(status_code=503, detail=f"Cannot connect to sd-server: {e}") from e
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"sd-server error: {e}") from e

    @router.get("/schedulers")
    async def list_schedulers(request: Request) -> dict[str, Any]:
        """List available schedulers from sd-server."""
        sd_server_url = request.app.state.sd_server_url
        url = f"{sd_server_url}/sdapi/v1/schedulers"

        try:
            headers = get_sd_headers(request)
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                return {"schedulers": response.json()}
        except httpx.ConnectError as e:
            raise HTTPException(status_code=503, detail=f"Cannot connect to sd-server: {e}") from e
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"sd-server error: {e}") from e

    return router
