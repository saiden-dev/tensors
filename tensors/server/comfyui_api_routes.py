"""FastAPI route handlers for ComfyUI programmatic API endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

from tensors.comfyui import (
    clear_queue,
    generate_image,
    get_history,
    get_image,
    get_loaded_models,
    get_queue_status,
    get_system_stats,
    queue_prompt,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/comfyui", tags=["ComfyUI API"])


# =============================================================================
# Request/Response Models
# =============================================================================


class GenerateRequest(PydanticBaseModel):
    """Request body for text-to-image generation."""

    prompt: str = Field(..., description="Positive prompt text")
    negative_prompt: str = Field(default="", description="Negative prompt text")
    model: str | None = Field(default=None, description="Checkpoint model name")
    width: int = Field(default=1024, ge=64, le=4096, description="Image width")
    height: int = Field(default=1024, ge=64, le=4096, description="Image height")
    steps: int = Field(default=20, ge=1, le=150, description="Sampling steps")
    cfg: float = Field(default=7.0, ge=1.0, le=30.0, description="CFG scale")
    seed: int = Field(default=-1, description="Random seed (-1 for random)")
    sampler: str = Field(default="euler", description="Sampler name")
    scheduler: str = Field(default="normal", description="Scheduler name")
    vae: str | None = Field(default=None, description="VAE model name (defaults to sdxl_vae.safetensors)")


class GenerateResponse(PydanticBaseModel):
    """Response from text-to-image generation."""

    success: bool
    prompt_id: str
    images: list[str] = Field(default_factory=list)
    errors: dict[str, Any] = Field(default_factory=dict)


class WorkflowRequest(PydanticBaseModel):
    """Request body for running arbitrary workflow."""

    workflow: dict[str, Any] = Field(..., description="ComfyUI API-format workflow")


class WorkflowResponse(PydanticBaseModel):
    """Response from workflow execution."""

    success: bool
    prompt_id: str
    number: int | None = None
    error: str | None = None
    node_errors: dict[str, Any] = Field(default_factory=dict)


class QueueStatusResponse(PydanticBaseModel):
    """Queue status response."""

    queue_running: list[Any] = Field(default_factory=list)
    queue_pending: list[Any] = Field(default_factory=list)


class SystemStatsResponse(PydanticBaseModel):
    """System stats response."""

    system: dict[str, Any] = Field(default_factory=dict)
    devices: list[dict[str, Any]] = Field(default_factory=list)


class ModelsResponse(PydanticBaseModel):
    """Available models response."""

    checkpoints: list[str] = Field(default_factory=list)
    loras: list[str] = Field(default_factory=list)
    vae: list[str] = Field(default_factory=list)
    clip: list[str] = Field(default_factory=list)
    controlnet: list[str] = Field(default_factory=list)
    upscale_models: list[str] = Field(default_factory=list)


class HistoryEntry(PydanticBaseModel):
    """Single history entry."""

    prompt_id: str
    status: str
    outputs: dict[str, Any] = Field(default_factory=dict)
    images: list[str] = Field(default_factory=list)


# =============================================================================
# Query Endpoints
# =============================================================================


@router.get("/status", response_model=SystemStatsResponse)
def comfyui_status() -> dict[str, Any]:
    """Get ComfyUI system stats (GPU, RAM, etc.)."""
    stats = get_system_stats()
    if not stats:
        raise HTTPException(status_code=502, detail="Could not connect to ComfyUI")
    return stats


@router.get("/queue", response_model=QueueStatusResponse)
def comfyui_queue() -> dict[str, Any]:
    """Get ComfyUI queue status."""
    queue = get_queue_status()
    if queue is None:
        raise HTTPException(status_code=502, detail="Could not connect to ComfyUI")
    return queue


@router.delete("/queue")
def comfyui_clear_queue() -> dict[str, Any]:
    """Clear the ComfyUI queue."""
    success = clear_queue()
    if not success:
        raise HTTPException(status_code=502, detail="Could not clear queue")
    return {"cleared": True}


@router.get("/models", response_model=ModelsResponse)
def comfyui_models() -> dict[str, Any]:
    """List available models in ComfyUI."""
    models = get_loaded_models()
    if models is None:
        raise HTTPException(status_code=502, detail="Could not fetch models from ComfyUI")
    return models


@router.get("/history")
def comfyui_history_list(
    limit: int = Query(default=20, le=100, description="Max history items"),
) -> dict[str, Any]:
    """List ComfyUI generation history."""
    history = get_history(max_items=limit)
    if history is None:
        raise HTTPException(status_code=502, detail="Could not fetch history from ComfyUI")

    # Transform to list format with summary
    items: list[dict[str, Any]] = []
    for prompt_id, entry in history.items():
        status = entry.get("status", {}).get("status_str", "unknown")
        outputs = entry.get("outputs", {})

        # Extract image filenames
        images: list[str] = []
        for _node_id, output in outputs.items():
            if "images" in output:
                for img in output["images"]:
                    images.append(img.get("filename", ""))

        items.append(
            {
                "prompt_id": prompt_id,
                "status": status,
                "image_count": len(images),
                "images": images,
            }
        )

    return {"items": items, "total": len(items)}


@router.get("/history/{prompt_id}")
def comfyui_history_detail(prompt_id: str) -> dict[str, Any]:
    """Get details for a specific history entry."""
    history = get_history(prompt_id=prompt_id)
    if history is None:
        raise HTTPException(status_code=502, detail="Could not fetch history from ComfyUI")

    if prompt_id not in history:
        raise HTTPException(status_code=404, detail="Prompt not found in history")

    entry = history[prompt_id]
    status = entry.get("status", {})
    outputs = entry.get("outputs", {})

    # Extract image filenames
    images: list[str] = []
    for _node_id, output in outputs.items():
        if "images" in output:
            for img in output["images"]:
                images.append(img.get("filename", ""))

    return {
        "prompt_id": prompt_id,
        "status": status.get("status_str", "unknown"),
        "completed": status.get("completed", False),
        "outputs": outputs,
        "images": images,
    }


# =============================================================================
# Generation Endpoints
# =============================================================================


@router.post("/generate", response_model=GenerateResponse)
def comfyui_generate(request: GenerateRequest) -> dict[str, Any]:
    """Generate an image using a simple text-to-image workflow.

    This uses the built-in SDXL/Flux compatible workflow template.
    For custom workflows, use the /workflow endpoint instead.
    """
    logger.info(
        "Generate request: model=%s, size=%dx%d, steps=%d, prompt=%r",
        request.model or "default",
        request.width,
        request.height,
        request.steps,
        request.prompt[:100] + "..." if len(request.prompt) > 100 else request.prompt,
    )
    if request.negative_prompt:
        logger.debug("Negative prompt: %r", request.negative_prompt[:100])

    result = generate_image(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        model=request.model,
        width=request.width,
        height=request.height,
        steps=request.steps,
        cfg=request.cfg,
        seed=request.seed,
        sampler=request.sampler,
        scheduler=request.scheduler,
        vae=request.vae,
    )

    if not result:
        logger.error("Generation failed to queue")
        raise HTTPException(status_code=502, detail="Failed to queue generation")

    if result.success:
        logger.info("Generation complete: prompt_id=%s, images=%d", result.prompt_id, len(result.images))
    else:
        logger.warning("Generation failed: prompt_id=%s, errors=%s", result.prompt_id, result.node_errors)

    return {
        "success": result.success,
        "prompt_id": result.prompt_id,
        "images": [str(img) for img in result.images],
        "errors": result.node_errors,
    }


@router.post("/workflow", response_model=WorkflowResponse)
def comfyui_workflow(request: WorkflowRequest) -> dict[str, Any]:
    """Queue an arbitrary ComfyUI workflow for execution.

    The workflow should be in ComfyUI API format (exported via "Save (API Format)").
    This endpoint queues the workflow and returns immediately with the prompt_id.
    Use /history/{prompt_id} to check the result.
    """
    result = queue_prompt(workflow=request.workflow)

    if not result:
        raise HTTPException(status_code=502, detail="Failed to queue workflow")

    if "error" in result:
        return {
            "success": False,
            "prompt_id": "",
            "error": result.get("error"),
            "node_errors": result.get("node_errors", {}),
        }

    return {
        "success": True,
        "prompt_id": result.get("prompt_id", ""),
        "number": result.get("number"),
    }


@router.get("/image/{filename}")
def comfyui_image(
    filename: str,
    subfolder: str = Query(default="", description="Subfolder within output directory"),
    folder_type: str = Query(default="output", description="Folder type: output, input, temp"),
) -> Response:
    """Fetch a generated image from ComfyUI.

    Use this to retrieve images by filename after generation.
    """
    image_data = get_image(filename=filename, subfolder=subfolder, folder_type=folder_type)
    if image_data is None:
        raise HTTPException(status_code=404, detail="Image not found")

    # Determine content type from filename
    content_type = "image/png"
    if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
        content_type = "image/jpeg"
    elif filename.lower().endswith(".webp"):
        content_type = "image/webp"

    return Response(content=image_data, media_type=content_type)


def create_comfyui_api_router() -> APIRouter:
    """Return the ComfyUI API router."""
    return router
