"""ComfyUI API client for programmatic workflow execution."""

from __future__ import annotations

import copy
import json
import random
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import websocket
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from tensors.config import get_comfyui_url

if TYPE_CHECKING:
    from rich.console import Console

# WebSocket timeout for receiving messages (seconds)
_WS_RECV_TIMEOUT = 1.0


def _get_comfyui_url() -> str:
    """Get ComfyUI URL from config (env var -> config file -> default)."""
    return get_comfyui_url()


# Map ComfyUI validation input_name → CLI override flag that would fix it.
# When a node fails value_not_in_list validation we hint at the appropriate
# escape hatch instead of just dumping the raw error dict.
_INPUT_HINT_FLAGS: dict[str, str] = {
    "vae_name": "--vae <name>  (or --family <key> to switch workflow family)",
    "ckpt_name": "--model <name>",
    "unet_name": "--model <name>  (UNet-only flux; check filename)",
    "clip_name1": "--family <key>  (CLIP/T5 file missing for chosen family)",
    "clip_name2": "--family <key>  (CLIP/T5 file missing for chosen family)",
    "lora_name": "--lora <name>",
}


def _print_validation_hint(console: Console, node_errors: dict[str, Any]) -> None:
    """Surface actionable hints for ComfyUI value_not_in_list errors.

    Parses node_errors from a prompt validation failure and, for each
    `value_not_in_list` entry on a known input (vae_name, ckpt_name, etc),
    prints the received value, a truncated list of valid choices, and the
    CLI flag that would override the selection. Falls back to the raw
    error dump when nothing actionable is recognized.
    """
    actionable = False
    for node_id, payload in node_errors.items():
        if not isinstance(payload, dict):
            continue
        class_type = payload.get("class_type", "?")
        for err in payload.get("errors", []) or []:
            if err.get("type") != "value_not_in_list":
                continue
            extra = err.get("extra_info") or {}
            input_name = extra.get("input_name", "?")
            received = extra.get("received_value", "?")
            valid_list = (extra.get("input_config") or [[]])[0]
            valid_preview = ", ".join(str(v) for v in valid_list[:8])
            if len(valid_list) > 8:
                valid_preview += f", … (+{len(valid_list) - 8} more)"
            hint = _INPUT_HINT_FLAGS.get(input_name)
            console.print(
                f"  [yellow]Node {node_id} ({class_type}):[/yellow] "
                f"[red]{input_name}={received!r} not available on this backend[/red]"
            )
            console.print(f"    available: [dim]{valid_preview}[/dim]")
            if hint:
                console.print(f"    [cyan]hint:[/cyan] retry with [bold]{hint}[/bold]")
            actionable = True
    if not actionable:
        # Fall back to the raw dump so we never hide errors entirely.
        for node_id, payload in node_errors.items():
            console.print(f"  [yellow]Node {node_id}:[/yellow] {payload}")


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class GenerationResult:
    """Result from image generation."""

    prompt_id: str
    images: list[Path] = field(default_factory=list)
    node_errors: dict[str, Any] = field(default_factory=dict)
    success: bool = True


@dataclass
class WorkflowResult:
    """Result from workflow execution."""

    prompt_id: str
    outputs: dict[str, Any] = field(default_factory=dict)
    node_errors: dict[str, Any] = field(default_factory=dict)
    success: bool = True


# ============================================================================
# Progress Callback Type
# ============================================================================

# (current_step, total_steps, status_message)
ProgressCallback = Callable[[int, int, str], None]


# ============================================================================
# Basic Query Functions
# ============================================================================


def get_system_stats(url: str | None = None, console: Console | None = None) -> dict[str, Any] | None:
    """Get ComfyUI system stats (GPU, RAM, etc.).

    Args:
        url: ComfyUI base URL (defaults to COMFYUI_URL env var or localhost:8188)
        console: Rich console for progress/error output

    Returns:
        System stats dict or None on error
    """
    base_url = url or _get_comfyui_url()

    def _do_fetch() -> dict[str, Any] | None:
        try:
            response = httpx.get(f"{base_url}/system_stats", timeout=10.0)
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
        except httpx.HTTPStatusError as e:
            if console:
                console.print(f"[red]API error: {e.response.status_code}[/red]")
            return None
        except httpx.RequestError as e:
            if console:
                console.print(f"[red]Connection error: {e}[/red]")
            return None

    if console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("[cyan]Fetching system stats...", total=None)
            return _do_fetch()
    else:
        return _do_fetch()


def get_queue_status(url: str | None = None, console: Console | None = None) -> dict[str, Any] | None:
    """Get ComfyUI queue status.

    Args:
        url: ComfyUI base URL
        console: Rich console for output

    Returns:
        Queue status dict with 'queue_running' and 'queue_pending' lists, or None on error
    """
    base_url = url or _get_comfyui_url()

    def _do_fetch() -> dict[str, Any] | None:
        try:
            response = httpx.get(f"{base_url}/queue", timeout=10.0)
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
        except httpx.HTTPStatusError as e:
            if console:
                console.print(f"[red]API error: {e.response.status_code}[/red]")
            return None
        except httpx.RequestError as e:
            if console:
                console.print(f"[red]Connection error: {e}[/red]")
            return None

    if console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("[cyan]Fetching queue status...", total=None)
            return _do_fetch()
    else:
        return _do_fetch()


def clear_queue(url: str | None = None, console: Console | None = None) -> bool:
    """Clear the ComfyUI queue.

    Args:
        url: ComfyUI base URL
        console: Rich console for output

    Returns:
        True if successful, False on error
    """
    base_url = url or _get_comfyui_url()

    try:
        # Clear both pending and running
        response = httpx.post(f"{base_url}/queue", json={"clear": True}, timeout=10.0)
        response.raise_for_status()
        if console:
            console.print("[green]Queue cleared[/green]")
        return True
    except httpx.HTTPStatusError as e:
        if console:
            console.print(f"[red]API error: {e.response.status_code}[/red]")
        return False
    except httpx.RequestError as e:
        if console:
            console.print(f"[red]Connection error: {e}[/red]")
        return False


def get_object_info(url: str | None = None, console: Console | None = None) -> dict[str, Any] | None:
    """Get ComfyUI object info (available nodes and their configurations).

    Args:
        url: ComfyUI base URL
        console: Rich console for output

    Returns:
        Object info dict or None on error
    """
    base_url = url or _get_comfyui_url()

    def _do_fetch() -> dict[str, Any] | None:
        try:
            response = httpx.get(f"{base_url}/object_info", timeout=30.0)
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
        except httpx.HTTPStatusError as e:
            if console:
                console.print(f"[red]API error: {e.response.status_code}[/red]")
            return None
        except httpx.RequestError as e:
            if console:
                console.print(f"[red]Connection error: {e}[/red]")
            return None

    if console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("[cyan]Fetching object info...", total=None)
            return _do_fetch()
    else:
        return _do_fetch()


def get_loaded_models(url: str | None = None, console: Console | None = None) -> dict[str, list[str]] | None:
    """Get list of loaded/available models (checkpoints, loras, etc.).

    Args:
        url: ComfyUI base URL
        console: Rich console for output

    Returns:
        Dict mapping model type to list of model names, or None on error
    """
    base_url = url or _get_comfyui_url()

    def _do_fetch() -> dict[str, list[str]] | None:
        result: dict[str, list[str]] = {}

        # Model type to node class and input name mapping
        model_types = {
            "checkpoints": ("CheckpointLoaderSimple", "ckpt_name"),
            "diffusion_models": ("UNETLoader", "unet_name"),
            "loras": ("LoraLoader", "lora_name"),
            "vae": ("VAELoader", "vae_name"),
            "clip": ("CLIPLoader", "clip_name"),
            "controlnet": ("ControlNetLoader", "control_net_name"),
            "upscale_models": ("UpscaleModelLoader", "model_name"),
        }

        try:
            response = httpx.get(f"{base_url}/object_info", timeout=30.0)
            response.raise_for_status()
            object_info: dict[str, Any] = response.json()

            for model_type, (node_class, input_name) in model_types.items():
                if node_class in object_info:
                    node_info = object_info[node_class]
                    inputs = node_info.get("input", {}).get("required", {})
                    if input_name in inputs:
                        input_def = inputs[input_name]
                        if isinstance(input_def, list) and len(input_def) > 0 and isinstance(input_def[0], list):
                            result[model_type] = input_def[0]

            return result

        except httpx.HTTPStatusError as e:
            if console:
                console.print(f"[red]API error: {e.response.status_code}[/red]")
            return None
        except httpx.RequestError as e:
            if console:
                console.print(f"[red]Connection error: {e}[/red]")
            return None

    if console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("[cyan]Fetching loaded models...", total=None)
            return _do_fetch()
    else:
        return _do_fetch()


def get_history(
    url: str | None = None,
    prompt_id: str | None = None,
    max_items: int = 100,
    console: Console | None = None,
) -> dict[str, Any] | None:
    """Get ComfyUI history.

    Args:
        url: ComfyUI base URL
        prompt_id: Specific prompt ID to fetch (if None, fetches recent history)
        max_items: Maximum number of history items to return
        console: Rich console for output

    Returns:
        History dict (keyed by prompt_id) or None on error
    """
    base_url = url or _get_comfyui_url()

    def _do_fetch() -> dict[str, Any] | None:
        try:
            endpoint = f"{base_url}/history/{prompt_id}" if prompt_id else f"{base_url}/history?max_items={max_items}"
            response = httpx.get(endpoint, timeout=30.0)
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
        except httpx.HTTPStatusError as e:
            if console:
                console.print(f"[red]API error: {e.response.status_code}[/red]")
            return None
        except httpx.RequestError as e:
            if console:
                console.print(f"[red]Connection error: {e}[/red]")
            return None

    if console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("[cyan]Fetching history...", total=None)
            return _do_fetch()
    else:
        return _do_fetch()


# ============================================================================
# Workflow Execution
# ============================================================================


def queue_prompt(
    workflow: dict[str, Any],
    url: str | None = None,
    client_id: str | None = None,
    console: Console | None = None,
) -> dict[str, Any] | None:
    """Queue a workflow prompt for execution.

    Args:
        workflow: ComfyUI workflow dict (API format)
        url: ComfyUI base URL
        client_id: Client ID for WebSocket tracking
        console: Rich console for output

    Returns:
        Response dict with 'prompt_id' and 'number', or None on error
    """
    base_url = url or _get_comfyui_url()
    client_id = client_id or str(uuid.uuid4())

    try:
        payload = {"prompt": workflow, "client_id": client_id}
        response = httpx.post(f"{base_url}/prompt", json=payload, timeout=30.0)
        response.raise_for_status()
        result: dict[str, Any] = response.json()

        if "error" in result:
            if console:
                console.print(f"[red]Workflow error: {result['error']}[/red]")
                if "node_errors" in result:
                    _print_validation_hint(console, result["node_errors"])
            return None

        return result
    except httpx.HTTPStatusError as e:
        if console:
            console.print(f"[red]API error: {e.response.status_code}[/red]")
            try:
                error_detail = e.response.json()
                if "error" in error_detail:
                    console.print(f"  [yellow]{error_detail['error']}[/yellow]")
                if "node_errors" in error_detail:
                    _print_validation_hint(console, error_detail["node_errors"])
            except Exception:
                pass
        return None
    except httpx.RequestError as e:
        if console:
            console.print(f"[red]Connection error: {e}[/red]")
        return None


def _wait_for_completion_ws(
    prompt_id: str,
    url: str,
    client_id: str,
    timeout: float = 600.0,
    on_progress: ProgressCallback | None = None,
) -> WorkflowResult:
    """Wait for workflow completion using WebSocket for real-time progress.

    Args:
        prompt_id: The prompt ID to track
        url: ComfyUI base URL (http://...)
        client_id: Client ID used when queueing the prompt
        timeout: Maximum wait time in seconds
        on_progress: Optional callback for progress updates (step, total, status)

    Returns:
        WorkflowResult with outputs or errors
    """
    # Convert http(s) URL to ws(s) URL
    ws_url = url.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_url}/ws?clientId={client_id}"

    start_time = time.time()
    outputs: dict[str, Any] = {}
    node_errors: dict[str, Any] = {}
    current_node: str | None = None

    try:
        ws = websocket.create_connection(ws_url, timeout=timeout)
    except Exception:
        # Fall back to polling if WebSocket fails
        return _poll_for_completion_fallback(prompt_id, url, timeout, on_progress)

    try:
        while time.time() - start_time < timeout:
            try:
                ws.settimeout(_WS_RECV_TIMEOUT)
                msg = ws.recv()
                if not msg:
                    continue

                data = json.loads(msg)
                msg_type = data.get("type", "")
                msg_data = data.get("data", {})

                # Only process messages for our prompt
                if msg_data.get("prompt_id") and msg_data.get("prompt_id") != prompt_id:
                    continue

                if msg_type == "execution_start":
                    if on_progress:
                        on_progress(0, 0, "Starting...")

                elif msg_type == "execution_cached":
                    # Some nodes were cached
                    cached_nodes = msg_data.get("nodes", [])
                    if on_progress and cached_nodes:
                        on_progress(0, 0, f"Cached {len(cached_nodes)} node(s)")

                elif msg_type == "executing":
                    # A node is being executed
                    current_node = msg_data.get("node")
                    if current_node is None:
                        # Execution finished (node=None means done)
                        break
                    # Don't update progress for non-sampler nodes to preserve step display

                elif msg_type == "progress":
                    # Sampling progress: {"value": 5, "max": 20}
                    value = msg_data.get("value", 0)
                    max_val = msg_data.get("max", 0)
                    if on_progress and max_val > 0:
                        on_progress(value, max_val, f"Step {value}/{max_val}")

                elif msg_type == "executed":
                    # A node finished, may have output
                    node_id = msg_data.get("node")
                    output = msg_data.get("output", {})
                    if node_id and output:
                        outputs[node_id] = output

                elif msg_type == "execution_error":
                    # Execution failed
                    node_id = msg_data.get("node_id", "unknown")
                    error_msg = msg_data.get("exception_message", "Unknown error")
                    node_errors[node_id] = error_msg
                    ws.close()
                    return WorkflowResult(
                        prompt_id=prompt_id,
                        outputs=outputs,
                        node_errors=node_errors,
                        success=False,
                    )

                elif msg_type == "execution_success":
                    # Explicitly done
                    break

            except websocket.WebSocketTimeoutException:
                # No message received, continue waiting
                continue
            except websocket.WebSocketConnectionClosedException:
                break

    finally:
        try:
            ws.close()
        except Exception:
            pass

    # Fetch final outputs from history to ensure we have everything
    try:
        response = httpx.get(f"{url}/history/{prompt_id}", timeout=10.0)
        response.raise_for_status()
        history = response.json()
        if prompt_id in history:
            entry = history[prompt_id]
            outputs = entry.get("outputs", outputs)
            status_info = entry.get("status", {})
            if status_info.get("status_str") == "error":
                return WorkflowResult(
                    prompt_id=prompt_id,
                    outputs=outputs,
                    node_errors=status_info.get("messages", {}),
                    success=False,
                )
    except Exception:
        pass

    return WorkflowResult(
        prompt_id=prompt_id,
        outputs=outputs,
        node_errors=node_errors,
        success=len(node_errors) == 0,
    )


def _poll_for_completion_fallback(
    prompt_id: str,
    url: str,
    timeout: float = 600.0,
    on_progress: ProgressCallback | None = None,
) -> WorkflowResult:
    """Fallback polling method when WebSocket is unavailable."""
    start_time = time.time()
    poll_interval = 0.5

    while time.time() - start_time < timeout:
        try:
            response = httpx.get(f"{url}/history/{prompt_id}", timeout=10.0)
            response.raise_for_status()
            history = response.json()

            if prompt_id in history:
                entry = history[prompt_id]
                outputs = entry.get("outputs", {})
                status_info = entry.get("status", {})

                if status_info.get("status_str") == "error":
                    return WorkflowResult(
                        prompt_id=prompt_id,
                        outputs=outputs,
                        node_errors=status_info.get("messages", {}),
                        success=False,
                    )

                return WorkflowResult(
                    prompt_id=prompt_id,
                    outputs=outputs,
                    success=True,
                )

            if on_progress:
                on_progress(0, 0, "Running...")

        except httpx.RequestError:
            pass

        time.sleep(poll_interval)

    return WorkflowResult(
        prompt_id=prompt_id,
        node_errors={"timeout": f"Workflow did not complete within {timeout}s"},
        success=False,
    )


def run_workflow(
    workflow: dict[str, Any] | Path,
    url: str | None = None,
    console: Console | None = None,
    on_progress: ProgressCallback | None = None,
    timeout: float = 600.0,
) -> WorkflowResult | None:
    """Run a workflow and wait for completion.

    Args:
        workflow: ComfyUI workflow dict (API format) or path to JSON file
        url: ComfyUI base URL
        console: Rich console for progress output
        on_progress: Optional callback for progress updates
        timeout: Maximum wait time in seconds

    Returns:
        WorkflowResult with outputs, or None if queuing failed
    """
    base_url = url or _get_comfyui_url()

    # Load workflow from file if needed
    workflow_dict: dict[str, Any]
    if isinstance(workflow, Path):
        if not workflow.exists():
            if console:
                console.print(f"[red]Workflow file not found: {workflow}[/red]")
            return None
        workflow_dict = json.loads(workflow.read_text())
    else:
        workflow_dict = workflow

    # Generate client_id for WebSocket tracking
    client_id = str(uuid.uuid4())

    # Queue the workflow
    if console:
        console.print("[cyan]Queueing workflow...[/cyan]")

    result = queue_prompt(workflow_dict, url=base_url, client_id=client_id, console=console)
    if not result:
        return None

    prompt_id = result["prompt_id"]
    if console:
        console.print(f"[dim]Prompt ID: {prompt_id}[/dim]")

    # Wait for completion with WebSocket progress
    if console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=20),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Starting...", total=None)

            def _console_progress(step: int, total: int, status: str) -> None:
                if total > 0:
                    # Update to determinate progress bar
                    progress.update(task, completed=step, total=total, description=f"[cyan]{status}[/cyan]")
                else:
                    # Indeterminate spinner
                    progress.update(task, description=f"[cyan]{status}[/cyan]")
                if on_progress:
                    on_progress(step, total, status)

            return _wait_for_completion_ws(prompt_id, base_url, client_id, timeout, on_progress=_console_progress)
    else:
        return _wait_for_completion_ws(prompt_id, base_url, client_id, timeout, on_progress=on_progress)


# ============================================================================
# Simple Text-to-Image Generation
# ============================================================================

# LoRA loader node template (inserted between checkpoint and sampler)
LORA_LOADER_NODE: dict[str, Any] = {
    "class_type": "LoraLoader",
    "inputs": {
        "lora_name": "",
        "strength_model": 1.0,
        "strength_clip": 1.0,
        "model": ["4", 0],  # From checkpoint
        "clip": ["4", 1],  # From checkpoint
    },
}

# Flux.1 Dev / Schnell workflow template (CheckpointLoaderSimple-based).
#
# Differs from DEFAULT_WORKFLOW_TEMPLATE in three load-bearing ways:
# 1. KSampler.cfg is HARDCODED to 1.0. Flux is guidance-distilled; raising
#    KSampler.cfg burns the image. Source: https://comfyanonymous.github.io/ComfyUI_examples/flux/
# 2. The user-facing "cfg/guidance" dial is wired into the FluxGuidance node
#    (default 3.5), which feeds the model's distilled guidance embedding.
# 3. Negative prompt is routed through ConditioningZeroOut — Flux ignores
#    classifier-free guidance, so negatives must be zero conditioning.
# 4. ModelSamplingFlux applies the resolution-dependent shift schedule
#    (defaults max_shift=1.15, base_shift=0.5) which sharpens output at
#    non-1024² aspect ratios.
# 5. EmptySD3LatentImage replaces EmptyLatentImage (Flux uses SD3-style latents).
#
# Use the all-in-one fp8 checkpoint (flux1-dev-fp8.safetensors) for the simplest
# path; for the split-file release (UNETLoader + DualCLIPLoader + VAELoader),
# see examples/flux1-dev/workflow.json.
FLUX_WORKFLOW_TEMPLATE: dict[str, Any] = {
    "100": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": ""},
    },
    "120": {
        "class_type": "ModelSamplingFlux",
        "inputs": {
            "model": ["100", 0],
            "max_shift": 1.15,
            "base_shift": 0.5,
            "width": 1024,
            "height": 1024,
        },
    },
    "130": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "", "clip": ["100", 1]},
    },
    "131": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "", "clip": ["100", 1]},
    },
    "132": {
        "class_type": "ConditioningZeroOut",
        "inputs": {"conditioning": ["131", 0]},
    },
    "140": {
        "class_type": "FluxGuidance",
        "inputs": {"conditioning": ["130", 0], "guidance": 3.5},
    },
    "150": {
        "class_type": "EmptySD3LatentImage",
        "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
    },
    "160": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 0,
            "steps": 20,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
            "model": ["120", 0],
            "positive": ["140", 0],
            "negative": ["132", 0],
            "latent_image": ["150", 0],
        },
    },
    "170": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["160", 0], "vae": ["100", 2]},
    },
    "180": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "flux", "images": ["170", 0]},
    },
}


# Flux UNet-only workflow template.
#
# Used for checkpoints that ship without CLIP/T5/VAE baked in (e.g. Flux Dev's
# split-file release, lust_v10, cyberrealisticFlux, getphatFLUXReality,
# moodyDesireMix, fcFluxPony*). Structurally identical to FLUX_WORKFLOW_TEMPLATE
# but the single CheckpointLoaderSimple (node "100") is replaced by three
# separate loaders that share the same node IDs the rest of the graph already
# expects:
#
#   "100" UNETLoader      → outputs MODEL (slot 0)
#   "101" DualCLIPLoader  → outputs CLIP  (slot 0)
#   "102" VAELoader       → outputs VAE   (slot 0)
#
# Downstream nodes that referenced ["100", 1] (clip) now read ["101", 0];
# downstream nodes that referenced ["100", 2] (vae) now read ["102", 0].
FLUX_UNET_WORKFLOW_TEMPLATE: dict[str, Any] = {
    "100": {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": "", "weight_dtype": "default"},
    },
    "101": {
        "class_type": "DualCLIPLoader",
        "inputs": {
            "clip_name1": "clip_l.safetensors",
            "clip_name2": "t5xxl_fp16.safetensors",
            "type": "flux",
        },
    },
    "102": {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "ae.safetensors"},
    },
    "120": {
        "class_type": "ModelSamplingFlux",
        "inputs": {
            "model": ["100", 0],
            "max_shift": 1.15,
            "base_shift": 0.5,
            "width": 1024,
            "height": 1024,
        },
    },
    "130": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "", "clip": ["101", 0]},
    },
    "131": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "", "clip": ["101", 0]},
    },
    "132": {
        "class_type": "ConditioningZeroOut",
        "inputs": {"conditioning": ["131", 0]},
    },
    "140": {
        "class_type": "FluxGuidance",
        "inputs": {"conditioning": ["130", 0], "guidance": 3.5},
    },
    "150": {
        "class_type": "EmptySD3LatentImage",
        "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
    },
    "160": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 0,
            "steps": 20,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
            "model": ["120", 0],
            "positive": ["140", 0],
            "negative": ["132", 0],
            "latent_image": ["150", 0],
        },
    },
    "170": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["160", 0], "vae": ["102", 0]},
    },
    "180": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "flux", "images": ["170", 0]},
    },
}


# Flux.2 Klein 9B workflow template — different architecture from Flux.1:
# - Single Qwen3-8B text encoder via CLIPLoader(type=flux2), produces 12288-dim
#   conditioning (3 stacked hidden layers)
# - EmptyFlux2LatentImage instead of EmptySD3LatentImage (different latent shape)
# - Custom-sampling pipeline (Flux2Scheduler + BasicGuider + RandomNoise +
#   KSamplerSelect + SamplerCustomAdvanced) instead of plain KSampler
# - Dedicated VAE (flux2-vae.safetensors), not the Flux.1 ae.safetensors
# Verified end-to-end against ComfyUI on madcat with lust_v10.safetensors.
FLUX2_KLEIN_WORKFLOW_TEMPLATE: dict[str, Any] = {
    "100": {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": "", "weight_dtype": "default"},
    },
    "101": {
        "class_type": "CLIPLoader",
        "inputs": {
            "clip_name": "qwen_3_8b_fp8mixed.safetensors",
            "type": "flux2",
        },
    },
    "102": {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "flux2-vae.safetensors"},
    },
    "130": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "", "clip": ["101", 0]},
    },
    "131": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "", "clip": ["101", 0]},
    },
    "140": {
        "class_type": "FluxGuidance",
        "inputs": {"conditioning": ["130", 0], "guidance": 3.5},
    },
    "150": {
        "class_type": "EmptyFlux2LatentImage",
        "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
    },
    "151": {
        "class_type": "RandomNoise",
        "inputs": {"noise_seed": 0},
    },
    "152": {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": "euler"},
    },
    "153": {
        "class_type": "Flux2Scheduler",
        "inputs": {"steps": 20, "width": 1024, "height": 1024},
    },
    "154": {
        "class_type": "BasicGuider",
        "inputs": {"model": ["100", 0], "conditioning": ["140", 0]},
    },
    "160": {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["151", 0],
            "guider": ["154", 0],
            "sampler": ["152", 0],
            "sigmas": ["153", 0],
            "latent_image": ["150", 0],
        },
    },
    "170": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["160", 0], "vae": ["102", 0]},
    },
    "180": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "flux2", "images": ["170", 0]},
    },
}


# Default SDXL/Illustrious/Pony compatible workflow template
# Uses separate VAE loader for better quality with modern models
DEFAULT_WORKFLOW_TEMPLATE: dict[str, Any] = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 0,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
        },
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": ""},
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "", "clip": ["4", 1]},
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "", "clip": ["4", 1]},
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["3", 0], "vae": ["11", 0]},
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "comfy", "images": ["8", 0]},
    },
    "11": {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "sdxl_vae.safetensors"},
    },
}


def _resolve_flux_guidance(
    guidance: float | None,
    cfg: float | None,
    defaults: dict[str, Any],
) -> float:
    """Resolve the FluxGuidance value with the precedence:

    explicit ``guidance`` > caller's ``cfg`` (re-interpreted as guidance for Flux) >
    family preset's ``guidance`` > 3.5 (BFL recommendation).
    """
    if guidance is not None:
        return float(guidance)
    if cfg is not None:
        return float(cfg)
    return float(defaults.get("guidance", 3.5))


def _build_flux_workflow(
    prompt: str,
    model: str | None,
    seed: int,
    steps: int,
    sampler: str,
    scheduler: str,
    width: int,
    height: int,
    batch_size: int,
    lora_name: str | None,
    lora_strength: float,
    vae: str | None,
    guidance: float,
) -> dict[str, Any]:
    """Build a Flux Dev/Schnell workflow.

    KSampler.cfg is force-locked to 1.0; the caller's intended CFG/guidance is
    routed to the FluxGuidance node. ModelSamplingFlux is wired with width/height
    matching the latent so the noise-shift schedule is correct.
    """
    workflow = copy.deepcopy(FLUX_WORKFLOW_TEMPLATE)

    # Set seed (random if -1)
    actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)

    # Checkpoint
    if model:
        workflow["100"]["inputs"]["ckpt_name"] = model

    # ModelSamplingFlux must match the latent dimensions
    workflow["120"]["inputs"]["width"] = width
    workflow["120"]["inputs"]["height"] = height

    # Prompts (positive only — negative is zero'd via ConditioningZeroOut)
    workflow["130"]["inputs"]["text"] = prompt

    # FluxGuidance carries the real prompt-adherence dial
    workflow["140"]["inputs"]["guidance"] = guidance

    # Latent
    workflow["150"]["inputs"]["width"] = width
    workflow["150"]["inputs"]["height"] = height
    workflow["150"]["inputs"]["batch_size"] = batch_size

    # KSampler — cfg stays 1.0
    workflow["160"]["inputs"]["seed"] = actual_seed
    workflow["160"]["inputs"]["steps"] = steps
    workflow["160"]["inputs"]["sampler_name"] = sampler
    workflow["160"]["inputs"]["scheduler"] = scheduler

    # Optional external VAE — fall back to checkpoint's built-in if not provided
    if vae:
        workflow["171"] = {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae},
        }
        workflow["170"]["inputs"]["vae"] = ["171", 0]

    # Optional LoRA injected between checkpoint and ModelSamplingFlux
    if lora_name:
        workflow["110"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["100", 0],
                "clip": ["100", 1],
                "lora_name": lora_name,
                "strength_model": lora_strength,
                "strength_clip": lora_strength,
            },
        }
        # Reroute downstream consumers from checkpoint to LoRA outputs
        workflow["120"]["inputs"]["model"] = ["110", 0]
        workflow["130"]["inputs"]["clip"] = ["110", 1]
        workflow["131"]["inputs"]["clip"] = ["110", 1]

    return workflow


def _build_flux_unet_workflow(
    prompt: str,
    model: str | None,
    seed: int,
    steps: int,
    sampler: str,
    scheduler: str,
    width: int,
    height: int,
    batch_size: int,
    lora_name: str | None,
    lora_strength: float,
    vae: str | None,
    guidance: float,
    clip_l: str,
    clip_t5: str,
) -> dict[str, Any]:
    """Build a Flux workflow for UNet-only checkpoints (split CLIP/T5/VAE).

    Structurally identical to ``_build_flux_workflow`` but uses UNETLoader +
    DualCLIPLoader + VAELoader (nodes 100/101/102) instead of a single
    CheckpointLoaderSimple. The rest of the graph (ModelSamplingFlux, KSampler,
    FluxGuidance, etc.) is unchanged.
    """
    workflow = copy.deepcopy(FLUX_UNET_WORKFLOW_TEMPLATE)

    # Seed (random if -1)
    actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)

    # UNet checkpoint
    if model:
        workflow["100"]["inputs"]["unet_name"] = model

    # CLIP/T5 (configurable so future variants can swap fp16 → fp8, etc.)
    workflow["101"]["inputs"]["clip_name1"] = clip_l
    workflow["101"]["inputs"]["clip_name2"] = clip_t5

    # External VAE — fall back to ae.safetensors from the template if unset
    if vae:
        workflow["102"]["inputs"]["vae_name"] = vae

    # ModelSamplingFlux must match the latent dimensions
    workflow["120"]["inputs"]["width"] = width
    workflow["120"]["inputs"]["height"] = height

    # Prompts (positive only — negative is zero'd via ConditioningZeroOut)
    workflow["130"]["inputs"]["text"] = prompt

    # FluxGuidance carries the real prompt-adherence dial
    workflow["140"]["inputs"]["guidance"] = guidance

    # Latent
    workflow["150"]["inputs"]["width"] = width
    workflow["150"]["inputs"]["height"] = height
    workflow["150"]["inputs"]["batch_size"] = batch_size

    # KSampler — cfg stays 1.0
    workflow["160"]["inputs"]["seed"] = actual_seed
    workflow["160"]["inputs"]["steps"] = steps
    workflow["160"]["inputs"]["sampler_name"] = sampler
    workflow["160"]["inputs"]["scheduler"] = scheduler

    # Optional LoRA injected between UNet/CLIP loaders and downstream consumers
    if lora_name:
        workflow["110"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["100", 0],
                "clip": ["101", 0],
                "lora_name": lora_name,
                "strength_model": lora_strength,
                "strength_clip": lora_strength,
            },
        }
        workflow["120"]["inputs"]["model"] = ["110", 0]
        workflow["130"]["inputs"]["clip"] = ["110", 1]
        workflow["131"]["inputs"]["clip"] = ["110", 1]

    return workflow


def _build_flux2_klein_workflow(
    prompt: str,
    model: str | None,
    seed: int,
    steps: int,
    sampler: str,
    width: int,
    height: int,
    batch_size: int,
    lora_name: str | None,
    lora_strength: float,
    vae: str | None,
    guidance: float,
    clip_encoder: str,
    clip_type: str,
) -> dict[str, Any]:
    """Build a Flux.2 Klein 9B workflow (single Qwen3 encoder, custom sampling).

    The graph differs from Flux.1 in three ways:
    1. Single-encoder ``CLIPLoader`` (type=flux2) instead of ``DualCLIPLoader``.
    2. ``EmptyFlux2LatentImage`` for the Flux2-specific latent shape.
    3. Custom-sampling pipeline: ``Flux2Scheduler`` produces SIGMAS, fed into
       ``SamplerCustomAdvanced`` along with ``BasicGuider``/``RandomNoise``/
       ``KSamplerSelect``. There is no standalone ``KSampler`` node, so the
       caller-provided ``scheduler`` is ignored (Flux2Scheduler is the only
       supported sigma source).
    """
    workflow = copy.deepcopy(FLUX2_KLEIN_WORKFLOW_TEMPLATE)

    actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)

    # UNet checkpoint
    if model:
        workflow["100"]["inputs"]["unet_name"] = model

    # Text encoder (Qwen3-8B variant — fp8 default, callers can override via
    # family preset's clip_encoder field if a different quantization is desired).
    workflow["101"]["inputs"]["clip_name"] = clip_encoder
    workflow["101"]["inputs"]["type"] = clip_type

    # External VAE — fall back to flux2-vae.safetensors from the template if unset
    if vae:
        workflow["102"]["inputs"]["vae_name"] = vae

    # Positive prompt (negative is unused for Flux — guidance is distilled)
    workflow["130"]["inputs"]["text"] = prompt

    # FluxGuidance carries the real prompt-adherence dial
    workflow["140"]["inputs"]["guidance"] = guidance

    # Latent dimensions
    workflow["150"]["inputs"]["width"] = width
    workflow["150"]["inputs"]["height"] = height
    workflow["150"]["inputs"]["batch_size"] = batch_size

    # Noise seed (separate node in custom-sampling pipeline)
    workflow["151"]["inputs"]["noise_seed"] = actual_seed

    # Sampler selection
    workflow["152"]["inputs"]["sampler_name"] = sampler

    # Flux2Scheduler — must receive matching width/height for correct sigma schedule
    workflow["153"]["inputs"]["steps"] = steps
    workflow["153"]["inputs"]["width"] = width
    workflow["153"]["inputs"]["height"] = height

    # Optional LoRA: injected between UNet/CLIP loaders and BasicGuider/
    # CLIPTextEncode consumers. Mirrors the flux_unet wiring pattern.
    if lora_name:
        workflow["110"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["100", 0],
                "clip": ["101", 0],
                "lora_name": lora_name,
                "strength_model": lora_strength,
                "strength_clip": lora_strength,
            },
        }
        # Re-route consumers from raw loaders (100/101) to LoRA outputs (110).
        workflow["154"]["inputs"]["model"] = ["110", 0]
        workflow["130"]["inputs"]["clip"] = ["110", 1]
        workflow["131"]["inputs"]["clip"] = ["110", 1]

    return workflow


def _build_workflow(
    prompt: str,
    negative_prompt: str = "",
    model: str | None = None,
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    cfg: float | None = None,
    seed: int = -1,
    sampler: str | None = None,
    scheduler: str | None = None,
    lora_name: str | None = None,
    lora_strength: float = 1.0,
    batch_size: int = 1,
    vae: str | None = None,
    orientation: str = "square",
    guidance: float | None = None,
) -> dict[str, Any]:
    """Build a text-to-image workflow from parameters.

    Parameters set to None are auto-resolved from the checkpoint's family preset
    via config.get_model_generation_defaults(). User-provided values always win.

    For Flux Dev/Schnell models, the workflow dispatches to FLUX_WORKFLOW_TEMPLATE
    which wires FluxGuidance + ConditioningZeroOut + ModelSamplingFlux around a
    KSampler locked to cfg=1.0 (Flux is guidance-distilled, real prompt-adherence
    lives on FluxGuidance). The ``guidance`` param maps to FluxGuidance; if not
    provided, falls back to ``cfg`` (treated as guidance for Flux), then preset.

    Args:
        prompt: Positive prompt text
        negative_prompt: Negative prompt text (zeroed-out for Flux)
        model: Checkpoint filename (if None, uses first available)
        width: Image width (None = use preset for orientation)
        height: Image height (None = use preset for orientation)
        steps: Number of sampling steps (None = use preset)
        cfg: CFG scale (None = use preset). For Flux models, this is interpreted
            as the FluxGuidance value if ``guidance`` is not explicitly set.
        seed: Random seed (-1 for random)
        sampler: Sampler name (None = use preset)
        scheduler: Scheduler name (None = use preset)
        lora_name: LoRA model filename (optional)
        lora_strength: LoRA strength (default 1.0)
        batch_size: Number of images to generate in one workflow (default 1)
        vae: VAE filename (None = use preset)
        orientation: Resolution orientation: "square", "portrait", or "landscape"
        guidance: FluxGuidance value (Flux only; default = preset 3.5)

    Returns:
        ComfyUI workflow dict
    """
    from tensors.config import get_model_generation_defaults, resolve_orientation  # noqa: PLC0415

    # Get preset defaults for this checkpoint family
    defaults = get_model_generation_defaults(model or "") if model else get_model_generation_defaults("")
    family = defaults.get("family")

    # Resolve orientation-based resolution
    res_w, res_h = resolve_orientation(family, orientation)

    # Merge: user overrides > preset defaults
    resolved_sampler = sampler if sampler is not None else defaults.get("sampler", "euler")
    resolved_scheduler = scheduler if scheduler is not None else defaults.get("scheduler", "normal")
    resolved_cfg = cfg if cfg is not None else defaults.get("cfg", 7.0)
    resolved_steps = steps if steps is not None else defaults.get("steps", 20)
    resolved_width = width if width is not None else res_w
    resolved_height = height if height is not None else res_h
    resolved_vae = vae if vae is not None else defaults.get("vae")

    # Dispatch to Flux-specific template when the family is flux/flux_schnell.
    if family in ("flux", "flux_schnell"):
        return _build_flux_workflow(
            prompt=prompt,
            model=model,
            seed=seed,
            steps=resolved_steps,
            sampler=resolved_sampler,
            scheduler=resolved_scheduler,
            width=resolved_width,
            height=resolved_height,
            batch_size=batch_size,
            lora_name=lora_name,
            lora_strength=lora_strength,
            vae=resolved_vae,
            guidance=_resolve_flux_guidance(guidance, cfg, defaults),
        )

    # Flux.2 Klein 9B: different architecture (single Qwen3 encoder, custom
    # sampling pipeline, Flux2 latent format). Must dispatch BEFORE flux_unet
    # since Klein checkpoints also set external_clip=True.
    if family == "flux2_klein":
        return _build_flux2_klein_workflow(
            prompt=prompt,
            model=model,
            seed=seed,
            steps=resolved_steps,
            sampler=resolved_sampler,
            width=resolved_width,
            height=resolved_height,
            batch_size=batch_size,
            lora_name=lora_name,
            lora_strength=lora_strength,
            vae=resolved_vae,
            guidance=_resolve_flux_guidance(guidance, cfg, defaults),
            clip_encoder=defaults.get("clip_encoder", "qwen_3_8b_fp8mixed.safetensors"),
            clip_type=defaults.get("clip_type", "flux2"),
        )

    # UNet-only Flux checkpoints (no baked-in CLIP/T5/VAE): use the split-loader
    # variant. Triggered by family="flux_unet" — also covers any family whose
    # preset opts in via external_clip=True.
    if family == "flux_unet" or defaults.get("external_clip"):
        return _build_flux_unet_workflow(
            prompt=prompt,
            model=model,
            seed=seed,
            steps=resolved_steps,
            sampler=resolved_sampler,
            scheduler=resolved_scheduler,
            width=resolved_width,
            height=resolved_height,
            batch_size=batch_size,
            lora_name=lora_name,
            lora_strength=lora_strength,
            vae=resolved_vae,
            guidance=_resolve_flux_guidance(guidance, cfg, defaults),
            clip_l=defaults.get("clip_l", "clip_l.safetensors"),
            clip_t5=defaults.get("clip_t5", "t5xxl_fp16.safetensors"),
        )

    workflow = copy.deepcopy(DEFAULT_WORKFLOW_TEMPLATE)

    # Set seed (random if -1)
    actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)

    # Update KSampler settings
    workflow["3"]["inputs"]["seed"] = actual_seed
    workflow["3"]["inputs"]["steps"] = resolved_steps
    workflow["3"]["inputs"]["cfg"] = resolved_cfg
    workflow["3"]["inputs"]["sampler_name"] = resolved_sampler
    workflow["3"]["inputs"]["scheduler"] = resolved_scheduler

    # Set model
    if model:
        workflow["4"]["inputs"]["ckpt_name"] = model

    # Set dimensions and batch size
    workflow["5"]["inputs"]["width"] = resolved_width
    workflow["5"]["inputs"]["height"] = resolved_height
    workflow["5"]["inputs"]["batch_size"] = batch_size

    # Set prompts
    workflow["6"]["inputs"]["text"] = prompt
    workflow["7"]["inputs"]["text"] = negative_prompt

    # Set VAE - use preset VAE if available, otherwise use checkpoint's built-in
    if resolved_vae:
        # Use external VAE loader (node 11)
        workflow["11"]["inputs"]["vae_name"] = resolved_vae
    else:
        # Use VAE from checkpoint (node 4, output index 2) - fallback for unknown models
        # Remove VAELoader node and connect VAEDecode directly to checkpoint
        del workflow["11"]
        workflow["8"]["inputs"]["vae"] = ["4", 2]

    # Inject LoRA loader if specified
    if lora_name:
        # Add LoRA loader node (node 10)
        lora_node = copy.deepcopy(LORA_LOADER_NODE)
        lora_node["inputs"]["lora_name"] = lora_name
        lora_node["inputs"]["strength_model"] = lora_strength
        lora_node["inputs"]["strength_clip"] = lora_strength
        # LoRA takes model/clip from checkpoint (node 4)
        lora_node["inputs"]["model"] = ["4", 0]
        lora_node["inputs"]["clip"] = ["4", 1]
        workflow["10"] = lora_node

        # Reroute KSampler model input from checkpoint (4) to LoRA (10)
        workflow["3"]["inputs"]["model"] = ["10", 0]

        # Reroute CLIP text encoders from checkpoint (4) to LoRA (10)
        workflow["6"]["inputs"]["clip"] = ["10", 1]
        workflow["7"]["inputs"]["clip"] = ["10", 1]

    return workflow


def generate_image(
    prompt: str,
    url: str | None = None,
    negative_prompt: str = "",
    model: str | None = None,
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    cfg: float | None = None,
    seed: int = -1,
    sampler: str | None = None,
    scheduler: str | None = None,
    console: Console | None = None,
    on_progress: ProgressCallback | None = None,
    timeout: float = 600.0,
    lora_name: str | None = None,
    lora_strength: float = 1.0,
    batch_size: int = 1,
    vae: str | None = None,
    orientation: str = "square",
    guidance: float | None = None,
) -> GenerationResult | None:
    """Generate an image using a simple text-to-image workflow.

    Parameters set to None are auto-resolved from the checkpoint's family preset.
    User-provided values always override preset defaults. For Flux Dev/Schnell
    checkpoints, ``guidance`` controls the FluxGuidance node (defaults to 3.5);
    KSampler cfg is locked to 1.0 by the Flux template.

    Args:
        prompt: Positive prompt text
        url: ComfyUI base URL
        negative_prompt: Negative prompt text
        model: Checkpoint filename (if None, must be pre-loaded in ComfyUI)
        width: Image width (None = use preset for orientation)
        height: Image height (None = use preset for orientation)
        steps: Number of sampling steps (None = use preset)
        cfg: CFG scale (None = use preset)
        seed: Random seed (-1 for random)
        sampler: Sampler name (None = use preset)
        scheduler: Scheduler name (None = use preset)
        console: Rich console for progress output
        on_progress: Optional callback for progress updates
        timeout: Maximum wait time in seconds
        lora_name: LoRA model filename (optional)
        lora_strength: LoRA strength (default 1.0)
        batch_size: Number of images to generate in one workflow (default 1)
        vae: VAE filename (None = use preset)
        orientation: Resolution orientation: "square", "portrait", or "landscape"

    Returns:
        GenerationResult with image paths, or None if generation failed
    """
    base_url = url or _get_comfyui_url()

    # Get available models if none specified
    if not model:
        models = get_loaded_models(url=base_url)
        if models and models.get("checkpoints"):
            model = models["checkpoints"][0]
            if console:
                console.print(f"[dim]Using model: {model}[/dim]")
        else:
            if console:
                console.print("[red]No checkpoints available. Specify a model with --model[/red]")
            return None

    # Build workflow
    workflow = _build_workflow(
        prompt=prompt,
        negative_prompt=negative_prompt,
        model=model,
        width=width,
        height=height,
        steps=steps,
        cfg=cfg,
        seed=seed,
        sampler=sampler,
        scheduler=scheduler,
        lora_name=lora_name,
        lora_strength=lora_strength,
        batch_size=batch_size,
        vae=vae,
        orientation=orientation,
        guidance=guidance,
    )

    # Run workflow
    result = run_workflow(
        workflow=workflow,
        url=base_url,
        console=console,
        on_progress=on_progress,
        timeout=timeout,
    )

    if not result:
        return None

    if not result.success:
        if console:
            console.print("[red]Generation failed[/red]")
            for node_id, errors in result.node_errors.items():
                console.print(f"  [yellow]Node {node_id}:[/yellow] {errors}")
        return GenerationResult(
            prompt_id=result.prompt_id,
            node_errors=result.node_errors,
            success=False,
        )

    # Extract image paths from outputs
    images: list[Path] = []
    for _node_id, output in result.outputs.items():
        if "images" in output:
            for img_info in output["images"]:
                filename = img_info.get("filename", "")
                subfolder = img_info.get("subfolder", "")
                img_type = img_info.get("type", "output")

                # Construct path (ComfyUI default output structure)
                if img_type == "output":
                    img_path = Path(subfolder) / filename if subfolder else Path(filename)
                    images.append(img_path)

    if console and images:
        console.print(f"[green]Generated {len(images)} image(s)[/green]")
        for img in images:
            console.print(f"  [dim]{img}[/dim]")

    return GenerationResult(
        prompt_id=result.prompt_id,
        images=images,
        success=True,
    )


def get_image(
    filename: str,
    url: str | None = None,
    subfolder: str = "",
    folder_type: str = "output",
) -> bytes | None:
    """Download a generated image from ComfyUI.

    Args:
        filename: Image filename
        url: ComfyUI base URL
        subfolder: Subfolder within the output directory
        folder_type: Folder type (output, input, temp)

    Returns:
        Image bytes or None on error
    """
    base_url = url or _get_comfyui_url()

    try:
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        response = httpx.get(f"{base_url}/view", params=params, timeout=30.0)
        response.raise_for_status()
        return response.content
    except httpx.RequestError:
        return None
