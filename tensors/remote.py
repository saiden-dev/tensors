"""HTTP client for calling tensors server API remotely."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from rich.console import Console

from tensors.config import get_server_api_key, resolve_remote


def _build_client(base_url: str, timeout: float = 300.0) -> httpx.Client:
    """Build an httpx client with API key auth."""
    api_key = get_server_api_key()
    headers: dict[str, str] = {}
    if api_key:
        headers["X-API-Key"] = api_key
    return httpx.Client(base_url=base_url, headers=headers, timeout=timeout)


def remote_generate(
    remote: str,
    prompt: str,
    *,
    negative_prompt: str = "",
    model: str | None = None,
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
    cfg: float = 7.0,
    seed: int = -1,
    sampler: str = "euler",
    scheduler: str = "normal",
    vae: str | None = None,
    lora_name: str | None = None,
    lora_strength: float = 0.8,
    console: Console | None = None,
) -> dict[str, Any] | None:
    """Generate an image via remote tensors server.

    Args:
        remote: Remote name or URL (resolved via config)
        prompt: Positive prompt text
        console: Rich console for error output

    Returns:
        Response dict with success, prompt_id, images, errors — or None on connection error
    """
    base_url = resolve_remote(remote)
    if not base_url:
        if console:
            console.print("[red]Error: Could not resolve remote server[/red]")
        return None

    payload: dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg": cfg,
        "seed": seed,
        "sampler": sampler,
        "scheduler": scheduler,
        "lora_strength": lora_strength,
    }
    if model:
        payload["model"] = model
    if vae:
        payload["vae"] = vae
    if lora_name:
        payload["lora_name"] = lora_name

    try:
        with _build_client(base_url) as client:
            response = client.post("/api/comfyui/generate", json=payload)
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
    except httpx.HTTPStatusError as e:
        if console:
            console.print(f"[red]Remote API error: {e.response.status_code}[/red]")
            try:
                detail = e.response.json().get("detail", "")
                if detail:
                    console.print(f"  [yellow]{detail}[/yellow]")
            except Exception:
                pass
        return None
    except httpx.RequestError as e:
        if console:
            console.print(f"[red]Remote connection error: {e}[/red]")
        return None


def remote_get_image(remote: str, filename: str) -> bytes | None:
    """Download a generated image from remote tensors server.

    Args:
        remote: Remote name or URL
        filename: Image filename from generation result

    Returns:
        Image bytes or None on error
    """
    base_url = resolve_remote(remote)
    if not base_url:
        return None

    try:
        with _build_client(base_url) as client:
            response = client.get("/api/comfyui/image/" + filename)
            response.raise_for_status()
            return response.content
    except (httpx.HTTPStatusError, httpx.RequestError):
        return None


def remote_models(
    remote: str,
    console: Console | None = None,
) -> dict[str, list[str]] | None:
    """List available models from remote tensors server.

    Args:
        remote: Remote name or URL
        console: Rich console for error output

    Returns:
        Dict mapping model type to list of model names, or None on error
    """
    base_url = resolve_remote(remote)
    if not base_url:
        if console:
            console.print("[red]Error: Could not resolve remote server[/red]")
        return None

    try:
        with _build_client(base_url) as client:
            response = client.get("/api/comfyui/models")
            response.raise_for_status()
            result: dict[str, list[str]] = response.json()
            return result
    except httpx.HTTPStatusError as e:
        if console:
            console.print(f"[red]Remote API error: {e.response.status_code}[/red]")
        return None
    except httpx.RequestError as e:
        if console:
            console.print(f"[red]Remote connection error: {e}[/red]")
        return None


def remote_search(
    remote: str,
    *,
    query: str | None = None,
    model_type: str | None = None,
    base_model: str | None = None,
    sort: str = "downloads",
    limit: int = 20,
    page: int | None = None,
    nsfw: str | None = None,
    sfw: bool = False,
    console: Console | None = None,
) -> dict[str, Any] | None:
    """Search CivitAI models via remote tensors server.

    Args:
        remote: Remote name or URL
        console: Rich console for error output

    Returns:
        Search results dict or None on error
    """
    base_url = resolve_remote(remote)
    if not base_url:
        if console:
            console.print("[red]Error: Could not resolve remote server[/red]")
        return None

    params: dict[str, Any] = {
        "provider": "civitai",
        "sort": sort,
        "limit": limit,
    }
    if query:
        params["query"] = query
    if model_type:
        params["types"] = model_type
    if base_model:
        params["baseModels"] = base_model
    if page:
        params["page"] = page
    if sfw:
        params["sfw"] = True
    elif nsfw:
        params["nsfw"] = nsfw

    try:
        with _build_client(base_url) as client:
            response = client.get("/api/search", params=params)
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            # The remote API wraps CivitAI results under "civitai" key
            return result.get("civitai", result)
    except httpx.HTTPStatusError as e:
        if console:
            console.print(f"[red]Remote API error: {e.response.status_code}[/red]")
        return None
    except httpx.RequestError as e:
        if console:
            console.print(f"[red]Remote connection error: {e}[/red]")
        return None


def remote_download(
    remote: str,
    *,
    version_id: int | None = None,
    model_id: int | None = None,
    output_dir: str | None = None,
    console: Console | None = None,
) -> dict[str, Any] | None:
    """Start a model download on remote tensors server.

    Args:
        remote: Remote name or URL
        version_id: CivitAI version ID
        model_id: CivitAI model ID (downloads latest version)
        output_dir: Override output directory on the remote
        console: Rich console for error output

    Returns:
        Download status dict with download_id, or None on error
    """
    base_url = resolve_remote(remote)
    if not base_url:
        if console:
            console.print("[red]Error: Could not resolve remote server[/red]")
        return None

    payload: dict[str, Any] = {}
    if version_id:
        payload["version_id"] = version_id
    if model_id:
        payload["model_id"] = model_id
    if output_dir:
        payload["output_dir"] = output_dir

    try:
        with _build_client(base_url) as client:
            response = client.post("/api/download", json=payload)
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
    except httpx.HTTPStatusError as e:
        if console:
            console.print(f"[red]Remote API error: {e.response.status_code}[/red]")
            try:
                detail = e.response.json().get("detail", "")
                if detail:
                    console.print(f"  [yellow]{detail}[/yellow]")
            except Exception:
                pass
        return None
    except httpx.RequestError as e:
        if console:
            console.print(f"[red]Remote connection error: {e}[/red]")
        return None


def remote_download_status(
    remote: str,
    download_id: str,
) -> dict[str, Any] | None:
    """Check download status on remote tensors server.

    Args:
        remote: Remote name or URL
        download_id: Download ID from start_download response

    Returns:
        Download status dict or None on error
    """
    base_url = resolve_remote(remote)
    if not base_url:
        return None

    try:
        with _build_client(base_url) as client:
            response = client.get(f"/api/download/status/{download_id}")
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
    except (httpx.HTTPStatusError, httpx.RequestError):
        return None
