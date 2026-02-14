"""HTTP client for remote tsr server API."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from collections.abc import Iterator


class TsrClientError(Exception):
    """Error from TsrClient operations."""


class TsrClient:
    """HTTP client wrapper for tsr server API.

    Usage:
        with TsrClient("http://junkpile:8080") as client:
            images = client.list_images()
            result = client.generate("a cat")
    """

    def __init__(self, base_url: str, timeout: float = 300.0) -> None:
        """Initialize client with server URL."""
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.Client | None = None

    def __enter__(self) -> TsrClient:
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._client:
            self._client.close()
            self._client = None

    @property
    def client(self) -> httpx.Client:
        """Get the HTTP client, creating if needed."""
        if self._client is None:
            self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        return self._client

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Make GET request."""
        try:
            resp = self.client.get(path, params=params)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise TsrClientError(f"HTTP {e.response.status_code}: {e.response.text}") from e
        except httpx.RequestError as e:
            raise TsrClientError(f"Request failed: {e}") from e

    def _post(self, path: str, json: dict[str, Any] | None = None) -> Any:
        """Make POST request."""
        try:
            resp = self.client.post(path, json=json)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise TsrClientError(f"HTTP {e.response.status_code}: {e.response.text}") from e
        except httpx.RequestError as e:
            raise TsrClientError(f"Request failed: {e}") from e

    def _delete(self, path: str) -> Any:
        """Make DELETE request."""
        try:
            resp = self.client.delete(path)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise TsrClientError(f"HTTP {e.response.status_code}: {e.response.text}") from e
        except httpx.RequestError as e:
            raise TsrClientError(f"Request failed: {e}") from e

    # =========================================================================
    # Server Status
    # =========================================================================

    def status(self) -> dict[str, Any]:
        """Get server status."""
        return dict(self._get("/status"))

    # =========================================================================
    # Gallery / Images
    # =========================================================================

    def list_images(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List images in gallery."""
        return dict(self._get("/api/images", params={"limit": limit, "offset": offset}))

    def get_image_meta(self, image_id: str) -> dict[str, Any]:
        """Get metadata for an image."""
        return dict(self._get(f"/api/images/{image_id}/meta"))

    def delete_image(self, image_id: str) -> dict[str, Any]:
        """Delete an image."""
        return dict(self._delete(f"/api/images/{image_id}"))

    def edit_image(self, image_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        """Update image metadata."""
        return dict(self._post(f"/api/images/{image_id}/edit", json=updates))

    def download_image(self, image_id: str) -> bytes:
        """Download image file bytes."""
        try:
            resp = self.client.get(f"/api/images/{image_id}")
            resp.raise_for_status()
            return resp.content
        except httpx.HTTPStatusError as e:
            raise TsrClientError(f"HTTP {e.response.status_code}: {e.response.text}") from e
        except httpx.RequestError as e:
            raise TsrClientError(f"Request failed: {e}") from e

    # =========================================================================
    # Models
    # =========================================================================

    def list_models(self) -> dict[str, Any]:
        """List available models."""
        return dict(self._get("/api/models"))

    def get_active_model(self) -> dict[str, Any]:
        """Get currently active model."""
        return dict(self._get("/api/models/active"))

    def switch_model(self, model_path: str) -> dict[str, Any]:
        """Switch to a different model."""
        return dict(self._post("/api/models/switch", json={"model": model_path}))

    def list_loras(self) -> dict[str, Any]:
        """List available LoRAs."""
        return dict(self._get("/api/models/loras"))

    def scan_models(self) -> dict[str, Any]:
        """Scan model directories."""
        return dict(self._get("/api/models/scan"))

    # =========================================================================
    # Generation
    # =========================================================================

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg_scale: float = 7.0,
        seed: int = -1,
        sampler_name: str = "",
        scheduler: str = "",
        batch_size: int = 1,
        save_to_gallery: bool = True,
        return_base64: bool = False,
    ) -> dict[str, Any]:
        """Generate images."""
        body = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "batch_size": batch_size,
            "save_to_gallery": save_to_gallery,
            "return_base64": return_base64,
        }
        return dict(self._post("/api/generate", json=body))

    def list_samplers(self) -> dict[str, Any]:
        """List available samplers."""
        return dict(self._get("/api/samplers"))

    def list_schedulers(self) -> dict[str, Any]:
        """List available schedulers."""
        return dict(self._get("/api/schedulers"))

    # =========================================================================
    # Download
    # =========================================================================

    def start_download(
        self,
        version_id: int | None = None,
        model_id: int | None = None,
        hash_val: str | None = None,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        """Start a model download from CivitAI."""
        body: dict[str, Any] = {}
        if version_id:
            body["version_id"] = version_id
        if model_id:
            body["model_id"] = model_id
        if hash_val:
            body["hash"] = hash_val
        if output_dir:
            body["output_dir"] = output_dir
        return dict(self._post("/api/download", json=body))

    def get_download_status(self, download_id: str) -> dict[str, Any]:
        """Get download status."""
        return dict(self._get(f"/api/download/status/{download_id}"))

    def list_downloads(self) -> dict[str, Any]:
        """List active downloads."""
        return dict(self._get("/api/download/active"))

    # =========================================================================
    # Database
    # =========================================================================

    def db_list_files(self) -> list[dict[str, Any]]:
        """List local files in database."""
        return list(self._get("/api/db/files"))

    def db_search_models(
        self,
        query: str | None = None,
        model_type: str | None = None,
        base_model: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search cached models."""
        params: dict[str, Any] = {"limit": limit}
        if query:
            params["query"] = query
        if model_type:
            params["type"] = model_type
        if base_model:
            params["base"] = base_model
        return list(self._get("/api/db/models", params=params))

    def db_get_model(self, civitai_id: int) -> dict[str, Any]:
        """Get cached model by CivitAI ID."""
        return dict(self._get(f"/api/db/models/{civitai_id}"))

    def db_get_triggers(self, file_path: str | None = None, version_id: int | None = None) -> list[str]:
        """Get trigger words."""
        if version_id:
            return list(self._get(f"/api/db/triggers/{version_id}"))
        if file_path:
            return list(self._get("/api/db/triggers", params={"file_path": file_path}))
        return []

    def db_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        return dict(self._get("/api/db/stats"))

    def db_scan(self, directory: str) -> dict[str, Any]:
        """Scan directory for safetensor files."""
        return dict(self._post("/api/db/scan", json={"directory": directory}))

    def db_link(self) -> dict[str, Any]:
        """Link unlinked files to CivitAI."""
        return dict(self._post("/api/db/link"))

    def db_cache(self, model_id: int) -> dict[str, Any]:
        """Cache CivitAI model data."""
        return dict(self._post("/api/db/cache", json={"model_id": model_id}))

    # =========================================================================
    # Streaming Downloads
    # =========================================================================

    def stream_image(self, image_id: str) -> Iterator[bytes]:
        """Stream image download in chunks."""
        try:
            with self.client.stream("GET", f"/api/images/{image_id}") as resp:
                resp.raise_for_status()
                yield from resp.iter_bytes(chunk_size=1024 * 64)
        except httpx.HTTPStatusError as e:
            raise TsrClientError(f"HTTP {e.response.status_code}") from e
        except httpx.RequestError as e:
            raise TsrClientError(f"Request failed: {e}") from e

    def save_image_to(self, image_id: str, dest: Path) -> Path:
        """Download and save image to file."""
        content = self.download_image(image_id)
        dest.write_bytes(content)
        return dest
