"""HTTP client utilities for sd-server communication."""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import Request


def get_sd_headers(request: Request) -> dict[str, str]:
    """Get headers for sd-server requests, including API key if configured."""
    headers: dict[str, str] = {}
    api_key = getattr(request.app.state, "sd_server_api_key", None)
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


async def sd_get(request: Request, path: str, *, timeout: float = 30) -> httpx.Response:
    """Make a GET request to sd-server."""
    url = f"{request.app.state.sd_server_url}/{path.lstrip('/')}"
    headers = get_sd_headers(request)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response


async def sd_post(request: Request, path: str, *, json: dict[str, Any] | None = None, timeout: float = 300) -> httpx.Response:
    """Make a POST request to sd-server."""
    url = f"{request.app.state.sd_server_url}/{path.lstrip('/')}"
    headers = get_sd_headers(request)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=json, headers=headers)
        response.raise_for_status()
        return response
