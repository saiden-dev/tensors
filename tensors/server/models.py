"""Pydantic models for the sd-server wrapper API."""

from __future__ import annotations

from pydantic import BaseModel

DEFAULT_PORT = 1234


class ReloadRequest(BaseModel):
    model: str


class ServerConfig(BaseModel):
    model: str
    port: int = DEFAULT_PORT
    args: list[str] = []
