"""sd-server process lifecycle management."""

from __future__ import annotations

import asyncio
import logging
import shutil
import signal
import subprocess
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from tensors.server.models import ServerConfig

logger = logging.getLogger(__name__)

_HTTP_OK = 200

SD_SERVER_BIN = shutil.which("sd-server") or "sd-server"


class ProcessManager:
    def __init__(self) -> None:
        self.proc: subprocess.Popen[bytes] | None = None
        self.config: ServerConfig | None = None

    def build_cmd(self) -> list[str]:
        if self.config is None:
            raise RuntimeError("No config set")
        cmd = [SD_SERVER_BIN, "-m", self.config.model, "--port", str(self.config.port)]
        cmd.extend(self.config.args)
        return cmd

    def start(self, config: ServerConfig) -> None:
        if self.proc is not None and self.proc.poll() is None:
            raise RuntimeError("Server already running — stop it first")
        self.config = config
        cmd = self.build_cmd()
        self.proc = subprocess.Popen(cmd)
        logger.info("started sd-server pid=%d cmd=%s", self.proc.pid, cmd)

    def stop(self) -> bool:
        if self.proc is None or self.proc.poll() is not None:
            self.proc = None
            return False
        self.proc.send_signal(signal.SIGTERM)
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)
        logger.info("stopped sd-server")
        self.proc = None
        return True

    def status(self) -> dict[str, Any]:
        if self.proc is None:
            return {"running": False}
        rc = self.proc.poll()
        if rc is not None:
            return {"running": False, "exit_code": rc}
        return {
            "running": True,
            "pid": self.proc.pid,
            "model": self.config.model if self.config else None,
            "cmd": self.build_cmd(),
        }

    async def wait_ready(self, timeout: float = 120) -> bool:
        """Poll sd-server /health until it responds or timeout."""
        if self.config is None:
            return False
        url = f"http://127.0.0.1:{self.config.port}/health"
        deadline = asyncio.get_event_loop().time() + timeout
        async with httpx.AsyncClient() as client:
            while asyncio.get_event_loop().time() < deadline:
                if self.proc is not None and self.proc.poll() is not None:
                    return False
                try:
                    r = await client.get(url, timeout=2)
                    if r.status_code == _HTTP_OK:
                        return True
                except httpx.ConnectError:
                    pass
                await asyncio.sleep(1)
        return False
