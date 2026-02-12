"""Tests for tensors.server package (FastAPI sd-server proxy wrapper)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from tensors.server import create_app
from tensors.server.models import ServerConfig
from tensors.server.process import ProcessManager


@pytest.fixture()
def pm() -> ProcessManager:
    return ProcessManager()


@pytest.fixture()
def api() -> TestClient:
    return TestClient(create_app())


def _get_pm(api: TestClient) -> ProcessManager:
    return api.app.state.pm  # type: ignore[no-any-return, attr-defined]


class TestStatus:
    def test_not_running(self, api: TestClient) -> None:
        r = api.get("/status")
        assert r.status_code == 200
        assert r.json()["running"] is False

    def test_running(self, api: TestClient) -> None:
        pm = _get_pm(api)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 999
        pm.proc = mock_proc
        pm.config = ServerConfig(model="/m.safetensors")
        r = api.get("/status")
        data = r.json()
        assert data["running"] is True
        assert data["pid"] == 999

    def test_exited(self, api: TestClient) -> None:
        pm = _get_pm(api)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        pm.proc = mock_proc
        r = api.get("/status")
        data = r.json()
        assert data["running"] is False
        assert data["exit_code"] == 1


class TestReload:
    @patch.object(ProcessManager, "wait_ready", new_callable=AsyncMock, return_value=True)
    @patch("tensors.server.process.subprocess.Popen")
    def test_reload_swaps_model(self, mock_popen: MagicMock, mock_ready: AsyncMock, api: TestClient) -> None:
        pm = _get_pm(api)
        pm.config = ServerConfig(model="/old.gguf", port=5555, args=["--fa"])
        mock_popen.return_value.pid = 42
        mock_popen.return_value.poll.return_value = None
        r = api.post("/reload", json={"model": "/new.gguf"})
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert data["model"] == "/new.gguf"
        assert data["pid"] == 42
        # Verify new config preserved port and args from previous config
        assert pm.config is not None
        assert pm.config.port == 5555
        assert pm.config.args == ["--fa"]
        assert pm.config.model == "/new.gguf"

    @patch.object(ProcessManager, "wait_ready", new_callable=AsyncMock, return_value=False)
    @patch("tensors.server.process.subprocess.Popen")
    def test_reload_fails_when_not_ready(self, mock_popen: MagicMock, mock_ready: AsyncMock, api: TestClient) -> None:
        pm = _get_pm(api)
        pm.config = ServerConfig(model="/old.gguf")
        mock_popen.return_value.pid = 43
        mock_popen.return_value.poll.return_value = None
        r = api.post("/reload", json={"model": "/bad.gguf"})
        assert r.status_code == 503
        assert "failed" in r.json()["error"]

    def test_reload_requires_model(self, api: TestClient) -> None:
        r = api.post("/reload", json={})
        assert r.status_code == 422


class TestProxy:
    def test_proxy_503_when_not_running(self, api: TestClient) -> None:
        r = api.get("/v1/models")
        assert r.status_code == 503
        assert "not running" in r.json()["error"]

    def test_proxy_forwards_request(self, api: TestClient) -> None:
        pm = _get_pm(api)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 100
        pm.proc = mock_proc
        pm.config = ServerConfig(model="/m.gguf", port=1234)

        upstream_response = httpx.Response(
            200,
            json={"data": [{"id": "model-1"}]},
            headers={"content-type": "application/json"},
        )
        mock_client = AsyncMock()
        mock_client.request.return_value = upstream_response
        api.app.state.client = mock_client  # type: ignore[attr-defined]

        r = api.get("/v1/models")
        assert r.status_code == 200
        assert r.json() == {"data": [{"id": "model-1"}]}
        mock_client.request.assert_called_once()

    def test_proxy_forwards_post_with_body(self, api: TestClient) -> None:
        pm = _get_pm(api)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 100
        pm.proc = mock_proc
        pm.config = ServerConfig(model="/m.gguf", port=1234)

        upstream_response = httpx.Response(200, json={"ok": True})
        mock_client = AsyncMock()
        mock_client.request.return_value = upstream_response
        api.app.state.client = mock_client  # type: ignore[attr-defined]

        r = api.post("/v1/chat/completions", json={"prompt": "hello"})
        assert r.status_code == 200
        mock_client.request.assert_called_once()


class TestProcessManager:
    def test_status_not_running(self, pm: ProcessManager) -> None:
        assert pm.status() == {"running": False}

    def test_build_cmd(self, pm: ProcessManager) -> None:
        pm.config = ServerConfig(model="/m.gguf", port=1234, args=["--fa"])
        cmd = pm.build_cmd()
        assert "/m.gguf" in cmd
        assert "--fa" in cmd
        assert "1234" in cmd

    def test_build_cmd_no_config(self, pm: ProcessManager) -> None:
        with pytest.raises(RuntimeError, match="No config"):
            pm.build_cmd()

    @patch("tensors.server.process.subprocess.Popen")
    def test_start_and_stop(self, mock_popen: MagicMock, pm: ProcessManager) -> None:
        mock_popen.return_value.pid = 77
        mock_popen.return_value.poll.return_value = None
        mock_popen.return_value.wait.return_value = 0
        pm.start(ServerConfig(model="/m.gguf"))
        assert pm.proc is not None
        assert pm.stop() is True
        assert pm.proc is None

    def test_server_config_defaults(self) -> None:
        cfg = ServerConfig(model="/m.gguf")
        assert cfg.port == 1234
        assert cfg.args == []
