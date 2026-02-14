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


# =============================================================================
# Gallery Endpoint Tests
# =============================================================================


@pytest.fixture
def temp_gallery(tmp_path):
    """Create a temporary gallery for testing."""
    from tensors.server.gallery import Gallery  # noqa: PLC0415

    gallery_dir = tmp_path / "gallery"
    gallery_dir.mkdir()
    return Gallery(gallery_dir=gallery_dir)


@pytest.fixture
def gallery_with_images(temp_gallery):
    """Gallery with some test images."""
    # Create test images
    for i in range(3):
        # Create a minimal PNG (1x1 pixel)
        image_data = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
            b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
            b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        metadata = {"prompt": f"test prompt {i}", "seed": i, "width": 512, "height": 512}
        temp_gallery.save_image(image_data, metadata=metadata, seed=i)

    return temp_gallery


@pytest.fixture
def gallery_api(temp_gallery) -> TestClient:
    """Test client for gallery API with temp gallery."""
    from fastapi import FastAPI  # noqa: PLC0415

    # Override the gallery singleton
    from tensors.server import gallery_routes  # noqa: PLC0415
    from tensors.server.gallery_routes import create_gallery_router  # noqa: PLC0415

    gallery_routes._gallery = temp_gallery

    app = FastAPI()
    app.include_router(create_gallery_router())
    return TestClient(app)


class TestGalleryList:
    """Tests for gallery list endpoint."""

    def test_list_images_empty(self, gallery_api: TestClient) -> None:
        """Test listing empty gallery."""
        response = gallery_api.get("/api/images")
        assert response.status_code == 200
        data = response.json()
        assert data["images"] == []
        assert data["total"] == 0

    def test_list_images_with_data(self, gallery_api: TestClient, gallery_with_images) -> None:
        """Test listing gallery with images."""
        from tensors.server import gallery_routes  # noqa: PLC0415

        gallery_routes._gallery = gallery_with_images

        response = gallery_api.get("/api/images")
        assert response.status_code == 200
        data = response.json()
        assert len(data["images"]) == 3
        assert data["total"] == 3

    def test_list_images_pagination(self, gallery_api: TestClient, gallery_with_images) -> None:
        """Test pagination parameters."""
        from tensors.server import gallery_routes  # noqa: PLC0415

        gallery_routes._gallery = gallery_with_images

        response = gallery_api.get("/api/images?limit=2&offset=1")
        assert response.status_code == 200
        data = response.json()
        assert len(data["images"]) == 2


class TestGalleryGetImage:
    """Tests for getting individual images."""

    def test_get_image_not_found(self, gallery_api: TestClient) -> None:
        """Test getting non-existent image returns 404."""
        response = gallery_api.get("/api/images/nonexistent")
        assert response.status_code == 404

    def test_get_image_success(self, gallery_api: TestClient, gallery_with_images) -> None:
        """Test getting an image file."""
        from tensors.server import gallery_routes  # noqa: PLC0415

        gallery_routes._gallery = gallery_with_images

        list_response = gallery_api.get("/api/images")
        images = list_response.json()["images"]
        image_id = images[0]["id"]

        response = gallery_api.get(f"/api/images/{image_id}")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"


class TestGalleryMetadata:
    """Tests for image metadata endpoints."""

    def test_get_metadata_not_found(self, gallery_api: TestClient) -> None:
        """Test getting metadata for non-existent image."""
        response = gallery_api.get("/api/images/nonexistent/meta")
        assert response.status_code == 404

    def test_get_metadata_success(self, gallery_api: TestClient, gallery_with_images) -> None:
        """Test getting image metadata."""
        from tensors.server import gallery_routes  # noqa: PLC0415

        gallery_routes._gallery = gallery_with_images

        list_response = gallery_api.get("/api/images")
        images = list_response.json()["images"]
        image_id = images[0]["id"]

        response = gallery_api.get(f"/api/images/{image_id}/meta")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == image_id
        assert "metadata" in data

    def test_edit_metadata(self, gallery_api: TestClient, gallery_with_images) -> None:
        """Test updating image metadata."""
        from tensors.server import gallery_routes  # noqa: PLC0415

        gallery_routes._gallery = gallery_with_images

        list_response = gallery_api.get("/api/images")
        images = list_response.json()["images"]
        image_id = images[0]["id"]

        response = gallery_api.post(
            f"/api/images/{image_id}/edit",
            json={"tags": ["test", "favorite"], "rating": 5},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["tags"] == ["test", "favorite"]
        assert data["metadata"]["rating"] == 5

    def test_edit_metadata_not_found(self, gallery_api: TestClient) -> None:
        """Test editing non-existent image metadata."""
        response = gallery_api.post(
            "/api/images/nonexistent/edit",
            json={"tags": ["test"]},
        )
        assert response.status_code == 404


class TestGalleryDelete:
    """Tests for deleting images."""

    def test_delete_image_not_found(self, gallery_api: TestClient) -> None:
        """Test deleting non-existent image."""
        response = gallery_api.delete("/api/images/nonexistent")
        assert response.status_code == 404

    def test_delete_image_success(self, gallery_api: TestClient, gallery_with_images) -> None:
        """Test deleting an image."""
        from tensors.server import gallery_routes  # noqa: PLC0415

        gallery_routes._gallery = gallery_with_images

        list_response = gallery_api.get("/api/images")
        initial_count = list_response.json()["total"]
        image_id = list_response.json()["images"][0]["id"]

        response = gallery_api.delete(f"/api/images/{image_id}")
        assert response.status_code == 200
        assert response.json()["deleted"] is True

        list_response = gallery_api.get("/api/images")
        assert list_response.json()["total"] == initial_count - 1


class TestGalleryStats:
    """Tests for gallery statistics endpoint."""

    def test_stats_empty(self, gallery_api: TestClient) -> None:
        """Test stats on empty gallery."""
        response = gallery_api.get("/api/images/stats/summary")
        assert response.status_code == 200
        data = response.json()
        assert data["total_images"] == 0


# =============================================================================
# Database Endpoint Tests
# =============================================================================


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database for testing."""
    from tensors.db import Database  # noqa: PLC0415

    db_path = tmp_path / "test_models.db"
    db = Database(db_path=db_path)
    db.init_schema()
    return db


@pytest.fixture
def db_api(temp_db, monkeypatch) -> TestClient:
    """Test client for db API with temp database."""
    from fastapi import FastAPI  # noqa: PLC0415

    # Monkeypatch Database to use temp_db path
    from tensors import db as db_module  # noqa: PLC0415
    from tensors.server.db_routes import create_db_router  # noqa: PLC0415

    monkeypatch.setattr(db_module, "DB_PATH", temp_db.db_path)

    app = FastAPI()
    app.include_router(create_db_router())
    return TestClient(app)


class TestDbEndpoints:
    """Tests for database API endpoints."""

    def test_list_files_empty(self, db_api: TestClient) -> None:
        """Test listing files from empty database."""
        response = db_api.get("/api/db/files")
        assert response.status_code == 200
        assert response.json() == []

    def test_search_models_empty(self, db_api: TestClient) -> None:
        """Test searching models in empty database."""
        response = db_api.get("/api/db/models")
        assert response.status_code == 200
        assert response.json() == []

    def test_search_models_with_query(self, db_api: TestClient, temp_db, monkeypatch) -> None:
        """Test searching models with query parameters."""
        from tensors import db as db_module  # noqa: PLC0415

        monkeypatch.setattr(db_module, "DB_PATH", temp_db.db_path)

        model_data = {
            "id": 12345,
            "name": "Test Model",
            "type": "LORA",
            "tags": [],
            "modelVersions": [],
        }
        temp_db.cache_model(model_data)

        response = db_api.get("/api/db/models?query=Test")
        assert response.status_code == 200
        results = response.json()
        assert len(results) >= 1

    def test_get_model_not_found(self, db_api: TestClient) -> None:
        """Test getting non-existent model."""
        response = db_api.get("/api/db/models/999999")
        assert response.status_code == 404

    def test_get_model_success(self, db_api: TestClient, temp_db, monkeypatch) -> None:
        """Test getting cached model."""
        from tensors import db as db_module  # noqa: PLC0415

        monkeypatch.setattr(db_module, "DB_PATH", temp_db.db_path)

        model_data = {
            "id": 12345,
            "name": "Test Model",
            "type": "Checkpoint",
            "tags": [],
            "modelVersions": [],
        }
        temp_db.cache_model(model_data)

        response = db_api.get("/api/db/models/12345")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Model"

    def test_get_stats(self, db_api: TestClient) -> None:
        """Test getting database stats."""
        response = db_api.get("/api/db/stats")
        assert response.status_code == 200
        data = response.json()
        assert "local_files" in data
        assert "models" in data


# =============================================================================
# Gallery Class Unit Tests
# =============================================================================


class TestGalleryClass:
    """Unit tests for the Gallery class."""

    def test_save_image(self, temp_gallery) -> None:
        """Test saving an image to gallery."""
        image_data = b"\x89PNG test data"
        metadata = {"prompt": "test", "seed": 42}

        result = temp_gallery.save_image(image_data, metadata=metadata, seed=42)

        assert result.id is not None
        assert result.path.exists()
        assert result.meta_path.exists()

    def test_list_images_empty(self, temp_gallery) -> None:
        """Test listing empty gallery."""
        images = temp_gallery.list_images()
        assert images == []

    def test_list_images_sorted(self, gallery_with_images) -> None:
        """Test images are sorted by creation time."""
        images = gallery_with_images.list_images(newest_first=True)
        assert len(images) == 3

        times = [img.created_at for img in images]
        assert times == sorted(times, reverse=True)

    def test_get_image(self, gallery_with_images) -> None:
        """Test getting image by ID."""
        images = gallery_with_images.list_images()
        image_id = images[0].id

        result = gallery_with_images.get_image(image_id)
        assert result is not None
        assert result.id == image_id

    def test_get_image_not_found(self, temp_gallery) -> None:
        """Test getting non-existent image."""
        result = temp_gallery.get_image("nonexistent")
        assert result is None

    def test_delete_image(self, gallery_with_images) -> None:
        """Test deleting an image."""
        images = gallery_with_images.list_images()
        image_id = images[0].id
        image_path = images[0].path

        result = gallery_with_images.delete_image(image_id)
        assert result is True
        assert not image_path.exists()

    def test_delete_image_not_found(self, temp_gallery) -> None:
        """Test deleting non-existent image."""
        result = temp_gallery.delete_image("nonexistent")
        assert result is False

    def test_count(self, gallery_with_images) -> None:
        """Test counting images."""
        assert gallery_with_images.count() == 3

    def test_update_metadata(self, gallery_with_images) -> None:
        """Test updating metadata."""
        images = gallery_with_images.list_images()
        image_id = images[0].id

        result = gallery_with_images.update_metadata(image_id, {"custom_field": "value"})
        assert result is not None
        assert result["custom_field"] == "value"

    def test_update_metadata_not_found(self, temp_gallery) -> None:
        """Test updating metadata for non-existent image."""
        result = temp_gallery.update_metadata("nonexistent", {"field": "value"})
        assert result is None
