"""Tests for tensors.server package (gallery and CivitAI management)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tensors.server import create_app


@pytest.fixture()
def api() -> TestClient:
    """Create test client."""
    return TestClient(create_app())


class TestStatus:
    def test_status_ok(self, api: TestClient) -> None:
        """Test status endpoint returns ok."""
        r = api.get("/status")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"


# =============================================================================
# Gallery Endpoint Tests
# =============================================================================


@pytest.fixture
def temp_gallery(tmp_path):
    """Create a temporary gallery for testing."""
    from tensors.server.gallery import Gallery

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
    from fastapi import FastAPI

    # Override the gallery singleton
    from tensors.server import gallery_routes
    from tensors.server.gallery_routes import create_gallery_router

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
        from tensors.server import gallery_routes

        gallery_routes._gallery = gallery_with_images

        response = gallery_api.get("/api/images")
        assert response.status_code == 200
        data = response.json()
        assert len(data["images"]) == 3
        assert data["total"] == 3

    def test_list_images_pagination(self, gallery_api: TestClient, gallery_with_images) -> None:
        """Test pagination parameters."""
        from tensors.server import gallery_routes

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
        from tensors.server import gallery_routes

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
        from tensors.server import gallery_routes

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
        from tensors.server import gallery_routes

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
        from tensors.server import gallery_routes

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
    from tensors.db import Database

    db_path = tmp_path / "test_models.db"
    db = Database(db_path=db_path)
    db.init_schema()
    return db


@pytest.fixture
def db_api(temp_db, monkeypatch) -> TestClient:
    """Test client for db API with temp database."""
    from fastapi import FastAPI

    # Monkeypatch Database to use temp_db path
    from tensors import db as db_module
    from tensors.server.db_routes import create_db_router

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
        from tensors import db as db_module

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
        from tensors import db as db_module

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


# =============================================================================
# Auth Tests
# =============================================================================


class TestAuth:
    """Tests for API key authentication."""

    def test_no_auth_when_no_key_configured(self, monkeypatch) -> None:
        """Test auth is disabled when no API key is configured."""
        from tensors.server.auth import verify_api_key

        monkeypatch.setattr("tensors.server.auth.get_server_api_key", lambda: None)
        result = verify_api_key(header_key=None, query_key=None)
        assert result is None

    def test_auth_required_when_key_configured(self, monkeypatch) -> None:
        """Test auth is required when API key is configured."""
        from tensors.server.auth import verify_api_key

        monkeypatch.setattr("tensors.server.auth.get_server_api_key", lambda: "secret-key")

        with pytest.raises(Exception) as exc_info:
            verify_api_key(header_key=None, query_key=None)
        assert exc_info.value.status_code == 401

    def test_valid_header_key(self, monkeypatch) -> None:
        """Test valid API key via header."""
        from tensors.server.auth import verify_api_key

        monkeypatch.setattr("tensors.server.auth.get_server_api_key", lambda: "secret-key")
        result = verify_api_key(header_key="secret-key", query_key=None)
        assert result == "secret-key"

    def test_valid_query_key(self, monkeypatch) -> None:
        """Test valid API key via query param."""
        from tensors.server.auth import verify_api_key

        monkeypatch.setattr("tensors.server.auth.get_server_api_key", lambda: "secret-key")
        result = verify_api_key(header_key=None, query_key="secret-key")
        assert result == "secret-key"

    def test_invalid_key(self, monkeypatch) -> None:
        """Test invalid API key returns 403."""
        from tensors.server.auth import verify_api_key

        monkeypatch.setattr("tensors.server.auth.get_server_api_key", lambda: "secret-key")

        with pytest.raises(Exception) as exc_info:
            verify_api_key(header_key="wrong-key", query_key=None)
        assert exc_info.value.status_code == 403

    def test_header_takes_precedence(self, monkeypatch) -> None:
        """Test header key takes precedence over query key."""
        from tensors.server.auth import verify_api_key

        monkeypatch.setattr("tensors.server.auth.get_server_api_key", lambda: "secret-key")
        result = verify_api_key(header_key="secret-key", query_key="wrong-key")
        assert result == "secret-key"


# =============================================================================
# CivitAI Routes Tests
# =============================================================================


@pytest.fixture
def civitai_api(monkeypatch) -> TestClient:
    """Test client for CivitAI API."""
    from fastapi import FastAPI

    from tensors.server.civitai_routes import create_civitai_router

    # Disable auth for testing
    monkeypatch.setattr("tensors.config.get_server_api_key", lambda: None)

    app = FastAPI()
    app.include_router(create_civitai_router())
    return TestClient(app)


class TestCivitAIGetModel:
    """Tests for CivitAI get model endpoint."""

    def test_get_model_success(self, civitai_api: TestClient, respx_mock, temp_db, monkeypatch) -> None:
        """Test getting a model by ID."""
        import respx

        from tensors import db as db_module

        monkeypatch.setattr(db_module, "DB_PATH", temp_db.db_path)

        respx_mock.get("https://civitai.com/api/v1/models/12345").mock(
            return_value=respx.MockResponse(
                200,
                json={
                    "id": 12345,
                    "name": "Test Model",
                    "type": "LORA",
                    "tags": [],
                    "modelVersions": [],
                },
            )
        )

        response = civitai_api.get("/api/civitai/model/12345")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == 12345
        assert data["name"] == "Test Model"

    def test_get_model_not_found(self, civitai_api: TestClient, respx_mock) -> None:
        """Test getting non-existent model."""
        import respx

        respx_mock.get("https://civitai.com/api/v1/models/99999").mock(
            return_value=respx.MockResponse(404, json={"error": "Not found"})
        )

        response = civitai_api.get("/api/civitai/model/99999")
        assert response.status_code == 404

    def test_get_model_network_error(self, civitai_api: TestClient, respx_mock) -> None:
        """Test get model handles network errors."""
        import httpx

        respx_mock.get("https://civitai.com/api/v1/models/12345").mock(side_effect=httpx.RequestError("Connection failed"))

        response = civitai_api.get("/api/civitai/model/12345")
        assert response.status_code == 500


# =============================================================================
# Download Routes Tests
# =============================================================================


@pytest.fixture
def download_api(monkeypatch) -> TestClient:
    """Test client for Download API."""
    from fastapi import FastAPI

    from tensors.server.download_routes import create_download_router

    # Disable auth for testing
    monkeypatch.setattr("tensors.config.get_server_api_key", lambda: None)

    app = FastAPI()
    app.include_router(create_download_router())
    return TestClient(app)


class TestDownloadRoutes:
    """Tests for download endpoints."""

    def test_list_active_downloads_empty(self, download_api: TestClient) -> None:
        """Test listing active downloads when none exist."""
        response = download_api.get("/api/download/active")
        assert response.status_code == 200
        data = response.json()
        assert data["downloads"] == []
        assert data["total"] == 0

    def test_get_download_status_not_found(self, download_api: TestClient) -> None:
        """Test getting status of non-existent download."""
        response = download_api.get("/api/download/status/nonexistent-id")
        assert response.status_code == 404

    def test_start_download_no_identifier(self, download_api: TestClient) -> None:
        """Test starting download without any identifier returns 404."""
        response = download_api.post("/api/download", json={})
        # No version_id, model_id, or hash provided - can't find model
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


# =============================================================================
# DB Routes Additional Tests
# =============================================================================


class TestDbRoutesExtended:
    """Extended tests for database routes."""

    def test_get_file_not_found(self, db_api: TestClient) -> None:
        """Test getting non-existent file."""
        response = db_api.get("/api/db/files/99999")
        assert response.status_code == 404

    def test_get_triggers_by_path(self, db_api: TestClient) -> None:
        """Test getting triggers by path (returns empty for non-existent)."""
        response = db_api.get("/api/db/triggers", params={"file_path": "/nonexistent/path.safetensors"})
        assert response.status_code == 200
        assert response.json() == []

    def test_get_triggers_by_version(self, db_api: TestClient) -> None:
        """Test getting triggers by version (returns empty for non-existent)."""
        response = db_api.get("/api/db/triggers/99999")
        assert response.status_code == 200
        assert response.json() == []

    def test_cache_model(self, db_api: TestClient, temp_db, monkeypatch, respx_mock) -> None:
        """Test caching a model."""
        import respx

        from tensors import db as db_module

        monkeypatch.setattr(db_module, "DB_PATH", temp_db.db_path)

        respx_mock.get("https://civitai.com/api/v1/models/12345").mock(
            return_value=respx.MockResponse(
                200,
                json={
                    "id": 12345,
                    "name": "Cached Model",
                    "type": "LORA",
                    "tags": [],
                    "modelVersions": [],
                },
            )
        )

        response = db_api.post("/api/db/cache", json={"model_id": 12345})
        assert response.status_code == 200
        data = response.json()
        assert "model_id" in data

    def test_scan_directory_not_found(self, db_api: TestClient) -> None:
        """Test scanning non-existent directory returns 400."""
        response = db_api.post("/api/db/scan", json={"directory": "/nonexistent/directory"})
        assert response.status_code == 400

    def test_link_files(self, db_api: TestClient, temp_db, monkeypatch) -> None:
        """Test linking files endpoint."""
        from tensors import db as db_module

        monkeypatch.setattr(db_module, "DB_PATH", temp_db.db_path)

        response = db_api.post("/api/db/link")
        assert response.status_code == 200
        data = response.json()
        assert "linked" in data

    def test_search_models_with_type_filter(self, db_api: TestClient, temp_db, monkeypatch) -> None:
        """Test searching models with type filter."""
        from tensors import db as db_module

        monkeypatch.setattr(db_module, "DB_PATH", temp_db.db_path)

        # Cache some models
        temp_db.cache_model({"id": 1, "name": "LORA Model", "type": "LORA", "tags": [], "modelVersions": []})
        temp_db.cache_model({"id": 2, "name": "Checkpoint", "type": "Checkpoint", "tags": [], "modelVersions": []})

        response = db_api.get("/api/db/models?type=LORA")
        assert response.status_code == 200
        results = response.json()
        assert all(r.get("type") == "LORA" for r in results if r.get("type"))

    def test_search_models_with_base_filter(self, db_api: TestClient, temp_db, monkeypatch) -> None:
        """Test searching models with base model filter."""
        from tensors import db as db_module

        monkeypatch.setattr(db_module, "DB_PATH", temp_db.db_path)

        response = db_api.get("/api/db/models?base=SD 1.5")
        assert response.status_code == 200

    def test_search_models_with_limit(self, db_api: TestClient, temp_db, monkeypatch) -> None:
        """Test searching models with limit."""
        from tensors import db as db_module

        monkeypatch.setattr(db_module, "DB_PATH", temp_db.db_path)

        for i in range(10):
            temp_db.cache_model({"id": 100 + i, "name": f"Model {i}", "type": "LORA", "tags": [], "modelVersions": []})

        response = db_api.get("/api/db/models?limit=5")
        assert response.status_code == 200
        results = response.json()
        assert len(results) <= 5

    def test_cache_model_not_found(self, db_api: TestClient, respx_mock) -> None:
        """Test caching a model that doesn't exist on CivitAI."""
        import respx

        respx_mock.get("https://civitai.com/api/v1/models/99999").mock(
            return_value=respx.MockResponse(404, json={"error": "Not found"})
        )

        response = db_api.post("/api/db/cache", json={"model_id": 99999})
        assert response.status_code == 404

    def test_scan_directory_success(self, db_api: TestClient, temp_db, monkeypatch, tmp_path) -> None:
        """Test scanning a valid directory."""
        from tensors import db as db_module

        monkeypatch.setattr(db_module, "DB_PATH", temp_db.db_path)

        # Create a temporary directory (empty, no safetensors)
        scan_dir = tmp_path / "models"
        scan_dir.mkdir()

        response = db_api.post("/api/db/scan", json={"directory": str(scan_dir)})
        assert response.status_code == 200
        data = response.json()
        assert "scanned" in data
        assert data["scanned"] == 0  # Empty directory


# =============================================================================
# Download Routes Helper Function Tests
# =============================================================================


class TestDownloadHelpers:
    """Tests for download route helper functions."""

    def test_format_size_bytes(self) -> None:
        """Test formatting bytes."""
        from tensors.server.download_routes import _format_size

        assert _format_size(500) == "500 B"
        assert _format_size(0) == "0 B"

    def test_format_size_kb(self) -> None:
        """Test formatting kilobytes."""
        from tensors.server.download_routes import _format_size

        assert _format_size(1024) == "1.0 KB"
        assert _format_size(2048) == "2.0 KB"
        assert _format_size(1536) == "1.5 KB"

    def test_format_size_mb(self) -> None:
        """Test formatting megabytes."""
        from tensors.server.download_routes import _format_size

        assert _format_size(1024 * 1024) == "1.0 MB"
        assert _format_size(50 * 1024 * 1024) == "50.0 MB"

    def test_format_size_gb(self) -> None:
        """Test formatting gigabytes."""
        from tensors.server.download_routes import _format_size

        assert _format_size(1024 * 1024 * 1024) == "1.0 GB"
        assert _format_size(2 * 1024 * 1024 * 1024) == "2.0 GB"

    def test_get_output_dir_with_override(self) -> None:
        """Test output dir with override."""
        from tensors.server.download_routes import _get_output_dir

        result = _get_output_dir({}, "/custom/path")
        assert str(result) == "/custom/path"

    def test_get_output_dir_checkpoint(self) -> None:
        """Test output dir for checkpoint."""
        from tensors.server.download_routes import _get_output_dir

        version_info = {"model": {"type": "Checkpoint"}}
        result = _get_output_dir(version_info, None)
        assert "checkpoints" in str(result)

    def test_get_output_dir_lora(self) -> None:
        """Test output dir for LORA."""
        from tensors.server.download_routes import _get_output_dir

        version_info = {"model": {"type": "LORA"}}
        result = _get_output_dir(version_info, None)
        assert "loras" in str(result)

    def test_get_output_dir_locon(self) -> None:
        """Test output dir for LoCon."""
        from tensors.server.download_routes import _get_output_dir

        version_info = {"model": {"type": "LoCon"}}
        result = _get_output_dir(version_info, None)
        assert "loras" in str(result)

    def test_get_output_dir_textual_inversion(self) -> None:
        """Test output dir for TextualInversion."""
        from tensors.server.download_routes import _get_output_dir

        version_info = {"model": {"type": "TextualInversion"}}
        result = _get_output_dir(version_info, None)
        assert "embeddings" in str(result)

    def test_get_output_dir_vae(self) -> None:
        """Test output dir for VAE."""
        from tensors.server.download_routes import _get_output_dir

        version_info = {"model": {"type": "VAE"}}
        result = _get_output_dir(version_info, None)
        assert "vae" in str(result)

    def test_get_output_dir_controlnet(self) -> None:
        """Test output dir for Controlnet."""
        from tensors.server.download_routes import _get_output_dir

        version_info = {"model": {"type": "Controlnet"}}
        result = _get_output_dir(version_info, None)
        assert "controlnet" in str(result)

    def test_get_output_dir_unknown(self) -> None:
        """Test output dir for unknown type."""
        from tensors.server.download_routes import _get_output_dir

        version_info = {"model": {"type": "UnknownType"}}
        result = _get_output_dir(version_info, None)
        assert "other" in str(result)

    def test_get_output_dir_no_type(self) -> None:
        """Test output dir with missing type (defaults to Checkpoint)."""
        from tensors.server.download_routes import _get_output_dir

        version_info = {"model": {}}
        result = _get_output_dir(version_info, None)
        assert "checkpoints" in str(result)


class TestResolveVersionId:
    """Tests for _resolve_version_id helper."""

    def test_resolve_with_version_id(self, respx_mock) -> None:
        """Test resolving by version ID."""
        import respx

        from tensors.server.download_routes import _resolve_version_id

        respx_mock.get("https://civitai.com/api/v1/model-versions/12345").mock(
            return_value=respx.MockResponse(200, json={"id": 12345, "name": "v1.0"})
        )

        version_id, info = _resolve_version_id(12345, None, None, None)
        assert version_id == 12345
        assert info is not None
        assert info["id"] == 12345

    def test_resolve_with_hash(self, respx_mock) -> None:
        """Test resolving by hash."""
        import respx

        from tensors.server.download_routes import _resolve_version_id

        respx_mock.get("https://civitai.com/api/v1/model-versions/by-hash/ABC123").mock(
            return_value=respx.MockResponse(200, json={"id": 555, "modelId": 100})
        )

        version_id, info = _resolve_version_id(None, None, "abc123", None)
        assert version_id == 555
        assert info is not None

    def test_resolve_with_hash_not_found(self, respx_mock) -> None:
        """Test resolving by hash when not found."""
        import respx

        from tensors.server.download_routes import _resolve_version_id

        respx_mock.get("https://civitai.com/api/v1/model-versions/by-hash/NOTFOUND").mock(
            return_value=respx.MockResponse(404, json={"error": "Not found"})
        )

        version_id, info = _resolve_version_id(None, None, "notfound", None)
        assert version_id is None
        assert info is None

    def test_resolve_with_model_id(self, respx_mock) -> None:
        """Test resolving by model ID (uses latest version)."""
        import respx

        from tensors.server.download_routes import _resolve_version_id

        respx_mock.get("https://civitai.com/api/v1/models/999").mock(
            return_value=respx.MockResponse(
                200,
                json={
                    "id": 999,
                    "modelVersions": [{"id": 1001, "name": "Latest"}, {"id": 1000, "name": "Old"}],
                },
            )
        )

        version_id, info = _resolve_version_id(None, 999, None, None)
        assert version_id == 1001
        assert info is not None
        assert info["name"] == "Latest"

    def test_resolve_with_model_id_no_versions(self, respx_mock) -> None:
        """Test resolving by model ID with no versions."""
        import respx

        from tensors.server.download_routes import _resolve_version_id

        respx_mock.get("https://civitai.com/api/v1/models/888").mock(
            return_value=respx.MockResponse(200, json={"id": 888, "modelVersions": []})
        )

        version_id, info = _resolve_version_id(None, 888, None, None)
        assert version_id is None
        assert info is None

    def test_resolve_with_model_id_not_found(self, respx_mock) -> None:
        """Test resolving by model ID when model not found."""
        import respx

        from tensors.server.download_routes import _resolve_version_id

        respx_mock.get("https://civitai.com/api/v1/models/777").mock(
            return_value=respx.MockResponse(404, json={"error": "Not found"})
        )

        version_id, info = _resolve_version_id(None, 777, None, None)
        assert version_id is None
        assert info is None

    def test_resolve_with_nothing(self) -> None:
        """Test resolving with no identifiers."""
        from tensors.server.download_routes import _resolve_version_id

        version_id, info = _resolve_version_id(None, None, None, None)
        assert version_id is None
        assert info is None


class TestDownloadEndpoints:
    """Tests for download API endpoints."""

    def test_start_download_success(self, download_api: TestClient, respx_mock, tmp_path, monkeypatch) -> None:
        """Test starting a download successfully."""
        import respx

        from tensors import config as config_module

        monkeypatch.setattr(config_module, "MODELS_DIR", tmp_path)

        respx_mock.get("https://civitai.com/api/v1/model-versions/12345").mock(
            return_value=respx.MockResponse(
                200,
                json={
                    "id": 12345,
                    "name": "v1.0",
                    "model": {"name": "Test Model", "type": "LORA"},
                    "files": [{"name": "test-model.safetensors", "primary": True}],
                },
            )
        )

        response = download_api.post("/api/download", json={"version_id": 12345})
        assert response.status_code == 200
        data = response.json()
        assert "download_id" in data
        assert data["status"] == "queued"
        assert data["version_id"] == 12345

    def test_start_download_no_files(self, download_api: TestClient, respx_mock, tmp_path, monkeypatch) -> None:
        """Test starting download with no files returns 400."""
        import respx

        from tensors import config as config_module

        monkeypatch.setattr(config_module, "MODELS_DIR", tmp_path)

        respx_mock.get("https://civitai.com/api/v1/model-versions/99999").mock(
            return_value=respx.MockResponse(
                200,
                json={
                    "id": 99999,
                    "name": "v1.0",
                    "model": {"name": "Empty Model", "type": "LORA"},
                    "files": [],
                },
            )
        )

        response = download_api.post("/api/download", json={"version_id": 99999})
        assert response.status_code == 400
        assert "No files found" in response.json()["detail"]

    def test_start_download_with_hash(self, download_api: TestClient, respx_mock, tmp_path, monkeypatch) -> None:
        """Test starting download using hash."""
        import respx

        from tensors import config as config_module

        monkeypatch.setattr(config_module, "MODELS_DIR", tmp_path)

        respx_mock.get("https://civitai.com/api/v1/model-versions/by-hash/ABCD1234").mock(
            return_value=respx.MockResponse(
                200,
                json={
                    "id": 555,
                    "modelId": 100,
                    "name": "v1.0",
                    "model": {"name": "Hash Model", "type": "Checkpoint"},
                    "files": [{"name": "model.safetensors", "primary": True}],
                },
            )
        )

        response = download_api.post("/api/download", json={"hash": "abcd1234"})
        assert response.status_code == 200
        data = response.json()
        assert data["version_id"] == 555

    def test_start_download_with_model_id(self, download_api: TestClient, respx_mock, tmp_path, monkeypatch) -> None:
        """Test starting download using model ID (picks latest version)."""
        import respx

        from tensors import config as config_module

        monkeypatch.setattr(config_module, "MODELS_DIR", tmp_path)

        respx_mock.get("https://civitai.com/api/v1/models/200").mock(
            return_value=respx.MockResponse(
                200,
                json={
                    "id": 200,
                    "name": "Model With Versions",
                    "modelVersions": [
                        {
                            "id": 2001,
                            "name": "Latest",
                            "model": {"name": "Model", "type": "LORA"},
                            "files": [{"name": "latest.safetensors", "primary": True}],
                        }
                    ],
                },
            )
        )

        response = download_api.post("/api/download", json={"model_id": 200})
        assert response.status_code == 200
        data = response.json()
        assert data["version_id"] == 2001

    def test_start_download_with_output_dir(self, download_api: TestClient, respx_mock, tmp_path, monkeypatch) -> None:
        """Test starting download with custom output directory."""
        import respx

        from tensors import config as config_module

        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        monkeypatch.setattr(config_module, "MODELS_DIR", tmp_path)

        respx_mock.get("https://civitai.com/api/v1/model-versions/333").mock(
            return_value=respx.MockResponse(
                200,
                json={
                    "id": 333,
                    "name": "v1.0",
                    "model": {"name": "Custom Dir Model", "type": "LORA"},
                    "files": [{"name": "custom.safetensors", "primary": True}],
                },
            )
        )

        response = download_api.post("/api/download", json={"version_id": 333, "output_dir": str(custom_dir)})
        assert response.status_code == 200
        assert str(custom_dir) in response.json()["destination"]

    def test_get_download_status_success(self, download_api: TestClient, respx_mock, tmp_path, monkeypatch) -> None:
        """Test getting status of an existing download."""
        import respx

        from tensors import config as config_module

        monkeypatch.setattr(config_module, "MODELS_DIR", tmp_path)

        # First create a download
        respx_mock.get("https://civitai.com/api/v1/model-versions/444").mock(
            return_value=respx.MockResponse(
                200,
                json={
                    "id": 444,
                    "name": "v1.0",
                    "model": {"name": "Status Test", "type": "LORA"},
                    "files": [{"name": "status.safetensors", "primary": True}],
                },
            )
        )

        create_response = download_api.post("/api/download", json={"version_id": 444})
        download_id = create_response.json()["download_id"]

        # Now get its status
        status_response = download_api.get(f"/api/download/status/{download_id}")
        assert status_response.status_code == 200
        data = status_response.json()
        assert data["id"] == download_id

    def test_list_active_downloads_with_data(self, download_api: TestClient, respx_mock, tmp_path, monkeypatch) -> None:
        """Test listing active downloads after creating some."""
        import respx

        from tensors import config as config_module
        from tensors.server import download_routes

        # Clear any existing downloads
        download_routes._active_downloads.clear()
        monkeypatch.setattr(config_module, "MODELS_DIR", tmp_path)

        respx_mock.get("https://civitai.com/api/v1/model-versions/555").mock(
            return_value=respx.MockResponse(
                200,
                json={
                    "id": 555,
                    "name": "v1.0",
                    "model": {"name": "Active Test", "type": "LORA"},
                    "files": [{"name": "active.safetensors", "primary": True}],
                },
            )
        )

        download_api.post("/api/download", json={"version_id": 555})

        response = download_api.get("/api/download/active")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        assert len(data["downloads"]) >= 1


# =============================================================================
# CivitAI Routes Extended Tests
# =============================================================================


class TestCivitAIRoutesExtended:
    """Extended tests for CivitAI routes."""

    def test_get_model_caches_result(self, civitai_api: TestClient, respx_mock, temp_db, monkeypatch) -> None:
        """Test that getting a model caches it in the database."""
        import respx

        from tensors import db as db_module

        monkeypatch.setattr(db_module, "DB_PATH", temp_db.db_path)

        respx_mock.get("https://civitai.com/api/v1/models/77777").mock(
            return_value=respx.MockResponse(
                200,
                json={
                    "id": 77777,
                    "name": "Cacheable Model",
                    "type": "Checkpoint",
                    "tags": ["anime"],
                    "modelVersions": [{"id": 77778, "name": "v1"}],
                },
            )
        )

        response = civitai_api.get("/api/civitai/model/77777")
        assert response.status_code == 200
        assert response.json()["name"] == "Cacheable Model"

        # Verify it was cached
        cached = temp_db.get_model(77777)
        assert cached is not None
        assert cached["name"] == "Cacheable Model"


# =============================================================================
# Gallery Routes Extended Tests
# =============================================================================


class TestGalleryRoutesExtended:
    """Extended tests for gallery routes."""

    def test_list_images_oldest_first(self, gallery_api: TestClient, gallery_with_images) -> None:
        """Test listing images sorted oldest first."""
        from tensors.server import gallery_routes

        gallery_routes._gallery = gallery_with_images

        response = gallery_api.get("/api/images?newest_first=false")
        assert response.status_code == 200
        data = response.json()
        assert len(data["images"]) == 3

    def test_edit_metadata_partial_update(self, gallery_api: TestClient, gallery_with_images) -> None:
        """Test partial metadata update (only some fields)."""
        from tensors.server import gallery_routes

        gallery_routes._gallery = gallery_with_images

        list_response = gallery_api.get("/api/images")
        image_id = list_response.json()["images"][0]["id"]

        # Only update notes, not tags
        response = gallery_api.post(f"/api/images/{image_id}/edit", json={"notes": "Test note"})
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["notes"] == "Test note"

    def test_edit_metadata_favorite(self, gallery_api: TestClient, gallery_with_images) -> None:
        """Test setting favorite flag."""
        from tensors.server import gallery_routes

        gallery_routes._gallery = gallery_with_images

        list_response = gallery_api.get("/api/images")
        image_id = list_response.json()["images"][0]["id"]

        response = gallery_api.post(f"/api/images/{image_id}/edit", json={"favorite": True})
        assert response.status_code == 200
        assert response.json()["metadata"]["favorite"] is True


# =============================================================================
# Download Background Task Tests
# =============================================================================


class TestDownloadBackgroundTasks:
    """Tests for download background task functions."""

    @staticmethod
    def _patch_db_noop(monkeypatch, download_routes_module) -> dict:
        """Replace Database with a no-op stub; return a dict capturing register calls."""
        captured: dict = {}

        class StubDB:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def init_schema(self):
                pass

            def register_downloaded_file(self, dest_path, version_info, api_key=None, console=None):
                captured["dest_path"] = dest_path
                captured["version_info"] = version_info
                captured["api_key"] = api_key
                return {"file_id": 42, "sha256": "deadbeef", "linked": True, "cached": True, "error": None}

        monkeypatch.setattr(download_routes_module, "Database", StubDB)
        return captured

    def test_do_download_success(self, monkeypatch, tmp_path) -> None:
        """Test successful download task execution."""
        from tensors.server import download_routes
        from tensors.server.download_routes import _do_download

        # Set up tracking entry
        download_id = "test_123"
        download_routes._active_downloads[download_id] = {"id": download_id, "status": "queued"}

        # Mock the download function
        def mock_download(version_id, dest_path, api_key, on_progress, resume):
            # Simulate progress callback
            on_progress(1024, 2048, 100.0)
            on_progress(2048, 2048, 200.0)
            return True

        monkeypatch.setattr(download_routes, "download_model_with_progress", mock_download)
        captured = self._patch_db_noop(monkeypatch, download_routes)

        dest_path = tmp_path / "model.safetensors"
        version_info = {"id": 999, "modelId": 888, "name": "v1"}
        _do_download(12345, dest_path, None, download_id, version_info)

        assert download_routes._active_downloads[download_id]["status"] == "completed"
        assert download_routes._active_downloads[download_id]["progress"] == 100
        assert download_routes._active_downloads[download_id]["db_file_id"] == 42
        assert download_routes._active_downloads[download_id]["db_linked"] is True
        assert download_routes._active_downloads[download_id]["db_cached"] is True
        assert captured["version_info"] == version_info

        # Cleanup
        del download_routes._active_downloads[download_id]

    def test_do_download_failure(self, monkeypatch, tmp_path) -> None:
        """Test failed download task execution."""
        from tensors.server import download_routes
        from tensors.server.download_routes import _do_download

        download_id = "test_fail_123"
        download_routes._active_downloads[download_id] = {"id": download_id, "status": "queued"}

        # Mock download to return failure
        monkeypatch.setattr(download_routes, "download_model_with_progress", lambda *args, **kwargs: False)

        dest_path = tmp_path / "model.safetensors"
        _do_download(12345, dest_path, None, download_id, {"id": 1, "modelId": 1})

        assert download_routes._active_downloads[download_id]["status"] == "failed"
        assert "error" in download_routes._active_downloads[download_id]

        del download_routes._active_downloads[download_id]

    def test_do_download_exception(self, monkeypatch, tmp_path) -> None:
        """Test download task with exception."""
        from tensors.server import download_routes
        from tensors.server.download_routes import _do_download

        download_id = "test_exc_123"
        download_routes._active_downloads[download_id] = {"id": download_id, "status": "queued"}

        # Mock download to raise exception
        def mock_download(*args, **kwargs):
            raise RuntimeError("Network error")

        monkeypatch.setattr(download_routes, "download_model_with_progress", mock_download)

        dest_path = tmp_path / "model.safetensors"
        _do_download(12345, dest_path, None, download_id, {"id": 1, "modelId": 1})

        assert download_routes._active_downloads[download_id]["status"] == "failed"
        assert "Network error" in download_routes._active_downloads[download_id]["error"]

        del download_routes._active_downloads[download_id]

    def test_on_progress_callback(self, monkeypatch, tmp_path) -> None:
        """Test progress callback updates correctly."""
        from tensors.server import download_routes
        from tensors.server.download_routes import _do_download

        download_id = "test_progress_123"
        download_routes._active_downloads[download_id] = {"id": download_id, "status": "queued"}

        progress_calls = []

        def mock_download(version_id, dest_path, api_key, on_progress, resume):
            # Test with different sizes
            on_progress(512, 1024, 50.0)  # 512 B of 1 KB
            progress_calls.append(dict(download_routes._active_downloads[download_id]))

            on_progress(1024 * 500, 1024 * 1024, 1000.0)  # 500 KB of 1 MB
            progress_calls.append(dict(download_routes._active_downloads[download_id]))

            on_progress(1024 * 1024 * 500, 1024 * 1024 * 1024, 10000.0)  # 500 MB of 1 GB
            progress_calls.append(dict(download_routes._active_downloads[download_id]))

            return True

        monkeypatch.setattr(download_routes, "download_model_with_progress", mock_download)
        self._patch_db_noop(monkeypatch, download_routes)

        dest_path = tmp_path / "model.safetensors"
        _do_download(12345, dest_path, None, download_id, {"id": 1, "modelId": 1})

        # Check progress formatting was called
        assert len(progress_calls) == 3
        assert progress_calls[0]["downloaded_str"] == "512 B"
        assert progress_calls[0]["total_str"] == "1.0 KB"
        assert progress_calls[1]["downloaded_str"] == "500.0 KB"
        assert progress_calls[2]["downloaded_str"] == "500.0 MB"

        del download_routes._active_downloads[download_id]

    def test_on_progress_zero_total(self, monkeypatch, tmp_path) -> None:
        """Test progress callback with zero total (unknown size)."""
        from tensors.server import download_routes
        from tensors.server.download_routes import _do_download

        download_id = "test_zero_total"
        download_routes._active_downloads[download_id] = {"id": download_id, "status": "queued"}

        def mock_download(version_id, dest_path, api_key, on_progress, resume):
            on_progress(1024, 0, 100.0)  # Unknown total
            return True

        monkeypatch.setattr(download_routes, "download_model_with_progress", mock_download)
        self._patch_db_noop(monkeypatch, download_routes)

        dest_path = tmp_path / "model.safetensors"
        _do_download(12345, dest_path, None, download_id, {"id": 1, "modelId": 1})

        assert download_routes._active_downloads[download_id]["total_str"] == "Unknown"

        del download_routes._active_downloads[download_id]

    def test_do_download_db_error_surfaced(self, monkeypatch, tmp_path) -> None:
        """DB errors during register must be surfaced into _active_downloads."""
        from tensors.server import download_routes
        from tensors.server.download_routes import _do_download

        download_id = "test_db_err"
        download_routes._active_downloads[download_id] = {"id": download_id, "status": "queued"}

        monkeypatch.setattr(download_routes, "download_model_with_progress", lambda *a, **kw: True)

        class FailingDB:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def init_schema(self):
                pass

            def register_downloaded_file(self, *args, **kwargs):
                return {"file_id": None, "sha256": None, "linked": False, "cached": False, "error": "boom"}

        monkeypatch.setattr(download_routes, "Database", FailingDB)

        dest_path = tmp_path / "model.safetensors"
        _do_download(12345, dest_path, None, download_id, {"id": 1, "modelId": 1})

        # Download itself succeeded; DB layer is reported separately
        assert download_routes._active_downloads[download_id]["status"] == "completed"
        assert download_routes._active_downloads[download_id]["db_error"] == "boom"
        assert download_routes._active_downloads[download_id]["db_linked"] is False

        del download_routes._active_downloads[download_id]


# =============================================================================
# DB Routes - File Lookup Tests
# =============================================================================


class TestDbFileLookup:
    """Tests for database file lookup."""

    def test_get_file_success(self, db_api: TestClient, temp_db, monkeypatch, tmp_path) -> None:
        """Test getting an existing file."""
        from tensors import db as db_module

        monkeypatch.setattr(db_module, "DB_PATH", temp_db.db_path)

        # Create and scan a file to add it to the database
        test_file = tmp_path / "test.safetensors"

        # Create minimal valid safetensor header
        header_data = b'{"__metadata__": {}}'
        header_size = len(header_data)
        test_file.write_bytes(header_size.to_bytes(8, "little") + header_data)

        # Scan to add to database
        results = temp_db.scan_directory(tmp_path)
        assert len(results) > 0

        file_id = results[0]["id"]

        response = db_api.get(f"/api/db/files/{file_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == file_id

    def test_link_files_with_matches(self, db_api: TestClient, temp_db, monkeypatch, tmp_path, respx_mock) -> None:
        """Test linking files when CivitAI matches exist."""
        import respx

        from tensors import db as db_module

        monkeypatch.setattr(db_module, "DB_PATH", temp_db.db_path)

        # Create and scan a file
        test_file = tmp_path / "linkable.safetensors"
        header_data = b'{"__metadata__": {}}'
        header_size = len(header_data)
        test_file.write_bytes(header_size.to_bytes(8, "little") + header_data)

        temp_db.scan_directory(tmp_path)
        files = temp_db.list_local_files()
        assert len(files) > 0

        sha256 = files[0]["sha256"]

        # Mock CivitAI hash lookup to return a match
        respx_mock.get(f"https://civitai.com/api/v1/model-versions/by-hash/{sha256.upper()}").mock(
            return_value=respx.MockResponse(200, json={"id": 12345, "modelId": 67890, "name": "Found Model"})
        )

        response = db_api.post("/api/db/link")
        assert response.status_code == 200
        data = response.json()
        assert data["linked"] >= 1
        assert len(data["results"]) >= 1


# =============================================================================
# Server Init Tests
# =============================================================================


class TestServerInit:
    """Tests for server initialization."""

    def test_docs_endpoint(self, api: TestClient) -> None:
        """Test /docs endpoint returns HTML."""
        response = api.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_openapi_schema(self, api: TestClient) -> None:
        """Test OpenAPI schema is available."""
        response = api.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "tensors"
        assert "paths" in data

    def test_app_startup_with_auth(self, monkeypatch) -> None:
        """Test app startup logging with auth enabled."""
        monkeypatch.setattr("tensors.config.get_server_api_key", lambda: "test-key")

        from tensors.server import create_app

        app = create_app()
        # App should be created successfully
        assert app.title == "tensors"

    def test_app_startup_without_auth(self, monkeypatch) -> None:
        """Test app startup logging without auth."""
        monkeypatch.setattr("tensors.config.get_server_api_key", lambda: None)

        from tensors.server import create_app

        app = create_app()
        assert app.title == "tensors"


class TestCivitAICacheFailure:
    """Test CivitAI model caching failure handling."""

    def test_get_model_cache_failure_continues(self, civitai_api: TestClient, respx_mock, monkeypatch) -> None:
        """Test that cache failure doesn't prevent model retrieval."""
        import respx

        respx_mock.get("https://civitai.com/api/v1/models/88888").mock(
            return_value=respx.MockResponse(
                200,
                json={
                    "id": 88888,
                    "name": "Cache Fail Model",
                    "type": "LORA",
                    "tags": [],
                    "modelVersions": [],
                },
            )
        )

        # Make Database raise an exception
        class FailingDB:
            def __enter__(self):
                raise RuntimeError("Database error")

            def __exit__(self, *args):
                pass

        from tensors.server import civitai_routes

        monkeypatch.setattr(civitai_routes, "Database", FailingDB)

        # Should still return the model even though caching failed
        response = civitai_api.get("/api/civitai/model/88888")
        assert response.status_code == 200
        assert response.json()["name"] == "Cache Fail Model"


# =============================================================================
# Gallery Edge Cases
# =============================================================================


class TestGalleryEdgeCases:
    """Edge case tests for gallery functionality."""

    def test_gallery_get_metadata_for_image(self, gallery_api: TestClient, gallery_with_images) -> None:
        """Test getting metadata returns full image info."""
        from tensors.server import gallery_routes

        gallery_routes._gallery = gallery_with_images

        list_response = gallery_api.get("/api/images")
        image_id = list_response.json()["images"][0]["id"]

        response = gallery_api.get(f"/api/images/{image_id}/meta")
        assert response.status_code == 200
        data = response.json()
        assert "path" in data
        assert "created_at" in data
        assert "metadata" in data

    def test_gallery_stats_with_images(self, gallery_api: TestClient, gallery_with_images) -> None:
        """Test stats with actual images."""
        from tensors.server import gallery_routes

        gallery_routes._gallery = gallery_with_images

        response = gallery_api.get("/api/images/stats/summary")
        assert response.status_code == 200
        data = response.json()
        assert data["total_images"] == 3
        assert "gallery_dir" in data


class TestGalleryClassExtended:
    """Extended unit tests for Gallery class."""

    def test_save_image_with_seed(self, temp_gallery) -> None:
        """Test saving image with seed creates proper filename."""
        image_data = b"\x89PNG test data"
        result = temp_gallery.save_image(image_data, seed=12345)

        assert "12345" in result.path.name
        assert result.path.exists()

    def test_save_image_without_metadata(self, temp_gallery) -> None:
        """Test saving image without metadata."""
        image_data = b"\x89PNG test data"
        result = temp_gallery.save_image(image_data)

        assert result.path.exists()
        # No metadata file should exist
        assert not result.meta_path.exists()

    def test_list_images_with_offset(self, gallery_with_images) -> None:
        """Test list images with offset."""
        images = gallery_with_images.list_images(offset=1, limit=10)
        assert len(images) == 2  # 3 total - 1 offset = 2

    def test_get_metadata_returns_dict(self, gallery_with_images) -> None:
        """Test get_metadata returns metadata dict."""
        images = gallery_with_images.list_images()
        metadata = gallery_with_images.get_metadata(images[0].id)
        assert isinstance(metadata, dict)
        assert "prompt" in metadata

    def test_get_metadata_nonexistent(self, temp_gallery) -> None:
        """Test get_metadata for non-existent image returns None."""
        result = temp_gallery.get_metadata("nonexistent")
        assert result is None
