"""Tests for the TsrClient HTTP client module."""

from __future__ import annotations

import pytest
import respx
from httpx import Response

from tensors.client import TsrClient, TsrClientError

BASE_URL = "http://test-server:8080"


@pytest.fixture
def mock_server():
    """Activate respx mock for the test server."""
    with respx.mock(base_url=BASE_URL, assert_all_called=False) as rsps:
        yield rsps


@pytest.fixture
def client(mock_server) -> TsrClient:  # noqa: ARG001 - mock_server activates respx
    """TsrClient connected to mock server."""
    return TsrClient(BASE_URL)


# =============================================================================
# Status Tests
# =============================================================================


class TestStatus:
    """Tests for server status endpoint."""

    def test_status_success(self, client: TsrClient, mock_server) -> None:
        """Test getting server status."""
        mock_server.get("/status").mock(return_value=Response(200, json={"running": True, "pid": 12345, "model": "/test.gguf"}))

        with client:
            result = client.status()

        assert result["running"] is True
        assert result["pid"] == 12345

    def test_status_error(self, client: TsrClient, mock_server) -> None:
        """Test handling status error."""
        mock_server.get("/status").mock(return_value=Response(503, text="Service unavailable"))

        with client, pytest.raises(TsrClientError, match="HTTP 503"):
            client.status()


# =============================================================================
# Gallery Tests
# =============================================================================


class TestGalleryImages:
    """Tests for gallery image operations."""

    def test_list_images(self, client: TsrClient, mock_server) -> None:
        """Test listing gallery images."""
        mock_server.get("/api/images").mock(
            return_value=Response(
                200,
                json={
                    "images": [
                        {"id": "123_0", "filename": "123_0.png", "width": 512, "height": 512},
                        {"id": "124_1", "filename": "124_1.png", "width": 1024, "height": 1024},
                    ],
                    "total": 2,
                },
            )
        )

        with client:
            result = client.list_images()

        assert len(result["images"]) == 2
        assert result["total"] == 2

    def test_list_images_with_pagination(self, client: TsrClient, mock_server) -> None:
        """Test listing images with pagination."""
        mock_server.get("/api/images", params={"limit": 10, "offset": 5}).mock(
            return_value=Response(200, json={"images": [], "total": 100})
        )

        with client:
            result = client.list_images(limit=10, offset=5)

        assert result["total"] == 100

    def test_get_image_meta(self, client: TsrClient, mock_server) -> None:
        """Test getting image metadata."""
        mock_server.get("/api/images/123_0/meta").mock(
            return_value=Response(
                200,
                json={
                    "id": "123_0",
                    "path": "/gallery/123_0.png",
                    "metadata": {"prompt": "test prompt", "seed": 42},
                },
            )
        )

        with client:
            result = client.get_image_meta("123_0")

        assert result["id"] == "123_0"
        assert result["metadata"]["prompt"] == "test prompt"

    def test_delete_image(self, client: TsrClient, mock_server) -> None:
        """Test deleting an image."""
        mock_server.delete("/api/images/123_0").mock(return_value=Response(200, json={"deleted": True, "id": "123_0"}))

        with client:
            result = client.delete_image("123_0")

        assert result["deleted"] is True

    def test_edit_image(self, client: TsrClient, mock_server) -> None:
        """Test editing image metadata."""
        mock_server.post("/api/images/123_0/edit").mock(
            return_value=Response(200, json={"id": "123_0", "metadata": {"tags": ["favorite"], "rating": 5}})
        )

        with client:
            result = client.edit_image("123_0", {"tags": ["favorite"], "rating": 5})

        assert result["metadata"]["tags"] == ["favorite"]

    def test_download_image(self, client: TsrClient, mock_server) -> None:
        """Test downloading image bytes."""
        image_bytes = b"\x89PNG test image data"
        mock_server.get("/api/images/123_0").mock(return_value=Response(200, content=image_bytes))

        with client:
            result = client.download_image("123_0")

        assert result == image_bytes


# =============================================================================
# Models Tests
# =============================================================================


class TestModels:
    """Tests for model management operations."""

    def test_list_models(self, client: TsrClient, mock_server) -> None:
        """Test listing available models."""
        mock_server.get("/api/models").mock(
            return_value=Response(
                200,
                json={
                    "models": [
                        {"name": "sdxl_base", "path": "/models/sdxl_base.safetensors"},
                        {"name": "pony_v6", "path": "/models/pony_v6.safetensors"},
                    ],
                    "active": "/models/sdxl_base.safetensors",
                },
            )
        )

        with client:
            result = client.list_models()

        assert len(result["models"]) == 2
        assert result["active"] == "/models/sdxl_base.safetensors"

    def test_get_active_model(self, client: TsrClient, mock_server) -> None:
        """Test getting active model."""
        mock_server.get("/api/models/active").mock(return_value=Response(200, json={"model": "/models/sdxl_base.safetensors"}))

        with client:
            result = client.get_active_model()

        assert result["model"] == "/models/sdxl_base.safetensors"

    def test_switch_model(self, client: TsrClient, mock_server) -> None:
        """Test switching model."""
        mock_server.post("/api/models/switch").mock(
            return_value=Response(200, json={"status": "ok", "model": "/models/pony_v6.safetensors"})
        )

        with client:
            result = client.switch_model("/models/pony_v6.safetensors")

        assert result["status"] == "ok"

    def test_list_loras(self, client: TsrClient, mock_server) -> None:
        """Test listing LoRAs."""
        mock_server.get("/api/models/loras").mock(
            return_value=Response(
                200,
                json={
                    "loras": [
                        {"name": "detail_tweaker", "path": "/loras/detail_tweaker.safetensors"},
                    ]
                },
            )
        )

        with client:
            result = client.list_loras()

        assert len(result["loras"]) == 1

    def test_scan_models(self, client: TsrClient, mock_server) -> None:
        """Test scanning models."""
        mock_server.get("/api/models/scan").mock(return_value=Response(200, json={"scanned": 5}))

        with client:
            result = client.scan_models()

        assert result["scanned"] == 5


# =============================================================================
# Generation Tests
# =============================================================================


class TestGeneration:
    """Tests for image generation."""

    def test_generate(self, client: TsrClient, mock_server) -> None:
        """Test generating an image."""
        mock_server.post("/api/generate").mock(
            return_value=Response(
                200,
                json={
                    "images": [{"id": "999_42", "seed": 42}],
                    "parameters": {"prompt": "test prompt", "seed": 42},
                },
            )
        )

        with client:
            result = client.generate(
                prompt="test prompt",
                width=512,
                height=512,
                seed=42,
            )

        assert len(result["images"]) == 1
        assert result["images"][0]["seed"] == 42

    def test_generate_with_all_params(self, client: TsrClient, mock_server) -> None:
        """Test generation with all parameters."""
        mock_server.post("/api/generate").mock(return_value=Response(200, json={"images": []}))

        with client:
            result = client.generate(
                prompt="detailed test prompt",
                negative_prompt="bad quality",
                width=1024,
                height=1024,
                steps=30,
                cfg_scale=5.5,
                seed=12345,
                sampler_name="DPM++ 2M",
                scheduler="karras",
                batch_size=2,
                save_to_gallery=False,
                return_base64=True,
            )

        assert "images" in result

    def test_list_samplers(self, client: TsrClient, mock_server) -> None:
        """Test listing samplers."""
        mock_server.get("/api/samplers").mock(return_value=Response(200, json={"samplers": ["Euler", "DPM++ 2M", "Euler a"]}))

        with client:
            result = client.list_samplers()

        assert "samplers" in result

    def test_list_schedulers(self, client: TsrClient, mock_server) -> None:
        """Test listing schedulers."""
        mock_server.get("/api/schedulers").mock(
            return_value=Response(200, json={"schedulers": ["simple", "karras", "sgm_uniform"]})
        )

        with client:
            result = client.list_schedulers()

        assert "schedulers" in result


# =============================================================================
# Download Tests
# =============================================================================


class TestDownload:
    """Tests for CivitAI download operations."""

    def test_start_download_by_version(self, client: TsrClient, mock_server) -> None:
        """Test starting download by version ID."""
        mock_server.post("/api/download").mock(
            return_value=Response(200, json={"download_id": "abc123", "status": "started", "version_id": 12345})
        )

        with client:
            result = client.start_download(version_id=12345)

        assert result["download_id"] == "abc123"

    def test_start_download_by_hash(self, client: TsrClient, mock_server) -> None:
        """Test starting download by hash."""
        mock_server.post("/api/download").mock(return_value=Response(200, json={"download_id": "def456", "status": "started"}))

        with client:
            result = client.start_download(hash_val="ABC123DEF456")

        assert result["status"] == "started"

    def test_get_download_status(self, client: TsrClient, mock_server) -> None:
        """Test getting download status."""
        mock_server.get("/api/download/status/abc123").mock(
            return_value=Response(200, json={"download_id": "abc123", "status": "downloading", "progress": 0.5})
        )

        with client:
            result = client.get_download_status("abc123")

        assert result["progress"] == 0.5

    def test_list_downloads(self, client: TsrClient, mock_server) -> None:
        """Test listing active downloads."""
        mock_server.get("/api/download/active").mock(
            return_value=Response(200, json={"downloads": [{"id": "abc123", "progress": 0.75}]})
        )

        with client:
            result = client.list_downloads()

        assert len(result["downloads"]) == 1


# =============================================================================
# Database Tests
# =============================================================================


class TestDatabase:
    """Tests for database operations."""

    def test_db_list_files(self, client: TsrClient, mock_server) -> None:
        """Test listing local files."""
        mock_server.get("/api/db/files").mock(
            return_value=Response(200, json=[{"id": 1, "file_path": "/models/test.safetensors", "sha256": "abc123"}])
        )

        with client:
            result = client.db_list_files()

        assert len(result) == 1
        assert result[0]["sha256"] == "abc123"

    def test_db_search_models(self, client: TsrClient, mock_server) -> None:
        """Test searching cached models."""
        mock_server.get("/api/db/models").mock(
            return_value=Response(200, json=[{"civitai_id": 12345, "name": "Test Model", "type": "LORA"}])
        )

        with client:
            result = client.db_search_models(query="Test", model_type="LORA")

        assert len(result) == 1
        assert result[0]["name"] == "Test Model"

    def test_db_get_model(self, client: TsrClient, mock_server) -> None:
        """Test getting cached model."""
        mock_server.get("/api/db/models/12345").mock(
            return_value=Response(200, json={"civitai_id": 12345, "name": "Test Model", "type": "Checkpoint"})
        )

        with client:
            result = client.db_get_model(12345)

        assert result["name"] == "Test Model"

    def test_db_get_triggers(self, client: TsrClient, mock_server) -> None:
        """Test getting trigger words."""
        mock_server.get("/api/db/triggers/12345").mock(return_value=Response(200, json=["trigger1", "trigger2"]))

        with client:
            result = client.db_get_triggers(version_id=12345)

        assert result == ["trigger1", "trigger2"]

    def test_db_stats(self, client: TsrClient, mock_server) -> None:
        """Test getting database stats."""
        mock_server.get("/api/db/stats").mock(
            return_value=Response(200, json={"local_files": 10, "models": 5, "model_versions": 15})
        )

        with client:
            result = client.db_stats()

        assert result["local_files"] == 10

    def test_db_scan(self, client: TsrClient, mock_server) -> None:
        """Test scanning directory."""
        mock_server.post("/api/db/scan").mock(return_value=Response(200, json={"scanned": 3, "files": []}))

        with client:
            result = client.db_scan("/models")

        assert result["scanned"] == 3

    def test_db_link(self, client: TsrClient, mock_server) -> None:
        """Test linking files to CivitAI."""
        mock_server.post("/api/db/link").mock(return_value=Response(200, json={"linked": 2}))

        with client:
            result = client.db_link()

        assert result["linked"] == 2

    def test_db_cache(self, client: TsrClient, mock_server) -> None:
        """Test caching model data."""
        mock_server.post("/api/db/cache").mock(return_value=Response(200, json={"model_id": 12345, "cached": True}))

        with client:
            result = client.db_cache(12345)

        assert result["cached"] is True


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_http_error(self, client: TsrClient, mock_server) -> None:
        """Test HTTP error handling."""
        mock_server.get("/api/images").mock(return_value=Response(500, text="Internal server error"))

        with client, pytest.raises(TsrClientError, match="HTTP 500"):
            client.list_images()

    def test_not_found_error(self, client: TsrClient, mock_server) -> None:
        """Test 404 error handling."""
        mock_server.get("/api/images/nonexistent/meta").mock(return_value=Response(404, json={"detail": "Image not found"}))

        with client, pytest.raises(TsrClientError, match="HTTP 404"):
            client.get_image_meta("nonexistent")


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestContextManager:
    """Tests for context manager usage."""

    def test_context_manager(self, mock_server) -> None:
        """Test client works as context manager."""
        mock_server.get("/status").mock(return_value=Response(200, json={"running": True}))

        with TsrClient(BASE_URL) as client:
            result = client.status()
            assert result["running"] is True

    def test_client_without_context(self, mock_server) -> None:
        """Test client works without context manager."""
        mock_server.get("/status").mock(return_value=Response(200, json={"running": True}))

        client = TsrClient(BASE_URL)
        result = client.status()
        assert result["running"] is True
