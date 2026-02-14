"""Tests for the database module."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from tensors.db import Database


@pytest.fixture
def temp_db(tmp_path: Path) -> Database:
    """Create a temporary database for testing."""
    db_path = tmp_path / "test_models.db"
    db = Database(db_path=db_path)
    db.init_schema()
    return db


@pytest.fixture
def sample_safetensor(tmp_path: Path) -> Path:
    """Create a sample safetensor file for testing."""
    header = {
        "__metadata__": {
            "format": "pt",
            "test_key": "test_value",
        }
    }
    header_bytes = json.dumps(header).encode("utf-8")
    header_size = len(header_bytes)

    file_path = tmp_path / "models" / "test_lora.safetensors"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("wb") as f:
        f.write(struct.pack("<Q", header_size))
        f.write(header_bytes)

    return file_path


@pytest.fixture
def sample_civitai_model() -> dict:
    """Sample CivitAI model API response."""
    return {
        "id": 123456,
        "name": "Test LoRA",
        "description": "A test LoRA model",
        "type": "LORA",
        "nsfw": False,
        "poi": False,
        "minor": False,
        "tags": ["test", "lora", "anime"],
        "creator": {
            "username": "test_creator",
            "image": "https://example.com/avatar.png",
        },
        "stats": {
            "downloadCount": 1000,
            "thumbsUpCount": 500,
            "thumbsDownCount": 10,
            "commentCount": 50,
            "tippedAmountCount": 5,
        },
        "modelVersions": [
            {
                "id": 789012,
                "name": "v1.0",
                "description": "Initial release",
                "baseModel": "SDXL 1.0",
                "trainedWords": ["test_trigger", "lora_trigger"],
                "files": [
                    {
                        "id": 111222,
                        "name": "test_lora.safetensors",
                        "type": "Model",
                        "sizeKB": 150000,
                        "primary": True,
                        "hashes": {
                            "SHA256": "ABC123DEF456",
                            "BLAKE3": "789XYZ",
                        },
                        "metadata": {
                            "format": "SafeTensor",
                            "size": "full",
                            "fp": "fp16",
                        },
                    }
                ],
                "images": [
                    {
                        "id": 333444,
                        "url": "https://example.com/image.png",
                        "type": "image",
                        "width": 1024,
                        "height": 1024,
                        "meta": {
                            "prompt": "test prompt",
                            "negativePrompt": "bad quality",
                            "cfgScale": 7.0,
                        },
                    }
                ],
                "stats": {
                    "downloadCount": 1000,
                    "thumbsUpCount": 500,
                },
            }
        ],
    }


class TestDatabaseSchema:
    """Tests for database schema initialization."""

    def test_init_schema(self, temp_db: Database) -> None:
        """Test schema initialization creates tables."""
        cur = temp_db.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}

        expected = {
            "local_files",
            "safetensor_metadata",
            "models",
            "model_versions",
            "version_files",
            "file_hashes",
            "trained_words",
            "creators",
            "tags",
            "model_tags",
            "version_images",
            "image_generation_params",
            "image_resources",
        }
        assert expected.issubset(tables)

    def test_init_schema_creates_views(self, temp_db: Database) -> None:
        """Test schema creates required views."""
        cur = temp_db.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='view'")
        views = {row[0] for row in cur.fetchall()}

        assert "v_local_files_full" in views
        assert "v_models_with_latest" in views


class TestLocalFiles:
    """Tests for local file operations."""

    def test_scan_directory(self, temp_db: Database, sample_safetensor: Path) -> None:
        """Test scanning directory for safetensor files."""
        results = temp_db.scan_directory(sample_safetensor.parent)

        assert len(results) == 1
        assert results[0]["file_path"] == str(sample_safetensor.resolve())
        assert "sha256" in results[0]
        assert results[0]["sha256"]  # Should have hash

    def test_scan_directory_empty(self, temp_db: Database, tmp_path: Path) -> None:
        """Test scanning empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        results = temp_db.scan_directory(empty_dir)
        assert results == []

    def test_list_local_files(self, temp_db: Database, sample_safetensor: Path) -> None:
        """Test listing local files after scan."""
        temp_db.scan_directory(sample_safetensor.parent)
        files = temp_db.list_local_files()

        assert len(files) == 1
        assert files[0]["file_path"] == str(sample_safetensor.resolve())

    def test_get_local_file_by_path(self, temp_db: Database, sample_safetensor: Path) -> None:
        """Test getting local file by path."""
        temp_db.scan_directory(sample_safetensor.parent)

        file_info = temp_db.get_local_file_by_path(str(sample_safetensor.resolve()))
        assert file_info is not None
        assert file_info["file_path"] == str(sample_safetensor.resolve())

    def test_get_local_file_by_path_not_found(self, temp_db: Database) -> None:
        """Test getting non-existent file."""
        result = temp_db.get_local_file_by_path("/nonexistent/file.safetensors")
        assert result is None

    def test_get_unlinked_files(self, temp_db: Database, sample_safetensor: Path) -> None:
        """Test getting unlinked files."""
        temp_db.scan_directory(sample_safetensor.parent)
        unlinked = temp_db.get_unlinked_files()

        assert len(unlinked) == 1
        assert unlinked[0].get("civitai_model_id", True)

    def test_link_file_to_civitai(self, temp_db: Database, sample_safetensor: Path) -> None:
        """Test linking a file to CivitAI."""
        results = temp_db.scan_directory(sample_safetensor.parent)
        file_id = results[0]["id"]

        temp_db.link_file_to_civitai(file_id, model_id=123, version_id=456)

        # Should have no unlinked files now
        unlinked = temp_db.get_unlinked_files()
        assert len(unlinked) == 0

    def test_upsert_local_file_updates_existing(self, temp_db: Database, sample_safetensor: Path) -> None:
        """Test that scanning same file twice updates instead of inserting."""
        temp_db.scan_directory(sample_safetensor.parent)
        temp_db.scan_directory(sample_safetensor.parent)

        files = temp_db.list_local_files()
        assert len(files) == 1


class TestCivitAICache:
    """Tests for CivitAI model caching."""

    def test_cache_model(self, temp_db: Database, sample_civitai_model: dict) -> None:
        """Test caching a full CivitAI model."""
        model_id = temp_db.cache_model(sample_civitai_model)
        assert model_id > 0

    def test_cache_model_creates_creator(self, temp_db: Database, sample_civitai_model: dict) -> None:
        """Test that caching model creates creator record."""
        temp_db.cache_model(sample_civitai_model)

        cur = temp_db.conn.cursor()
        cur.execute("SELECT * FROM creators WHERE username = ?", ("test_creator",))
        creator = cur.fetchone()

        assert creator is not None
        assert creator["username"] == "test_creator"

    def test_cache_model_creates_tags(self, temp_db: Database, sample_civitai_model: dict) -> None:
        """Test that caching model creates tags."""
        temp_db.cache_model(sample_civitai_model)

        cur = temp_db.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM tags")
        count = cur.fetchone()[0]

        assert count == 3  # test, lora, anime

    def test_cache_model_creates_versions(self, temp_db: Database, sample_civitai_model: dict) -> None:
        """Test that caching model creates versions."""
        temp_db.cache_model(sample_civitai_model)

        cur = temp_db.conn.cursor()
        cur.execute("SELECT * FROM model_versions WHERE civitai_id = ?", (789012,))
        version = cur.fetchone()

        assert version is not None
        assert version["name"] == "v1.0"
        assert version["base_model"] == "SDXL 1.0"

    def test_cache_model_creates_trained_words(self, temp_db: Database, sample_civitai_model: dict) -> None:
        """Test that caching model creates trained words."""
        temp_db.cache_model(sample_civitai_model)

        cur = temp_db.conn.cursor()
        cur.execute("SELECT word FROM trained_words ORDER BY position")
        words = [row[0] for row in cur.fetchall()]

        assert words == ["test_trigger", "lora_trigger"]

    def test_cache_model_creates_files_and_hashes(self, temp_db: Database, sample_civitai_model: dict) -> None:
        """Test that caching model creates files and hashes."""
        temp_db.cache_model(sample_civitai_model)

        cur = temp_db.conn.cursor()
        cur.execute("SELECT * FROM version_files WHERE civitai_id = ?", (111222,))
        file_record = cur.fetchone()

        assert file_record is not None
        assert file_record["name"] == "test_lora.safetensors"
        assert file_record["is_primary"] == 1

        cur.execute("SELECT hash_type, hash_value FROM file_hashes WHERE file_id = ?", (file_record["id"],))
        hashes = {row[0]: row[1] for row in cur.fetchall()}

        assert hashes["SHA256"] == "ABC123DEF456"
        assert hashes["BLAKE3"] == "789XYZ"

    def test_cache_model_idempotent(self, temp_db: Database, sample_civitai_model: dict) -> None:
        """Test that caching same model twice is idempotent."""
        id1 = temp_db.cache_model(sample_civitai_model)
        id2 = temp_db.cache_model(sample_civitai_model)

        assert id1 == id2

        cur = temp_db.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM models")
        assert cur.fetchone()[0] == 1


class TestQueryOperations:
    """Tests for search and query operations."""

    def test_search_models_by_name(self, temp_db: Database, sample_civitai_model: dict) -> None:
        """Test searching models by name."""
        temp_db.cache_model(sample_civitai_model)
        results = temp_db.search_models(query="Test")

        assert len(results) == 1
        assert results[0]["name"] == "Test LoRA"

    def test_search_models_by_type(self, temp_db: Database, sample_civitai_model: dict) -> None:
        """Test searching models by type."""
        temp_db.cache_model(sample_civitai_model)
        results = temp_db.search_models(model_type="LORA")

        assert len(results) == 1

    def test_search_models_by_base_model(self, temp_db: Database, sample_civitai_model: dict) -> None:
        """Test searching models by base model."""
        temp_db.cache_model(sample_civitai_model)
        results = temp_db.search_models(base_model="SDXL")

        assert len(results) == 1

    def test_search_models_no_results(self, temp_db: Database, sample_civitai_model: dict) -> None:
        """Test search with no matching results."""
        temp_db.cache_model(sample_civitai_model)
        results = temp_db.search_models(query="nonexistent")

        assert len(results) == 0

    def test_search_models_limit(self, temp_db: Database, sample_civitai_model: dict) -> None:
        """Test search respects limit."""
        # Cache multiple models
        for i in range(5):
            model = sample_civitai_model.copy()
            model["id"] = 100000 + i
            model["name"] = f"Model {i}"
            temp_db.cache_model(model)

        results = temp_db.search_models(limit=3)
        assert len(results) == 3

    def test_get_model(self, temp_db: Database, sample_civitai_model: dict) -> None:
        """Test getting model by CivitAI ID."""
        temp_db.cache_model(sample_civitai_model)
        model = temp_db.get_model(123456)

        assert model is not None
        assert model["name"] == "Test LoRA"
        assert model["type"] == "LORA"

    def test_get_model_not_found(self, temp_db: Database) -> None:
        """Test getting non-existent model."""
        result = temp_db.get_model(999999)
        assert result is None

    def test_get_version_by_hash(self, temp_db: Database, sample_civitai_model: dict) -> None:
        """Test finding version by file hash."""
        temp_db.cache_model(sample_civitai_model)
        version = temp_db.get_version_by_hash("ABC123DEF456")

        assert version is not None
        assert version["model_name"] == "Test LoRA"
        assert version["version_name"] == "v1.0"

    def test_get_version_by_hash_case_insensitive(self, temp_db: Database, sample_civitai_model: dict) -> None:
        """Test hash lookup is case insensitive."""
        temp_db.cache_model(sample_civitai_model)
        version = temp_db.get_version_by_hash("abc123def456")

        assert version is not None


class TestTriggerWords:
    """Tests for trigger word operations."""

    def test_get_triggers_by_version(self, temp_db: Database, sample_civitai_model: dict) -> None:
        """Test getting triggers by version ID."""
        temp_db.cache_model(sample_civitai_model)
        triggers = temp_db.get_triggers_by_version(789012)

        assert triggers == ["test_trigger", "lora_trigger"]

    def test_get_triggers_by_file_path(self, temp_db: Database, sample_civitai_model: dict, sample_safetensor: Path) -> None:
        """Test getting triggers by linked file path."""
        temp_db.cache_model(sample_civitai_model)
        results = temp_db.scan_directory(sample_safetensor.parent)
        temp_db.link_file_to_civitai(results[0]["id"], model_id=123456, version_id=789012)

        triggers = temp_db.get_triggers(str(sample_safetensor.resolve()))
        assert triggers == ["test_trigger", "lora_trigger"]

    def test_get_triggers_empty(self, temp_db: Database) -> None:
        """Test getting triggers for unlinked file."""
        triggers = temp_db.get_triggers("/nonexistent/file.safetensors")
        assert triggers == []


class TestStatistics:
    """Tests for database statistics."""

    def test_get_stats_empty(self, temp_db: Database) -> None:
        """Test stats on empty database."""
        stats = temp_db.get_stats()

        assert stats["local_files"] == 0
        assert stats["models"] == 0
        assert stats["model_versions"] == 0

    def test_get_stats_with_data(self, temp_db: Database, sample_civitai_model: dict, sample_safetensor: Path) -> None:
        """Test stats with data."""
        temp_db.cache_model(sample_civitai_model)
        temp_db.scan_directory(sample_safetensor.parent)

        stats = temp_db.get_stats()

        assert stats["local_files"] == 1
        assert stats["models"] == 1
        assert stats["model_versions"] == 1
        assert stats["trained_words"] == 2
        assert stats["creators"] == 1
        assert stats["tags"] == 3


class TestContextManager:
    """Tests for database context manager."""

    def test_context_manager(self, tmp_path: Path) -> None:
        """Test database works as context manager."""
        db_path = tmp_path / "test.db"

        with Database(db_path=db_path) as db:
            db.init_schema()
            stats = db.get_stats()
            assert stats["local_files"] == 0

        # Connection should be closed
        assert db._conn is None

    def test_connection_reuse(self, tmp_path: Path) -> None:
        """Test that connection is reused within context."""
        db_path = tmp_path / "test.db"

        with Database(db_path=db_path) as db:
            db.init_schema()
            conn1 = db.conn
            conn2 = db.conn
            assert conn1 is conn2
