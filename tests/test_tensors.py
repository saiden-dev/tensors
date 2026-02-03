"""Tests for tensors module."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from rich.console import Console
from typer.testing import CliRunner

from tensors import config
from tensors.api import (
    download_model,
    fetch_civitai_by_hash,
    fetch_civitai_model,
    fetch_civitai_model_version,
    search_civitai,
)
from tensors.cli import app
from tensors.config import (
    BaseModel,
    ModelType,
    SortOrder,
    get_default_output_path,
    load_api_key,
    load_config,
    save_config,
)
from tensors.display import (
    _format_count,
    _format_size,
    display_civitai_data,
    display_file_info,
    display_local_metadata,
    display_model_info,
    display_search_results,
)
from tensors.safetensor import get_base_name, read_safetensor_metadata

runner = CliRunner()


class TestReadSafetensorMetadata:
    """Tests for read_safetensor_metadata function."""

    def test_reads_valid_safetensor(self, temp_safetensor: Path) -> None:
        """Test reading metadata from a valid safetensor file."""
        result = read_safetensor_metadata(temp_safetensor)

        assert "metadata" in result
        assert "tensor_count" in result
        assert "header_size" in result
        assert result["metadata"]["test_key"] == "test_value"
        assert result["tensor_count"] == 0  # No tensors, just metadata

    def test_raises_on_short_file(self, tmp_path: Path) -> None:
        """Test that short files raise ValueError."""
        short_file = tmp_path / "short.safetensors"
        short_file.write_bytes(b"short")

        with pytest.raises(ValueError, match="too short"):
            read_safetensor_metadata(short_file)

    def test_raises_on_truncated_header(self, tmp_path: Path) -> None:
        """Test that truncated headers raise ValueError."""
        truncated = tmp_path / "truncated.safetensors"
        # Write header size that claims 1000 bytes but only provide 10
        with truncated.open("wb") as f:
            f.write(struct.pack("<Q", 1000))
            f.write(b"x" * 10)

        with pytest.raises(ValueError, match="truncated"):
            read_safetensor_metadata(truncated)

    def test_raises_on_huge_header_size(self, tmp_path: Path) -> None:
        """Test that unreasonably large header sizes raise ValueError."""
        huge = tmp_path / "huge.safetensors"
        with huge.open("wb") as f:
            f.write(struct.pack("<Q", 200_000_000))  # 200MB header

        with pytest.raises(ValueError, match="Invalid header size"):
            read_safetensor_metadata(huge)


class TestGetBaseName:
    """Tests for get_base_name function."""

    def test_removes_safetensors_extension(self) -> None:
        """Test that .safetensors extension is removed."""
        assert get_base_name(Path("model.safetensors")) == "model"

    def test_removes_sft_extension(self) -> None:
        """Test that .sft extension is removed."""
        assert get_base_name(Path("model.sft")) == "model"

    def test_handles_uppercase_extension(self) -> None:
        """Test that uppercase extensions are handled."""
        assert get_base_name(Path("model.SAFETENSORS")) == "model"

    def test_preserves_name_without_known_extension(self) -> None:
        """Test that unknown extensions use stem."""
        assert get_base_name(Path("model.bin")) == "model"


class TestGetDefaultOutputPath:
    """Tests for get_default_output_path function."""

    def test_returns_checkpoint_path(self) -> None:
        """Test that Checkpoint type returns checkpoints directory."""
        result = get_default_output_path("Checkpoint")
        assert result is not None
        assert "checkpoints" in str(result)

    def test_returns_lora_path(self) -> None:
        """Test that LORA type returns loras directory."""
        result = get_default_output_path("LORA")
        assert result is not None
        assert "loras" in str(result)

    def test_returns_none_for_unknown_type(self) -> None:
        """Test that unknown types return None."""
        assert get_default_output_path("UnknownType") is None
        assert get_default_output_path(None) is None


class TestLoadApiKey:
    """Tests for load_api_key function."""

    def test_returns_env_var_if_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that environment variable takes precedence."""
        monkeypatch.setenv("CIVITAI_API_KEY", "test-key-from-env")
        assert load_api_key() == "test-key-from-env"

    def test_returns_none_if_no_key(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test that None is returned when no key is available."""
        monkeypatch.delenv("CIVITAI_API_KEY", raising=False)
        # Point config and legacy files to nonexistent paths
        monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "nonexistent" / "config.toml")
        monkeypatch.setattr(config, "LEGACY_RC_FILE", tmp_path / "nonexistent")
        assert load_api_key() is None

    def test_returns_key_from_config_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test that key is loaded from TOML config file."""
        monkeypatch.delenv("CIVITAI_API_KEY", raising=False)
        config_file = tmp_path / "config.toml"
        config_file.write_text('[api]\ncivitai_key = "key-from-config"\n')
        monkeypatch.setattr(config, "CONFIG_FILE", config_file)
        monkeypatch.setattr(config, "LEGACY_RC_FILE", tmp_path / "nonexistent")
        assert load_api_key() == "key-from-config"

    def test_returns_key_from_legacy_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test that key is loaded from legacy RC file when no config exists."""
        monkeypatch.delenv("CIVITAI_API_KEY", raising=False)
        legacy_file = tmp_path / ".sftrc"
        legacy_file.write_text("legacy-key")
        monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "nonexistent" / "config.toml")
        monkeypatch.setattr(config, "LEGACY_RC_FILE", legacy_file)
        assert load_api_key() == "legacy-key"


class TestSaveConfig:
    """Tests for save_config function."""

    def test_saves_simple_config(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test saving a simple config."""
        config_dir = tmp_path / "config"
        config_file = config_dir / "config.toml"
        monkeypatch.setattr(config, "CONFIG_DIR", config_dir)
        monkeypatch.setattr(config, "CONFIG_FILE", config_file)

        save_config({"key": "value"})

        assert config_file.exists()
        content = config_file.read_text()
        assert 'key = "value"' in content

    def test_saves_nested_config(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test saving a nested config with sections."""
        config_dir = tmp_path / "config"
        config_file = config_dir / "config.toml"
        monkeypatch.setattr(config, "CONFIG_DIR", config_dir)
        monkeypatch.setattr(config, "CONFIG_FILE", config_file)

        save_config({"api": {"civitai_key": "test-key"}})

        content = config_file.read_text()
        assert "[api]" in content
        assert 'civitai_key = "test-key"' in content

    def test_saves_numeric_values(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test saving numeric values without quotes."""
        config_dir = tmp_path / "config"
        config_file = config_dir / "config.toml"
        monkeypatch.setattr(config, "CONFIG_DIR", config_dir)
        monkeypatch.setattr(config, "CONFIG_FILE", config_file)

        save_config({"timeout": 30})

        content = config_file.read_text()
        assert "timeout = 30" in content


class TestLoadConfig:
    """Tests for load_config function."""

    def test_returns_empty_dict_if_no_config(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test that empty dict is returned when config file doesn't exist."""
        monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "nonexistent.toml")
        assert load_config() == {}


class TestEnums:
    """Tests for enum to_api methods."""

    def test_model_type_to_api(self) -> None:
        """Test ModelType enum to_api conversion."""
        assert ModelType.checkpoint.to_api() == "Checkpoint"
        assert ModelType.lora.to_api() == "LORA"
        assert ModelType.embedding.to_api() == "TextualInversion"
        assert ModelType.vae.to_api() == "VAE"
        assert ModelType.controlnet.to_api() == "Controlnet"
        assert ModelType.locon.to_api() == "LoCon"

    def test_base_model_to_api(self) -> None:
        """Test BaseModel enum to_api conversion."""
        assert BaseModel.sd15.to_api() == "SD 1.5"
        assert BaseModel.sdxl.to_api() == "SDXL 1.0"
        assert BaseModel.pony.to_api() == "Pony"
        assert BaseModel.flux.to_api() == "Flux.1 D"
        assert BaseModel.illustrious.to_api() == "Illustrious"

    def test_sort_order_to_api(self) -> None:
        """Test SortOrder enum to_api conversion."""
        assert SortOrder.downloads.to_api() == "Most Downloaded"
        assert SortOrder.rating.to_api() == "Highest Rated"
        assert SortOrder.newest.to_api() == "Newest"


class TestDisplayFormatters:
    """Tests for display formatting functions."""

    def test_format_size_kb(self) -> None:
        """Test formatting sizes in KB."""
        assert _format_size(500) == "500 KB"
        assert _format_size(1023) == "1023 KB"

    def test_format_size_mb(self) -> None:
        """Test formatting sizes in MB."""
        assert _format_size(1024) == "1.0 MB"
        assert _format_size(2048) == "2.0 MB"
        assert _format_size(1024 * 500) == "500.0 MB"

    def test_format_size_gb(self) -> None:
        """Test formatting sizes in GB."""
        assert _format_size(1024 * 1024) == "1.00 GB"
        assert _format_size(1024 * 1024 * 2.5) == "2.50 GB"

    def test_format_count_small(self) -> None:
        """Test formatting small counts."""
        assert _format_count(0) == "0"
        assert _format_count(999) == "999"

    def test_format_count_thousands(self) -> None:
        """Test formatting counts in thousands."""
        assert _format_count(1000) == "1.0K"
        assert _format_count(5500) == "5.5K"
        assert _format_count(999999) == "1000.0K"

    def test_format_count_millions(self) -> None:
        """Test formatting counts in millions."""
        assert _format_count(1_000_000) == "1.0M"
        assert _format_count(2_500_000) == "2.5M"


class TestDisplayFunctions:
    """Tests for display functions with console output."""

    def test_display_file_info(self, temp_safetensor: Path) -> None:
        """Test display_file_info renders without error."""
        console = Console(force_terminal=True, width=80)
        metadata = read_safetensor_metadata(temp_safetensor)
        # Should not raise
        display_file_info(temp_safetensor, metadata, "ABC123", console)

    def test_display_local_metadata_with_data(self) -> None:
        """Test display_local_metadata with metadata."""
        console = Console(force_terminal=True, width=80)
        metadata = {"metadata": {"key1": "value1", "key2": "value2"}, "tensor_count": 0, "header_size": 100}
        # Should not raise
        display_local_metadata(metadata, console)

    def test_display_local_metadata_empty(self) -> None:
        """Test display_local_metadata with no metadata."""
        console = Console(force_terminal=True, width=80)
        metadata: dict[str, Any] = {"metadata": {}, "tensor_count": 0, "header_size": 100}
        # Should not raise
        display_local_metadata(metadata, console)

    def test_display_local_metadata_with_filter(self) -> None:
        """Test display_local_metadata with key filter."""
        console = Console(force_terminal=True, width=80)
        metadata = {"metadata": {"key1": "value1", "key2": "value2"}, "tensor_count": 0, "header_size": 100}
        # Should not raise
        display_local_metadata(metadata, console, keys_filter=["key1"])

    def test_display_civitai_data_none(self) -> None:
        """Test display_civitai_data with None."""
        console = Console(force_terminal=True, width=80)
        # Should not raise
        display_civitai_data(None, console)

    def test_display_civitai_data_with_data(self) -> None:
        """Test display_civitai_data with model data."""
        console = Console(force_terminal=True, width=80)
        data = {
            "modelId": 123,
            "id": 456,
            "name": "Test Model v1",
            "baseModel": "SDXL 1.0",
            "createdAt": "2024-01-01",
            "trainedWords": ["word1", "word2"],
            "downloadUrl": "https://example.com/download",
            "files": [
                {
                    "primary": True,
                    "name": "model.safetensors",
                    "sizeKB": 5000,
                    "metadata": {"format": "SafeTensor", "fp": "fp16", "size": "full"},
                }
            ],
        }
        # Should not raise
        display_civitai_data(data, console)

    def test_display_model_info(self) -> None:
        """Test display_model_info with model data."""
        console = Console(force_terminal=True, width=80)
        data = {
            "id": 123,
            "name": "Test Model",
            "type": "LORA",
            "nsfw": False,
            "creator": {"username": "testuser"},
            "tags": ["tag1", "tag2"],
            "stats": {"downloadCount": 1000, "thumbsUpCount": 100},
            "modelVersions": [
                {
                    "id": 456,
                    "name": "v1.0",
                    "baseModel": "SDXL 1.0",
                    "createdAt": "2024-01-01",
                    "files": [{"primary": True, "name": "model.safetensors", "sizeKB": 5000}],
                }
            ],
        }
        # Should not raise
        display_model_info(data, console)

    def test_display_search_results_empty(self) -> None:
        """Test display_search_results with no results."""
        console = Console(force_terminal=True, width=80)
        # Should not raise
        display_search_results({"items": []}, console)

    def test_display_search_results_with_data(self) -> None:
        """Test display_search_results with results."""
        console = Console(force_terminal=True, width=80)
        results = {
            "items": [
                {
                    "id": 123,
                    "name": "Test Model",
                    "type": "LORA",
                    "modelVersions": [{"baseModel": "SDXL 1.0", "files": [{"primary": True, "sizeKB": 5000}]}],
                    "stats": {"downloadCount": 1000, "thumbsUpCount": 100},
                }
            ],
            "metadata": {"totalItems": 1},
        }
        # Should not raise
        display_search_results(results, console)


class TestAPIFunctions:
    """Tests for API functions with mocked HTTP."""

    @respx.mock
    def test_fetch_model_version_success(self) -> None:
        """Test successful model version fetch."""
        console = Console(force_terminal=True, width=80)
        respx.get("https://civitai.com/api/v1/model-versions/123").mock(
            return_value=httpx.Response(200, json={"id": 123, "name": "Test"})
        )

        result = fetch_civitai_model_version(123, None, console)
        assert result == {"id": 123, "name": "Test"}

    @respx.mock
    def test_fetch_model_version_not_found(self) -> None:
        """Test model version not found."""
        console = Console(force_terminal=True, width=80)
        respx.get("https://civitai.com/api/v1/model-versions/999").mock(return_value=httpx.Response(404))

        result = fetch_civitai_model_version(999, None, console)
        assert result is None

    @respx.mock
    def test_fetch_model_success(self) -> None:
        """Test successful model fetch."""
        console = Console(force_terminal=True, width=80)
        respx.get("https://civitai.com/api/v1/models/123").mock(
            return_value=httpx.Response(200, json={"id": 123, "name": "Test Model"})
        )

        result = fetch_civitai_model(123, None, console)
        assert result == {"id": 123, "name": "Test Model"}

    @respx.mock
    def test_fetch_model_not_found(self) -> None:
        """Test model not found."""
        console = Console(force_terminal=True, width=80)
        respx.get("https://civitai.com/api/v1/models/999").mock(return_value=httpx.Response(404))

        result = fetch_civitai_model(999, None, console)
        assert result is None

    @respx.mock
    def test_fetch_by_hash_success(self) -> None:
        """Test successful hash lookup."""
        console = Console(force_terminal=True, width=80)
        respx.get("https://civitai.com/api/v1/model-versions/by-hash/ABC123").mock(
            return_value=httpx.Response(200, json={"id": 456, "name": "Found"})
        )

        result = fetch_civitai_by_hash("ABC123", None, console)
        assert result == {"id": 456, "name": "Found"}

    @respx.mock
    def test_fetch_by_hash_not_found(self) -> None:
        """Test hash not found."""
        console = Console(force_terminal=True, width=80)
        respx.get("https://civitai.com/api/v1/model-versions/by-hash/NOTFOUND").mock(return_value=httpx.Response(404))

        result = fetch_civitai_by_hash("NOTFOUND", None, console)
        assert result is None

    @respx.mock
    def test_search_civitai_success(self) -> None:
        """Test successful search."""
        console = Console(force_terminal=True, width=80)
        respx.get("https://civitai.com/api/v1/models").mock(
            return_value=httpx.Response(200, json={"items": [{"id": 1}], "metadata": {}})
        )

        result = search_civitai("test", None, None, SortOrder.downloads, 20, None, console)
        assert result is not None
        assert len(result["items"]) == 1

    @respx.mock
    def test_search_civitai_with_filters(self) -> None:
        """Test search with type and base model filters."""
        console = Console(force_terminal=True, width=80)
        respx.get("https://civitai.com/api/v1/models").mock(
            return_value=httpx.Response(200, json={"items": [{"id": 1, "name": "Test LORA"}], "metadata": {}})
        )

        result = search_civitai("test", ModelType.lora, BaseModel.sdxl, SortOrder.downloads, 20, None, console)
        assert result is not None

    @respx.mock
    def test_download_model_success(self, tmp_path: Path) -> None:
        """Test successful model download."""
        console = Console(force_terminal=True, width=80)
        dest = tmp_path / "model.safetensors"
        respx.get("https://civitai.com/api/download/models/123").mock(
            return_value=httpx.Response(200, content=b"fake model data")
        )

        result = download_model(123, dest, None, console, resume=False)
        assert result is True
        assert dest.exists()
        assert dest.read_bytes() == b"fake model data"

    @respx.mock
    def test_download_model_unauthorized(self, tmp_path: Path) -> None:
        """Test download with 401 unauthorized."""
        console = Console(force_terminal=True, width=80)
        dest = tmp_path / "model.safetensors"
        respx.get("https://civitai.com/api/download/models/123").mock(return_value=httpx.Response(401))

        result = download_model(123, dest, None, console, resume=False)
        assert result is False


class TestCLI:
    """Tests for CLI commands."""

    def test_help(self) -> None:
        """Test --help works."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "safetensor" in result.stdout.lower()

    def test_info_file_not_found(self, tmp_path: Path) -> None:
        """Test info command with non-existent file."""
        result = runner.invoke(app, ["info", str(tmp_path / "nonexistent.safetensors")])
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    def test_info_with_safetensor(self, temp_safetensor: Path) -> None:
        """Test info command with valid safetensor file."""
        result = runner.invoke(app, ["info", str(temp_safetensor), "--skip-civitai"])
        assert result.exit_code == 0

    def test_info_json_output(self, temp_safetensor: Path) -> None:
        """Test info command with JSON output."""
        result = runner.invoke(app, ["info", str(temp_safetensor), "--skip-civitai", "--json"])
        assert result.exit_code == 0
        assert "sha256" in result.stdout

    def test_info_meta_filter(self, temp_safetensor: Path) -> None:
        """Test info command with metadata filter."""
        result = runner.invoke(app, ["info", str(temp_safetensor), "--meta", "test_key"])
        assert result.exit_code == 0
        assert "test_value" in result.stdout

    @respx.mock
    def test_search_command(self) -> None:
        """Test search command."""
        respx.get("https://civitai.com/api/v1/models").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [{"id": 1, "name": "Test", "type": "LORA", "modelVersions": [], "stats": {}}],
                    "metadata": {"totalItems": 1},
                },
            )
        )

        result = runner.invoke(app, ["search", "test"])
        assert result.exit_code == 0

    @respx.mock
    def test_get_command(self) -> None:
        """Test get command."""
        respx.get("https://civitai.com/api/v1/models/123").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": 123,
                    "name": "Test Model",
                    "type": "LORA",
                    "nsfw": False,
                    "stats": {},
                    "modelVersions": [],
                },
            )
        )

        result = runner.invoke(app, ["get", "123"])
        assert result.exit_code == 0

    @respx.mock
    def test_get_command_not_found(self) -> None:
        """Test get command with non-existent model."""
        respx.get("https://civitai.com/api/v1/models/999").mock(return_value=httpx.Response(404))

        result = runner.invoke(app, ["get", "999"])
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    def test_config_show(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test config --show command."""
        monkeypatch.delenv("CIVITAI_API_KEY", raising=False)
        monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.toml")
        monkeypatch.setattr(config, "LEGACY_RC_FILE", tmp_path / "nonexistent")

        result = runner.invoke(app, ["config", "--show"])
        assert result.exit_code == 0
        assert "config file" in result.stdout.lower()

    def test_download_no_args(self) -> None:
        """Test dl command with no arguments."""
        result = runner.invoke(app, ["dl"])
        assert result.exit_code == 1
        assert "must specify" in result.stdout.lower()
