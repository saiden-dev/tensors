"""Tests for tensors module."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

import tensors
from tensors import (
    get_base_name,
    get_default_output_path,
    load_api_key,
    read_safetensor_metadata,
)


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
        # Temporarily point RC_FILE to nonexistent file
        monkeypatch.setattr(tensors, "RC_FILE", tmp_path / "nonexistent")
        assert load_api_key() is None
