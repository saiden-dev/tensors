"""Pytest configuration and fixtures."""

from __future__ import annotations

import json
import struct

import pytest


@pytest.fixture
def temp_safetensor(tmp_path):
    """Create a minimal valid safetensor file for testing."""

    # Create minimal safetensor with empty tensors and some metadata
    header = {
        "__metadata__": {
            "format": "pt",
            "test_key": "test_value",
        }
    }
    header_bytes = json.dumps(header).encode("utf-8")
    header_size = len(header_bytes)

    file_path = tmp_path / "test_model.safetensors"
    with file_path.open("wb") as f:
        f.write(struct.pack("<Q", header_size))
        f.write(header_bytes)

    return file_path
