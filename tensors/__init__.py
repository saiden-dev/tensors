"""tsr: Read safetensor metadata, search and download CivitAI models."""

__version__ = "0.1.19"

from tensors.cli import main
from tensors.config import (
    CONFIG_DIR,
    CONFIG_FILE,
    LEGACY_RC_FILE,
    get_default_output_path,
    load_api_key,
    load_config,
    save_config,
)
from tensors.safetensor import get_base_name, read_safetensor_metadata

__all__ = [
    "CONFIG_DIR",
    "CONFIG_FILE",
    "LEGACY_RC_FILE",
    "get_base_name",
    "get_default_output_path",
    "load_api_key",
    "load_config",
    "main",
    "read_safetensor_metadata",
    "save_config",
]
