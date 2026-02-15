"""Configuration, constants, and enums for tsr CLI."""

from __future__ import annotations

import os
import tomllib
from enum import Enum
from pathlib import Path
from typing import Any

# ============================================================================
# XDG Base Directory Configuration
# ============================================================================

# Config: ~/.config/tensors/config.toml
# Data:   ~/.local/share/tensors/models/, ~/.local/share/tensors/metadata/
CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "tensors"
CONFIG_FILE = CONFIG_DIR / "config.toml"

DATA_DIR = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "tensors"
MODELS_DIR = DATA_DIR / "models"
METADATA_DIR = DATA_DIR / "metadata"
GALLERY_DIR = DATA_DIR / "gallery"

# Legacy config for migration
LEGACY_RC_FILE = Path.home() / ".sftrc"

# Default download paths by model type
DEFAULT_PATHS: dict[str, Path] = {
    "Checkpoint": MODELS_DIR / "checkpoints",
    "LORA": MODELS_DIR / "loras",
    "LoCon": MODELS_DIR / "loras",
}

CIVITAI_API_BASE = "https://civitai.com/api/v1"
CIVITAI_DOWNLOAD_BASE = "https://civitai.com/api/download/models"


# ============================================================================
# Enums for CLI
# ============================================================================


class ModelType(str, Enum):
    """CivitAI model types."""

    checkpoint = "checkpoint"
    lora = "lora"
    embedding = "embedding"
    vae = "vae"
    controlnet = "controlnet"
    locon = "locon"
    hypernetwork = "hypernetwork"
    poses = "poses"
    upscaler = "upscaler"
    motionmodule = "motionmodule"
    wildcards = "wildcards"
    workflows = "workflows"
    other = "other"

    def to_api(self) -> str:
        """Convert to CivitAI API value."""
        mapping = {
            "checkpoint": "Checkpoint",
            "lora": "LORA",
            "embedding": "TextualInversion",
            "vae": "VAE",
            "controlnet": "Controlnet",
            "locon": "LoCon",
            "hypernetwork": "Hypernetwork",
            "poses": "Poses",
            "upscaler": "Upscaler",
            "motionmodule": "MotionModule",
            "wildcards": "Wildcards",
            "workflows": "Workflows",
            "other": "Other",
        }
        return mapping[self.value]


class BaseModel(str, Enum):
    """Common base models."""

    # Stable Diffusion 1.x
    sd14 = "sd14"
    sd15 = "sd15"
    sd15_lcm = "sd15_lcm"
    sd15_hyper = "sd15_hyper"
    # Stable Diffusion 2.x
    sd20 = "sd20"
    sd21 = "sd21"
    # SDXL variants
    sdxl = "sdxl"
    sdxl_turbo = "sdxl_turbo"
    sdxl_lightning = "sdxl_lightning"
    sdxl_hyper = "sdxl_hyper"
    # Pony / Illustrious
    pony = "pony"
    illustrious = "illustrious"
    # Flux variants
    flux_dev = "flux_dev"
    flux_schnell = "flux_schnell"
    # SD 3.x
    sd35_large = "sd35_large"
    sd35_medium = "sd35_medium"
    # Other
    cascade = "cascade"
    svd = "svd"
    other = "other"

    def to_api(self) -> str:
        """Convert to CivitAI API value."""
        mapping = {
            "sd14": "SD 1.4",
            "sd15": "SD 1.5",
            "sd15_lcm": "SD 1.5 LCM",
            "sd15_hyper": "SD 1.5 Hyper",
            "sd20": "SD 2.0",
            "sd21": "SD 2.1",
            "sdxl": "SDXL 1.0",
            "sdxl_turbo": "SDXL Turbo",
            "sdxl_lightning": "SDXL Lightning",
            "sdxl_hyper": "SDXL Hyper",
            "pony": "Pony",
            "illustrious": "Illustrious",
            "flux_dev": "Flux.1 D",
            "flux_schnell": "Flux.1 S",
            "sd35_large": "SD 3.5 Large",
            "sd35_medium": "SD 3.5 Medium",
            "cascade": "Stable Cascade",
            "svd": "SVD",
            "other": "Other",
        }
        return mapping[self.value]


class SortOrder(str, Enum):
    """Sort options for search."""

    downloads = "downloads"
    rating = "rating"
    newest = "newest"

    def to_api(self) -> str:
        """Convert to CivitAI API value."""
        mapping = {
            "downloads": "Most Downloaded",
            "rating": "Highest Rated",
            "newest": "Newest",
        }
        return mapping[self.value]


class Period(str, Enum):
    """Time period for sorting/filtering."""

    all = "all"
    year = "year"
    month = "month"
    week = "week"
    day = "day"

    def to_api(self) -> str:
        """Convert to CivitAI API value."""
        mapping = {
            "all": "AllTime",
            "year": "Year",
            "month": "Month",
            "week": "Week",
            "day": "Day",
        }
        return mapping[self.value]


class NsfwLevel(str, Enum):
    """NSFW content filter level."""

    none = "none"
    soft = "soft"
    mature = "mature"
    x = "x"

    def to_api(self) -> str:
        """Convert to CivitAI API value."""
        # For models endpoint, this maps to the nsfw param
        # none = exclude NSFW, others = specific levels
        return self.value.capitalize() if self.value != "none" else "None"


class CommercialUse(str, Enum):
    """Commercial use permissions."""

    none = "none"
    image = "image"
    rent = "rent"
    sell = "sell"

    def to_api(self) -> str:
        """Convert to CivitAI API value."""
        return self.value.capitalize()


# ============================================================================
# Config Functions
# ============================================================================


def load_config() -> dict[str, Any]:
    """Load configuration from TOML config file."""
    if CONFIG_FILE.exists():
        with CONFIG_FILE.open("rb") as f:
            return tomllib.load(f)
    return {}


def save_config(config: dict[str, Any]) -> None:
    """Save configuration to TOML config file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    # Write scalar values first (before any sections)
    for key, value in config.items():
        if not isinstance(value, dict):
            if isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            else:
                lines.append(f"{key} = {value}")

    if lines:
        lines.append("")

    # Then write sections (dicts)
    for key, value in config.items():
        if isinstance(value, dict):
            lines.append(f"[{key}]")
            for k, v in value.items():
                if isinstance(v, str):
                    lines.append(f'{k} = "{v}"')
                else:
                    lines.append(f"{k} = {v}")
            lines.append("")

    CONFIG_FILE.write_text("\n".join(lines) + "\n")


def load_api_key() -> str | None:
    """Load API key from config file or CIVITAI_API_KEY env var."""
    # Check environment variable first
    env_key = os.environ.get("CIVITAI_API_KEY")
    if env_key:
        return env_key

    # Check TOML config file
    config = load_config()
    api_section = config.get("api", {})
    if isinstance(api_section, dict):
        key = api_section.get("civitai_key")
        if key:
            return str(key)

    # Fall back to legacy RC file for migration
    if LEGACY_RC_FILE.exists():
        content = LEGACY_RC_FILE.read_text().strip()
        if content:
            return content
    return None


def get_default_output_path(model_type: str | None) -> Path | None:
    """Get default output path based on model type."""
    if model_type and model_type in DEFAULT_PATHS:
        return DEFAULT_PATHS[model_type]
    return None


# ============================================================================
# Remote Server Configuration
# ============================================================================


def get_remotes() -> dict[str, str]:
    """Get configured remote servers.

    Returns a dict mapping names to URLs, e.g., {"junkpile": "http://junkpile:8080"}
    """
    config = load_config()
    remotes = config.get("remotes", {})
    return dict(remotes) if isinstance(remotes, dict) else {}


def get_default_remote() -> str | None:
    """Get the default remote name or URL."""
    config = load_config()
    return config.get("default_remote")


def resolve_remote(remote: str | None) -> str | None:
    """Resolve a remote name or URL to a full URL.

    Args:
        remote: Remote name (from config), URL, or None

    Returns:
        Full URL or None if no remote specified and no default
    """
    if remote is None:
        # Check for default remote
        default = get_default_remote()
        if default:
            remote = default
        else:
            return None

    # Check if it's a URL (starts with http:// or https://)
    if remote.startswith(("http://", "https://")):
        return remote

    # Look up in configured remotes
    remotes = get_remotes()
    if remote in remotes:
        return remotes[remote]

    # Treat as hostname with default port
    return f"http://{remote}:8080"


def save_remote(name: str, url: str) -> None:
    """Save a remote server configuration."""
    config = load_config()
    if "remotes" not in config:
        config["remotes"] = {}
    config["remotes"][name] = url
    save_config(config)


def set_default_remote(name: str | None) -> None:
    """Set the default remote."""
    config = load_config()
    if name is None:
        config.pop("default_remote", None)
    else:
        config["default_remote"] = name
    save_config(config)


# ============================================================================
# SD Server Configuration
# ============================================================================

SD_SERVER_DEFAULT_URL = "http://localhost:1234"


def get_sd_server_url() -> str:
    """Get the sd-server URL.

    Resolution order:
    1. SD_SERVER_URL environment variable
    2. config.toml [server].sd_server_url
    3. Default: http://localhost:1234
    """
    # Check environment variable first
    env_url = os.environ.get("SD_SERVER_URL")
    if env_url:
        return env_url

    # Check config file
    config = load_config()
    server_config = config.get("server", {})
    if isinstance(server_config, dict):
        url = server_config.get("sd_server_url")
        if url:
            return str(url)

    return SD_SERVER_DEFAULT_URL


def get_sd_server_api_key() -> str | None:
    """Get the sd-server API key.

    Resolution order:
    1. SD_SERVER_API_KEY environment variable
    2. config.toml [server].sd_server_api_key
    3. None (no authentication)
    """
    # Check environment variable first
    env_key = os.environ.get("SD_SERVER_API_KEY")
    if env_key:
        return env_key

    # Check config file
    config = load_config()
    server_config = config.get("server", {})
    if isinstance(server_config, dict):
        key = server_config.get("sd_server_api_key")
        if key:
            return str(key)

    return None
