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

# Default download paths by model type (can be overridden in config.toml [paths])
DEFAULT_PATHS: dict[str, Path] = {
    "Checkpoint": MODELS_DIR / "checkpoints",
    "LORA": MODELS_DIR / "loras",
    "LoCon": MODELS_DIR / "loras",
    "TextualInversion": MODELS_DIR / "embeddings",
    "VAE": MODELS_DIR / "vae",
    "Controlnet": MODELS_DIR / "controlnet",
    "Upscaler": MODELS_DIR / "upscalers",
    "Other": MODELS_DIR / "other",
}

CIVITAI_API_BASE = "https://civitai.com/api/v1"
CIVITAI_DOWNLOAD_BASE = "https://civitai.com/api/download/models"


# ============================================================================
# Enums for CLI
# ============================================================================


class Provider(str, Enum):
    """Model search providers."""

    civitai = "civitai"
    hf = "hf"
    all = "all"


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


def get_model_paths() -> dict[str, Path]:
    """Get model paths from config, with defaults.

    Config format in config.toml:
        [paths]
        checkpoints = "/opt/comfyui/models/checkpoints"
        loras = "/opt/comfyui/models/loras"
        embeddings = "/opt/comfyui/models/embeddings"
        vae = "/opt/comfyui/models/vae"
        controlnet = "/opt/comfyui/models/controlnet"
        upscalers = "/opt/comfyui/models/upscale_models"
        other = "/opt/comfyui/models/other"

    Returns dict mapping CivitAI model types to paths.
    """
    config = load_config()
    paths_config = config.get("paths", {})

    # Map config keys to CivitAI model types
    key_to_types = {
        "checkpoints": ["Checkpoint"],
        "loras": ["LORA", "LoCon"],
        "embeddings": ["TextualInversion"],
        "vae": ["VAE"],
        "controlnet": ["Controlnet"],
        "upscalers": ["Upscaler"],
        "other": ["Other"],
    }

    # Start with defaults
    result = dict(DEFAULT_PATHS)

    # Override with config values
    if isinstance(paths_config, dict):
        for key, types in key_to_types.items():
            if key in paths_config:
                path = Path(paths_config[key])
                for model_type in types:
                    result[model_type] = path

    return result


def get_default_output_path(model_type: str | None) -> Path | None:
    """Get default output path based on model type.

    Checks config.toml [paths] section first, falls back to defaults.
    """
    if not model_type:
        return None

    paths = get_model_paths()
    return paths.get(model_type)


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


# ============================================================================
# Tensors Server API Key
# ============================================================================


def get_server_api_key() -> str | None:
    """Get the tensors server API key for authentication.

    Resolution order:
    1. TENSORS_API_KEY environment variable
    2. config.toml [server].api_key
    3. None (no authentication required)
    """
    # Check environment variable first
    env_key = os.environ.get("TENSORS_API_KEY")
    if env_key:
        return env_key

    # Check config file
    config = load_config()
    server_config = config.get("server", {})
    if isinstance(server_config, dict):
        key = server_config.get("api_key")
        if key:
            return str(key)

    return None


# ============================================================================
# ComfyUI Configuration
# ============================================================================

COMFYUI_DEFAULT_URL = "http://127.0.0.1:8188"

# Default generation parameters
COMFYUI_DEFAULT_WIDTH = 1024
COMFYUI_DEFAULT_HEIGHT = 1024
COMFYUI_DEFAULT_STEPS = 20
COMFYUI_DEFAULT_CFG = 7.0
COMFYUI_DEFAULT_SAMPLER = "euler"
COMFYUI_DEFAULT_SCHEDULER = "normal"

# ============================================================================
# Model Family Defaults (Quality Tags, Negative Prompts, etc.)
# ============================================================================

# Rating tags per model family — maps (family, rating) to the tag to inject
# Families not listed here have no rating tag system (prompt-driven only)
RATING_TAGS: dict[str, dict[str, str]] = {
    "pony": {
        "safe": "rating_safe",
        "questionable": "rating_questionable",
        "explicit": "rating_explicit",
    },
    "illustrious": {
        "safe": "rating:safe",
        "questionable": "rating:questionable",
        "explicit": "rating:explicit",
    },
}
# NoobAI uses same tags as Illustrious
RATING_TAGS["noobai"] = RATING_TAGS["illustrious"]


def get_rating_tag(family: str | None, rating: str) -> str | None:
    """Get the rating tag for a model family and rating level.

    Args:
        family: Model family key (e.g. "pony", "illustrious") or None
        rating: One of "safe", "questionable", "explicit"

    Returns:
        Rating tag string to inject into prompt, or None if family has no rating system
    """
    if not family:
        return None
    tags = RATING_TAGS.get(family)
    if not tags:
        return None
    return tags.get(rating)


MODEL_FAMILY_DEFAULTS: dict[str, dict[str, Any]] = {
    "pony": {
        "quality_prefix": "score_9, score_8_up, score_7_up",
        "negative_prompt": "score_5, score_4, ugly, deformed, blurry, bad anatomy, bad hands, missing fingers",
        "width": 1024,
        "height": 1024,
        "portrait": (832, 1216),
        "landscape": (1216, 832),
        "cfg": 6.5,
        "clip_skip": 2,
        "sampler": "euler_ancestral",
        "scheduler": "normal",
        "steps": 25,
        "vae": "ponyStandardVAE_v10.safetensors",
    },
    "illustrious": {
        "quality_prefix": "masterpiece, best quality, highres",
        "negative_prompt": "worst quality, bad quality, low quality, lowres, bad anatomy, bad hands, jpeg artifacts, watermark",
        "width": 1024,
        "height": 1024,
        "portrait": (832, 1216),
        "landscape": (1216, 832),
        "cfg": 6.0,
        "sampler": "euler_ancestral",
        "scheduler": "normal",
        "steps": 25,
        "vae": "illustriousXLV20_v10.safetensors",
    },
    "sdxl": {
        "quality_prefix": "",
        "negative_prompt": "ugly, deformed, bad anatomy, bad hands, extra fingers, missing fingers, blurry, watermark",
        "width": 1024,
        "height": 1024,
        "portrait": (832, 1216),
        "landscape": (1216, 832),
        "cfg": 7.0,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
        "steps": 25,
        "vae": "sdxl_vae.safetensors",
    },
    "sdxl_lightning": {
        "quality_prefix": "",
        "negative_prompt": "ugly, deformed, bad anatomy, bad hands, extra fingers, missing fingers, blurry, watermark",
        "width": 1024,
        "height": 1024,
        "portrait": (832, 1216),
        "landscape": (1216, 832),
        "cfg": 2.0,
        "sampler": "euler",
        "scheduler": "sgm_uniform",
        "steps": 8,
        "vae": "sdxl_vae.safetensors",
    },
    "sdxl_turbo": {
        "quality_prefix": "",
        "negative_prompt": "",
        "width": 1024,
        "height": 1024,
        "portrait": (832, 1216),
        "landscape": (1216, 832),
        "cfg": 1.0,
        "sampler": "euler_ancestral",
        "scheduler": "normal",
        "steps": 4,
        "vae": "sdxl_vae.safetensors",
    },
    "sd15": {
        "quality_prefix": "masterpiece, best quality",
        "negative_prompt": (
            "(worst quality:2), (low quality:2), bad anatomy, bad hands, extra fingers, "
            "missing fingers, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, watermark"
        ),
        "width": 512,
        "height": 512,
        "portrait": (512, 768),
        "landscape": (768, 512),
        "cfg": 7.0,
        "sampler": "euler_ancestral",
        "scheduler": "normal",
        "steps": 25,
        "vae": "vae-ft-mse-840000-ema-pruned.safetensors",
    },
    "sd15_lcm": {
        "quality_prefix": "masterpiece, best quality",
        "negative_prompt": "",
        "width": 512,
        "height": 512,
        "portrait": (512, 768),
        "landscape": (768, 512),
        "cfg": 1.5,
        "sampler": "lcm",
        "scheduler": "normal",
        "steps": 6,
        "vae": "vae-ft-mse-840000-ema-pruned.safetensors",
    },
    "flux": {
        "quality_prefix": "",
        "negative_prompt": "",
        "width": 1024,
        "height": 1024,
        "portrait": (832, 1216),
        "landscape": (1216, 832),
        # Flux Dev is guidance-distilled: KSampler.cfg MUST be 1.0.
        # Real prompt-adherence dial lives on the FluxGuidance node (see "guidance" below).
        # Source: https://comfyanonymous.github.io/ComfyUI_examples/flux/
        "cfg": 1.0,
        "guidance": 3.5,
        "sampler": "euler",
        "scheduler": "simple",
        "steps": 20,
        "vae": "ae.safetensors",
    },
    "flux_schnell": {
        "quality_prefix": "",
        "negative_prompt": "",
        "width": 1024,
        "height": 1024,
        "portrait": (832, 1216),
        "landscape": (1216, 832),
        # Schnell is also distilled; FluxGuidance is typically left at 3.5 but
        # has minimal effect since the model is trained for 4 steps regardless.
        "cfg": 1.0,
        "guidance": 3.5,
        "sampler": "euler",
        "scheduler": "simple",
        "steps": 4,
        "vae": "ae.safetensors",
    },
    # UNet-only Flux checkpoints — same architecture as "flux" but the file
    # ships without CLIP/T5/VAE baked in. Workflow must load them externally
    # via UNETLoader + DualCLIPLoader + VAELoader instead of CheckpointLoaderSimple.
    # external_clip=True signals this to the workflow builder.
    "flux_unet": {
        "quality_prefix": "",
        "negative_prompt": "",
        "width": 1024,
        "height": 1024,
        "portrait": (832, 1216),
        "landscape": (1216, 832),
        "cfg": 1.0,
        "guidance": 3.5,
        "sampler": "euler",
        "scheduler": "simple",
        "steps": 20,
        "vae": "ae.safetensors",
        "external_clip": True,
        "clip_l": "clip_l.safetensors",
        "clip_t5": "t5xxl_fp16.safetensors",
    },
    # Flux.2 Klein 9B — newer Black Forest Labs release. Different architecture
    # from Flux.1: single Qwen3-8B text encoder (12288-dim conditioning, 3 stacked
    # hidden layers), Flux2 latent format, custom Flux2Scheduler, dedicated VAE
    # (flux2-vae.safetensors). Workflow uses CLIPLoader (type=flux2) instead of
    # DualCLIPLoader, and the custom-sampling pipeline (SamplerCustomAdvanced +
    # BasicGuider + Flux2Scheduler) instead of plain KSampler.
    "flux2_klein": {
        "quality_prefix": "",
        "negative_prompt": "",
        "width": 1024,
        "height": 1024,
        "portrait": (832, 1216),
        "landscape": (1216, 832),
        "cfg": 1.0,
        "guidance": 3.5,
        "sampler": "euler",
        "scheduler": "simple",  # unused — Flux2Scheduler provides sigmas
        "steps": 20,
        "vae": "flux2-vae.safetensors",
        "external_clip": True,
        "clip_encoder": "qwen_3_8b_fp8mixed.safetensors",
        "clip_type": "flux2",
    },
    "zimage": {
        "quality_prefix": "",
        "negative_prompt": "",
        "width": 1024,
        "height": 1024,
        "portrait": (832, 1216),
        "landscape": (1216, 832),
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "steps": 4,
        "vae": "ae.safetensors",
    },
}


# UNet-only Flux checkpoint filename substrings (case-insensitive). These ship
# without baked-in CLIP/T5/VAE, so they require UNETLoader + DualCLIPLoader +
# VAELoader instead of CheckpointLoaderSimple. Matched on the lowercased
# filename via simple substring containment.
#
# Add new patterns here as we encounter them — order doesn't matter, first
# match wins.
FLUX_UNET_ONLY_PATTERNS: tuple[str, ...] = (
    "lust_",             # lust_v10.safetensors (Flux.2 Klein 9B-base)
    # Note: bare "lust" would falsely match "illustrious" — keep the underscore.
    "cyberrealisticflux",  # cyberrealisticFlux_v25.safetensors
    "getphatflux",       # getphatFLUXReality_v11Softcore.safetensors
    "moodydesire",       # moodyDesireMix_v20PRO.safetensors
    "fcfluxpony",        # fcFluxPonyPerfectBase_fcFluxPerfectBase.safetensors
)


def _is_flux_unet_only(name_lower: str) -> bool:
    """True if the lowercased filename matches a known UNet-only Flux pattern."""
    return any(p in name_lower for p in FLUX_UNET_ONLY_PATTERNS)


# Flux.2 Klein 9B filename substrings (case-insensitive). These checkpoints are
# UNet-only AND require the Flux.2 architecture (Qwen3-8B encoder, Flux2
# scheduler). Detection is primarily via base_model field ("Flux.2 Klein"); the
# filename patterns are a fallback for checkpoints with missing/wrong DB
# metadata. Filename match wins over base_model.
FLUX2_KLEIN_PATTERNS: tuple[str, ...] = (
    "lust_",             # lust_v10.safetensors
    "moodydesire",       # moodyDesireMix_v20PRO.safetensors
)


def _is_flux2_klein(name_lower: str, base_lower: str) -> bool:
    """True if the model is a Flux.2 Klein 9B checkpoint.

    Detects via base_model field ("flux.2 klein", "flux2 klein") first,
    then filename pattern fallback.
    """
    if "flux.2 klein" in base_lower or "flux2 klein" in base_lower:
        return True
    return any(p in name_lower for p in FLUX2_KLEIN_PATTERNS)


def detect_model_family(model_name: str, base_model: str | None = None) -> str | None:  # noqa: PLR0911
    """Detect model family from filename or CivitAI base_model field.

    Args:
        model_name: Filename of the model (e.g., "ponyDiffusionV6XL.safetensors")
        base_model: Optional CivitAI base_model field (e.g., "Pony", "SDXL 1.0")

    Returns:
        Model family key (pony, illustrious, sdxl, sdxl_lightning, sdxl_turbo,
        sd15, sd15_lcm, flux, flux_schnell, flux_unet, flux2_klein, zimage)
        or None if unknown
    """
    name_lower = model_name.lower()
    base_lower = (base_model or "").lower()

    # Flux.2 Klein 9B override: must run BEFORE flux_unet (Klein patterns like
    # "lust_" and "moodydesire" also appear in FLUX_UNET_ONLY_PATTERNS) AND
    # before the generic flux check. Detection prefers base_model field but
    # falls back to filename pattern for checkpoints with missing metadata.
    if _is_flux2_klein(name_lower, base_lower):
        return "flux2_klein"

    # UNet-only Flux override: must run BEFORE the generic flux check below,
    # since some patterns ("cyberrealisticflux", "getphatflux", "fcfluxpony")
    # also contain the substring "flux". Filename wins over base_model
    # field — these checkpoints are often mis-tagged on CivitAI.
    if _is_flux_unet_only(name_lower):
        return "flux_unet"

    # Architecture override: filename containing "flux" wins over any base_model
    # field (handles hybrid models like "FluxPony" that CivitAI tags as "Pony"
    # but are architecturally Flux and need the Flux workflow).
    if "flux" in name_lower:
        if "schnell" in name_lower:
            return "flux_schnell"
        return "flux"

    # Check base_model field first (most reliable from CivitAI)
    if base_lower:
        if "pony" in base_lower:
            return "pony"
        if "illustrious" in base_lower:
            return "illustrious"
        # Flux variants (check specific variants before generic flux)
        if "flux" in base_lower and "schnell" in base_lower:
            return "flux_schnell"
        if "flux" in base_lower:
            return "flux"
        # ZImageTurbo
        if "zimage" in base_lower:
            return "zimage"
        # SD 1.5 variants
        if "lcm" in base_lower and ("sd 1.5" in base_lower or "sd 1.4" in base_lower):
            return "sd15_lcm"
        if "sd 1.5" in base_lower or "sd 1.4" in base_lower:
            return "sd15"
        # SDXL variants (check specific variants before generic sdxl)
        if "sdxl" in base_lower and "lightning" in base_lower:
            return "sdxl_lightning"
        if "sdxl" in base_lower and "turbo" in base_lower:
            return "sdxl_turbo"
        if "sdxl" in base_lower:
            return "sdxl"

    # Fall back to filename heuristics (check specific variants first)
    # Flux variants take precedence — architecture wins over training dataset
    # (e.g. "FluxPony" hybrids are Flux models trained on Pony data, not SDXL/Pony)
    if "flux" in name_lower and "schnell" in name_lower:
        return "flux_schnell"
    if "flux" in name_lower:
        return "flux"
    if "pony" in name_lower:
        return "pony"
    if "illustrious" in name_lower or "noob" in name_lower:
        return "illustrious"
    # ZImageTurbo
    if "zimage" in name_lower:
        return "zimage"
    # SDXL variants
    if "lightning" in name_lower and any(x in name_lower for x in ["sdxl", "xl"]):
        return "sdxl_lightning"
    if "turbo" in name_lower and any(x in name_lower for x in ["sdxl", "xl"]):
        return "sdxl_turbo"
    # SD 1.5 variants
    if "lcm" in name_lower and any(x in name_lower for x in ["sd15", "sd1.5", "sd_1.5"]):
        return "sd15_lcm"
    if any(x in name_lower for x in ["sd15", "sd1.5", "sd_1.5", "dreamshaper", "realistic", "deliberate", "anything"]):
        return "sd15"
    if any(x in name_lower for x in ["sdxl", "xl_"]):
        return "sdxl"

    return None


def get_model_generation_defaults(model_name: str, base_model: str | None = None) -> dict[str, Any]:
    """Get generation defaults for a model based on its family.

    Detects the model family and returns appropriate default settings for:
    - sampler, scheduler, steps, cfg, width, height
    - quality_prefix, negative_prompt

    Args:
        model_name: Filename of the model
        base_model: Optional CivitAI base_model field

    Returns:
        Dict with generation defaults. Falls back to global SDXL defaults if family unknown.
    """
    family = detect_model_family(model_name, base_model)

    # Get family-specific defaults or fall back to SDXL defaults
    if family and family in MODEL_FAMILY_DEFAULTS:
        defaults = dict(MODEL_FAMILY_DEFAULTS[family])
    else:
        # Default to SDXL settings for unknown models
        defaults = dict(MODEL_FAMILY_DEFAULTS.get("sdxl", {}))

    # Ensure all expected keys are present with fallbacks
    defaults.setdefault("sampler", COMFYUI_DEFAULT_SAMPLER)
    defaults.setdefault("scheduler", COMFYUI_DEFAULT_SCHEDULER)
    defaults.setdefault("steps", COMFYUI_DEFAULT_STEPS)
    defaults.setdefault("cfg", COMFYUI_DEFAULT_CFG)
    defaults.setdefault("width", COMFYUI_DEFAULT_WIDTH)
    defaults.setdefault("height", COMFYUI_DEFAULT_HEIGHT)
    defaults.setdefault("quality_prefix", "")
    defaults.setdefault("negative_prompt", "")

    # Include the detected family for reference
    defaults["family"] = family

    return defaults


def resolve_orientation(family: str | None, orientation: str = "square") -> tuple[int, int]:
    """Get width/height for a model family and orientation.

    Args:
        family: Model family key (e.g. "pony", "sd15", "sdxl") or None for default
        orientation: One of "square", "portrait", "landscape"

    Returns:
        (width, height) tuple
    """
    defaults = MODEL_FAMILY_DEFAULTS.get(family or "sdxl", MODEL_FAMILY_DEFAULTS["sdxl"])
    w: int = defaults["width"]
    h: int = defaults["height"]
    fallback = (w, h)

    if orientation == "portrait":
        pair: tuple[int, int] = defaults.get("portrait", fallback)
        return pair
    if orientation == "landscape":
        pair = defaults.get("landscape", fallback)
        return pair
    return fallback


def get_comfyui_url() -> str:
    """Get the ComfyUI server URL.

    Resolution order:
    1. COMFYUI_URL environment variable
    2. config.toml [comfyui].url
    3. Default: http://127.0.0.1:8188

    Config example:
        [comfyui]
        url = "http://192.168.1.100:8188"
    """
    # Check environment variable first
    env_url = os.environ.get("COMFYUI_URL")
    if env_url:
        return env_url

    # Check config file
    config = load_config()
    comfyui_config = config.get("comfyui", {})
    if isinstance(comfyui_config, dict):
        url = comfyui_config.get("url")
        if url:
            return str(url)

    return COMFYUI_DEFAULT_URL


def get_comfyui_defaults() -> dict[str, Any]:
    """Get default ComfyUI generation parameters.

    Resolution order (per parameter):
    1. config.toml [comfyui] section values
    2. Built-in defaults

    Config example:
        [comfyui]
        url = "http://127.0.0.1:8188"
        default_model = "flux1-dev-fp8.safetensors"
        width = 1024
        height = 1024
        steps = 20
        cfg = 7.0
        sampler = "euler"
        scheduler = "normal"

    Returns dict with keys: model, width, height, steps, cfg, sampler, scheduler
    """
    config = load_config()
    comfyui_config = config.get("comfyui", {})

    defaults: dict[str, Any] = {
        "model": None,
        "width": COMFYUI_DEFAULT_WIDTH,
        "height": COMFYUI_DEFAULT_HEIGHT,
        "steps": COMFYUI_DEFAULT_STEPS,
        "cfg": COMFYUI_DEFAULT_CFG,
        "sampler": COMFYUI_DEFAULT_SAMPLER,
        "scheduler": COMFYUI_DEFAULT_SCHEDULER,
    }

    if isinstance(comfyui_config, dict):
        # Override with config values if present
        if "default_model" in comfyui_config:
            defaults["model"] = str(comfyui_config["default_model"])
        if "width" in comfyui_config:
            defaults["width"] = int(comfyui_config["width"])
        if "height" in comfyui_config:
            defaults["height"] = int(comfyui_config["height"])
        if "steps" in comfyui_config:
            defaults["steps"] = int(comfyui_config["steps"])
        if "cfg" in comfyui_config:
            defaults["cfg"] = float(comfyui_config["cfg"])
        if "sampler" in comfyui_config:
            defaults["sampler"] = str(comfyui_config["sampler"])
        if "scheduler" in comfyui_config:
            defaults["scheduler"] = str(comfyui_config["scheduler"])

    return defaults
