#!/usr/bin/env python3
"""
tsr: Read safetensor metadata, search and download CivitAI models.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import sys
import tomllib
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

# ============================================================================
# App and Console Setup
# ============================================================================

app = typer.Typer(
    name="tsr",
    help="Read safetensor metadata, search and download CivitAI models.",
    no_args_is_help=True,
)
console = Console()

# ============================================================================
# Configuration
# ============================================================================

# XDG Base Directory spec
# Config: ~/.config/tensors/config.toml
# Data:   ~/.local/share/tensors/models/, ~/.local/share/tensors/metadata/
CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "tensors"
CONFIG_FILE = CONFIG_DIR / "config.toml"

DATA_DIR = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "tensors"
MODELS_DIR = DATA_DIR / "models"
METADATA_DIR = DATA_DIR / "metadata"

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

    def to_api(self) -> str:
        """Convert to CivitAI API value."""
        mapping = {
            "checkpoint": "Checkpoint",
            "lora": "LORA",
            "embedding": "TextualInversion",
            "vae": "VAE",
            "controlnet": "Controlnet",
            "locon": "LoCon",
        }
        return mapping[self.value]


class BaseModel(str, Enum):
    """Common base models."""

    sd15 = "sd15"
    sdxl = "sdxl"
    pony = "pony"
    flux = "flux"
    illustrious = "illustrious"

    def to_api(self) -> str:
        """Convert to CivitAI API value."""
        mapping = {
            "sd15": "SD 1.5",
            "sdxl": "SDXL 1.0",
            "pony": "Pony",
            "flux": "Flux.1 D",
            "illustrious": "Illustrious",
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
    for key, value in config.items():
        if isinstance(value, dict):
            lines.append(f"[{key}]")
            for k, v in value.items():
                if isinstance(v, str):
                    lines.append(f'{k} = "{v}"')
                else:
                    lines.append(f"{k} = {v}")
            lines.append("")
        elif isinstance(value, str):
            lines.append(f'{key} = "{value}"')
        else:
            lines.append(f"{key} = {value}")

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
# Safetensor Functions
# ============================================================================


def read_safetensor_metadata(file_path: Path) -> dict[str, Any]:
    """Read metadata from a safetensor file header."""
    with file_path.open("rb") as f:
        # First 8 bytes are the header size (little-endian u64)
        header_size_bytes = f.read(8)
        if len(header_size_bytes) < 8:
            raise ValueError("Invalid safetensor file: too short")

        header_size = struct.unpack("<Q", header_size_bytes)[0]

        if header_size > 100_000_000:  # 100MB sanity check
            raise ValueError(f"Invalid header size: {header_size}")

        header_bytes = f.read(header_size)
        if len(header_bytes) < header_size:
            raise ValueError("Invalid safetensor file: header truncated")

        header: dict[str, Any] = json.loads(header_bytes.decode("utf-8"))

    # Extract __metadata__ if present
    metadata: dict[str, Any] = header.get("__metadata__", {})

    # Count tensors (keys that aren't __metadata__)
    tensor_count = sum(1 for k in header if k != "__metadata__")

    return {
        "metadata": metadata,
        "tensor_count": tensor_count,
        "header_size": header_size,
    }


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file with progress display."""
    file_size = file_path.stat().st_size
    sha256 = hashlib.sha256()
    chunk_size = 1024 * 1024 * 8  # 8MB chunks

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"[cyan]Hashing {file_path.name}...", total=file_size)

        with file_path.open("rb") as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
                progress.update(task, advance=len(chunk))

    return sha256.hexdigest().upper()


def get_base_name(file_path: Path) -> str:
    """Get base filename without .safetensors extension."""
    name = file_path.name
    for ext in (".safetensors", ".sft"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return file_path.stem


# ============================================================================
# CivitAI API Functions
# ============================================================================


def _get_headers(api_key: str | None) -> dict[str, str]:
    """Get headers for CivitAI API requests."""
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def fetch_civitai_model_version(
    version_id: int, api_key: str | None = None
) -> dict[str, Any] | None:
    """Fetch model version information from CivitAI by version ID."""
    url = f"{CIVITAI_API_BASE}/model-versions/{version_id}"

    try:
        response = httpx.get(url, headers=_get_headers(api_key), timeout=30.0)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result
    except httpx.HTTPStatusError as e:
        console.print(f"[red]API error: {e.response.status_code}[/red]")
        return None
    except httpx.RequestError as e:
        console.print(f"[red]Request error: {e}[/red]")
        return None


def fetch_civitai_model(model_id: int, api_key: str | None = None) -> dict[str, Any] | None:
    """Fetch model information from CivitAI by model ID."""
    url = f"{CIVITAI_API_BASE}/models/{model_id}"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("[cyan]Fetching model from CivitAI...", total=None)

        try:
            response = httpx.get(url, headers=_get_headers(api_key), timeout=30.0)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
        except httpx.HTTPStatusError as e:
            console.print(f"[red]API error: {e.response.status_code}[/red]")
            return None
        except httpx.RequestError as e:
            console.print(f"[red]Request error: {e}[/red]")
            return None


def fetch_civitai_by_hash(sha256_hash: str, api_key: str | None = None) -> dict[str, Any] | None:
    """Fetch model information from CivitAI by SHA256 hash."""
    url = f"{CIVITAI_API_BASE}/model-versions/by-hash/{sha256_hash}"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("[cyan]Fetching from CivitAI...", total=None)

        try:
            response = httpx.get(url, headers=_get_headers(api_key), timeout=30.0)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
        except httpx.HTTPStatusError as e:
            console.print(f"[red]API error: {e.response.status_code}[/red]")
            return None
        except httpx.RequestError as e:
            console.print(f"[red]Request error: {e}[/red]")
            return None


def search_civitai(
    query: str | None = None,
    model_type: ModelType | None = None,
    base_model: BaseModel | None = None,
    sort: SortOrder = SortOrder.downloads,
    limit: int = 20,
    api_key: str | None = None,
) -> dict[str, Any] | None:
    """Search CivitAI models."""
    params: dict[str, Any] = {
        "limit": min(limit, 100),
        "nsfw": "true",
    }

    # API quirk: query + filters don't work reliably together
    # If we have filters, skip query and filter client-side
    has_filters = model_type is not None or base_model is not None

    if query and not has_filters:
        params["query"] = query

    if model_type:
        params["types"] = model_type.to_api()

    if base_model:
        params["baseModels"] = base_model.to_api()

    params["sort"] = sort.to_api()

    # Request more if we need client-side filtering
    if query and has_filters:
        params["limit"] = 100

    url = f"{CIVITAI_API_BASE}/models"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("[cyan]Searching CivitAI...", total=None)

        try:
            response = httpx.get(url, params=params, headers=_get_headers(api_key), timeout=30.0)
            response.raise_for_status()
            result: dict[str, Any] = response.json()

            # Client-side filtering when query + filters combined
            if query and has_filters:
                q_lower = query.lower()
                result["items"] = [
                    m for m in result.get("items", []) if q_lower in m.get("name", "").lower()
                ][:limit]

            return result
        except httpx.HTTPStatusError as e:
            console.print(f"[red]API error: {e.response.status_code}[/red]")
            return None
        except httpx.RequestError as e:
            console.print(f"[red]Request error: {e}[/red]")
            return None


def download_model(
    version_id: int,
    dest_path: Path,
    api_key: str | None = None,
    resume: bool = True,
) -> bool:
    """Download a model from CivitAI by version ID with resume support."""
    url = f"{CIVITAI_DOWNLOAD_BASE}/{version_id}"
    params: dict[str, str] = {}
    if api_key:
        params["token"] = api_key

    headers: dict[str, str] = {}
    mode = "wb"
    initial_size = 0

    # Check for existing partial download
    if resume and dest_path.exists():
        initial_size = dest_path.stat().st_size
        headers["Range"] = f"bytes={initial_size}-"
        mode = "ab"
        console.print(f"[cyan]Resuming download from {initial_size / (1024**2):.1f} MB[/cyan]")

    try:
        with httpx.stream(
            "GET",
            url,
            params=params,
            headers=headers,
            follow_redirects=True,
            timeout=httpx.Timeout(30.0, read=None),
        ) as response:
            if response.status_code == 416:
                console.print("[green]File already fully downloaded.[/green]")
                return True

            response.raise_for_status()

            content_length = response.headers.get("content-length")
            total_size = int(content_length) + initial_size if content_length else 0

            content_disp = response.headers.get("content-disposition", "")
            if "filename=" in content_disp:
                match = re.search(r'filename="?([^";\n]+)"?', content_disp)
                if match and dest_path.is_dir():
                    dest_path = dest_path / match.group(1)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"[cyan]Downloading {dest_path.name}...",
                    total=total_size if total_size > 0 else None,
                    completed=initial_size,
                )

                with dest_path.open(mode) as f:
                    for chunk in response.iter_bytes(1024 * 1024):
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))

        console.print(f"[green]Downloaded:[/green] {dest_path}")
        return True

    except httpx.HTTPStatusError as e:
        console.print(f"[red]Download error: HTTP {e.response.status_code}[/red]")
        if e.response.status_code == 401:
            console.print("[yellow]Hint: This model may require an API key.[/yellow]")
        return False
    except httpx.RequestError as e:
        console.print(f"[red]Download error: {e}[/red]")
        return False


# ============================================================================
# Display Functions
# ============================================================================


def _format_size(size_kb: float) -> str:
    """Format size in KB to human-readable string."""
    if size_kb < 1024:
        return f"{size_kb:.0f} KB"
    if size_kb < 1024 * 1024:
        return f"{size_kb / 1024:.1f} MB"
    return f"{size_kb / 1024 / 1024:.2f} GB"


def _format_count(count: int) -> str:
    """Format large numbers with K/M suffix."""
    if count < 1000:
        return str(count)
    if count < 1_000_000:
        return f"{count / 1000:.1f}K"
    return f"{count / 1_000_000:.1f}M"


def _display_file_info(file_path: Path, local_metadata: dict[str, Any], sha256_hash: str) -> None:
    """Display file information table."""
    file_table = Table(title="File Information", show_header=True, header_style="bold magenta")
    file_table.add_column("Property", style="cyan")
    file_table.add_column("Value", style="green")

    file_table.add_row("File", str(file_path.name))
    file_table.add_row("Path", str(file_path.parent))
    file_table.add_row("Size", f"{file_path.stat().st_size / (1024**3):.2f} GB")
    file_table.add_row("SHA256", sha256_hash)
    file_table.add_row("Header Size", f"{local_metadata['header_size']:,} bytes")
    file_table.add_row("Tensor Count", str(local_metadata["tensor_count"]))

    console.print()
    console.print(file_table)


def _display_local_metadata(local_metadata: dict[str, Any]) -> None:
    """Display local safetensor metadata table."""
    if local_metadata["metadata"]:
        meta_table = Table(
            title="Safetensor Metadata", show_header=True, header_style="bold magenta"
        )
        meta_table.add_column("Key", style="cyan")
        meta_table.add_column("Value", style="green", max_width=80)

        for key, value in sorted(local_metadata["metadata"].items()):
            display_value = str(value)
            if len(display_value) > 200:
                display_value = display_value[:200] + "..."
            meta_table.add_row(key, display_value)

        console.print()
        console.print(meta_table)
    else:
        console.print()
        console.print("[yellow]No embedded metadata found in safetensor file.[/yellow]")


def _display_civitai_data(civitai_data: dict[str, Any] | None) -> None:
    """Display CivitAI model information table."""
    if not civitai_data:
        console.print()
        console.print("[yellow]Model not found on CivitAI.[/yellow]")
        return

    civit_table = Table(
        title="CivitAI Model Information", show_header=True, header_style="bold magenta"
    )
    civit_table.add_column("Property", style="cyan")
    civit_table.add_column("Value", style="green", max_width=80)

    civit_table.add_row("Model ID", str(civitai_data.get("modelId", "N/A")))
    civit_table.add_row("Version ID", str(civitai_data.get("id", "N/A")))
    civit_table.add_row("Version Name", str(civitai_data.get("name", "N/A")))
    civit_table.add_row("Base Model", str(civitai_data.get("baseModel", "N/A")))
    civit_table.add_row("Created At", str(civitai_data.get("createdAt", "N/A")))

    trained_words: list[str] = civitai_data.get("trainedWords", [])
    if trained_words:
        civit_table.add_row("Trigger Words", ", ".join(trained_words))

    download_url = str(civitai_data.get("downloadUrl", "N/A"))
    civit_table.add_row("Download URL", download_url)

    files: list[dict[str, Any]] = civitai_data.get("files", [])
    for f in files:
        if f.get("primary"):
            civit_table.add_row("Primary File", str(f.get("name", "N/A")))
            civit_table.add_row("File Size (CivitAI)", _format_size(f.get("sizeKB", 0)))
            meta: dict[str, Any] = f.get("metadata", {})
            if meta:
                civit_table.add_row("Format", str(meta.get("format", "N/A")))
                civit_table.add_row("Precision", str(meta.get("fp", "N/A")))
                civit_table.add_row("Size Type", str(meta.get("size", "N/A")))

    console.print()
    console.print(civit_table)

    model_id = civitai_data.get("modelId")
    if model_id:
        console.print()
        console.print(
            f"[bold blue]View on CivitAI:[/bold blue] https://civitai.com/models/{model_id}"
        )


def _display_model_info(model_data: dict[str, Any]) -> None:
    """Display full CivitAI model information."""
    model_table = Table(title="Model Information", show_header=True, header_style="bold magenta")
    model_table.add_column("Property", style="cyan")
    model_table.add_column("Value", style="green", max_width=80)

    model_table.add_row("ID", str(model_data.get("id", "N/A")))
    model_table.add_row("Name", str(model_data.get("name", "N/A")))
    model_table.add_row("Type", str(model_data.get("type", "N/A")))
    model_table.add_row("NSFW", str(model_data.get("nsfw", False)))

    creator = model_data.get("creator", {})
    if creator:
        model_table.add_row("Creator", str(creator.get("username", "N/A")))

    tags: list[str] = model_data.get("tags", [])
    if tags:
        model_table.add_row("Tags", ", ".join(tags[:10]) + ("..." if len(tags) > 10 else ""))

    stats: dict[str, Any] = model_data.get("stats", {})
    if stats:
        model_table.add_row("Downloads", f"{stats.get('downloadCount', 0):,}")
        model_table.add_row("Favorites", f"{stats.get('favoriteCount', 0):,}")
        model_table.add_row(
            "Rating", f"{stats.get('rating', 0):.1f} ({stats.get('ratingCount', 0)} ratings)"
        )

    mode = model_data.get("mode")
    if mode:
        model_table.add_row("Status", str(mode))

    console.print()
    console.print(model_table)

    versions: list[dict[str, Any]] = model_data.get("modelVersions", [])
    if versions:
        ver_table = Table(title="Model Versions", show_header=True, header_style="bold magenta")
        ver_table.add_column("ID", style="cyan")
        ver_table.add_column("Name", style="green")
        ver_table.add_column("Base Model", style="yellow")
        ver_table.add_column("Created", style="blue")
        ver_table.add_column("Primary File", style="white")

        for ver in versions:
            files: list[dict[str, Any]] = ver.get("files", [])
            primary_file = next((f for f in files if f.get("primary")), files[0] if files else None)
            file_info = ""
            if primary_file:
                file_info = (
                    f"{primary_file.get('name', 'N/A')} "
                    f"({_format_size(primary_file.get('sizeKB', 0))})"
                )

            created = str(ver.get("createdAt", "N/A"))[:10]
            ver_table.add_row(
                str(ver.get("id", "N/A")),
                str(ver.get("name", "N/A")),
                str(ver.get("baseModel", "N/A")),
                created,
                file_info,
            )

        console.print()
        console.print(ver_table)

    model_id = model_data.get("id")
    if model_id:
        console.print()
        console.print(
            f"[bold blue]View on CivitAI:[/bold blue] https://civitai.com/models/{model_id}"
        )


def _display_search_results(results: dict[str, Any]) -> None:
    """Display search results in a table."""
    items = results.get("items", [])
    if not items:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan", justify="right")
    table.add_column("Name", style="green", max_width=40)
    table.add_column("Type", style="yellow")
    table.add_column("Base", style="blue")
    table.add_column("Size", justify="right")
    table.add_column("DLs", justify="right")
    table.add_column("Rating", justify="right")

    for model in items:
        model_id = str(model.get("id", ""))
        name = model.get("name", "N/A")
        if len(name) > 40:
            name = name[:37] + "..."
        model_type = model.get("type", "N/A")

        # Get latest version info
        versions = model.get("modelVersions", [])
        base_model = "N/A"
        size = "N/A"
        if versions:
            latest = versions[0]
            base_model = latest.get("baseModel", "N/A")
            files = latest.get("files", [])
            primary = next((f for f in files if f.get("primary")), files[0] if files else None)
            if primary:
                size = _format_size(primary.get("sizeKB", 0))

        stats = model.get("stats", {})
        downloads = _format_count(stats.get("downloadCount", 0))
        rating = f"{stats.get('rating', 0):.1f}"

        table.add_row(model_id, name, model_type, base_model, size, downloads, rating)

    console.print()
    console.print(table)

    metadata = results.get("metadata", {})
    total = metadata.get("totalItems", len(items))
    console.print(f"\n[dim]Showing {len(items)} of {total:,} results[/dim]")
    console.print("[dim]Use 'tsr get <id>' to view details or 'tsr dl -m <id>' to download[/dim]")


# ============================================================================
# CLI Commands
# ============================================================================


@app.command()
def info(
    file: Annotated[Path, typer.Argument(help="Path to the safetensor file")],
    api_key: Annotated[str | None, typer.Option("--api-key", help="CivitAI API key")] = None,
    skip_civitai: Annotated[
        bool, typer.Option("--skip-civitai", help="Skip CivitAI API lookup")
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
    save_to: Annotated[
        Path | None, typer.Option("--save-to", help="Save metadata to directory")
    ] = None,
) -> None:
    """Read safetensor metadata and fetch CivitAI info."""
    file_path = file.resolve()

    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)

    if file_path.suffix.lower() not in (".safetensors", ".sft"):
        console.print("[yellow]Warning: File does not have .safetensors extension[/yellow]")

    try:
        console.print(f"[bold]Reading safetensor file:[/bold] {file_path.name}")
        local_metadata = read_safetensor_metadata(file_path)
        sha256_hash = compute_sha256(file_path)

        civitai_data = None
        if not skip_civitai:
            key = api_key or load_api_key()
            civitai_data = fetch_civitai_by_hash(sha256_hash, key)

        if json_output:
            output = {
                "file": str(file_path),
                "sha256": sha256_hash,
                "header_size": local_metadata["header_size"],
                "tensor_count": local_metadata["tensor_count"],
                "metadata": local_metadata["metadata"],
                "civitai": civitai_data,
            }
            console.print_json(data=output)
        else:
            _display_file_info(file_path, local_metadata, sha256_hash)
            _display_local_metadata(local_metadata)
            _display_civitai_data(civitai_data)

        if save_to:
            output_dir = save_to.resolve()
            if not output_dir.exists() or not output_dir.is_dir():
                console.print(f"[red]Error: Invalid directory: {output_dir}[/red]")
                raise typer.Exit(1)

            base_name = get_base_name(file_path)
            json_path = output_dir / f"{base_name}.json"
            sha_path = output_dir / f"{base_name}.sha256"

            output = {
                "file": str(file_path),
                "sha256": sha256_hash,
                "header_size": local_metadata["header_size"],
                "tensor_count": local_metadata["tensor_count"],
                "metadata": local_metadata["metadata"],
                "civitai": civitai_data,
            }
            json_path.write_text(json.dumps(output, indent=2))
            sha_path.write_text(f"{sha256_hash}  {file_path.name}\n")

            console.print()
            console.print(f"[green]Saved:[/green] {json_path}")
            console.print(f"[green]Saved:[/green] {sha_path}")

    except ValueError as e:
        console.print(f"[red]Error reading safetensor: {e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def search(
    query: Annotated[str | None, typer.Argument(help="Search query (optional)")] = None,
    model_type: Annotated[
        ModelType | None, typer.Option("-t", "--type", help="Model type filter")
    ] = None,
    base: Annotated[
        BaseModel | None, typer.Option("-b", "--base", help="Base model filter")
    ] = None,
    sort: Annotated[
        SortOrder, typer.Option("-s", "--sort", help="Sort order")
    ] = SortOrder.downloads,
    limit: Annotated[int, typer.Option("-n", "--limit", help="Max results")] = 20,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key", help="CivitAI API key")] = None,
) -> None:
    """Search CivitAI models."""
    key = api_key or load_api_key()

    results = search_civitai(
        query=query,
        model_type=model_type,
        base_model=base,
        sort=sort,
        limit=limit,
        api_key=key,
    )

    if not results:
        console.print("[red]Search failed.[/red]")
        raise typer.Exit(1)

    if json_output:
        console.print_json(data=results)
    else:
        _display_search_results(results)


@app.command()
def get(
    id_value: Annotated[int, typer.Argument(help="CivitAI model ID or version ID")],
    version: Annotated[
        bool, typer.Option("-v", "--version", help="Treat ID as version ID instead of model ID")
    ] = False,
    api_key: Annotated[str | None, typer.Option("--api-key", help="CivitAI API key")] = None,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Fetch model information from CivitAI by model ID or version ID."""
    key = api_key or load_api_key()

    if version:
        # Fetch by version ID
        version_data = fetch_civitai_model_version(id_value, key)
        if not version_data:
            console.print(f"[red]Error: Version {id_value} not found on CivitAI.[/red]")
            raise typer.Exit(1)

        if json_output:
            console.print_json(data=version_data)
        else:
            _display_civitai_data(version_data)
    else:
        # Fetch by model ID
        model_data = fetch_civitai_model(id_value, key)
        if not model_data:
            console.print(f"[red]Error: Model {id_value} not found on CivitAI.[/red]")
            raise typer.Exit(1)

        if json_output:
            console.print_json(data=model_data)
        else:
            _display_model_info(model_data)


def _resolve_version_id(
    version_id: int | None,
    hash_val: str | None,
    model_id: int | None,
    api_key: str | None,
) -> int | None:
    """Resolve version ID from hash or model ID."""
    if version_id:
        return version_id

    if hash_val:
        console.print(f"[cyan]Looking up model by hash: {hash_val[:16]}...[/cyan]")
        civitai_data = fetch_civitai_by_hash(hash_val.upper(), api_key)
        if not civitai_data:
            console.print("[red]Error: Model not found on CivitAI for this hash.[/red]")
            return None
        vid: int | None = civitai_data.get("id")
        if vid:
            console.print(f"[green]Found:[/green] {civitai_data.get('name', 'N/A')}")
        return vid

    if model_id:
        console.print(f"[cyan]Looking up model {model_id}...[/cyan]")
        model_data = fetch_civitai_model(model_id, api_key)
        if not model_data:
            console.print(f"[red]Error: Model {model_id} not found.[/red]")
            return None
        versions = model_data.get("modelVersions", [])
        if not versions:
            console.print("[red]Error: Model has no versions.[/red]")
            return None
        latest = versions[0]
        latest_vid: int | None = latest.get("id")
        if latest_vid:
            name = latest.get("name", "N/A")
            console.print(f"[green]Found latest:[/green] {name} (ID: {latest_vid})")
        return latest_vid

    return None


def _prepare_download_dir(output: Path | None, model_type_str: str | None) -> Path | None:
    """Prepare output directory for download."""
    if output is None:
        output_dir = get_default_output_path(model_type_str)
        if output_dir is None:
            console.print(
                f"[red]Error: No default path for type '{model_type_str}'. "
                "Use --output to specify.[/red]"
            )
            return None
        console.print(f"[dim]Using default path for {model_type_str}: {output_dir}[/dim]")
    else:
        output_dir = output.resolve()

    if not output_dir.exists():
        console.print(f"[cyan]Creating directory: {output_dir}[/cyan]")
        output_dir.mkdir(parents=True, exist_ok=True)
    elif not output_dir.is_dir():
        console.print(f"[red]Error: Not a directory: {output_dir}[/red]")
        return None

    return output_dir


@app.command("dl")
def download(
    version_id: Annotated[
        int | None, typer.Option("-v", "--version-id", help="Model version ID")
    ] = None,
    model_id: Annotated[
        int | None, typer.Option("-m", "--model-id", help="Model ID (downloads latest)")
    ] = None,
    hash_val: Annotated[
        str | None, typer.Option("-H", "--hash", help="SHA256 hash to look up")
    ] = None,
    output: Annotated[Path | None, typer.Option("-o", "--output", help="Output directory")] = None,
    no_resume: Annotated[
        bool, typer.Option("--no-resume", help="Don't resume partial downloads")
    ] = False,
    api_key: Annotated[str | None, typer.Option("--api-key", help="CivitAI API key")] = None,
) -> None:
    """Download a model from CivitAI."""
    key = api_key or load_api_key()

    resolved_version_id = _resolve_version_id(version_id, hash_val, model_id, key)
    if not resolved_version_id:
        if not version_id and not hash_val and not model_id:
            console.print("[red]Error: Must specify --version-id, --model-id, or --hash[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Fetching version info for {resolved_version_id}...[/cyan]")
    version_info = fetch_civitai_model_version(resolved_version_id, key)
    if not version_info:
        console.print("[red]Error: Could not fetch model version info.[/red]")
        raise typer.Exit(1)

    model_type_str: str | None = version_info.get("model", {}).get("type")
    output_dir = _prepare_download_dir(output, model_type_str)
    if not output_dir:
        raise typer.Exit(1)

    files: list[dict[str, Any]] = version_info.get("files", [])
    primary_file = next((f for f in files if f.get("primary")), files[0] if files else None)
    if not primary_file:
        console.print("[red]Error: No files found for this version.[/red]")
        raise typer.Exit(1)

    filename = primary_file.get("name", f"model-{resolved_version_id}.safetensors")
    dest_path = output_dir / filename

    table = Table(title="Model Download", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Version", version_info.get("name", "N/A"))
    table.add_row("Base Model", version_info.get("baseModel", "N/A"))
    table.add_row("File", filename)
    table.add_row("Size", _format_size(primary_file.get("sizeKB", 0)))
    table.add_row("Destination", str(dest_path))
    console.print()
    console.print(table)
    console.print()

    success = download_model(resolved_version_id, dest_path, key, resume=not no_resume)
    if not success:
        raise typer.Exit(1)


@app.command()
def config(
    show: Annotated[bool, typer.Option("--show", help="Show current config")] = False,
    set_key: Annotated[str | None, typer.Option("--set-key", help="Set CivitAI API key")] = None,
) -> None:
    """Manage configuration."""
    if set_key:
        cfg = load_config()
        if "api" not in cfg:
            cfg["api"] = {}
        cfg["api"]["civitai_key"] = set_key
        save_config(cfg)
        console.print(f"[green]API key saved to {CONFIG_FILE}[/green]")
        return

    if show or (not set_key):
        console.print(f"[bold]Config file:[/bold] {CONFIG_FILE}")
        console.print(f"[bold]Config exists:[/bold] {CONFIG_FILE.exists()}")

        key = load_api_key()
        if key:
            masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "***"
            console.print(f"[bold]API key:[/bold] {masked}")
        else:
            console.print("[bold]API key:[/bold] [yellow]Not set[/yellow]")

        console.print()
        console.print("[dim]Set API key with: tsr config --set-key YOUR_KEY[/dim]")


def main() -> int:
    """Main entry point."""
    # Handle legacy invocation: tsr <file.safetensors> -> tsr info <file>
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        arg = sys.argv[1]
        if arg not in ("info", "search", "get", "dl", "download", "config") and (
            arg.endswith(".safetensors") or arg.endswith(".sft") or Path(arg).exists()
        ):
            sys.argv = [sys.argv[0], "info", *sys.argv[1:]]

    app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
