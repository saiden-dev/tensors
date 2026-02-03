#!/usr/bin/env python3
"""
sft-get: Read safetensor metadata and fetch CivitAI model information.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
from pathlib import Path
from typing import Any

import httpx
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

console = Console()

RC_FILE = Path.home() / ".sftrc"

# Default download paths by model type
DEFAULT_PATHS: dict[str, Path] = {
    "Checkpoint": Path.home() / ".xm" / "models" / "checkpoints",
    "LORA": Path.home() / ".xm" / "models" / "loras",
    "LoCon": Path.home() / ".xm" / "models" / "loras",
}


def load_api_key() -> str | None:
    """Load API key from ~/.sftrc or CIVITAI_API_KEY env var."""
    # Check environment variable first
    env_key = os.environ.get("CIVITAI_API_KEY")
    if env_key:
        return env_key

    # Fall back to RC file
    if RC_FILE.exists():
        content = RC_FILE.read_text().strip()
        if content:
            return content
    return None


def get_default_output_path(model_type: str | None) -> Path | None:
    """Get default output path based on model type."""
    if model_type and model_type in DEFAULT_PATHS:
        return DEFAULT_PATHS[model_type]
    return None


CIVITAI_API_BASE = "https://civitai.com/api/v1"
CIVITAI_DOWNLOAD_BASE = "https://civitai.com/api/download/models"


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


def fetch_civitai_model_version(
    version_id: int, api_key: str | None = None
) -> dict[str, Any] | None:
    """Fetch model version information from CivitAI by version ID."""
    url = f"{CIVITAI_API_BASE}/model-versions/{version_id}"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = httpx.get(url, headers=headers, timeout=30.0)
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
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("[cyan]Fetching model from CivitAI...", total=None)

        try:
            response = httpx.get(url, headers=headers, timeout=30.0)
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
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("[cyan]Fetching from CivitAI...", total=None)

        try:
            response = httpx.get(url, headers=headers, timeout=30.0)
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


def download_model(
    version_id: int,
    dest_path: Path,
    api_key: str | None = None,
    resume: bool = True,
) -> bool:
    """Download a model from CivitAI by version ID with resume support.

    Returns True on success, False on failure.
    """
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
            timeout=httpx.Timeout(30.0, read=None),  # No read timeout for large files
        ) as response:
            # Handle 416 Range Not Satisfiable (file already complete)
            if response.status_code == 416:
                console.print("[green]File already fully downloaded.[/green]")
                return True

            response.raise_for_status()

            # Get total size from Content-Length or Content-Range
            content_length = response.headers.get("content-length")
            total_size = int(content_length) + initial_size if content_length else 0

            # Get filename from Content-Disposition if available
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
                    for chunk in response.iter_bytes(1024 * 1024):  # 1MB chunks
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

    # Trained words
    trained_words: list[str] = civitai_data.get("trainedWords", [])
    if trained_words:
        civit_table.add_row("Trigger Words", ", ".join(trained_words))

    # Download URL
    download_url = str(civitai_data.get("downloadUrl", "N/A"))
    civit_table.add_row("Download URL", download_url)

    # File info from CivitAI
    files: list[dict[str, Any]] = civitai_data.get("files", [])
    for f in files:
        if f.get("primary"):
            civit_table.add_row("Primary File", str(f.get("name", "N/A")))
            civit_table.add_row(
                "File Size (CivitAI)",
                f"{f.get('sizeKB', 0) / 1024:.2f} MB",
            )
            meta: dict[str, Any] = f.get("metadata", {})
            if meta:
                civit_table.add_row("Format", str(meta.get("format", "N/A")))
                civit_table.add_row("Precision", str(meta.get("fp", "N/A")))
                civit_table.add_row("Size Type", str(meta.get("size", "N/A")))

    console.print()
    console.print(civit_table)

    # Model page link
    model_id = civitai_data.get("modelId")
    if model_id:
        console.print()
        console.print(
            f"[bold blue]View on CivitAI:[/bold blue] https://civitai.com/models/{model_id}"
        )


def _display_model_info(model_data: dict[str, Any]) -> None:
    """Display full CivitAI model information."""
    # Main model info table
    model_table = Table(title="Model Information", show_header=True, header_style="bold magenta")
    model_table.add_column("Property", style="cyan")
    model_table.add_column("Value", style="green", max_width=80)

    model_table.add_row("ID", str(model_data.get("id", "N/A")))
    model_table.add_row("Name", str(model_data.get("name", "N/A")))
    model_table.add_row("Type", str(model_data.get("type", "N/A")))
    model_table.add_row("NSFW", str(model_data.get("nsfw", False)))

    # Creator info
    creator = model_data.get("creator", {})
    if creator:
        model_table.add_row("Creator", str(creator.get("username", "N/A")))

    # Tags
    tags: list[str] = model_data.get("tags", [])
    if tags:
        model_table.add_row("Tags", ", ".join(tags[:10]) + ("..." if len(tags) > 10 else ""))

    # Stats
    stats: dict[str, Any] = model_data.get("stats", {})
    if stats:
        model_table.add_row("Downloads", f"{stats.get('downloadCount', 0):,}")
        model_table.add_row("Favorites", f"{stats.get('favoriteCount', 0):,}")
        model_table.add_row(
            "Rating", f"{stats.get('rating', 0):.1f} ({stats.get('ratingCount', 0)} ratings)"
        )

    # Mode (archived/taken down)
    mode = model_data.get("mode")
    if mode:
        model_table.add_row("Status", str(mode))

    console.print()
    console.print(model_table)

    # Versions table
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
                size_kb = primary_file.get("sizeKB", 0)
                size_str = (
                    f"{size_kb / 1024:.0f} MB"
                    if size_kb < 1024 * 1024
                    else f"{size_kb / 1024 / 1024:.1f} GB"
                )
                file_info = f"{primary_file.get('name', 'N/A')} ({size_str})"

            created = str(ver.get("createdAt", "N/A"))[:10]  # Just date portion
            ver_table.add_row(
                str(ver.get("id", "N/A")),
                str(ver.get("name", "N/A")),
                str(ver.get("baseModel", "N/A")),
                created,
                file_info,
            )

        console.print()
        console.print(ver_table)

    # Model page link
    model_id = model_data.get("id")
    if model_id:
        console.print()
        console.print(
            f"[bold blue]View on CivitAI:[/bold blue] https://civitai.com/models/{model_id}"
        )


def display_results(
    file_path: Path,
    local_metadata: dict[str, Any],
    sha256_hash: str,
    civitai_data: dict[str, Any] | None,
) -> None:
    """Display results in rich tables."""
    _display_file_info(file_path, local_metadata, sha256_hash)
    _display_local_metadata(local_metadata)
    _display_civitai_data(civitai_data)


def get_base_name(file_path: Path) -> str:
    """Get base filename without .safetensors extension."""
    name = file_path.name
    for ext in (".safetensors", ".sft"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return file_path.stem


def save_metadata(
    file_path: Path,
    sha256_hash: str,
    local_metadata: dict[str, Any],
    civitai_data: dict[str, Any] | None,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Save metadata JSON and SHA256 hash to the specified output directory."""
    base_name = get_base_name(file_path)

    # Save JSON metadata
    json_path = output_dir / f"{base_name}-xm.json"
    output = {
        "file": str(file_path),
        "sha256": sha256_hash,
        "header_size": local_metadata["header_size"],
        "tensor_count": local_metadata["tensor_count"],
        "metadata": local_metadata["metadata"],
        "civitai": civitai_data,
    }
    json_path.write_text(json.dumps(output, indent=2))

    # Save SHA256 hash
    sha_path = output_dir / f"{base_name}-xm.sha256"
    sha_path.write_text(f"{sha256_hash}  {file_path.name}\n")

    return json_path, sha_path


def cmd_info(args: argparse.Namespace) -> int:
    """Handle the info subcommand (default behavior)."""
    file_path: Path = args.file.resolve()

    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        return 1

    if file_path.suffix.lower() not in (".safetensors", ".sft"):
        console.print("[yellow]Warning: File does not have .safetensors extension[/yellow]")

    try:
        # Read local metadata
        console.print(f"[bold]Reading safetensor file:[/bold] {file_path.name}")
        local_metadata = read_safetensor_metadata(file_path)

        # Compute SHA256
        sha256_hash = compute_sha256(file_path)

        # Fetch from CivitAI
        civitai_data = None
        if not args.skip_civitai:
            api_key = args.api_key or load_api_key()
            civitai_data = fetch_civitai_by_hash(sha256_hash, api_key)

        if args.json_output:
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
            display_results(file_path, local_metadata, sha256_hash, civitai_data)

        # Save files if requested
        if args.save_to:
            output_dir: Path = args.save_to.resolve()
            if not output_dir.exists():
                console.print(f"[red]Error: Output directory not found: {output_dir}[/red]")
                return 1
            if not output_dir.is_dir():
                console.print(f"[red]Error: Not a directory: {output_dir}[/red]")
                return 1
            json_path, sha_path = save_metadata(
                file_path, sha256_hash, local_metadata, civitai_data, output_dir
            )
            console.print()
            console.print(f"[green]Saved:[/green] {json_path}")
            console.print(f"[green]Saved:[/green] {sha_path}")

        return 0

    except ValueError as e:
        console.print(f"[red]Error reading safetensor: {e}[/red]")
        return 1
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        return 1


def _resolve_version_id(
    version_id: int | None,
    sha256_hash: str | None,
    model_id: int | None,
    api_key: str | None,
) -> int | None:
    """Resolve version ID from hash or model ID if needed."""
    if version_id:
        return version_id
    if sha256_hash:
        console.print(f"[cyan]Looking up model by hash: {sha256_hash[:16]}...[/cyan]")
        civitai_data = fetch_civitai_by_hash(sha256_hash.upper(), api_key)
        if not civitai_data:
            console.print("[red]Error: Model not found on CivitAI for this hash.[/red]")
            return None
        vid = civitai_data.get("id")
        if vid:
            console.print(f"[green]Found model version:[/green] {civitai_data.get('name', 'N/A')}")
        else:
            console.print("[red]Error: Could not determine version ID from CivitAI response.[/red]")
        return vid
    if model_id:
        console.print(f"[cyan]Looking up model {model_id}...[/cyan]")
        model_data = fetch_civitai_model(model_id, api_key)
        if not model_data:
            console.print(f"[red]Error: Model {model_id} not found on CivitAI.[/red]")
            return None
        versions: list[dict[str, Any]] = model_data.get("modelVersions", [])
        if not versions:
            console.print("[red]Error: Model has no versions.[/red]")
            return None
        # First version is the latest
        latest = versions[0]
        vid = latest.get("id")
        if vid:
            console.print(
                f"[green]Found latest version:[/green] {latest.get('name', 'N/A')} (ID: {vid})"
            )
        return vid
    return None


def cmd_download(args: argparse.Namespace) -> int:
    """Handle the download subcommand."""
    api_key: str | None = args.api_key or load_api_key()

    # Resolve version ID from hash or model ID if needed
    version_id = _resolve_version_id(
        args.version_id, args.hash, getattr(args, "model_id", None), api_key
    )
    if not version_id:
        if not args.version_id and not args.hash and not getattr(args, "model_id", None):
            console.print("[red]Error: Must specify --version-id, --model-id, or --hash[/red]")
        return 1

    # Fetch version info to get filename and model type
    console.print(f"[cyan]Fetching model info for version {version_id}...[/cyan]")
    version_info = fetch_civitai_model_version(version_id, api_key)

    if not version_info:
        console.print("[red]Error: Could not fetch model version info.[/red]")
        return 1

    # Determine model type for default path
    model_type: str | None = version_info.get("model", {}).get("type")

    # Determine output directory
    if args.output is None:
        # Use model type-based default
        output_dir = get_default_output_path(model_type)
        if output_dir is None:
            console.print(
                f"[red]Error: No default path for model type '{model_type}'. "
                "Use --output to specify.[/red]"
            )
            return 1
        console.print(f"[dim]Using default path for {model_type}: {output_dir}[/dim]")
    else:
        output_dir = args.output.resolve()

    # Create directory if it doesn't exist
    if not output_dir.exists():
        console.print(f"[cyan]Creating directory: {output_dir}[/cyan]")
        output_dir.mkdir(parents=True, exist_ok=True)
    elif not output_dir.is_dir():
        console.print(f"[red]Error: Not a directory: {output_dir}[/red]")
        return 1

    # Find primary file or first file
    files: list[dict[str, Any]] = version_info.get("files", [])
    primary_file = next((f for f in files if f.get("primary")), files[0] if files else None)

    if not primary_file:
        console.print("[red]Error: No files found for this model version.[/red]")
        return 1

    filename = primary_file.get("name", f"model-{version_id}.safetensors")
    dest_path = output_dir / filename

    # Display model info
    model_table = Table(title="Model Download", show_header=True, header_style="bold magenta")
    model_table.add_column("Property", style="cyan")
    model_table.add_column("Value", style="green")
    model_table.add_row("Version", version_info.get("name", "N/A"))
    model_table.add_row("Base Model", version_info.get("baseModel", "N/A"))
    model_table.add_row("File", filename)
    model_table.add_row("Size", f"{primary_file.get('sizeKB', 0) / 1024:.2f} MB")
    model_table.add_row("Destination", str(dest_path))
    console.print()
    console.print(model_table)
    console.print()

    # Download
    success = download_model(version_id, dest_path, api_key, resume=not args.no_resume)
    return 0 if success else 1


def cmd_get(args: argparse.Namespace) -> int:
    """Handle the get subcommand - fetch model info by ID."""
    model_id: int = args.model_id
    api_key: str | None = args.api_key or load_api_key()

    model_data = fetch_civitai_model(model_id, api_key)

    if not model_data:
        console.print(f"[red]Error: Model {model_id} not found on CivitAI.[/red]")
        return 1

    if args.json_output:
        console.print_json(data=model_data)
    else:
        _display_model_info(model_data)

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Read safetensor metadata and download CivitAI models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Info command (default)
    info_parser = subparsers.add_parser(
        "info",
        help="Read safetensor metadata and fetch CivitAI info (default)",
    )
    info_parser.add_argument(
        "file",
        type=Path,
        help="Path to the safetensor file",
    )
    info_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="CivitAI API key for authenticated requests",
    )
    info_parser.add_argument(
        "--skip-civitai",
        action="store_true",
        help="Skip CivitAI API lookup",
    )
    info_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON",
    )
    info_parser.add_argument(
        "--save-to",
        type=Path,
        metavar="DIR",
        help="Save metadata JSON and SHA256 hash to the specified directory",
    )
    info_parser.set_defaults(func=cmd_info)

    # Download command
    dl_parser = subparsers.add_parser(
        "download",
        aliases=["dl"],
        help="Download a model from CivitAI",
    )
    dl_parser.add_argument(
        "--version-id",
        "-v",
        type=int,
        help="CivitAI model version ID to download",
    )
    dl_parser.add_argument(
        "--model-id",
        "-m",
        type=int,
        help="CivitAI model ID (downloads latest version)",
    )
    dl_parser.add_argument(
        "--hash",
        "-H",
        type=str,
        help="SHA256 hash to look up and download",
    )
    dl_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="CivitAI API key for authenticated requests",
    )
    dl_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory (default: type-based, e.g. ~/.xm/models/checkpoints for Checkpoint)",
    )
    dl_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume partial downloads, start fresh",
    )
    dl_parser.set_defaults(func=cmd_download)

    # Get command
    get_parser = subparsers.add_parser(
        "get",
        help="Fetch model information from CivitAI by model ID",
    )
    get_parser.add_argument(
        "model_id",
        type=int,
        help="CivitAI model ID",
    )
    get_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="CivitAI API key for authenticated requests",
    )
    get_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON",
    )
    get_parser.set_defaults(func=cmd_get)

    # Parse and handle default command
    args = parser.parse_args()

    # If no command specified and file argument given, assume 'info' command
    if args.command is None:
        # Check if there's a positional argument (file path)
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            # Re-parse with 'info' prepended
            args = parser.parse_args(["info", *sys.argv[1:]])
        else:
            parser.print_help()
            return 0

    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
