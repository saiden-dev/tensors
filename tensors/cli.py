"""CLI application and commands for tsr."""

from __future__ import annotations

import json
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any

import click
import typer
from rich.console import Console
from rich.table import Table

from tensors.api import (
    download_model,
    fetch_civitai_by_hash,
    fetch_civitai_model,
    fetch_civitai_model_version,
    search_civitai,
)
from tensors.config import (
    COMFYUI_DEFAULT_CFG,
    COMFYUI_DEFAULT_HEIGHT,
    COMFYUI_DEFAULT_SAMPLER,
    COMFYUI_DEFAULT_SCHEDULER,
    COMFYUI_DEFAULT_STEPS,
    COMFYUI_DEFAULT_WIDTH,
    CONFIG_FILE,
    MODEL_FAMILY_DEFAULTS,
    BaseModel,
    CommercialUse,
    ModelType,
    NsfwLevel,
    Period,
    Provider,
    SortOrder,
    detect_model_family,
    get_default_output_path,
    get_model_paths,
    load_api_key,
    load_config,
    save_config,
)
from tensors.db import DB_PATH, Database
from tensors.display import (
    _format_size,
    display_civitai_data,
    display_file_info,
    display_hf_model_info,
    display_hf_search_results,
    display_local_metadata,
    display_model_info,
    display_search_results,
)
from tensors.hf import (
    download_all_safetensors,
    download_hf_safetensor,
    get_hf_model,
    list_safetensor_files,
    search_hf_models,
)
from tensors.remote import (
    remote_download,
    remote_download_status,
    remote_generate,
    remote_get_image,
    remote_models,
    remote_search,
)
from tensors.safetensor import compute_sha256, get_base_name, read_safetensor_metadata

# Key masking threshold
MIN_KEY_LENGTH_FOR_MASKING = 8

# Display truncation limits
MAX_QUEUE_DISPLAY = 10
MAX_MODEL_LIST_DISPLAY = 20
MAX_PROMPT_ID_DISPLAY = 36


def _cache_model_quietly(model_data: dict[str, Any]) -> None:
    """Cache model data to database without output."""
    try:
        with Database() as db:
            db.init_schema()
            db.cache_model(model_data)
    except Exception:
        pass  # Silently ignore cache failures


def _cache_models_quietly(models: list[dict[str, Any]]) -> None:
    """Cache multiple models to database without output."""
    if not models:
        return
    try:
        with Database() as db:
            db.init_schema()
            for model_data in models:
                db.cache_model(model_data)
    except Exception:
        pass  # Silently ignore cache failures


def _version_callback(value: bool) -> None:
    if value:
        print(f"tsr {version('tensors')}")
        raise typer.Exit


app = typer.Typer(
    name="tsr",
    help="Read safetensor metadata, search and download CivitAI models.",
    no_args_is_help=True,
)


@app.callback()
def _main(
    _version: Annotated[
        bool,
        typer.Option("--version", "-V", callback=_version_callback, is_eager=True, help="Show version and exit."),
    ] = False,
) -> None:
    """Read safetensor metadata, search and download CivitAI models."""


console = Console()


@app.command()
def info(
    file: Annotated[Path, typer.Argument(help="Path to the safetensor file")],
    meta: Annotated[list[str] | None, typer.Option("--meta", "-m", help="Show specific metadata key(s) in full")] = None,
    api_key: Annotated[str | None, typer.Option("--api-key", help="CivitAI API key")] = None,
    skip_civitai: Annotated[bool, typer.Option("--skip-civitai", help="Skip CivitAI API lookup")] = False,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
    save_to: Annotated[Path | None, typer.Option("--save-to", help="Save metadata to directory")] = None,
) -> None:
    """Read safetensor metadata and fetch CivitAI info."""
    file_path = file.resolve()

    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)

    if file_path.suffix.lower() not in (".safetensors", ".sft"):
        console.print("[yellow]Warning: File does not have .safetensors extension[/yellow]")

    try:
        local_metadata = read_safetensor_metadata(file_path)

        if meta:
            display_local_metadata(local_metadata, console, keys_filter=meta)
            return

        console.print(f"[bold]Reading safetensor file:[/bold] {file_path.name}")
        sha256_hash = compute_sha256(file_path, console)

        civitai_data = None
        if not skip_civitai:
            key = api_key or load_api_key()
            civitai_data = fetch_civitai_by_hash(sha256_hash, key, console)

        if json_output:
            _output_info_json(file_path, sha256_hash, local_metadata, civitai_data)
        else:
            display_file_info(file_path, local_metadata, sha256_hash, console)
            display_local_metadata(local_metadata, console)
            display_civitai_data(civitai_data, console)

        if save_to:
            _save_metadata(save_to, file_path, sha256_hash, local_metadata, civitai_data)

    except ValueError as e:
        console.print(f"[red]Error reading safetensor: {e}[/red]")
        raise typer.Exit(1) from e


def _output_info_json(
    file_path: Path,
    sha256_hash: str,
    local_metadata: dict[str, Any],
    civitai_data: dict[str, Any] | None,
) -> None:
    """Output info command result as JSON."""
    output = {
        "file": str(file_path),
        "sha256": sha256_hash,
        "header_size": local_metadata["header_size"],
        "tensor_count": local_metadata["tensor_count"],
        "metadata": local_metadata["metadata"],
        "civitai": civitai_data,
    }
    console.print_json(data=output)


def _save_metadata(
    save_to: Path,
    file_path: Path,
    sha256_hash: str,
    local_metadata: dict[str, Any],
    civitai_data: dict[str, Any] | None,
) -> None:
    """Save metadata to directory."""
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


@app.command()
def search(
    query: Annotated[str | None, typer.Argument(help="Search query (optional)")] = None,
    provider: Annotated[Provider, typer.Option("--provider", "-P", help="Search provider")] = Provider.all,
    model_type: Annotated[ModelType | None, typer.Option("-t", "--type", help="Model type filter (CivitAI)")] = None,
    base: Annotated[BaseModel | None, typer.Option("-b", "--base", help="Base model filter (CivitAI)")] = None,
    sort: Annotated[SortOrder, typer.Option("-s", "--sort", help="Sort order")] = SortOrder.downloads,
    limit: Annotated[int, typer.Option("-n", "--limit", help="Max results per provider")] = 20,
    period: Annotated[Period | None, typer.Option("-p", "--period", help="Time period (CivitAI)")] = None,
    tag: Annotated[str | None, typer.Option("--tag", help="Filter by tag")] = None,
    username: Annotated[str | None, typer.Option("-u", "--user", "-a", "--author", help="Filter by creator/author")] = None,
    page: Annotated[int | None, typer.Option("--page", help="Page number (CivitAI)")] = None,
    nsfw: Annotated[NsfwLevel | None, typer.Option("--nsfw", help="NSFW filter level (CivitAI)")] = None,
    sfw: Annotated[bool, typer.Option("--sfw", help="Exclude NSFW content (CivitAI)")] = False,
    commercial: Annotated[CommercialUse | None, typer.Option("--commercial", help="Commercial use filter (CivitAI)")] = None,
    pipeline: Annotated[str | None, typer.Option("--pipeline", help="Pipeline tag (HuggingFace)")] = None,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key", help="CivitAI API key")] = None,
    remote: Annotated[str | None, typer.Option("-r", "--remote", help="Remote server name or URL")] = None,
) -> None:
    """Search models on CivitAI and/or Hugging Face.

    Examples:
        tsr search "flux"                     # Search both providers
        tsr search "anime" -P civitai         # CivitAI only
        tsr search "llama" -P hf              # Hugging Face only
        tsr search -t lora -b pony            # CivitAI LoRAs for Pony
        tsr search -a stabilityai -P hf       # HF by author
        tsr search --sfw -P civitai           # CivitAI SFW only
        tsr search "pony" --remote junkpile   # Search via remote server
    """
    # Remote mode: delegate to remote tensors server
    if remote:
        civitai_results = remote_search(
            remote,
            query=query,
            model_type=model_type.to_api() if model_type else None,
            base_model=base.to_api() if base else None,
            sort=sort.value,
            limit=limit,
            page=page,
            nsfw=nsfw.value if nsfw else None,
            sfw=sfw,
            console=console,
        )
        if not civitai_results:
            console.print("[red]Remote search failed.[/red]")
            raise typer.Exit(1)

        if json_output:
            console.print_json(data={"civitai": civitai_results})
        else:
            display_search_results(civitai_results, console)
        return

    key = api_key or load_api_key()
    civitai_results: dict[str, Any] | None = None
    hf_results: list[dict[str, Any]] | None = None

    # Search CivitAI
    if provider in (Provider.civitai, Provider.all):
        nsfw_filter: NsfwLevel | bool | None = NsfwLevel.none if sfw else nsfw
        civitai_results = search_civitai(
            query=query,
            model_type=model_type,
            base_model=base,
            sort=sort,
            limit=limit,
            api_key=key,
            console=console if provider == Provider.civitai else None,
            period=period,
            nsfw=nsfw_filter,
            tag=tag,
            username=username,
            page=page,
            commercial_use=commercial,
        )
        if civitai_results:
            _cache_models_quietly(civitai_results.get("items", []))

    # Search Hugging Face
    if provider in (Provider.hf, Provider.all):
        tags = [tag] if tag else None
        hf_results = search_hf_models(
            query=query,
            author=username,
            tags=tags,
            pipeline_tag=pipeline,
            sort="downloads" if sort == SortOrder.downloads else "likes" if sort == SortOrder.rating else "created_at",
            limit=limit,
            console=console if provider == Provider.hf else None,
        )

    # Output results
    if json_output:
        output: dict[str, Any] = {}
        if civitai_results:
            output["civitai"] = civitai_results
        if hf_results:
            output["huggingface"] = hf_results
        console.print_json(data=output)
        return

    # Display based on provider
    if provider == Provider.civitai:
        if not civitai_results:
            console.print("[red]CivitAI search failed.[/red]")
            raise typer.Exit(1)
        display_search_results(civitai_results, console)
    elif provider == Provider.hf:
        if hf_results is None:
            console.print("[red]Hugging Face search failed.[/red]")
            raise typer.Exit(1)
        display_hf_search_results(hf_results, console)
    else:
        # Both providers
        if civitai_results and civitai_results.get("items"):
            console.print("\n[bold cyan]═══ CivitAI Results ═══[/bold cyan]")
            display_search_results(civitai_results, console)

        if hf_results:
            console.print("\n[bold cyan]═══ Hugging Face Results ═══[/bold cyan]")
            display_hf_search_results(hf_results, console)

        if not (civitai_results and civitai_results.get("items")) and not hf_results:
            console.print("[yellow]No results found on either provider.[/yellow]")


@app.command()
def get(
    id_value: Annotated[int, typer.Argument(help="CivitAI model ID or version ID")],
    version: Annotated[bool, typer.Option("-v", "--version", help="Treat ID as version ID instead of model ID")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key", help="CivitAI API key")] = None,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Don't cache to local database")] = False,
) -> None:
    """Fetch model information from CivitAI by model ID or version ID."""
    key = api_key or load_api_key()

    if version:
        version_data = fetch_civitai_model_version(id_value, key, console)
        if not version_data:
            console.print(f"[red]Error: Version {id_value} not found on CivitAI.[/red]")
            raise typer.Exit(1)

        # Auto-cache version data (need to fetch full model for complete cache)
        if not no_cache:
            model_id = version_data.get("modelId")
            if model_id:
                model_data = fetch_civitai_model(model_id, key)
                if model_data:
                    _cache_model_quietly(model_data)

        if json_output:
            console.print_json(data=version_data)
        else:
            display_civitai_data(version_data, console)
    else:
        model_data = fetch_civitai_model(id_value, key, console)
        if not model_data:
            console.print(f"[red]Error: Model {id_value} not found on CivitAI.[/red]")
            raise typer.Exit(1)

        # Auto-cache model data
        if not no_cache:
            _cache_model_quietly(model_data)

        if json_output:
            console.print_json(data=model_data)
        else:
            display_model_info(model_data, console)


def _resolve_by_hash(hash_val: str, api_key: str | None) -> int | None:
    """Resolve version ID from SHA256 hash."""
    console.print(f"[cyan]Looking up model by hash: {hash_val[:16]}...[/cyan]")
    civitai_data = fetch_civitai_by_hash(hash_val.upper(), api_key, console)
    if not civitai_data:
        console.print("[red]Error: Model not found on CivitAI for this hash.[/red]")
        return None
    vid: int | None = civitai_data.get("id")
    if vid:
        console.print(f"[green]Found:[/green] {civitai_data.get('name', 'N/A')}")
    return vid


def _resolve_by_model_id(model_id: int, api_key: str | None) -> int | None:
    """Resolve latest version ID from model ID."""
    console.print(f"[cyan]Looking up model {model_id}...[/cyan]")
    model_data = fetch_civitai_model(model_id, api_key, console)
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
        console.print(f"[green]Found latest:[/green] {latest.get('name', 'N/A')} (ID: {latest_vid})")
    return latest_vid


def _resolve_version_id(
    version_id: int | None,
    hash_val: str | None,
    model_id: int | None,
    api_key: str | None,
) -> int | None:
    """Resolve version ID from direct ID, hash, or model ID."""
    if version_id:
        return version_id
    if hash_val:
        return _resolve_by_hash(hash_val, api_key)
    if model_id:
        return _resolve_by_model_id(model_id, api_key)
    return None


def _prepare_download_dir(output: Path | None, model_type_str: str | None) -> Path | None:
    """Prepare output directory for download."""
    if output is None:
        output_dir = get_default_output_path(model_type_str)
        if output_dir is None:
            console.print(f"[red]Error: No default path for type '{model_type_str}'. Use --output to specify.[/red]")
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


def _poll_remote_download(remote_name: str, download_id: str) -> None:
    """Poll a remote download for completion with a progress bar."""
    import time  # noqa: PLC0415

    from rich.progress import BarColumn, DownloadColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn  # noqa: PLC0415

    status: dict[str, Any] | None = None
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        DownloadColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Downloading...", total=100)

        while True:
            status = remote_download_status(remote_name, download_id)
            if not status:
                break

            dl_status = status.get("status", "")
            pct = status.get("progress", 0)
            progress.update(task, completed=pct, description=f"[cyan]{dl_status.title()}...")

            if dl_status in ("completed", "failed"):
                break

            time.sleep(1)

    if status and status.get("status") == "completed":
        console.print(f"[green]Download complete:[/green] {status.get('path', 'N/A')}")
    elif status and status.get("status") == "failed":
        console.print(f"[red]Download failed:[/red] {status.get('error', 'Unknown error')}")


def _download_remote(
    remote_name: str,
    version_id: int | None,
    model_id: int | None,
    hash_val: str | None,
    output: Path | None,
) -> None:
    """Handle remote download flow."""
    if not version_id and not model_id:
        if hash_val:
            console.print("[yellow]Remote download does not support --hash. Use --version-id or --model-id.[/yellow]")
        else:
            console.print("[red]Error: Must specify --version-id or --model-id for remote download[/red]")
        raise typer.Exit(1)

    console.print("[dim]Starting download on remote server...[/dim]")
    result = remote_download(
        remote_name,
        version_id=version_id,
        model_id=model_id,
        output_dir=str(output) if output else None,
        console=console,
    )
    if not result:
        raise typer.Exit(1)

    console.print(f"[green]Download started:[/green] {result.get('model_name', 'N/A')}")
    console.print(f"[dim]Version: {result.get('version_name', 'N/A')}[/dim]")
    console.print(f"[dim]Destination: {result.get('destination', 'N/A')}[/dim]")

    download_id = result.get("download_id")
    if download_id:
        _poll_remote_download(remote_name, download_id)


@app.command("dl")
def download(
    version_id: Annotated[int | None, typer.Option("-v", "--version-id", help="Model version ID")] = None,
    model_id: Annotated[int | None, typer.Option("-m", "--model-id", help="Model ID (downloads latest)")] = None,
    hash_val: Annotated[str | None, typer.Option("-H", "--hash", help="SHA256 hash to look up")] = None,
    output: Annotated[Path | None, typer.Option("-o", "--output", help="Output directory")] = None,
    no_resume: Annotated[bool, typer.Option("--no-resume", help="Don't resume partial downloads")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key", help="CivitAI API key")] = None,
    remote: Annotated[str | None, typer.Option("-r", "--remote", help="Remote server name or URL")] = None,
) -> None:
    """Download a model from CivitAI.

    When --remote is specified, the download happens on the remote server.

    Examples:
        tsr dl -v 12345                       # Download by version ID
        tsr dl -m 67890                       # Download latest version of model
        tsr dl -v 12345 --remote junkpile     # Download on remote server
    """
    if remote:
        _download_remote(remote, version_id, model_id, hash_val, output)
        return

    key = api_key or load_api_key()

    resolved_version_id = _resolve_version_id(version_id, hash_val, model_id, key)
    if not resolved_version_id:
        if not version_id and not hash_val and not model_id:
            console.print("[red]Error: Must specify --version-id, --model-id, or --hash[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Fetching version info for {resolved_version_id}...[/cyan]")
    version_info = fetch_civitai_model_version(resolved_version_id, key, console)
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

    _display_download_info(version_info, filename, primary_file, dest_path)

    success = download_model(resolved_version_id, dest_path, key, console, resume=not no_resume)
    if not success:
        raise typer.Exit(1)

    # Add downloaded file to database and link to CivitAI
    _add_downloaded_file_to_db(dest_path, version_info)


def _add_downloaded_file_to_db(dest_path: Path, version_info: dict[str, Any]) -> None:
    """Add a downloaded file to the database and link to CivitAI.

    Args:
        dest_path: Path to the downloaded file
        version_info: CivitAI version info response
    """
    try:
        console.print("[dim]Adding to database...[/dim]")

        # Compute SHA256 hash
        sha256 = compute_sha256(dest_path, console)

        # Read safetensor metadata
        metadata = read_safetensor_metadata(dest_path)

        # Extract CivitAI IDs
        civitai_version_id = version_info.get("id")
        civitai_model_id = version_info.get("modelId") or version_info.get("model", {}).get("id")

        with Database() as db:
            db.init_schema()
            with db.session() as session:
                # Add local file record
                local_file = db._upsert_local_file(
                    session,
                    file_path=str(dest_path.resolve()),
                    sha256=sha256,
                    header_size=metadata.get("header_size"),
                    tensor_count=metadata.get("tensor_count"),
                )

                # Store safetensor metadata
                db._store_safetensor_metadata(session, local_file.id, metadata.get("metadata", {}))

                # Link to CivitAI if we have the IDs
                if civitai_model_id and civitai_version_id:
                    local_file.civitai_model_id = civitai_model_id
                    local_file.civitai_version_id = civitai_version_id
                    session.add(local_file)

                session.commit()
                file_id = local_file.id

        # Report success
        console.print(f"[green]Added to database (id={file_id})[/green]")
        if civitai_model_id and civitai_version_id:
            console.print(f"[green]Linked to CivitAI model={civitai_model_id} version={civitai_version_id}[/green]")

    except Exception as e:
        console.print(f"[yellow]Warning: Could not add to database: {e}[/yellow]")


def _display_download_info(
    version_info: dict[str, Any],
    filename: str,
    primary_file: dict[str, Any],
    dest_path: Path,
) -> None:
    """Display download info table."""
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


@app.command()
def config(
    show: Annotated[bool, typer.Option("--show", help="Show current config")] = False,
    set_key: Annotated[str | None, typer.Option("--set-key", help="Set CivitAI API key")] = None,
    set_path: Annotated[str | None, typer.Option("--set-path", help="Set model path (TYPE=PATH)")] = None,
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

    if set_path:
        # Parse TYPE=PATH format
        if "=" not in set_path:
            console.print("[red]Error: Use format TYPE=PATH (e.g., checkpoints=/opt/models/checkpoints)[/red]")
            raise typer.Exit(1)

        path_type, path_value = set_path.split("=", 1)
        path_type = path_type.lower().strip()
        valid_types = ["checkpoints", "loras", "embeddings", "vae", "controlnet", "upscalers", "other"]

        if path_type not in valid_types:
            console.print(f"[red]Error: Invalid type '{path_type}'. Valid: {', '.join(valid_types)}[/red]")
            raise typer.Exit(1)

        cfg = load_config()
        if "paths" not in cfg:
            cfg["paths"] = {}
        cfg["paths"][path_type] = path_value.strip()
        save_config(cfg)
        console.print(f"[green]Path for {path_type} set to: {path_value}[/green]")
        return

    if show or (not set_key and not set_path):
        console.print(f"[bold]Config file:[/bold] {CONFIG_FILE}")
        console.print(f"[bold]Config exists:[/bold] {CONFIG_FILE.exists()}")

        key = load_api_key()
        if key:
            masked = key[:4] + "..." + key[-4:] if len(key) > MIN_KEY_LENGTH_FOR_MASKING else "***"
            console.print(f"[bold]API key:[/bold] {masked}")
        else:
            console.print("[bold]API key:[/bold] [yellow]Not set[/yellow]")

        console.print()
        console.print("[bold]Model paths:[/bold]")
        paths = get_model_paths()
        # Group by unique paths to show cleanly
        shown_paths: dict[str, list[str]] = {}
        for model_type, path in paths.items():
            path_str = str(path)
            if path_str not in shown_paths:
                shown_paths[path_str] = []
            shown_paths[path_str].append(model_type)

        cfg = load_config()
        configured_paths = cfg.get("paths", {})

        for path_str, types in sorted(shown_paths.items(), key=lambda x: x[0]):
            is_custom = any(
                path_str == configured_paths.get(k)
                for k in ["checkpoints", "loras", "embeddings", "vae", "controlnet", "upscalers", "other"]
            )
            marker = " [green](custom)[/green]" if is_custom else " [dim](default)[/dim]"
            console.print(f"  {', '.join(sorted(types))}: {path_str}{marker}")

        console.print()
        console.print("[dim]Set API key with: tsr config --set-key YOUR_KEY[/dim]")
        console.print("[dim]Set paths with:   tsr config --set-path checkpoints=/path/to/models[/dim]")


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Listen address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Listen port.")] = 51200,
    log_level: Annotated[str, typer.Option(help="Log level.")] = "info",
) -> None:
    """Start the tensors server (gallery and CivitAI management)."""
    try:
        import uvicorn  # noqa: PLC0415

        from tensors.server import create_app  # noqa: PLC0415
    except ImportError:
        console.print("[red]Missing server dependencies. Install with:[/red]")
        console.print("  pip install tensors[server]")
        raise typer.Exit(1) from None

    uvicorn.run(create_app(), host=host, port=port, log_level=log_level)


# =============================================================================
# Top-Level Generate Command
# =============================================================================


@app.command(context_settings={"allow_extra_args": False})
def generate(  # noqa: PLR0915
    ctx: typer.Context,
    prompt: Annotated[str | None, typer.Argument(help="Positive prompt text", show_default=False)] = None,
    model: Annotated[str | None, typer.Option("-m", "--model", help="Checkpoint model name")] = None,
    width: Annotated[int | None, typer.Option("-W", "--width", help="Image width (auto from checkpoint)")] = None,
    height: Annotated[int | None, typer.Option("-H", "--height", help="Image height (auto from checkpoint)")] = None,
    steps: Annotated[int | None, typer.Option("--steps", help="Sampling steps (auto from checkpoint)")] = None,
    cfg: Annotated[float | None, typer.Option("--cfg", help="CFG scale (auto from checkpoint)")] = None,
    seed: Annotated[int, typer.Option("--seed", "-s", help="Random seed (-1 for random)")] = -1,
    sampler: Annotated[str | None, typer.Option("--sampler", help="Sampler name (auto from checkpoint)")] = None,
    scheduler: Annotated[str | None, typer.Option("--scheduler", help="Scheduler name (auto from checkpoint)")] = None,
    vae: Annotated[str | None, typer.Option("--vae", help="VAE model name (auto from checkpoint)")] = None,
    orientation: Annotated[str, typer.Option("-O", "--orientation", help="Resolution: square, portrait, landscape")] = "square",
    lora: Annotated[str | None, typer.Option("-l", "--lora", help="LoRA model name")] = None,
    lora_strength: Annotated[float, typer.Option("--lora-strength", help="LoRA strength")] = 0.8,
    negative: Annotated[str, typer.Option("-n", "--negative-prompt", help="Negative prompt")] = "",
    count: Annotated[int, typer.Option("-c", "--count", help="Number of images to generate")] = 1,
    rating: Annotated[str | None, typer.Option("--rating", "-R", help="Content rating: safe, questionable, explicit (Pony/Illustrious)")] = None,
    no_quality: Annotated[bool, typer.Option("--no-quality", help="Disable auto quality tags")] = False,
    no_negative: Annotated[bool, typer.Option("--no-negative", help="Disable auto negative prompt")] = False,
    output: Annotated[Path | None, typer.Option("-o", "--output", help="Save path (default: current dir)")] = None,
    remote: Annotated[str | None, typer.Option("-r", "--remote", help="Remote server name or URL")] = None,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
    json_input: Annotated[str | None, typer.Option("--input", "-I", help="JSON params (keys match CLI options)")] = None,
) -> None:
    """Generate an image using text-to-image.

    Auto-detects optimal sampler, scheduler, CFG, resolution, and VAE from the checkpoint
    model family. All auto-detected values can be overridden with explicit flags.

    Calls ComfyUI directly when local, or the remote tensors API when --remote is given.
    Accepts --input with a JSON object whose keys match CLI option names. CLI flags override JSON values.

    Examples:
        tsr generate "a cat on a windowsill"
        tsr generate "portrait photo" -m ponyDiffusionV6XL_v6.safetensors -O portrait
        tsr generate "cyberpunk city" -o output.png --count 4
        tsr generate "landscape" --remote junkpile
        tsr generate --input '{"prompt": "a mech", "model": "flux1-dev-fp8.safetensors"}'
        tsr generate "raw prompt" --no-quality --no-negative
    """
    import random as rng  # noqa: PLC0415

    # ---- JSON input merging ----
    if json_input is not None:
        # Support file paths and raw JSON strings
        json_path = Path(json_input)
        if json_path.is_file():
            json_text = json_path.read_text()
        elif json_input.lstrip().startswith("{"):
            json_text = json_input
        else:
            console.print(f"[red]Not a JSON string or file:[/red] {json_input}")
            raise typer.Exit(1)

        try:
            ji = json.loads(json_text)
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid JSON input:[/red] {e}")
            raise typer.Exit(1) from e

        if not isinstance(ji, dict):
            console.print("[red]JSON input must be an object[/red]")
            raise typer.Exit(1)

        # Map JSON keys to parameter names (handle aliases)
        key_map = {"negative_prompt": "negative", "lora_name": "lora"}
        mapped: dict[str, Any] = {}
        for k, v in ji.items():
            mapped[key_map.get(k, k)] = v

        # Determine which CLI params the user explicitly set
        click_ctx = ctx._context if hasattr(ctx, "_context") else ctx
        explicit = (
            {
                p.name
                for p in click_ctx.command.params
                if click_ctx.get_parameter_source(p.name) == click.core.ParameterSource.COMMANDLINE
            }
            if hasattr(click_ctx, "get_parameter_source")
            else set()
        )

        # Apply JSON values for anything not explicitly set on CLI
        if "prompt" in mapped and ("prompt" not in explicit and prompt is None):
            prompt = mapped["prompt"]
        if "model" in mapped and "model" not in explicit:
            model = mapped["model"]
        if "width" in mapped and "width" not in explicit:
            width = int(mapped["width"])
        if "height" in mapped and "height" not in explicit:
            height = int(mapped["height"])
        if "steps" in mapped and "steps" not in explicit:
            steps = int(mapped["steps"])
        if "cfg" in mapped and "cfg" not in explicit:
            cfg = float(mapped["cfg"])
        if "seed" in mapped and "seed" not in explicit:
            seed = int(mapped["seed"])
        if "sampler" in mapped and "sampler" not in explicit:
            sampler = mapped["sampler"]
        if "scheduler" in mapped and "scheduler" not in explicit:
            scheduler = mapped["scheduler"]
        if "vae" in mapped and "vae" not in explicit:
            vae = mapped["vae"]
        if "lora" in mapped and "lora" not in explicit:
            lora = mapped["lora"]
        if "lora_strength" in mapped and "lora_strength" not in explicit:
            lora_strength = float(mapped["lora_strength"])
        if "negative" in mapped and "negative" not in explicit:
            negative = mapped["negative"]
        if "output" in mapped and "output" not in explicit:
            output = Path(mapped["output"])
        if "remote" in mapped and "remote" not in explicit:
            remote = mapped["remote"]
        if "count" in mapped and "count" not in explicit:
            count = int(mapped["count"])
        if "orientation" in mapped and "orientation" not in explicit:
            orientation = mapped["orientation"]
        if "no_quality" in mapped and "no_quality" not in explicit:
            no_quality = bool(mapped["no_quality"])
        if "no_negative" in mapped and "no_negative" not in explicit:
            no_negative = bool(mapped["no_negative"])
        if "rating" in mapped and "rating" not in explicit:
            rating = mapped["rating"]

    if not prompt:
        console.print("[red]Prompt is required (as argument or in --input JSON)[/red]")
        raise typer.Exit(1)

    # ---- Detect model family and enhance prompt/negative ----
    family_defaults: dict[str, Any] = {}
    model_family: str | None = None
    base_model_str: str | None = None
    if model:
        try:
            with Database() as db:
                db.init_schema()
                base_model_str = db.get_base_model_by_filename(model)
        except Exception:
            pass

        model_family = detect_model_family(model, base_model_str)
        if model_family:
            family_defaults = MODEL_FAMILY_DEFAULTS.get(model_family, {})
            if not json_output:
                console.print(f"[dim]Detected model family: {model_family}[/dim]")

    # Build enhanced prompt with quality prefix and LoRA trigger words
    prompt_parts: list[str] = []

    # Add LoRA trigger words if using LoRA
    if lora:
        try:
            with Database() as db:
                db.init_schema()
                trigger_words = db.get_trigger_words_by_filename(lora)
                if trigger_words:
                    prompt_parts.extend(trigger_words)
                    if not json_output:
                        console.print(f"[dim]LoRA trigger words: {', '.join(trigger_words)}[/dim]")
        except Exception:
            pass

    # Add quality prefix based on model family
    if not no_quality and family_defaults.get("quality_prefix"):
        prompt_parts.append(family_defaults["quality_prefix"])

    # Add rating tag based on model family (Pony/Illustrious)
    if rating:
        from tensors.config import get_rating_tag  # noqa: PLC0415

        rating_tag = get_rating_tag(model_family, rating.lower())
        if rating_tag:
            prompt_parts.append(rating_tag)
            if not json_output:
                console.print(f"[dim]Rating tag: {rating_tag}[/dim]")
        elif not json_output:
            console.print(f"[dim]Rating '{rating}' not applicable for {model_family or 'unknown'} family[/dim]")

    # Add user prompt
    prompt_parts.append(prompt)
    enhanced_prompt = ", ".join(prompt_parts) if len(prompt_parts) > 1 else prompt

    # Build enhanced negative prompt
    enhanced_negative = negative
    if not no_negative and family_defaults.get("negative_prompt"):
        family_negative = family_defaults["negative_prompt"]
        enhanced_negative = f"{negative}, {family_negative}" if negative else family_negative

    if not json_output and (enhanced_prompt != prompt or enhanced_negative != negative):
        if enhanced_prompt != prompt:
            truncated = enhanced_prompt[:100] + "..." if len(enhanced_prompt) > 100 else enhanced_prompt  # noqa: PLR2004
            console.print(f"[dim]Enhanced prompt: {truncated}[/dim]")
        if enhanced_negative != negative:
            truncated = enhanced_negative[:80] + "..." if len(enhanced_negative) > 80 else enhanced_negative  # noqa: PLR2004
            console.print(f"[dim]Enhanced negative: {truncated}[/dim]")

    # ---- Resolve preset defaults for None params (both remote and local need these) ----
    from tensors.config import resolve_orientation, resolve_remote as do_resolve_remote  # noqa: PLC0415

    # Use already-detected family_defaults from DB lookup above (not filename guessing)
    if family_defaults:
        res_w, res_h = resolve_orientation(model_family, orientation)
        if width is None:
            width = res_w
        if height is None:
            height = res_h
        if steps is None:
            steps = family_defaults.get("steps", 20)
        if cfg is None:
            cfg = family_defaults.get("cfg", 7.0)
        if sampler is None:
            sampler = family_defaults.get("sampler", "euler")
        if scheduler is None:
            scheduler = family_defaults.get("scheduler", "normal")
        if vae is None:
            vae = family_defaults.get("vae")

    # Fallback to global defaults when no model family was detected
    if width is None:
        width = COMFYUI_DEFAULT_WIDTH
    if height is None:
        height = COMFYUI_DEFAULT_HEIGHT
    if steps is None:
        steps = COMFYUI_DEFAULT_STEPS
    if cfg is None:
        cfg = COMFYUI_DEFAULT_CFG
    if sampler is None:
        sampler = COMFYUI_DEFAULT_SAMPLER
    if scheduler is None:
        scheduler = COMFYUI_DEFAULT_SCHEDULER

    # ---- Determine base seed ----
    base_seed = seed if seed >= 0 else rng.randint(0, 2**32 - 1)

    # Resolve remote (explicit flag, or default from config)
    remote_url = do_resolve_remote(remote) if remote else do_resolve_remote(None)

    all_results: list[dict[str, Any]] = []
    all_saved: list[Path] = []

    if remote_url:
        # ---- Remote mode: HTTP call to tensors server ----
        if not json_output:
            console.print(f"[dim]Remote: {remote_url}[/dim]")

        result = remote_generate(
            remote or remote_url,
            enhanced_prompt,
            negative_prompt=enhanced_negative,
            model=model,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            seed=base_seed,
            sampler=sampler,
            scheduler=scheduler,
            vae=vae,
            lora_name=lora,
            lora_strength=lora_strength,
            console=console,
        )

        if not result:
            if not json_output:
                console.print("[red]Generation failed[/red]")
            raise typer.Exit(1)

        if json_output:
            console.print_json(data=result)
            return

        if not result.get("success"):
            console.print("[red]Generation failed[/red]")
            errors = result.get("errors", {})
            for node_id, err in errors.items():
                console.print(f"  [yellow]Node {node_id}:[/yellow] {err}")
            raise typer.Exit(1)

        images = result.get("images", [])
        console.print(f"[green]Generated {len(images)} image(s)[/green]")
        console.print(f"[dim]Prompt ID: {result.get('prompt_id', 'N/A')}[/dim]")

        # Download and save images if --output specified
        if output and images:
            for i, img_name in enumerate(images):
                img_data = remote_get_image(remote or remote_url, img_name)
                if img_data:
                    save_path = output if len(images) == 1 else output.parent / f"{output.stem}_{i + 1:03d}{output.suffix}"
                    save_path.write_bytes(img_data)
                    console.print(f"[green]Saved:[/green] {save_path}")
                else:
                    console.print(f"[yellow]Could not download image: {img_name}[/yellow]")
        elif images:
            for img_name in images:
                console.print(f"  [dim]{img_name}[/dim]")

    else:
        # ---- Local mode: direct library call ----
        from tensors.comfyui import generate_image, get_image  # noqa: PLC0415

        result_local = generate_image(
            prompt=enhanced_prompt,
            negative_prompt=enhanced_negative,
            model=model,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            seed=base_seed,
            sampler=sampler,
            scheduler=scheduler,
            console=console if not json_output else None,
            lora_name=lora,
            lora_strength=lora_strength,
            batch_size=count,
            vae=vae,
            orientation=orientation,
        )

        if not result_local:
            if json_output:
                all_results.append({"success": False, "index": 0, "errors": {"generation": "Failed to generate"}})
            else:
                console.print("[red]Generation failed[/red]")
                raise typer.Exit(1)
        elif not result_local.success:
            if json_output:
                all_results.append({"success": False, "index": 0, "errors": result_local.node_errors})
            else:
                console.print("[red]Generation failed[/red]")
                for node_id, errors in result_local.node_errors.items():
                    console.print(f"  [yellow]Node {node_id}:[/yellow] {errors}")
                raise typer.Exit(1)
        else:
            # Save all output images
            for i, img_path in enumerate(result_local.images):
                saved_path: Path | None = None
                if output:
                    img_data = get_image(str(img_path))
                    if img_data:
                        save_path = (
                            output
                            if count == 1
                            else output.parent / f"{output.stem}_{i + 1:03d}{output.suffix}"
                        )
                        save_path.write_bytes(img_data)
                        saved_path = save_path
                        all_saved.append(save_path)
                        if not json_output:
                            console.print(f"[green]Saved:[/green] {save_path}")
                    elif not json_output:
                        console.print(f"[yellow]Could not download image: {img_path}[/yellow]")

                all_results.append(
                    {
                        "success": True,
                        "index": i,
                        "prompt_id": result_local.prompt_id,
                        "image": str(img_path),
                        "saved": str(saved_path) if saved_path else None,
                    }
                )

    if json_output:
        console.print_json(
            data={
                "success": all(r.get("success", False) for r in all_results),
                "count": len(all_results),
                "results": all_results,
            }
        )
        return

    console.print("[bold green]Generation complete![/bold green]")
    if count > 1:
        successful = sum(1 for r in all_results if r.get("success", False))
        console.print(f"[dim]Generated {successful}/{count} images[/dim]")
        if all_saved:
            console.print(f"[dim]Saved to: {all_saved[0].parent}/[/dim]")
    elif all_results and all_results[0].get("prompt_id"):
        console.print(f"[dim]Prompt ID: {all_results[0]['prompt_id']}[/dim]")


# =============================================================================
# Template Dump
# =============================================================================


@app.command()
def template(
    model: Annotated[str, typer.Option("-m", "--model", help="Checkpoint model name")],
    lora: Annotated[str | None, typer.Option("-l", "--lora", help="LoRA model name")] = None,
    lora_strength: Annotated[float, typer.Option("--lora-strength", help="LoRA strength")] = 0.8,
    orientation: Annotated[str, typer.Option("-O", "--orientation", help="Resolution: square, portrait, landscape")] = "square",
    rating: Annotated[str | None, typer.Option("--rating", "-R", help="Content rating: safe, questionable, explicit")] = None,
    output: Annotated[Path | None, typer.Option("-o", "--output", help="Save template to file")] = None,
) -> None:
    """Dump a JSON generation template with resolved defaults for a model.

    Outputs a ready-to-use JSON object with all parameters auto-resolved from the
    checkpoint family. Pipe to 'tsr generate --input' or save to a file for reuse.

    Examples:
        tsr template -m ponyDiffusionV6XL_v6StartWithThisOne.safetensors
        tsr template -m beautifulRealistic_v7.safetensors -O portrait
        tsr template -m waiIllustriousSDXL_v160.safetensors -l "Elvira iIlluLoRA.safetensors"
        tsr template -m ponyRealism_V22.safetensors -o pony_preset.json
        tsr generate --input "$(tsr template -m ponyRealism_V22.safetensors)" "a portrait"
    """
    from tensors.config import get_model_generation_defaults, resolve_orientation  # noqa: PLC0415

    # Look up base_model from DB for accurate family detection
    base_model_str: str | None = None
    try:
        with Database() as db:
            db.init_schema()
            base_model_str = db.get_base_model_by_filename(model)
    except Exception:
        pass

    family = detect_model_family(model, base_model_str)
    defaults = get_model_generation_defaults(model, base_model_str)
    res_w, res_h = resolve_orientation(family, orientation)

    # Build template
    tpl: dict[str, Any] = {
        "prompt": "",
        "negative_prompt": defaults.get("negative_prompt", ""),
        "model": model,
        "width": res_w,
        "height": res_h,
        "steps": defaults.get("steps"),
        "cfg": defaults.get("cfg"),
        "sampler": defaults.get("sampler"),
        "scheduler": defaults.get("scheduler"),
        "vae": defaults.get("vae"),
        "orientation": orientation,
        "seed": -1,
        "count": 1,
    }

    # Add quality prefix if the family has one
    quality_prefix = defaults.get("quality_prefix", "")
    if quality_prefix:
        tpl["quality_prefix"] = quality_prefix

    # Add rating tag if specified
    if rating:
        from tensors.config import get_rating_tag  # noqa: PLC0415

        rating_tag = get_rating_tag(family, rating.lower())
        if rating_tag:
            tpl["rating"] = rating
            tpl["rating_tag"] = rating_tag

    # Add LoRA info
    if lora:
        tpl["lora"] = lora
        tpl["lora_strength"] = lora_strength

        # Look up trigger words
        try:
            with Database() as db:
                db.init_schema()
                trigger_words = db.get_trigger_words_by_filename(lora)
                if trigger_words:
                    tpl["lora_triggers"] = trigger_words
        except Exception:
            pass

    # Add metadata (not used by generate, but informational)
    tpl["_family"] = family or "unknown"
    if base_model_str:
        tpl["_base_model"] = base_model_str

    json_str = json.dumps(tpl, indent=2)

    if output:
        output.write_text(json_str + "\n")
        console.print(f"[green]Saved template:[/green] {output}")
    else:
        console.print(json_str)


# =============================================================================
# Top-Level Models Command
# =============================================================================


@app.command()
def models(
    model_type: Annotated[str | None, typer.Option("-t", "--type", help="Filter by type (checkpoints, loras, vae)")] = None,
    remote: Annotated[str | None, typer.Option("-r", "--remote", help="Remote server name or URL")] = None,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """List available models from ComfyUI.

    Shows checkpoints, LoRAs, VAEs, and other model types loaded in ComfyUI.
    Uses --remote to query a remote tensors server instead of local ComfyUI.

    Examples:
        tsr models
        tsr models -t checkpoints
        tsr models --remote junkpile
        tsr models --json
    """
    from tensors.config import resolve_remote as do_resolve_remote  # noqa: PLC0415

    remote_url = do_resolve_remote(remote) if remote else do_resolve_remote(None)

    if remote_url:
        if not json_output:
            console.print(f"[dim]Remote: {remote_url}[/dim]")
        result = remote_models(remote or remote_url, console=console)
    else:
        from tensors.comfyui import get_loaded_models  # noqa: PLC0415

        result = get_loaded_models(console=console if not json_output else None)

    if not result:
        console.print("[red]Error: Could not fetch models[/red]")
        raise typer.Exit(1)

    # Filter by type if requested
    if model_type:
        key = model_type.lower()
        filtered = {k: v for k, v in result.items() if k.lower() == key}
        if not filtered:
            console.print(f"[yellow]No models found for type '{model_type}'[/yellow]")
            console.print(f"[dim]Available types: {', '.join(sorted(result.keys()))}[/dim]")
            raise typer.Exit(1)
        result = filtered

    if json_output:
        console.print_json(data=result)
        return

    console.print("[bold cyan]Available Models[/bold cyan]")

    for mtype, model_list in sorted(result.items()):
        console.print()
        console.print(f"[bold]{mtype}:[/bold] ({len(model_list)})")
        for name in model_list[:MAX_MODEL_LIST_DISPLAY]:
            console.print(f"  {name}")
        if len(model_list) > MAX_MODEL_LIST_DISPLAY:
            console.print(f"  ... and {len(model_list) - MAX_MODEL_LIST_DISPLAY} more")


# =============================================================================
# Database Commands
# =============================================================================

db_app = typer.Typer(
    name="db",
    help="Manage local models database and CivitAI cache.",
    no_args_is_help=True,
)
app.add_typer(db_app, name="db")


@db_app.command("scan")
def db_scan(
    directory: Annotated[Path, typer.Argument(help="Directory to scan for safetensor files")],
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Scan directory for safetensor files and add to database."""
    path = directory.resolve()
    if not path.exists() or not path.is_dir():
        console.print(f"[red]Error: Directory not found: {path}[/red]")
        raise typer.Exit(1)

    with Database() as db:
        db.init_schema()
        console.print(f"[cyan]Scanning {path}...[/cyan]")
        results = db.scan_directory(path, console if not json_output else None)

    if json_output:
        console.print_json(data=results)
    else:
        console.print(f"[green]Scanned {len(results)} file(s)[/green]")
        for f in results:
            console.print(f"  • {f['file_path']}")


@db_app.command("link")
def db_link(
    api_key: Annotated[str | None, typer.Option("--api-key", help="CivitAI API key")] = None,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Link unlinked local files to CivitAI by hash lookup."""
    key = api_key or load_api_key()

    with Database() as db:
        db.init_schema()
        unlinked = db.get_unlinked_files()

        if not unlinked:
            console.print("[green]All files already linked.[/green]")
            return

        console.print(f"[cyan]Found {len(unlinked)} unlinked file(s)[/cyan]")
        linked: list[dict[str, Any]] = []

        for file_info in unlinked:
            sha256 = file_info["sha256"]
            console.print(f"[dim]Looking up {sha256[:16]}...[/dim]")

            civitai_data = fetch_civitai_by_hash(sha256, key, console if not json_output else None)
            if civitai_data:
                version_id: int = civitai_data.get("id", 0)
                model_id: int = civitai_data.get("modelId", 0)
                if version_id and model_id:
                    db.link_file_to_civitai(file_info["id"], model_id, version_id)
                    linked.append(
                        {
                            "file": file_info["file_path"],
                            "model_id": model_id,
                            "version_id": version_id,
                            "name": civitai_data.get("name", ""),
                        }
                    )
                    if not json_output:
                        console.print(f"  [green]✓[/green] {civitai_data.get('name', 'N/A')}")

        if json_output:
            console.print_json(data=linked)
        else:
            console.print(f"[green]Linked {len(linked)} file(s)[/green]")

        # Cache model data for newly linked files
        if linked:
            _cache_linked_models(db, key, linked, json_output)


def _cache_linked_models(
    db: Database,
    api_key: str | None,
    linked: list[dict[str, Any]],
    json_output: bool,
) -> None:
    """Fetch and cache full model data for linked files.

    Args:
        db: Database instance (already initialized)
        api_key: CivitAI API key
        linked: List of linked file info dicts with model_id
        json_output: Whether to suppress console output
    """
    # Collect unique model IDs
    model_ids: set[int] = {item["model_id"] for item in linked if item.get("model_id")}

    # Find which models are not yet cached
    uncached_ids: list[int] = []
    for model_id in model_ids:
        if db.get_model(model_id) is None:
            uncached_ids.append(model_id)

    if not uncached_ids:
        return

    if not json_output:
        console.print(f"[cyan]Caching {len(uncached_ids)} model(s)...[/cyan]")

    cached: list[dict[str, Any]] = []
    for model_id in uncached_ids:
        model_data = fetch_civitai_model(model_id, api_key, console if not json_output else None)
        if model_data:
            db.cache_model(model_data)
            cached.append({"model_id": model_id, "name": model_data.get("name", "")})
            if not json_output:
                console.print(f"  [green]✓[/green] Cached: {model_data.get('name', 'N/A')}")

    if not json_output and cached:
        console.print(f"[green]Cached {len(cached)} model(s)[/green]")


@db_app.command("cache")
def db_cache(
    model_id: Annotated[int, typer.Argument(help="CivitAI model ID to cache")],
    api_key: Annotated[str | None, typer.Option("--api-key", help="CivitAI API key")] = None,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Fetch and cache full CivitAI model data."""
    key = api_key or load_api_key()

    model_data = fetch_civitai_model(model_id, key, console if not json_output else None)
    if not model_data:
        console.print(f"[red]Error: Model {model_id} not found on CivitAI.[/red]")
        raise typer.Exit(1)

    with Database() as db:
        db.init_schema()
        internal_id = db.cache_model(model_data)

    if json_output:
        console.print_json(data={"model_id": model_id, "internal_id": internal_id, "name": model_data.get("name")})
    else:
        console.print(f"[green]Cached:[/green] {model_data.get('name')} (internal ID: {internal_id})")


@db_app.command("list")
def db_list(
    model_type: Annotated[str | None, typer.Option("-t", "--type", help="Filter by model type (Checkpoint, LORA, VAE, etc.)")] = None,
    base: Annotated[str | None, typer.Option("-b", "--base", help="Filter by base model (Pony, Illustrious, SDXL 1.0, SD 1.5, etc.)")] = None,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """List local files with CivitAI info.

    Examples:
        tsr db list                          # All local files
        tsr db list -t Checkpoint            # Only checkpoints
        tsr db list -t LORA                  # Only LoRAs
        tsr db list -t Checkpoint -b Pony    # Pony checkpoints only
        tsr db list -b "SDXL 1.0"           # All SDXL 1.0 models
    """
    with Database() as db:
        db.init_schema()
        files = db.list_local_files()

    # Apply filters (case-insensitive substring match)
    if model_type:
        mt_lower = model_type.lower()
        files = [f for f in files if (f.get("model_type") or "").lower() == mt_lower]
    if base:
        base_lower = base.lower()
        files = [f for f in files if base_lower in (f.get("base_model") or "").lower()]

    if json_output:
        console.print_json(data=files)
        return

    if not files:
        console.print("[yellow]No files found. Try 'tsr db scan' or adjust filters.[/yellow]")
        return

    title = "Local Files"
    if model_type or base:
        parts = []
        if model_type:
            parts.append(model_type)
        if base:
            parts.append(base)
        title = f"Local Files ({', '.join(parts)})"

    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Path", style="cyan", max_width=50)
    table.add_column("Model", style="green")
    table.add_column("Version", style="white")
    table.add_column("Type", style="yellow")
    table.add_column("Base", style="dim")

    for f in files:
        path = Path(f["file_path"]).name
        model = f.get("model_name") or "[dim]unlinked[/dim]"
        version = f.get("version_name") or ""
        ft = f.get("model_type") or ""
        base_model = f.get("base_model") or ""
        table.add_row(path, model, version, ft, base_model)

    console.print(table)


@db_app.command("search")
def db_search(
    query: Annotated[str | None, typer.Argument(help="Search query")] = None,
    model_type: Annotated[str | None, typer.Option("-t", "--type", help="Model type filter")] = None,
    base_model: Annotated[str | None, typer.Option("-b", "--base", help="Base model filter")] = None,
    limit: Annotated[int, typer.Option("-n", "--limit", help="Max results")] = 20,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Search cached models offline."""
    with Database() as db:
        db.init_schema()
        results = db.search_models(query=query, model_type=model_type, base_model=base_model, limit=limit)

    if json_output:
        console.print_json(data=results)
        return

    if not results:
        console.print("[yellow]No models found.[/yellow]")
        return

    table = Table(title="Cached Models", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="yellow")
    table.add_column("Base", style="green")
    table.add_column("Creator", style="dim")
    table.add_column("Downloads", justify="right")

    for m in results:
        table.add_row(
            str(m.get("civitai_id", "")),
            m.get("name", ""),
            m.get("type", ""),
            m.get("base_model", ""),
            m.get("creator", ""),
            str(m.get("download_count", 0)),
        )

    console.print(table)


@db_app.command("triggers")
def db_triggers(
    file: Annotated[Path, typer.Argument(help="Path to safetensor file")],
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Show trigger words for a LoRA file."""
    file_path = file.resolve()
    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)

    with Database() as db:
        db.init_schema()
        triggers = db.get_triggers(str(file_path))

    if json_output:
        console.print_json(data=triggers)
        return

    if not triggers:
        console.print("[yellow]No trigger words found. File may not be linked to CivitAI.[/yellow]")
        console.print("[dim]Run 'tsr db link' to link files to CivitAI.[/dim]")
        return

    console.print(f"[bold]Trigger words for {file_path.name}:[/bold]")
    for word in triggers:
        console.print(f"  • {word}")


@db_app.command("stats")
def db_stats(
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Show database statistics."""
    with Database() as db:
        db.init_schema()
        stats = db.get_stats()

    if json_output:
        console.print_json(data={"db_path": str(DB_PATH), "stats": stats})
        return

    table = Table(title="Database Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Table", style="cyan")
    table.add_column("Count", style="green", justify="right")

    for table_name, count in stats.items():
        table.add_row(table_name, str(count))

    console.print(f"[dim]Database: {DB_PATH}[/dim]")
    console.print(table)


# =============================================================================
# Hugging Face Commands
# =============================================================================

hf_app = typer.Typer(name="hf", help="Hugging Face Hub commands for safetensor files.")
app.add_typer(hf_app)


@hf_app.command("get")
def hf_get(
    model_id: Annotated[str, typer.Argument(help="Model ID (e.g., stabilityai/stable-diffusion-xl-base-1.0)")],
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Get Hugging Face model info and list safetensor files."""
    model = get_hf_model(model_id, console=console)

    if not model:
        raise typer.Exit(1)

    if json_output:
        console.print_json(data=model)
        return

    display_hf_model_info(model, console)


@hf_app.command("files")
def hf_files(
    model_id: Annotated[str, typer.Argument(help="Model ID")],
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """List safetensor files in a Hugging Face model."""
    files = list_safetensor_files(model_id, console=console)

    if json_output:
        console.print_json(data=files)
        return

    if not files:
        console.print("[yellow]No safetensor files found.[/yellow]")
        return

    console.print(f"[bold]Safetensor files in {model_id}:[/bold]")
    for i, f in enumerate(files, 1):
        console.print(f"  {i}. {f}")


@hf_app.command("dl")
def hf_download(
    model_id: Annotated[str, typer.Argument(help="Model ID (e.g., stabilityai/stable-diffusion-xl-base-1.0)")],
    filename: Annotated[str | None, typer.Option("-f", "--file", help="Specific file to download")] = None,
    output: Annotated[Path | None, typer.Option("-o", "--output", help="Output directory")] = None,
    all_files: Annotated[bool, typer.Option("--all", "-a", help="Download all safetensor files")] = False,
) -> None:
    """Download safetensor files from Hugging Face.

    Examples:
        tsr hf dl stabilityai/stable-diffusion-xl-base-1.0 -f sd_xl_base_1.0.safetensors
        tsr hf dl author/model --all
    """
    output_dir = output or Path.cwd()

    if all_files:
        downloaded = download_all_safetensors(model_id, output_dir, console=console)
        if downloaded:
            console.print(f"[green]Downloaded {len(downloaded)} files[/green]")
        else:
            console.print("[red]No files downloaded[/red]")
            raise typer.Exit(1)
        return

    if not filename:
        # List files and prompt or show help
        files = list_safetensor_files(model_id, console=console)
        if not files:
            console.print("[red]No safetensor files found in model[/red]")
            raise typer.Exit(1)

        if len(files) == 1:
            filename = files[0]
            console.print(f"[dim]Downloading only safetensor file: {filename}[/dim]")
        else:
            console.print("[yellow]Multiple safetensor files found. Specify one with -f or use --all:[/yellow]")
            for i, f in enumerate(files, 1):
                console.print(f"  {i}. {f}")
            raise typer.Exit(1)

    result = download_hf_safetensor(model_id, filename, output_dir, console=console)
    if not result:
        raise typer.Exit(1)


# =============================================================================
# ComfyUI Commands
# =============================================================================

comfy_app = typer.Typer(name="comfy", help="ComfyUI integration for image generation.")
app.add_typer(comfy_app)


@comfy_app.command("status")
def comfy_status(
    url: Annotated[str | None, typer.Option("--url", "-u", help="ComfyUI server URL")] = None,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Show ComfyUI system status (GPU, RAM, queue)."""
    from tensors.comfyui import get_queue_status, get_system_stats  # noqa: PLC0415

    stats = get_system_stats(url=url, console=console if not json_output else None)
    if not stats:
        console.print("[red]Error: Could not connect to ComfyUI[/red]")
        raise typer.Exit(1)

    queue = get_queue_status(url=url)

    if json_output:
        output = {"system": stats, "queue": queue}
        console.print_json(data=output)
        return

    # Display system stats
    console.print("[bold cyan]ComfyUI System Status[/bold cyan]")
    console.print()

    # System info
    system_info = stats.get("system", {})
    console.print(f"[bold]OS:[/bold] {system_info.get('os', 'N/A')}")
    console.print(f"[bold]Python:[/bold] {system_info.get('python_version', 'N/A')}")
    console.print(f"[bold]PyTorch:[/bold] {system_info.get('pytorch_version', 'N/A')}")

    # GPU info
    devices = stats.get("devices", [])
    if devices:
        console.print()
        console.print("[bold]GPU Devices:[/bold]")
        for i, device in enumerate(devices):
            name = device.get("name", "Unknown")
            vram_total = device.get("vram_total", 0)
            vram_free = device.get("vram_free", 0)
            vram_used = vram_total - vram_free
            vram_pct = (vram_used / vram_total * 100) if vram_total > 0 else 0
            console.print(f"  [{i}] {name}")
            console.print(f"      VRAM: {vram_used / 1024**3:.1f} / {vram_total / 1024**3:.1f} GB ({vram_pct:.0f}%)")

    # Queue info
    if queue:
        running = len(queue.get("queue_running", []))
        pending = len(queue.get("queue_pending", []))
        console.print()
        console.print(f"[bold]Queue:[/bold] {running} running, {pending} pending")


@comfy_app.command("queue")
def comfy_queue(
    url: Annotated[str | None, typer.Option("--url", "-u", help="ComfyUI server URL")] = None,
    clear: Annotated[bool, typer.Option("--clear", "-c", help="Clear the queue")] = False,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Show or clear the ComfyUI queue."""
    from tensors.comfyui import clear_queue as do_clear_queue  # noqa: PLC0415
    from tensors.comfyui import get_queue_status  # noqa: PLC0415

    if clear:
        success = do_clear_queue(url=url, console=console)
        if not success:
            raise typer.Exit(1)
        return

    queue = get_queue_status(url=url, console=console if not json_output else None)
    if not queue:
        console.print("[red]Error: Could not connect to ComfyUI[/red]")
        raise typer.Exit(1)

    if json_output:
        console.print_json(data=queue)
        return

    running = queue.get("queue_running", [])
    pending = queue.get("queue_pending", [])

    console.print("[bold cyan]ComfyUI Queue[/bold cyan]")
    console.print()
    console.print(f"[bold]Running:[/bold] {len(running)}")
    console.print(f"[bold]Pending:[/bold] {len(pending)}")

    if running:
        console.print()
        console.print("[bold]Running Jobs:[/bold]")
        for job in running:
            prompt_id = job[1] if len(job) > 1 else "unknown"
            console.print(f"  • {prompt_id}")

    if pending:
        console.print()
        console.print("[bold]Pending Jobs:[/bold]")
        for job in pending[:MAX_QUEUE_DISPLAY]:
            prompt_id = job[1] if len(job) > 1 else "unknown"
            console.print(f"  • {prompt_id}")
        if len(pending) > MAX_QUEUE_DISPLAY:
            console.print(f"  ... and {len(pending) - MAX_QUEUE_DISPLAY} more")


@comfy_app.command("models")
def comfy_models(
    url: Annotated[str | None, typer.Option("--url", "-u", help="ComfyUI server URL")] = None,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """List available models in ComfyUI."""
    from tensors.comfyui import get_loaded_models  # noqa: PLC0415

    models = get_loaded_models(url=url, console=console if not json_output else None)
    if not models:
        console.print("[red]Error: Could not fetch models from ComfyUI[/red]")
        raise typer.Exit(1)

    if json_output:
        console.print_json(data=models)
        return

    console.print("[bold cyan]ComfyUI Available Models[/bold cyan]")

    for model_type, model_list in sorted(models.items()):
        console.print()
        console.print(f"[bold]{model_type}:[/bold] ({len(model_list)})")
        for name in model_list[:MAX_MODEL_LIST_DISPLAY]:
            console.print(f"  • {name}")
        if len(model_list) > MAX_MODEL_LIST_DISPLAY:
            console.print(f"  ... and {len(model_list) - MAX_MODEL_LIST_DISPLAY} more")


@comfy_app.command("history")
def comfy_history(
    prompt_id: Annotated[str | None, typer.Argument(help="Specific prompt ID to view")] = None,
    url: Annotated[str | None, typer.Option("--url", "-u", help="ComfyUI server URL")] = None,
    limit: Annotated[int, typer.Option("-n", "--limit", help="Max history items")] = 20,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """View ComfyUI generation history."""
    from tensors.comfyui import get_history  # noqa: PLC0415

    history = get_history(url=url, prompt_id=prompt_id, max_items=limit, console=console if not json_output else None)
    if history is None:
        console.print("[red]Error: Could not fetch history from ComfyUI[/red]")
        raise typer.Exit(1)

    if json_output:
        console.print_json(data=history)
        return

    if not history:
        console.print("[yellow]No history found.[/yellow]")
        return

    if prompt_id:
        # Show single entry details
        if prompt_id not in history:
            console.print(f"[yellow]Prompt {prompt_id} not found in history.[/yellow]")
            return

        entry = history[prompt_id]
        console.print(f"[bold cyan]Prompt: {prompt_id}[/bold cyan]")
        console.print()

        status = entry.get("status", {})
        console.print(f"[bold]Status:[/bold] {status.get('status_str', 'unknown')}")

        outputs = entry.get("outputs", {})
        if outputs:
            console.print()
            console.print("[bold]Outputs:[/bold]")
            for node_id, output in outputs.items():
                if "images" in output:
                    for img in output["images"]:
                        console.print(f"  [{node_id}] {img.get('filename', 'unknown')}")
    else:
        # Show list of history entries
        console.print("[bold cyan]ComfyUI History[/bold cyan]")
        console.print()

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Prompt ID", style="cyan", max_width=40)
        table.add_column("Status", style="green")
        table.add_column("Images", justify="right")

        for pid, entry in list(history.items())[:limit]:
            status = entry.get("status", {}).get("status_str", "unknown")
            outputs = entry.get("outputs", {})
            image_count = sum(len(o.get("images", [])) for o in outputs.values())
            display_pid = pid[:MAX_PROMPT_ID_DISPLAY] + "..." if len(pid) > MAX_PROMPT_ID_DISPLAY else pid
            table.add_row(display_pid, status, str(image_count))

        console.print(table)


@comfy_app.command("generate", deprecated=True)
def comfy_generate(
    prompt: Annotated[str, typer.Argument(help="Positive prompt text")],
    model: Annotated[str | None, typer.Option("-m", "--model", help="Checkpoint model name")] = None,
    negative: Annotated[str, typer.Option("-n", "--negative", help="Negative prompt")] = "",
    width: Annotated[int | None, typer.Option("-W", "--width")] = None,
    height: Annotated[int | None, typer.Option("-H", "--height")] = None,
    steps: Annotated[int | None, typer.Option("--steps")] = None,
    cfg: Annotated[float | None, typer.Option("--cfg")] = None,
    seed: Annotated[int, typer.Option("--seed", "-s")] = -1,
    sampler: Annotated[str | None, typer.Option("--sampler")] = None,
    scheduler: Annotated[str | None, typer.Option("--scheduler")] = None,
    orientation: Annotated[str, typer.Option("-O", "--orientation")] = "square",
    output: Annotated[Path | None, typer.Option("-o", "--output")] = None,
    count: Annotated[int, typer.Option("-c", "--count")] = 1,
    lora: Annotated[str | None, typer.Option("-l", "--lora")] = None,
    lora_strength: Annotated[float, typer.Option("--lora-strength")] = 0.8,
    no_quality: Annotated[bool, typer.Option("--no-quality")] = False,
    no_negative: Annotated[bool, typer.Option("--no-negative")] = False,
    json_output: Annotated[bool, typer.Option("--json", "-j")] = False,
) -> None:
    """[Deprecated] Use 'tsr generate' instead. All features have been merged into the top-level command."""
    console.print("[yellow]Warning: 'tsr comfy generate' is deprecated. Use 'tsr generate' instead.[/yellow]")
    # Delegate to the unified generate command via context invocation
    ctx = typer.Context(generate)
    generate(
        ctx=ctx,
        prompt=prompt,
        model=model,
        width=width,
        height=height,
        steps=steps,
        cfg=cfg,
        seed=seed,
        sampler=sampler,
        scheduler=scheduler,
        vae=None,
        orientation=orientation,
        lora=lora,
        lora_strength=lora_strength,
        negative=negative,
        count=count,
        no_quality=no_quality,
        no_negative=no_negative,
        output=output,
        remote=None,
        json_output=json_output,
        json_input=None,
    )


@comfy_app.command("run")
def comfy_run(
    workflow_file: Annotated[Path, typer.Argument(help="Path to workflow JSON file")],
    url: Annotated[str | None, typer.Option("--url", "-u", help="ComfyUI server URL")] = None,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Run an arbitrary ComfyUI workflow from a JSON file.

    The workflow should be in ComfyUI API format (exported via "Save (API Format)").
    """
    from tensors.comfyui import run_workflow  # noqa: PLC0415

    if not workflow_file.exists():
        console.print(f"[red]Error: Workflow file not found: {workflow_file}[/red]")
        raise typer.Exit(1)

    result = run_workflow(
        workflow=workflow_file,
        url=url,
        console=console if not json_output else None,
    )

    if not result:
        console.print("[red]Failed to queue workflow[/red]")
        raise typer.Exit(1)

    if not result.success:
        if json_output:
            console.print_json(data={"success": False, "prompt_id": result.prompt_id, "errors": result.node_errors})
        else:
            console.print("[red]Workflow execution failed[/red]")
            for node_id, errors in result.node_errors.items():
                console.print(f"  [yellow]Node {node_id}:[/yellow] {errors}")
        raise typer.Exit(1)

    if json_output:
        console.print_json(data={"success": True, "prompt_id": result.prompt_id, "outputs": result.outputs})
        return

    console.print("[bold green]Workflow complete![/bold green]")
    console.print(f"[dim]Prompt ID: {result.prompt_id}[/dim]")

    # Show output images
    for _node_id, output in result.outputs.items():
        if "images" in output:
            for img in output["images"]:
                console.print(f"  [green]Image:[/green] {img.get('filename', 'unknown')}")


def main() -> int:
    """Main entry point."""
    # Handle legacy invocation: tsr <file.safetensors> -> tsr info <file>
    known_commands = (
        "info",
        "search",
        "get",
        "dl",
        "download",
        "generate",
        "models",
        "config",
        "serve",
        "db",
        "hf",
        "comfy",
    )
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        arg = sys.argv[1]
        if arg not in known_commands and (arg.endswith(".safetensors") or arg.endswith(".sft") or Path(arg).exists()):
            sys.argv = [sys.argv[0], "info", *sys.argv[1:]]

    app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
