"""CLI application and commands for tsr."""

from __future__ import annotations

import json
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any

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
from tensors.client import TsrClient, TsrClientError
from tensors.config import (
    CONFIG_FILE,
    BaseModel,
    ModelType,
    SortOrder,
    get_default_output_path,
    get_remotes,
    load_api_key,
    load_config,
    resolve_remote,
    save_config,
    save_remote,
    set_default_remote,
)
from tensors.db import DB_PATH, Database
from tensors.display import (
    _format_size,
    display_civitai_data,
    display_file_info,
    display_local_metadata,
    display_model_info,
    display_search_results,
)
from tensors.safetensor import compute_sha256, get_base_name, read_safetensor_metadata

# Key masking threshold
MIN_KEY_LENGTH_FOR_MASKING = 8


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
    model_type: Annotated[ModelType | None, typer.Option("-t", "--type", help="Model type filter")] = None,
    base: Annotated[BaseModel | None, typer.Option("-b", "--base", help="Base model filter")] = None,
    sort: Annotated[SortOrder, typer.Option("-s", "--sort", help="Sort order")] = SortOrder.downloads,
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
        console=console,
    )

    if not results:
        console.print("[red]Search failed.[/red]")
        raise typer.Exit(1)

    if json_output:
        console.print_json(data=results)
    else:
        display_search_results(results, console)


@app.command()
def get(
    id_value: Annotated[int, typer.Argument(help="CivitAI model ID or version ID")],
    version: Annotated[bool, typer.Option("-v", "--version", help="Treat ID as version ID instead of model ID")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key", help="CivitAI API key")] = None,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Fetch model information from CivitAI by model ID or version ID."""
    key = api_key or load_api_key()

    if version:
        version_data = fetch_civitai_model_version(id_value, key, console)
        if not version_data:
            console.print(f"[red]Error: Version {id_value} not found on CivitAI.[/red]")
            raise typer.Exit(1)

        if json_output:
            console.print_json(data=version_data)
        else:
            display_civitai_data(version_data, console)
    else:
        model_data = fetch_civitai_model(id_value, key, console)
        if not model_data:
            console.print(f"[red]Error: Model {id_value} not found on CivitAI.[/red]")
            raise typer.Exit(1)

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


@app.command("dl")
def download(
    version_id: Annotated[int | None, typer.Option("-v", "--version-id", help="Model version ID")] = None,
    model_id: Annotated[int | None, typer.Option("-m", "--model-id", help="Model ID (downloads latest)")] = None,
    hash_val: Annotated[str | None, typer.Option("-H", "--hash", help="SHA256 hash to look up")] = None,
    output: Annotated[Path | None, typer.Option("-o", "--output", help="Output directory")] = None,
    no_resume: Annotated[bool, typer.Option("--no-resume", help="Don't resume partial downloads")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key", help="CivitAI API key")] = None,
    remote: Annotated[str | None, typer.Option("-r", "--remote", help="Remote server name or URL")] = None,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON (remote mode)")] = False,
) -> None:
    """Download a model from CivitAI (locally or to remote server)."""
    # Check if remote is specified or configured
    remote_url = resolve_remote(remote)

    if remote_url:
        # Remote mode: use TsrClient API
        if not version_id and not model_id and not hash_val:
            console.print("[red]Error: Must specify --version-id, --model-id, or --hash[/red]")
            raise typer.Exit(1)

        try:
            with TsrClient(remote_url) as client:
                console.print(f"[cyan]Starting download on {remote_url}...[/cyan]")
                result = client.start_download(
                    version_id=version_id,
                    model_id=model_id,
                    hash_val=hash_val,
                    output_dir=str(output) if output else None,
                )
        except TsrClientError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1) from e

        if json_output:
            console.print_json(data=result)
            return

        download_id = result.get("download_id")
        console.print(f"[green]Download started:[/green] {download_id}")
        console.print(f"[dim]Check status with: tsr images download-status {download_id} --remote {remote or 'default'}[/dim]")
    else:
        # Local mode: direct download
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
            masked = key[:4] + "..." + key[-4:] if len(key) > MIN_KEY_LENGTH_FOR_MASKING else "***"
            console.print(f"[bold]API key:[/bold] {masked}")
        else:
            console.print("[bold]API key:[/bold] [yellow]Not set[/yellow]")

        console.print()
        console.print("[dim]Set API key with: tsr config --set-key YOUR_KEY[/dim]")


@app.command()
def generate(
    prompt: Annotated[str, typer.Argument(help="Text prompt for image generation.")],
    remote: Annotated[str | None, typer.Option("-r", "--remote", help="Remote server name or URL")] = None,
    host: Annotated[str, typer.Option(help="sd-server address (local mode).")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="sd-server port (local mode).")] = 8080,
    output: Annotated[str, typer.Option("-o", help="Output directory (local mode).")] = ".",
    negative_prompt: Annotated[str, typer.Option("-n", help="Negative prompt.")] = "",
    width: Annotated[int, typer.Option("-W", help="Image width.")] = 512,
    height: Annotated[int, typer.Option("-H", help="Image height.")] = 512,
    steps: Annotated[int, typer.Option(help="Sampling steps.")] = 20,
    cfg_scale: Annotated[float, typer.Option(help="CFG scale.")] = 7.0,
    seed: Annotated[int, typer.Option("-s", help="RNG seed (-1 for random).")] = -1,
    sampler: Annotated[str, typer.Option(help="Sampler name.")] = "",
    scheduler: Annotated[str, typer.Option(help="Scheduler name.")] = "",
    batch_size: Annotated[int, typer.Option("-b", help="Number of images.")] = 1,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON (remote mode)")] = False,
) -> None:
    """Generate images using sd-server (local or remote)."""
    # Check if remote is specified or configured
    remote_url = resolve_remote(remote)

    if remote_url:
        # Remote mode: use TsrClient API
        try:
            with TsrClient(remote_url) as client:
                console.print(f"[cyan]Generating {batch_size} image(s) on {remote_url}...[/cyan]")
                result = client.generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    seed=seed,
                    sampler_name=sampler,
                    scheduler=scheduler,
                    batch_size=batch_size,
                )
        except TsrClientError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1) from e

        if json_output:
            console.print_json(data=result)
            return

        images = result.get("images", [])
        for img in images:
            console.print(f"[green]Generated:[/green] {img.get('id', 'unknown')}")
    else:
        # Local mode: direct sd-server connection
        from tensors.generate import SDClient, Txt2ImgParams, save_images  # noqa: PLC0415

        params = Txt2ImgParams(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            batch_size=batch_size,
            sampler_name=sampler,
            scheduler=scheduler,
        )

        with SDClient(host=host, port=port) as client:
            console.print(f"[cyan]Generating {batch_size} image(s)...[/cyan]")
            images = client.generate.txt2img(params)
            paths = save_images(images, output)
            for p in paths:
                console.print(f"[green]Saved:[/green] {p}")


@app.command()
def status(
    remote: Annotated[str | None, typer.Option("-r", "--remote", help="Remote server name or URL")] = None,
    host: Annotated[str, typer.Option(help="Wrapper API host (local mode).")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Wrapper API port (local mode).")] = 8080,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Show sd-server wrapper status."""
    # Check if remote is specified or configured
    remote_url = resolve_remote(remote)

    if remote_url:
        # Remote mode: use TsrClient API
        try:
            with TsrClient(remote_url) as client:
                data = client.status()
        except TsrClientError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1) from e
    else:
        # Local mode: direct HTTP call
        import httpx  # noqa: PLC0415

        url = f"http://{host}:{port}/status"
        try:
            resp = httpx.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            console.print(f"[red]Error: Could not reach wrapper at {url}: {e}[/red]")
            raise typer.Exit(1) from e

    if json_output:
        console.print_json(data=data)
        return

    table = Table(title="Server Status", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    for key, value in data.items():
        table.add_row(key, str(value))
    console.print(table)


@app.command()
def reload(
    model: Annotated[str, typer.Option(help="Path to model file for sd-server.")],
    host: Annotated[str, typer.Option(help="Wrapper API host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Wrapper API port.")] = 8080,
) -> None:
    """Reload sd-server with a new model."""
    import httpx  # noqa: PLC0415

    url = f"http://{host}:{port}/reload"
    console.print(f"[cyan]Reloading model: {model}[/cyan]")
    try:
        resp = httpx.post(url, json={"model": model}, timeout=300)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        console.print(f"[red]Error: Reload failed at {url}: {e}[/red]")
        raise typer.Exit(1) from e

    console.print(f"[green]{data.get('status', 'OK')}[/green]")


@app.command()
def serve(
    model: Annotated[str, typer.Option(help="Path to model file for sd-server.")],
    host: Annotated[str, typer.Option(help="Wrapper API listen address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Wrapper API listen port.")] = 8080,
    sd_port: Annotated[int, typer.Option(help="sd-server listen port.")] = 1234,
    log_level: Annotated[str, typer.Option(help="Log level.")] = "info",
) -> None:
    """Start the sd-server wrapper API (transparent proxy with hot reload)."""
    try:
        import uvicorn  # noqa: PLC0415

        from tensors.server import ServerConfig, create_app  # noqa: PLC0415
    except ImportError:
        console.print("[red]Missing server dependencies. Install with:[/red]")
        console.print("  pip install tensors[server]")
        raise typer.Exit(1) from None

    config = ServerConfig(model=model, port=sd_port)
    uvicorn.run(create_app(config), host=host, port=port, log_level=log_level)


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
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """List local files with CivitAI info."""
    with Database() as db:
        db.init_schema()
        files = db.list_local_files()

    if json_output:
        console.print_json(data=files)
        return

    if not files:
        console.print("[yellow]No files in database. Run 'tsr db scan' first.[/yellow]")
        return

    table = Table(title="Local Files", show_header=True, header_style="bold magenta")
    table.add_column("Path", style="cyan", max_width=50)
    table.add_column("Model", style="green")
    table.add_column("Version", style="white")
    table.add_column("Type", style="yellow")
    table.add_column("Base", style="dim")

    for f in files:
        path = Path(f["file_path"]).name
        model = f.get("model_name") or "[dim]unlinked[/dim]"
        version = f.get("version_name") or ""
        model_type = f.get("model_type") or ""
        base = f.get("base_model") or ""
        table.add_row(path, model, version, model_type, base)

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
# Images Commands (Remote)
# =============================================================================

images_app = typer.Typer(
    name="images",
    help="Manage images in remote gallery.",
    no_args_is_help=True,
)
app.add_typer(images_app, name="images")


def _get_client(remote: str | None) -> TsrClient:
    """Get TsrClient for remote or raise error."""
    url = resolve_remote(remote)
    if not url:
        console.print("[red]Error: No remote specified. Use --remote or set default_remote in config.[/red]")
        raise typer.Exit(1)
    return TsrClient(url)


@images_app.command("list")
def images_list(
    remote: Annotated[str | None, typer.Option("-r", "--remote", help="Remote server name or URL")] = None,
    limit: Annotated[int, typer.Option("-n", "--limit", help="Max results")] = 50,
    offset: Annotated[int, typer.Option("--offset", help="Offset for pagination")] = 0,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """List images in remote gallery."""
    try:
        with _get_client(remote) as client:
            result = client.list_images(limit=limit, offset=offset)
    except TsrClientError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e

    images = result.get("images", [])
    total = result.get("total", len(images))

    if json_output:
        console.print_json(data=result)
        return

    if not images:
        console.print("[yellow]No images in gallery.[/yellow]")
        return

    table = Table(title=f"Gallery Images ({len(images)}/{total})", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan")
    table.add_column("Filename", style="green")
    table.add_column("Size", style="white")
    table.add_column("Created", style="dim")

    for img in images:
        size = f"{img.get('width', '?')}x{img.get('height', '?')}"
        created = img.get("created_at", "")
        if isinstance(created, (int, float)):
            from datetime import datetime  # noqa: PLC0415

            created = datetime.fromtimestamp(created).strftime("%Y-%m-%d %H:%M")
        table.add_row(img.get("id", ""), img.get("filename", ""), size, str(created))

    console.print(table)


@images_app.command("show")
def images_show(
    image_id: Annotated[str, typer.Argument(help="Image ID to show")],
    remote: Annotated[str | None, typer.Option("-r", "--remote", help="Remote server name or URL")] = None,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Show image metadata."""
    try:
        with _get_client(remote) as client:
            meta = client.get_image_meta(image_id)
    except TsrClientError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e

    if json_output:
        console.print_json(data=meta)
        return

    table = Table(title=f"Image: {image_id}", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    for key, value in meta.items():
        display_value = json.dumps(value, indent=2) if isinstance(value, dict) else str(value)
        table.add_row(key, display_value)

    console.print(table)


@images_app.command("delete")
def images_delete(
    image_id: Annotated[str, typer.Argument(help="Image ID to delete")],
    remote: Annotated[str | None, typer.Option("-r", "--remote", help="Remote server name or URL")] = None,
    force: Annotated[bool, typer.Option("-f", "--force", help="Skip confirmation")] = False,
) -> None:
    """Delete an image from the gallery."""
    if not force:
        confirm = typer.confirm(f"Delete image {image_id}?")
        if not confirm:
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)

    try:
        with _get_client(remote) as client:
            client.delete_image(image_id)
    except TsrClientError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e

    console.print(f"[green]Deleted image: {image_id}[/green]")


@images_app.command("download")
def images_download(
    image_id: Annotated[str, typer.Argument(help="Image ID to download")],
    remote: Annotated[str | None, typer.Option("-r", "--remote", help="Remote server name or URL")] = None,
    output: Annotated[Path | None, typer.Option("-o", "--output", help="Output file or directory")] = None,
) -> None:
    """Download an image from the remote gallery."""
    try:
        with _get_client(remote) as client:
            content = client.download_image(image_id)
    except TsrClientError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e

    # Determine output path
    if output is None:
        dest = Path(f"{image_id}.png")
    elif output.is_dir():
        dest = output / f"{image_id}.png"
    else:
        dest = output

    dest.write_bytes(content)
    console.print(f"[green]Saved:[/green] {dest}")


# =============================================================================
# Models Commands (Remote)
# =============================================================================

models_app = typer.Typer(
    name="models",
    help="Manage models on remote server.",
    no_args_is_help=True,
)
app.add_typer(models_app, name="models")


@models_app.command("list")
def models_list(
    remote: Annotated[str | None, typer.Option("-r", "--remote", help="Remote server name or URL")] = None,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """List available models on remote server."""
    try:
        with _get_client(remote) as client:
            result = client.list_models()
    except TsrClientError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e

    if json_output:
        console.print_json(data=result)
        return

    models = result.get("models", [])
    active = result.get("active", "")

    if not models:
        console.print("[yellow]No models found.[/yellow]")
        return

    table = Table(title="Available Models", show_header=True, header_style="bold magenta")
    table.add_column("Status", style="dim", width=3)
    table.add_column("Name", style="cyan")
    table.add_column("Path", style="dim")

    for model in models:
        path = model.get("path", "")
        name = model.get("name", Path(path).stem if path else "")
        is_active = active in {path, name}
        status = "[green]✓[/green]" if is_active else ""
        table.add_row(status, name, path)

    console.print(table)


@models_app.command("active")
def models_active(
    remote: Annotated[str | None, typer.Option("-r", "--remote", help="Remote server name or URL")] = None,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Show currently active model."""
    try:
        with _get_client(remote) as client:
            result = client.get_active_model()
    except TsrClientError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e

    if json_output:
        console.print_json(data=result)
        return

    model = result.get("model", "None")
    console.print(f"[bold]Active model:[/bold] {model}")


@models_app.command("switch")
def models_switch(
    model: Annotated[str, typer.Argument(help="Model path or name to switch to")],
    remote: Annotated[str | None, typer.Option("-r", "--remote", help="Remote server name or URL")] = None,
) -> None:
    """Switch to a different model on the remote server."""
    console.print(f"[cyan]Switching to model: {model}[/cyan]")
    try:
        with _get_client(remote) as client:
            result = client.switch_model(model)
    except TsrClientError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e

    console.print(f"[green]{result.get('status', 'OK')}[/green]")


@models_app.command("loras")
def models_loras(
    remote: Annotated[str | None, typer.Option("-r", "--remote", help="Remote server name or URL")] = None,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """List available LoRAs on remote server."""
    try:
        with _get_client(remote) as client:
            result = client.list_loras()
    except TsrClientError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e

    if json_output:
        console.print_json(data=result)
        return

    loras = result.get("loras", [])
    if not loras:
        console.print("[yellow]No LoRAs found.[/yellow]")
        return

    table = Table(title="Available LoRAs", show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan")
    table.add_column("Path", style="dim")

    for lora in loras:
        path = lora.get("path", "")
        name = lora.get("name", Path(path).stem if path else "")
        table.add_row(name, path)

    console.print(table)


# =============================================================================
# Remote Configuration Commands
# =============================================================================

remote_app = typer.Typer(
    name="remote",
    help="Manage remote server configuration.",
    no_args_is_help=True,
)
app.add_typer(remote_app, name="remote")


@remote_app.command("list")
def remote_list(
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """List configured remotes."""
    from tensors.config import get_default_remote  # noqa: PLC0415

    remotes = get_remotes()
    default = get_default_remote()

    if json_output:
        console.print_json(data={"remotes": remotes, "default": default})
        return

    if not remotes:
        console.print("[yellow]No remotes configured.[/yellow]")
        console.print("[dim]Add one with: tsr remote add NAME URL[/dim]")
        return

    table = Table(title="Configured Remotes", show_header=True, header_style="bold magenta")
    table.add_column("Default", style="dim", width=3)
    table.add_column("Name", style="cyan")
    table.add_column("URL", style="green")

    for name, url in remotes.items():
        is_default = name == default
        status = "[green]✓[/green]" if is_default else ""
        table.add_row(status, name, url)

    console.print(table)


@remote_app.command("add")
def remote_add(
    name: Annotated[str, typer.Argument(help="Remote name")],
    url: Annotated[str, typer.Argument(help="Remote URL (e.g., http://host:8080)")],
) -> None:
    """Add a remote server."""
    save_remote(name, url)
    console.print(f"[green]Added remote:[/green] {name} → {url}")


@remote_app.command("default")
def remote_default(
    name: Annotated[str | None, typer.Argument(help="Remote name to set as default (omit to clear)")] = None,
) -> None:
    """Set or clear the default remote."""
    set_default_remote(name)
    if name:
        console.print(f"[green]Default remote set to:[/green] {name}")
    else:
        console.print("[green]Default remote cleared.[/green]")


def main() -> int:
    """Main entry point."""
    # Handle legacy invocation: tsr <file.safetensors> -> tsr info <file>
    known_commands = (
        "info",
        "search",
        "get",
        "dl",
        "download",
        "config",
        "generate",
        "serve",
        "status",
        "reload",
        "db",
        "images",
        "models",
        "remote",
    )
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        arg = sys.argv[1]
        if arg not in known_commands and (arg.endswith(".safetensors") or arg.endswith(".sft") or Path(arg).exists()):
            sys.argv = [sys.argv[0], "info", *sys.argv[1:]]

    app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
