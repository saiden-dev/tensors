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
from tensors.config import (
    CONFIG_FILE,
    BaseModel,
    ModelType,
    SortOrder,
    get_default_output_path,
    load_api_key,
    load_config,
    save_config,
)
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
) -> None:
    """Download a model from CivitAI."""
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
    host: Annotated[str, typer.Option(help="sd-server address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="sd-server port.")] = 1234,
    output: Annotated[str, typer.Option("-o", help="Output directory.")] = ".",
    negative_prompt: Annotated[str, typer.Option("-n", help="Negative prompt.")] = "",
    width: Annotated[int, typer.Option("-W", help="Image width.")] = 512,
    height: Annotated[int, typer.Option("-H", help="Image height.")] = 512,
    steps: Annotated[int, typer.Option(help="Sampling steps.")] = 20,
    cfg_scale: Annotated[float, typer.Option(help="CFG scale.")] = 7.0,
    seed: Annotated[int, typer.Option("-s", help="RNG seed (-1 for random).")] = -1,
    sampler: Annotated[str, typer.Option(help="Sampler name.")] = "",
    scheduler: Annotated[str, typer.Option(help="Scheduler name.")] = "",
    batch_size: Annotated[int, typer.Option("-b", help="Number of images.")] = 1,
) -> None:
    """Generate images using a running sd-server."""
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


def main() -> int:
    """Main entry point."""
    # Handle legacy invocation: tsr <file.safetensors> -> tsr info <file>
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        arg = sys.argv[1]
        if arg not in ("info", "search", "get", "dl", "download", "config", "generate", "serve") and (
            arg.endswith(".safetensors") or arg.endswith(".sft") or Path(arg).exists()
        ):
            sys.argv = [sys.argv[0], "info", *sys.argv[1:]]

    app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
