"""CivitAI API functions."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import httpx
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

from tensors.config import (
    CIVITAI_API_BASE,
    CIVITAI_DOWNLOAD_BASE,
    BaseModel,
    CommercialUse,
    ModelType,
    NsfwLevel,
    Period,
    SortOrder,
)

if TYPE_CHECKING:
    from rich.console import Console

# Progress update throttle interval (seconds)
_PROGRESS_UPDATE_INTERVAL = 0.25


def _get_headers(api_key: str | None) -> dict[str, str]:
    """Get headers for CivitAI API requests."""
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def fetch_civitai_model_version(version_id: int, api_key: str | None, console: Console | None = None) -> dict[str, Any] | None:
    """Fetch model version information from CivitAI by version ID."""
    url = f"{CIVITAI_API_BASE}/model-versions/{version_id}"

    try:
        response = httpx.get(url, headers=_get_headers(api_key), timeout=30.0)
        if response.status_code == HTTPStatus.NOT_FOUND:
            return None
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result
    except httpx.HTTPStatusError as e:
        if console:
            console.print(f"[red]API error: {e.response.status_code}[/red]")
        return None
    except httpx.RequestError as e:
        if console:
            console.print(f"[red]Request error: {e}[/red]")
        return None


def fetch_civitai_model(model_id: int, api_key: str | None, console: Console | None = None) -> dict[str, Any] | None:
    """Fetch model information from CivitAI by model ID."""
    url = f"{CIVITAI_API_BASE}/models/{model_id}"

    def _do_fetch() -> dict[str, Any] | None:
        try:
            response = httpx.get(url, headers=_get_headers(api_key), timeout=30.0)
            if response.status_code == HTTPStatus.NOT_FOUND:
                return None
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
        except httpx.HTTPStatusError as e:
            if console:
                console.print(f"[red]API error: {e.response.status_code}[/red]")
            return None
        except httpx.RequestError as e:
            if console:
                console.print(f"[red]Request error: {e}[/red]")
            return None

    if console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("[cyan]Fetching model from CivitAI...", total=None)
            return _do_fetch()
    else:
        return _do_fetch()


def fetch_civitai_by_hash(sha256_hash: str, api_key: str | None, console: Console | None = None) -> dict[str, Any] | None:
    """Fetch model information from CivitAI by SHA256 hash."""
    url = f"{CIVITAI_API_BASE}/model-versions/by-hash/{sha256_hash}"

    def _do_fetch() -> dict[str, Any] | None:
        try:
            response = httpx.get(url, headers=_get_headers(api_key), timeout=30.0)
            if response.status_code == HTTPStatus.NOT_FOUND:
                return None
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
        except httpx.HTTPStatusError as e:
            if console:
                console.print(f"[red]API error: {e.response.status_code}[/red]")
            return None
        except httpx.RequestError as e:
            if console:
                console.print(f"[red]Request error: {e}[/red]")
            return None

    if console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("[cyan]Fetching from CivitAI...", total=None)
            return _do_fetch()
    else:
        return _do_fetch()


def _build_search_params(
    query: str | None,
    model_type: ModelType | None,
    base_model: BaseModel | None,
    sort: SortOrder,
    limit: int,
    *,
    period: Period | None = None,
    nsfw: NsfwLevel | bool | None = None,
    tag: str | None = None,
    username: str | None = None,
    page: int | None = None,
    commercial_use: CommercialUse | None = None,
    allow_derivatives: bool | None = None,
    primary_file_only: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Build search parameters and return (params, has_filters).

    API Quirks / Workarounds:
    - query + filters don't work reliably together → we fetch more and filter client-side
    - nsfw=true is required to include NSFW content (default excludes it)
    - baseModels is undocumented but works
    """
    params: dict[str, Any] = {
        "limit": min(limit, 100),
    }

    # NSFW handling - default to including all content
    if nsfw is None:
        params["nsfw"] = "true"  # Include NSFW by default (like website)
    elif isinstance(nsfw, bool):
        params["nsfw"] = str(nsfw).lower()
    elif nsfw == NsfwLevel.none:
        params["nsfw"] = "false"  # Exclude NSFW
    else:
        params["nsfw"] = "true"  # Include for specific levels (API filters server-side)

    # API quirk: query + filters don't work reliably together
    has_filters = model_type is not None or base_model is not None or tag is not None

    if query and not has_filters:
        params["query"] = query

    if model_type:
        params["types"] = model_type.to_api()

    if base_model:
        params["baseModels"] = base_model.to_api()

    params["sort"] = sort.to_api()

    # Additional filters
    if period:
        params["period"] = period.to_api()

    if tag:
        params["tag"] = tag

    if username:
        params["username"] = username

    if page and page > 1:
        params["page"] = page

    if commercial_use:
        params["allowCommercialUse"] = commercial_use.to_api()

    if allow_derivatives is not None:
        params["allowDerivatives"] = str(allow_derivatives).lower()

    if primary_file_only:
        params["primaryFileOnly"] = "true"

    # Request more if we need client-side filtering
    if query and has_filters:
        params["limit"] = 100

    return params, has_filters


def _filter_results(result: dict[str, Any], query: str | None, has_filters: bool, limit: int) -> dict[str, Any]:
    """Apply client-side filtering when query + filters combined."""
    if query and has_filters:
        q_lower = query.lower()
        result["items"] = [m for m in result.get("items", []) if q_lower in m.get("name", "").lower()][:limit]
    return result


def search_civitai(
    query: str | None,
    model_type: ModelType | None,
    base_model: BaseModel | None,
    sort: SortOrder,
    limit: int,
    api_key: str | None,
    console: Console,
    *,
    period: Period | None = None,
    nsfw: NsfwLevel | bool | None = None,
    tag: str | None = None,
    username: str | None = None,
    page: int | None = None,
    commercial_use: CommercialUse | None = None,
    allow_derivatives: bool | None = None,
    primary_file_only: bool = False,
) -> dict[str, Any] | None:
    """Search CivitAI models.

    Implements workarounds for API limitations:
    - Query + filters: fetches more results and filters client-side
    - NSFW: defaults to including all content (like website behavior)
    """
    params, has_filters = _build_search_params(
        query,
        model_type,
        base_model,
        sort,
        limit,
        period=period,
        nsfw=nsfw,
        tag=tag,
        username=username,
        page=page,
        commercial_use=commercial_use,
        allow_derivatives=allow_derivatives,
        primary_file_only=primary_file_only,
    )
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
            return _filter_results(result, query, has_filters, limit)
        except httpx.HTTPStatusError as e:
            console.print(f"[red]API error: {e.response.status_code}[/red]")
            return None
        except httpx.RequestError as e:
            console.print(f"[red]Request error: {e}[/red]")
            return None


def _setup_resume(dest_path: Path, resume: bool, console: Console) -> tuple[dict[str, str], str, int]:
    """Set up resume headers and mode for download."""
    headers: dict[str, str] = {}
    mode = "wb"
    initial_size = 0

    if resume and dest_path.exists():
        initial_size = dest_path.stat().st_size
        headers["Range"] = f"bytes={initial_size}-"
        mode = "ab"
        console.print(f"[cyan]Resuming download from {initial_size / (1024**2):.1f} MB[/cyan]")

    return headers, mode, initial_size


def _get_dest_from_response(response: httpx.Response, dest_path: Path) -> Path:
    """Extract destination path from response headers if dest is directory."""
    content_disp = response.headers.get("content-disposition", "")
    if "filename=" in content_disp:
        match = re.search(r'filename="?([^";\n]+)"?', content_disp)
        if match and dest_path.is_dir():
            return dest_path / match.group(1)
    return dest_path


def _stream_download(
    response: httpx.Response,
    dest_path: Path,
    mode: str,
    initial_size: int,
    console: Console,
) -> bool:
    """Stream download content to file with progress."""
    content_length = response.headers.get("content-length")
    total_size = int(content_length) + initial_size if content_length else 0

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

    console.print()
    console.print(f'[magenta]Downloaded:[/magenta] [green]"{dest_path}"[/green]')
    return True


# Type alias for progress callback: (downloaded_bytes, total_bytes, speed_bytes_per_sec)
ProgressCallback = Callable[[int, int, float], None]


def _stream_download_with_callback(
    response: httpx.Response,
    dest_path: Path,
    mode: str,
    initial_size: int,
    on_progress: ProgressCallback | None = None,
) -> bool:
    """Stream download content to file with progress callback."""
    content_length = response.headers.get("content-length")
    total_size = int(content_length) + initial_size if content_length else 0
    downloaded = initial_size
    start_time = time.time()
    last_time = start_time

    with dest_path.open(mode) as f:
        for chunk in response.iter_bytes(1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)

            if on_progress:
                now = time.time()
                elapsed = now - start_time
                speed = downloaded / elapsed if elapsed > 0 else 0
                # Throttle updates
                if now - last_time >= _PROGRESS_UPDATE_INTERVAL:
                    on_progress(downloaded, total_size, speed)
                    last_time = now

    # Final progress update
    if on_progress:
        elapsed = time.time() - start_time
        speed = downloaded / elapsed if elapsed > 0 else 0
        on_progress(downloaded, total_size, speed)

    return True


def download_model_with_progress(
    version_id: int,
    dest_path: Path,
    api_key: str | None,
    on_progress: ProgressCallback | None = None,
    resume: bool = True,
) -> bool:
    """Download a model from CivitAI with progress callback instead of console output."""
    import logging  # noqa: PLC0415

    logger = logging.getLogger(__name__)
    url = f"{CIVITAI_DOWNLOAD_BASE}/{version_id}"
    params: dict[str, str] = {}
    if api_key:
        params["token"] = api_key

    # Set up resume
    headers: dict[str, str] = {}
    mode = "wb"
    initial_size = 0

    if resume and dest_path.exists():
        initial_size = dest_path.stat().st_size
        headers["Range"] = f"bytes={initial_size}-"
        mode = "ab"
        logger.info(f"Resuming download from {initial_size / (1024**2):.1f} MB")

    try:
        with httpx.stream(
            "GET",
            url,
            params=params,
            headers=headers,
            follow_redirects=True,
            timeout=httpx.Timeout(30.0, read=None),
        ) as response:
            if response.status_code == HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE:
                return True  # Already complete

            response.raise_for_status()
            dest_path = _get_dest_from_response(response, dest_path)
            return _stream_download_with_callback(response, dest_path, mode, initial_size, on_progress)

    except httpx.HTTPStatusError as e:
        logger.error(f"Download error: HTTP {e.response.status_code}")
        return False
    except httpx.RequestError as e:
        logger.error(f"Download error: {e}")
        return False


def download_model(
    version_id: int,
    dest_path: Path,
    api_key: str | None,
    console: Console,
    resume: bool = True,
) -> bool:
    """Download a model from CivitAI by version ID with resume support."""
    url = f"{CIVITAI_DOWNLOAD_BASE}/{version_id}"
    params: dict[str, str] = {}
    if api_key:
        params["token"] = api_key

    headers, mode, initial_size = _setup_resume(dest_path, resume, console)

    try:
        with httpx.stream(
            "GET",
            url,
            params=params,
            headers=headers,
            follow_redirects=True,
            timeout=httpx.Timeout(30.0, read=None),
        ) as response:
            if response.status_code == HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE:
                console.print("[green]File already fully downloaded.[/green]")
                return True

            response.raise_for_status()
            dest_path = _get_dest_from_response(response, dest_path)
            return _stream_download(response, dest_path, mode, initial_size, console)

    except httpx.HTTPStatusError as e:
        console.print(f"[red]Download error: HTTP {e.response.status_code}[/red]")
        if e.response.status_code == HTTPStatus.UNAUTHORIZED:
            console.print("[yellow]Hint: This model may require an API key.[/yellow]")
        return False
    except httpx.RequestError as e:
        console.print(f"[red]Download error: {e}[/red]")
        return False
