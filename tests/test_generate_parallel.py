"""Tests for the `tsr generate --parallel-queue` flag (parallel fanout path)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tensors import cli as cli_module
from tensors.cli import app

runner = CliRunner()


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record every _run_generation call and stub the disk-write side effect.

    The parallel fanout path invokes _run_generation N times (one per task);
    the sequential path invokes it once. By recording kwargs we can assert
    fanout behavior (per-task seeds, per-task output paths, count=1 per task)
    without round-tripping ComfyUI.
    """
    recorded: list[dict[str, Any]] = []

    def fake_run_generation(**kwargs: Any) -> None:
        recorded.append(kwargs)
        out: Path | None = kwargs.get("output")
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"fake-png")

    monkeypatch.setattr(cli_module, "_run_generation", fake_run_generation)
    return recorded


@pytest.fixture(autouse=True)
def _stub_model_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass ComfyUI's live model lookup so tests don't need a backend."""
    monkeypatch.setattr(
        cli_module,
        "_validate_model_available",
        lambda model, family, lora: (model, lora),
    )


# -----------------------------------------------------------------------------
# Validation / sanity
# -----------------------------------------------------------------------------


def test_parallel_queue_invalid_value_rejected(calls: list[dict[str, Any]]) -> None:
    """--parallel-queue 0 (or negative) exits non-zero before any work."""
    result = runner.invoke(
        app,
        ["generate", "test prompt", "-m", "x.safetensors", "--parallel-queue", "0"],
    )
    assert result.exit_code != 0
    assert "--parallel-queue must be >= 1" in result.output
    assert calls == []


def test_parallel_queue_one_is_sequential_path(calls: list[dict[str, Any]]) -> None:
    """-P 1 collapses to the legacy single _run_generation call with count=N.

    This is the key compatibility contract: existing scripts that don't pass
    -P must see identical behavior (one call, count forwarded as batch_size).
    """
    result = runner.invoke(
        app,
        ["generate", "test prompt", "-m", "x.safetensors", "-c", "4", "-P", "1"],
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["count"] == 4
    assert calls[0]["prompt"] == "test prompt"


def test_count_one_ignores_parallel_queue(calls: list[dict[str, Any]]) -> None:
    """count=1 always takes sequential path regardless of -P (no fanout point)."""
    result = runner.invoke(
        app,
        ["generate", "test prompt", "-m", "x.safetensors", "-c", "1", "-P", "8"],
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["count"] == 1


def test_json_output_incompatible_with_parallel(calls: list[dict[str, Any]]) -> None:
    """--json + -P>1 errors out cleanly (would skip disk-save inside tasks)."""
    result = runner.invoke(
        app,
        ["generate", "test prompt", "-m", "x.safetensors", "-c", "2", "-P", "2", "--json"],
    )
    assert result.exit_code != 0
    assert "--json is not supported with --parallel-queue > 1" in result.output
    assert calls == []


# -----------------------------------------------------------------------------
# Fanout behavior
# -----------------------------------------------------------------------------


def test_parallel_fanout_creates_n_tasks(tmp_path: Path, calls: list[dict[str, Any]]) -> None:
    """-c N -P M (M>1, N>1) → N independent _run_generation calls, each count=1."""
    out = tmp_path / "img.png"
    result = runner.invoke(
        app,
        [
            "generate",
            "test prompt",
            "-m",
            "x.safetensors",
            "-c",
            "4",
            "-P",
            "2",
            "--seed",
            "100",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 4
    # Each task generates exactly one image
    for c in calls:
        assert c["count"] == 1


def test_parallel_seeds_increment_from_base(tmp_path: Path, calls: list[dict[str, Any]]) -> None:
    """Explicit --seed → each task receives base+i (reproducible series)."""
    out = tmp_path / "img.png"
    runner.invoke(
        app,
        ["generate", "test prompt", "-m", "x.safetensors", "-c", "3", "-P", "3", "--seed", "500", "-o", str(out)],
    )
    seeds_seen = sorted(c["seed"] for c in calls)
    assert seeds_seen == [500, 501, 502]


def test_parallel_seeds_random_when_unset(tmp_path: Path, calls: list[dict[str, Any]]) -> None:
    """seed=-1 → each task gets a freshly-rolled random seed (not all the same).

    Vanishingly small chance of collision across 4 random ints; treat as flake
    threshold of "all distinct" rather than exact equality to any value.
    """
    out = tmp_path / "img.png"
    runner.invoke(
        app,
        ["generate", "test prompt", "-m", "x.safetensors", "-c", "4", "-P", "2", "-o", str(out)],
    )
    seeds = [c["seed"] for c in calls]
    # All non-negative (i.e. resolved from -1 to actual int) and distinct.
    assert all(s >= 0 for s in seeds)
    assert len(set(seeds)) == len(seeds)


def test_parallel_output_paths_indexed(tmp_path: Path, calls: list[dict[str, Any]]) -> None:
    """Per-task output paths use stem_NNN.suffix naming (matches sequential count>1)."""
    out = tmp_path / "scene.png"
    runner.invoke(
        app,
        ["generate", "test prompt", "-m", "x.safetensors", "-c", "3", "-P", "3", "--seed", "1", "-o", str(out)],
    )
    paths = sorted(str(c["output"]) for c in calls)
    assert paths == [
        str(tmp_path / "scene_001.png"),
        str(tmp_path / "scene_002.png"),
        str(tmp_path / "scene_003.png"),
    ]


def test_parallel_without_output_passes_none(tmp_path: Path, calls: list[dict[str, Any]]) -> None:
    """When --output is omitted, each task gets output=None (no disk write planned)."""
    runner.invoke(
        app,
        ["generate", "test prompt", "-m", "x.safetensors", "-c", "2", "-P", "2", "--seed", "1"],
    )
    assert len(calls) == 2
    assert all(c["output"] is None for c in calls)


def test_parallel_files_actually_written(tmp_path: Path, calls: list[dict[str, Any]]) -> None:
    """End-to-end: per-task stub writes its file → all N appear on disk.

    Guards against the bug where json_output=True short-circuits the save block
    inside _run_generation. Each task must use the non-JSON code path.
    """
    out = tmp_path / "shot.png"
    runner.invoke(
        app,
        ["generate", "test prompt", "-m", "x.safetensors", "-c", "3", "-P", "3", "--seed", "1", "-o", str(out)],
    )
    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == ["shot_001.png", "shot_002.png", "shot_003.png"]


def test_parallel_summary_reports_success_count(tmp_path: Path, calls: list[dict[str, Any]]) -> None:
    """Final summary line reports N/N success when all tasks complete."""
    out = tmp_path / "img.png"
    result = runner.invoke(
        app,
        ["generate", "test prompt", "-m", "x.safetensors", "-c", "3", "-P", "2", "--seed", "1", "-o", str(out)],
    )
    assert result.exit_code == 0
    assert "Generated 3/3 images" in result.output


def test_parallel_partial_failure_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If one task raises, summary shows partial count and command exits non-zero."""
    import typer

    call_indices: list[int] = []

    def flaky_run_generation(**kwargs: Any) -> None:
        # Fail every other call to simulate intermittent backend errors.
        idx = len(call_indices)
        call_indices.append(idx)
        if idx % 2 == 0:
            raise typer.Exit(1)
        out: Path | None = kwargs.get("output")
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"ok")

    monkeypatch.setattr(cli_module, "_run_generation", flaky_run_generation)

    out = tmp_path / "img.png"
    result = runner.invoke(
        app,
        ["generate", "test prompt", "-m", "x.safetensors", "-c", "4", "-P", "2", "--seed", "1", "-o", str(out)],
    )
    assert result.exit_code != 0
    # Two tasks failed; final summary should show 2/4.
    assert "Generated 2/4 images" in result.output


# -----------------------------------------------------------------------------
# --input integration
# -----------------------------------------------------------------------------


def test_parallel_queue_from_yaml_input(tmp_path: Path, calls: list[dict[str, Any]]) -> None:
    """parallel_queue can be set via --input YAML (mirrors other generate params)."""
    out = tmp_path / "img.png"
    yml = tmp_path / "spec.yml"
    yml.write_text(
        f'prompt: from-yaml\nmodel: x.safetensors\ncount: 3\nparallel_queue: 3\nseed: 7\noutput: "{out}"\n'
    )
    result = runner.invoke(app, ["generate", "--input", str(yml)])
    assert result.exit_code == 0, result.output
    assert len(calls) == 3
    assert sorted(c["seed"] for c in calls) == [7, 8, 9]


def test_cli_parallel_queue_overrides_yaml(tmp_path: Path, calls: list[dict[str, Any]]) -> None:
    """CLI --parallel-queue wins over YAML's parallel_queue (standard precedence)."""
    out = tmp_path / "img.png"
    yml = tmp_path / "spec.yml"
    yml.write_text(
        f'prompt: from-yaml\nmodel: x.safetensors\ncount: 2\nparallel_queue: 1\nseed: 10\noutput: "{out}"\n'
    )
    # YAML says P=1 (sequential), CLI overrides to P=2 (fanout)
    result = runner.invoke(app, ["generate", "--input", str(yml), "-P", "2"])
    assert result.exit_code == 0, result.output
    # Fanout path → 2 separate calls, each count=1
    assert len(calls) == 2
    assert all(c["count"] == 1 for c in calls)


# -----------------------------------------------------------------------------
# Concurrency assertion
# -----------------------------------------------------------------------------


def test_parallel_actually_runs_concurrently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity: P concurrent tasks really overlap in time (vs all-serial)."""
    import threading
    import time as _t

    in_flight = 0
    peak_in_flight = 0
    lock = threading.Lock()

    def slow_run_generation(**kwargs: Any) -> None:
        nonlocal in_flight, peak_in_flight
        with lock:
            in_flight += 1
            peak_in_flight = max(peak_in_flight, in_flight)
        _t.sleep(0.1)  # 100ms — long enough to overlap, short enough for fast tests
        with lock:
            in_flight -= 1
        out: Path | None = kwargs.get("output")
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"ok")

    monkeypatch.setattr(cli_module, "_run_generation", slow_run_generation)

    out = tmp_path / "img.png"
    result = runner.invoke(
        app,
        ["generate", "test prompt", "-m", "x.safetensors", "-c", "4", "-P", "4", "--seed", "1", "-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    # With P=4 and 4 tasks each sleeping 100ms, peak concurrency should hit 4.
    # Even allowing for thread-pool warmup quirks, ≥2 means parallelism is real.
    assert peak_in_flight >= 2, f"peak_in_flight={peak_in_flight} (expected ≥2 for parallel)"
