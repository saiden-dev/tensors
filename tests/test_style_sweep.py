"""Tests for the `tsr style-sweep` command."""

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
# Helpers
# -----------------------------------------------------------------------------


def _write_template(
    tmp_path: Path,
    *,
    output_dir: Path | str,
    styles: Any = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a minimal template JSON file and return its path."""
    body: dict[str, Any] = {
        "prompt": "a portrait of a person",
        "model": "test-model.safetensors",
        "seed": 12345,
        "orientation": "portrait",
        "output_dir": str(output_dir),
    }
    if styles is not None:
        body["styles"] = styles
    if extra:
        body.update(extra)
    path = tmp_path / "template.json"
    path.write_text(json.dumps(body))
    return path


def _write_styles_file(tmp_path: Path, entries: list[dict[str, str]]) -> Path:
    """Write a styles JSON file (object form with 'styles' key)."""
    path = tmp_path / "styles.json"
    path.write_text(json.dumps({"name": "test", "description": "", "styles": entries}))
    return path


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Patch `_run_generation` to record calls and create the output file."""
    recorded: list[dict[str, Any]] = []

    def fake_run_generation(**kwargs: Any) -> None:
        recorded.append(kwargs)
        out: Path | None = kwargs.get("output")
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"fake-png")

    monkeypatch.setattr(cli_module, "_run_generation", fake_run_generation)
    return recorded


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_loads_template_and_styles_from_files(tmp_path: Path, calls: list[dict[str, Any]]) -> None:
    """Template + external styles file → N generate calls with composed prompts."""
    out_dir = tmp_path / "out"
    styles_file = _write_styles_file(
        tmp_path,
        [
            {"slug": "01-foo", "suffix": "in the style of Foo"},
            {"slug": "02-bar", "suffix": "in the style of Bar"},
            {"slug": "03-baz", "suffix": "in the style of Baz"},
        ],
    )
    tpl = _write_template(tmp_path, output_dir=out_dir, styles=str(styles_file))

    result = runner.invoke(app, ["style-sweep", "--template", str(tpl)])

    assert result.exit_code == 0, result.output
    assert len(calls) == 3
    assert calls[0]["prompt"] == "a portrait of a person, in the style of Foo"
    assert calls[1]["prompt"] == "a portrait of a person, in the style of Bar"
    assert calls[2]["prompt"] == "a portrait of a person, in the style of Baz"
    # Each call writes to {output_dir}/{slug}.png
    assert calls[0]["output"] == out_dir / "01-foo.png"
    assert calls[2]["output"] == out_dir / "03-baz.png"
    # Template values propagated
    assert calls[0]["model"] == "test-model.safetensors"
    assert calls[0]["seed"] == 12345
    assert calls[0]["orientation"] == "portrait"


def test_skip_existing(tmp_path: Path, calls: list[dict[str, Any]]) -> None:
    """Pre-existing output file → that slug is skipped."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "01-foo.png").write_bytes(b"already here")

    styles_file = _write_styles_file(
        tmp_path,
        [
            {"slug": "01-foo", "suffix": "Foo"},
            {"slug": "02-bar", "suffix": "Bar"},
        ],
    )
    tpl = _write_template(tmp_path, output_dir=out_dir, styles=str(styles_file))

    result = runner.invoke(app, ["style-sweep", "--template", str(tpl)])

    assert result.exit_code == 0, result.output
    # Only 02-bar should have been generated
    assert len(calls) == 1
    assert calls[0]["output"] == out_dir / "02-bar.png"
    assert "skip" in result.output.lower()


def test_limit(tmp_path: Path, calls: list[dict[str, Any]]) -> None:
    """--limit 2 caps the sweep at 2 styles."""
    out_dir = tmp_path / "out"
    styles_file = _write_styles_file(
        tmp_path,
        [{"slug": f"{i:02d}-style", "suffix": f"style {i}"} for i in range(1, 6)],
    )
    tpl = _write_template(tmp_path, output_dir=out_dir, styles=str(styles_file))

    result = runner.invoke(app, ["style-sweep", "--template", str(tpl), "--limit", "2"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 2
    assert calls[0]["output"].name == "01-style.png"
    assert calls[1]["output"].name == "02-style.png"


def test_dry_run(tmp_path: Path, calls: list[dict[str, Any]]) -> None:
    """--dry-run prints plan but does not invoke generate."""
    out_dir = tmp_path / "out"
    styles_file = _write_styles_file(
        tmp_path,
        [
            {"slug": "01-foo", "suffix": "Foo style"},
            {"slug": "02-bar", "suffix": "Bar style"},
        ],
    )
    tpl = _write_template(tmp_path, output_dir=out_dir, styles=str(styles_file))

    result = runner.invoke(app, ["style-sweep", "--template", str(tpl), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 0
    assert "DRY RUN" in result.output
    assert "01-foo" in result.output
    assert "Foo style" in result.output
    # No manifest written on dry-run
    assert not (out_dir / "_sweep.json").exists()


def test_inline_styles_list(tmp_path: Path, calls: list[dict[str, Any]]) -> None:
    """Styles can be passed inline as a list inside the template."""
    out_dir = tmp_path / "out"
    inline_styles = [
        {"slug": "alpha", "suffix": "Alpha suffix"},
        {"slug": "beta", "suffix": "Beta suffix"},
    ]
    tpl = _write_template(tmp_path, output_dir=out_dir, styles=inline_styles)

    result = runner.invoke(app, ["style-sweep", "--template", str(tpl)])

    assert result.exit_code == 0, result.output
    assert len(calls) == 2
    assert calls[0]["prompt"].endswith("Alpha suffix")
    assert calls[1]["prompt"].endswith("Beta suffix")


def test_manifest_written(tmp_path: Path, calls: list[dict[str, Any]]) -> None:
    """A successful sweep produces {output_dir}/_sweep.json with expected keys."""
    out_dir = tmp_path / "out"
    styles_file = _write_styles_file(
        tmp_path,
        [
            {"slug": "01-foo", "suffix": "Foo"},
            {"slug": "02-bar", "suffix": "Bar"},
        ],
    )
    tpl = _write_template(tmp_path, output_dir=out_dir, styles=str(styles_file))

    result = runner.invoke(app, ["style-sweep", "--template", str(tpl)])

    assert result.exit_code == 0, result.output
    manifest_path = out_dir / "_sweep.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["template"] == str(tpl)
    assert manifest["styles_source"] == str(styles_file)
    assert len(manifest["results"]) == 2

    first = manifest["results"][0]
    for key in ("slug", "prompt", "output", "seed", "duration_sec", "success", "error"):
        assert key in first, f"missing manifest key {key}"
    assert first["slug"] == "01-foo"
    assert first["success"] is True
    assert first["error"] is None
    assert first["seed"] == 12345


def test_continue_on_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One failed style does not abort the sweep; manifest records the error."""
    out_dir = tmp_path / "out"
    styles_file = _write_styles_file(
        tmp_path,
        [
            {"slug": "01-ok", "suffix": "ok one"},
            {"slug": "02-bad", "suffix": "bad one"},
            {"slug": "03-ok", "suffix": "ok two"},
        ],
    )
    tpl = _write_template(tmp_path, output_dir=out_dir, styles=str(styles_file))

    def fake_run_generation(**kwargs: Any) -> None:
        out: Path = kwargs["output"]
        if "02-bad" in out.name:
            raise RuntimeError("simulated failure")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake")

    monkeypatch.setattr(cli_module, "_run_generation", fake_run_generation)

    result = runner.invoke(app, ["style-sweep", "--template", str(tpl)])

    # Sweep finished but exit code non-zero because one slug failed
    assert result.exit_code == 1, result.output
    assert "02-bad" in result.output
    assert "FAIL" in result.output

    manifest = json.loads((out_dir / "_sweep.json").read_text())
    assert len(manifest["results"]) == 3
    statuses = {r["slug"]: r for r in manifest["results"]}
    assert statuses["01-ok"]["success"] is True
    assert statuses["02-bad"]["success"] is False
    assert "simulated failure" in statuses["02-bad"]["error"]
    assert statuses["03-ok"]["success"] is True


def test_abort_on_error_stops_immediately(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--abort-on-error aborts at the first failure."""
    out_dir = tmp_path / "out"
    styles_file = _write_styles_file(
        tmp_path,
        [
            {"slug": "01-bad", "suffix": "bad"},
            {"slug": "02-skipped", "suffix": "never reached"},
        ],
    )
    tpl = _write_template(tmp_path, output_dir=out_dir, styles=str(styles_file))

    seen: list[str] = []

    def fake_run_generation(**kwargs: Any) -> None:
        seen.append(Path(kwargs["output"]).name)
        raise RuntimeError("boom")

    monkeypatch.setattr(cli_module, "_run_generation", fake_run_generation)

    result = runner.invoke(app, ["style-sweep", "--template", str(tpl), "--abort-on-error"])

    assert result.exit_code != 0
    assert seen == ["01-bad.png"]
    # Manifest was still written (with the one failed entry)
    manifest = json.loads((out_dir / "_sweep.json").read_text())
    assert len(manifest["results"]) == 1
    assert manifest["results"][0]["slug"] == "01-bad"


def test_missing_template_file_errors(tmp_path: Path) -> None:
    """A non-existent template path yields a clean error exit."""
    result = runner.invoke(app, ["style-sweep", "--template", str(tmp_path / "nope.json")])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_missing_styles_errors(tmp_path: Path) -> None:
    """A template without styles (and no --styles) errors out."""
    out_dir = tmp_path / "out"
    tpl_body = {
        "prompt": "a portrait",
        "output_dir": str(out_dir),
    }
    tpl = tmp_path / "template.json"
    tpl.write_text(json.dumps(tpl_body))

    result = runner.invoke(app, ["style-sweep", "--template", str(tpl)])
    assert result.exit_code != 0
    assert "styles" in result.output.lower()


def test_cli_output_dir_overrides_template(tmp_path: Path, calls: list[dict[str, Any]]) -> None:
    """--output-dir on the CLI overrides the template's output_dir."""
    tpl_out = tmp_path / "from-template"
    cli_out = tmp_path / "from-cli"
    styles_file = _write_styles_file(tmp_path, [{"slug": "x", "suffix": "X"}])
    tpl = _write_template(tmp_path, output_dir=tpl_out, styles=str(styles_file))

    result = runner.invoke(app, ["style-sweep", "--template", str(tpl), "--output-dir", str(cli_out)])

    assert result.exit_code == 0, result.output
    assert calls[0]["output"] == cli_out / "x.png"
    assert (cli_out / "_sweep.json").exists()
    assert not tpl_out.exists()


def test_remote_override(tmp_path: Path, calls: list[dict[str, Any]]) -> None:
    """--remote propagates through to _run_generation."""
    out_dir = tmp_path / "out"
    styles_file = _write_styles_file(tmp_path, [{"slug": "x", "suffix": "X"}])
    tpl = _write_template(tmp_path, output_dir=out_dir, styles=str(styles_file))

    result = runner.invoke(app, ["style-sweep", "--template", str(tpl), "--remote", "junkpile"])

    assert result.exit_code == 0, result.output
    assert calls[0]["remote"] == "junkpile"


# -----------------------------------------------------------------------------
# --list flag
# -----------------------------------------------------------------------------


def test_list_flag_prints_slugs(tmp_path: Path, calls: list[dict[str, Any]]) -> None:
    """--list prints all slugs and does not call generate."""
    out_dir = tmp_path / "out"
    slugs = [f"{i:02d}-style" for i in range(1, 5)]
    styles_file = _write_styles_file(tmp_path, [{"slug": s, "suffix": f"suffix for {s}"} for s in slugs])
    tpl = _write_template(tmp_path, output_dir=out_dir, styles=str(styles_file))

    result = runner.invoke(app, ["style-sweep", "--template", str(tpl), "--list"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 0
    for slug in slugs:
        assert slug in result.output
    # Header line names the file and count
    assert "4 entries" in result.output
    # No manifest written
    assert not (out_dir / "_sweep.json").exists()


def test_list_with_limit(tmp_path: Path, calls: list[dict[str, Any]]) -> None:
    """--list --limit N restricts the table to the first N entries."""
    out_dir = tmp_path / "out"
    styles_file = _write_styles_file(tmp_path, [{"slug": f"{i:02d}-x", "suffix": f"s{i}"} for i in range(1, 6)])
    tpl = _write_template(tmp_path, output_dir=out_dir, styles=str(styles_file))

    result = runner.invoke(app, ["style-sweep", "--template", str(tpl), "--list", "--limit", "2"])

    assert result.exit_code == 0, result.output
    assert "01-x" in result.output
    assert "02-x" in result.output
    assert "03-x" not in result.output
    assert "05-x" not in result.output


def test_list_without_template(tmp_path: Path, calls: list[dict[str, Any]]) -> None:
    """--styles + --list works without --template."""
    styles_file = _write_styles_file(
        tmp_path,
        [
            {"slug": "alpha", "suffix": "Alpha suffix"},
            {"slug": "beta", "suffix": "Beta suffix"},
        ],
    )

    result = runner.invoke(app, ["style-sweep", "--styles", str(styles_file), "--list"])

    assert result.exit_code == 0, result.output
    assert "alpha" in result.output
    assert "beta" in result.output
    assert "2 entries" in result.output


def test_list_long_suffix_truncated(tmp_path: Path) -> None:
    """Long suffixes are truncated with an ellipsis."""
    long_suffix = "very long " * 20  # ~200 chars
    styles_file = _write_styles_file(tmp_path, [{"slug": "long", "suffix": long_suffix}])

    result = runner.invoke(app, ["style-sweep", "--styles", str(styles_file), "--list"])

    assert result.exit_code == 0, result.output
    assert "long" in result.output
    assert "…" in result.output
    # Full suffix should not appear verbatim
    assert long_suffix not in result.output


# -----------------------------------------------------------------------------
# --style filter
# -----------------------------------------------------------------------------


def test_style_filter_single(tmp_path: Path, calls: list[dict[str, Any]]) -> None:
    """--style SLUG only runs the matching entry."""
    out_dir = tmp_path / "out"
    styles_file = _write_styles_file(
        tmp_path,
        [
            {"slug": "01-foo", "suffix": "Foo"},
            {"slug": "02-bar", "suffix": "Bar"},
            {"slug": "03-baz", "suffix": "Baz"},
        ],
    )
    tpl = _write_template(tmp_path, output_dir=out_dir, styles=str(styles_file))

    result = runner.invoke(app, ["style-sweep", "--template", str(tpl), "--style", "02-bar"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["output"] == out_dir / "02-bar.png"
    assert calls[0]["prompt"].endswith("Bar")


def test_style_filter_multiple(tmp_path: Path, calls: list[dict[str, Any]]) -> None:
    """Multiple --style flags select multiple entries (preserving styles-file order)."""
    out_dir = tmp_path / "out"
    styles_file = _write_styles_file(
        tmp_path,
        [
            {"slug": "01-a", "suffix": "A"},
            {"slug": "02-b", "suffix": "B"},
            {"slug": "03-c", "suffix": "C"},
            {"slug": "04-d", "suffix": "D"},
        ],
    )
    tpl = _write_template(tmp_path, output_dir=out_dir, styles=str(styles_file))

    # Note: pass in non-sorted order; filter should preserve source order.
    result = runner.invoke(
        app,
        [
            "style-sweep",
            "--template",
            str(tpl),
            "-S",
            "03-c",
            "-S",
            "01-a",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 2
    assert calls[0]["output"].name == "01-a.png"
    assert calls[1]["output"].name == "03-c.png"


def test_style_filter_unknown_slug(tmp_path: Path, calls: list[dict[str, Any]]) -> None:
    """An unknown slug aborts with exit 1 and lists available slugs."""
    out_dir = tmp_path / "out"
    styles_file = _write_styles_file(
        tmp_path,
        [
            {"slug": "01-foo", "suffix": "Foo"},
            {"slug": "02-bar", "suffix": "Bar"},
        ],
    )
    tpl = _write_template(tmp_path, output_dir=out_dir, styles=str(styles_file))

    result = runner.invoke(app, ["style-sweep", "--template", str(tpl), "--style", "99-nope"])

    assert result.exit_code == 1, result.output
    assert "99-nope" in result.output
    # Available slugs printed for the user
    assert "01-foo" in result.output
    assert "02-bar" in result.output
    # No generation
    assert len(calls) == 0


def test_style_filter_with_list(tmp_path: Path, calls: list[dict[str, Any]]) -> None:
    """--list --style SLUG shows only the filtered entry."""
    styles_file = _write_styles_file(
        tmp_path,
        [
            {"slug": "01-foo", "suffix": "Foo suffix"},
            {"slug": "02-bar", "suffix": "Bar suffix"},
            {"slug": "03-baz", "suffix": "Baz suffix"},
        ],
    )

    result = runner.invoke(
        app,
        [
            "style-sweep",
            "--styles",
            str(styles_file),
            "--list",
            "--style",
            "02-bar",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 0
    assert "02-bar" in result.output
    assert "01-foo" not in result.output
    assert "03-baz" not in result.output
    assert "1 entries" in result.output
