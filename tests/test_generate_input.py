"""Tests for the ``tsr generate --input`` JSON/YAML parser.

Covers the :func:`tensors.cli._parse_generate_input` helper directly (unit
level) and the end-to-end integration through the ``generate`` Typer command
(with ``_run_generation`` patched so nothing hits ComfyUI).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from tensors import cli as cli_module
from tensors.cli import _parse_generate_input, app

runner = CliRunner()


# -----------------------------------------------------------------------------
# Unit tests: _parse_generate_input
# -----------------------------------------------------------------------------


class TestParseGenerateInputInline:
    """Inline string arguments (not file paths)."""

    def test_inline_json_object(self) -> None:
        out = _parse_generate_input('{"prompt": "hi", "steps": 30}')
        assert out == {"prompt": "hi", "steps": 30}

    def test_inline_yaml_mapping(self) -> None:
        out = _parse_generate_input("prompt: hi\nsteps: 30\n")
        assert out == {"prompt": "hi", "steps": 30}

    def test_inline_yaml_with_list(self) -> None:
        out = _parse_generate_input("prompt: x\nscene:\n  - foo\n  - bar\n")
        assert out == {"prompt": "x", "scene": ["foo", "bar"]}

    def test_inline_json_with_leading_whitespace(self) -> None:
        out = _parse_generate_input('   {"prompt": "hi"}')
        assert out == {"prompt": "hi"}

    def test_inline_non_mapping_yaml_rejected(self) -> None:
        with pytest.raises(typer.Exit):
            _parse_generate_input("- just\n- a list\n")

    def test_inline_non_mapping_json_rejected(self) -> None:
        with pytest.raises(typer.Exit):
            _parse_generate_input("[1, 2, 3]")

    def test_inline_invalid_yaml_rejected(self) -> None:
        with pytest.raises(typer.Exit):
            _parse_generate_input("prompt: [unterminated\n")

    def test_inline_invalid_json_falls_to_yaml_and_fails(self) -> None:
        # Starts with '{' so JSON path is taken; malformed → Exit.
        with pytest.raises(typer.Exit):
            _parse_generate_input('{"prompt": "missing-close"')


class TestParseGenerateInputFiles:
    """File path arguments resolved by extension."""

    def test_json_file_by_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "scene.json"
        p.write_text(json.dumps({"prompt": "from-json", "steps": 20}))
        assert _parse_generate_input(str(p)) == {"prompt": "from-json", "steps": 20}

    def test_yaml_file_dot_yml(self, tmp_path: Path) -> None:
        p = tmp_path / "scene.yml"
        p.write_text("prompt: from-yml\nsteps: 25\n")
        assert _parse_generate_input(str(p)) == {"prompt": "from-yml", "steps": 25}

    def test_yaml_file_dot_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "scene.yaml"
        p.write_text("prompt: from-yaml\n")
        assert _parse_generate_input(str(p)) == {"prompt": "from-yaml"}

    def test_unknown_extension_sniffs_json(self, tmp_path: Path) -> None:
        p = tmp_path / "scene.txt"
        p.write_text('{"prompt": "sniffed"}')
        assert _parse_generate_input(str(p)) == {"prompt": "sniffed"}

    def test_unknown_extension_sniffs_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "scene.txt"
        p.write_text("prompt: sniffed-yaml\n")
        assert _parse_generate_input(str(p)) == {"prompt": "sniffed-yaml"}

    def test_yaml_file_with_full_draw_template(self, tmp_path: Path) -> None:
        """Smoke test against the exact shape used by ~/Projects/draw/templates/."""
        p = tmp_path / "scene.yml"
        p.write_text(
            'prompt: ""\n'
            'negative_prompt: ""\n'
            'model: "getphatFLUXReality_v5Hardcore.safetensors"\n'
            "width: 832\n"
            "height: 1216\n"
            "steps: 35\n"
            "cfg: 1.0\n"
            "guidance: 4.0\n"
            'sampler: "dpmpp_2m"\n'
            'scheduler: "sgm_uniform"\n'
            'vae: "ae.safetensors"\n'
            'orientation: "portrait"\n'
            "seed: -1\n"
            "count: 1\n"
            "scene:\n"
            '  - "first element with embedded \\nnewline"\n'
            '  - "second element"\n'
            '_scene_name: "demo_01"\n'
            '_family: "flux_unet"\n'
            '_base_model: "Flux.1 D"\n'
        )
        out = _parse_generate_input(str(p))
        assert out["model"] == "getphatFLUXReality_v5Hardcore.safetensors"
        assert out["width"] == 832
        assert out["height"] == 1216
        assert out["steps"] == 35
        assert out["cfg"] == 1.0
        assert out["guidance"] == 4.0
        assert out["sampler"] == "dpmpp_2m"
        assert out["scheduler"] == "sgm_uniform"
        assert out["vae"] == "ae.safetensors"
        assert out["orientation"] == "portrait"
        assert out["seed"] == -1
        assert out["count"] == 1
        assert isinstance(out["scene"], list)
        assert len(out["scene"]) == 2
        assert "embedded" in out["scene"][0]

    def test_malformed_yaml_file_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yml"
        p.write_text("prompt: [unterminated\n")
        with pytest.raises(typer.Exit):
            _parse_generate_input(str(p))


# -----------------------------------------------------------------------------
# Integration: generate --input through Typer
# -----------------------------------------------------------------------------


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Capture _run_generation kwargs without dispatching to ComfyUI."""
    sink: dict[str, Any] = {}

    def fake_run_generation(**kwargs: Any) -> None:
        sink.update(kwargs)

    monkeypatch.setattr(cli_module, "_run_generation", fake_run_generation)
    return sink


def test_generate_consumes_yaml_file(tmp_path: Path, captured: dict[str, Any]) -> None:
    """``tsr generate --input scene.yml`` plumbs YAML values through."""
    yml = tmp_path / "scene.yml"
    yml.write_text('prompt: a sunset\nmodel: "fluxmodel.safetensors"\nsteps: 28\nscene:\n  - "golden hour"\n  - "wide angle"\n')
    result = runner.invoke(app, ["generate", "--input", str(yml)])
    assert result.exit_code == 0, result.output
    assert captured["prompt"] == "a sunset"
    assert captured["model"] == "fluxmodel.safetensors"
    assert captured["steps"] == 28
    # YAML list under `scene` is joined into scene_prompt by existing logic.
    assert captured["scene_prompt"] == "golden hour, wide angle"


def test_generate_yaml_then_cli_flag_wins(tmp_path: Path, captured: dict[str, Any]) -> None:
    """Explicit CLI flags must override --input values (same contract as JSON)."""
    yml = tmp_path / "scene.yml"
    yml.write_text('prompt: from-yaml\nmodel: "yamlmodel.safetensors"\nsteps: 10\n')
    result = runner.invoke(app, ["generate", "--input", str(yml), "--steps", "99"])
    assert result.exit_code == 0, result.output
    assert captured["prompt"] == "from-yaml"
    assert captured["model"] == "yamlmodel.safetensors"
    assert captured["steps"] == 99  # CLI override wins


def test_generate_inline_yaml_string(captured: dict[str, Any]) -> None:
    result = runner.invoke(
        app,
        ["generate", "--input", "prompt: inline-yaml\nmodel: m.safetensors\n"],
    )
    assert result.exit_code == 0, result.output
    assert captured["prompt"] == "inline-yaml"
    assert captured["model"] == "m.safetensors"


def test_generate_inline_json_still_works(captured: dict[str, Any]) -> None:
    """Regression guard for the original JSON contract."""
    result = runner.invoke(
        app,
        ["generate", "--input", '{"prompt": "inline-json", "model": "j.safetensors"}'],
    )
    assert result.exit_code == 0, result.output
    assert captured["prompt"] == "inline-json"
    assert captured["model"] == "j.safetensors"


def test_generate_invalid_yaml_file_exits_nonzero(tmp_path: Path, captured: dict[str, Any]) -> None:
    yml = tmp_path / "bad.yml"
    yml.write_text("prompt: [oops\n")
    result = runner.invoke(app, ["generate", "--input", str(yml)])
    assert result.exit_code != 0
    assert "Invalid YAML input" in result.output
    assert captured == {}
