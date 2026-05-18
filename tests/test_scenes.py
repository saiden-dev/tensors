"""Tests for the scene library (tensors.scenes).

Mirrors test_characters.py — the underlying implementation is shared so the
focus here is verifying that scene-specific exports route through correctly
and that monkeypatching SCENES_DIR works the same way.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def scene_env(tmp_path, monkeypatch):
    """Redirect SCENES_DIR at fresh tmp_path per test."""
    from tensors import scenes as scene_mod

    scene_dir = tmp_path / "scenes"
    monkeypatch.setattr(scene_mod, "SCENES_DIR", scene_dir)
    return scene_mod, scene_dir


class TestSaveLoad:
    def test_save_creates_dir_and_file(self, scene_env):
        scene_mod, scene_dir = scene_env
        path = scene_mod.save_scene("penthouse", ["luxury penthouse", "volumetric lighting"])

        assert path == scene_dir / "penthouse.yml"
        assert path.is_file()
        assert path.read_text() == '- "luxury penthouse"\n- "volumetric lighting"\n'

    def test_load_roundtrip(self, scene_env):
        scene_mod, _ = scene_env
        elements = ["a", "b with spaces", "c, with, commas", 'd "quotes"']
        scene_mod.save_scene("mixed", elements)

        assert scene_mod.load_scene("mixed") == elements

    def test_load_missing_raises(self, scene_env):
        scene_mod, _ = scene_env
        with pytest.raises(FileNotFoundError, match=r"Scene 'nope' not found"):
            scene_mod.load_scene("nope")


class TestListAndDelete:
    def test_list_empty_when_dir_missing(self, scene_env):
        scene_mod, _ = scene_env
        assert scene_mod.list_scenes() == []

    def test_list_sorted_names(self, scene_env):
        scene_mod, _ = scene_env
        scene_mod.save_scene("zeta", ["x"])
        scene_mod.save_scene("alpha", ["y"])

        assert scene_mod.list_scenes() == ["alpha", "zeta"]

    def test_delete_existing(self, scene_env):
        scene_mod, _ = scene_env
        scene_mod.save_scene("doomed", ["x"])

        assert scene_mod.delete_scene("doomed") is True
        assert scene_mod.list_scenes() == []

    def test_delete_missing(self, scene_env):
        scene_mod, _ = scene_env
        assert scene_mod.delete_scene("ghost") is False


class TestResolveScene:
    def test_named_plus_inline_dedup(self, scene_env):
        scene_mod, _ = scene_env
        scene_mod.save_scene("base", ["a", "b"])

        assert scene_mod.resolve_scene(scene="base", scene_prompt="b, c") == ["a", "b", "c"]

    def test_extra_appends_last(self, scene_env):
        scene_mod, _ = scene_env

        assert scene_mod.resolve_scene(scene_prompt="a, b", extra=["b", "c"]) == ["a", "b", "c"]

    def test_no_args_empty(self, scene_env):
        scene_mod, _ = scene_env
        assert scene_mod.resolve_scene() == []


class TestNameValidation:
    @pytest.mark.parametrize("bad_name", ["", "foo/bar", "../etc", "with space", "a$b"])
    def test_invalid_names_rejected(self, scene_env, bad_name):
        scene_mod, _ = scene_env
        with pytest.raises(ValueError, match=r"Invalid scene name"):
            scene_mod.save_scene(bad_name, ["x"])

    def test_scene_path_validates(self, scene_env):
        scene_mod, _ = scene_env
        with pytest.raises(ValueError, match=r"Invalid scene name"):
            scene_mod.scene_path("bad name")
