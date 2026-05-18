"""Tests for the character library (tensors.characters)."""

from __future__ import annotations

import pytest


@pytest.fixture
def char_env(tmp_path, monkeypatch):
    """Redirect CHARACTERS_DIR (and DATA_DIR) at fresh tmp_path per test.

    The module captures CHARACTERS_DIR at import time, so we monkeypatch the
    attribute directly rather than rely on env vars (config.DATA_DIR is also
    captured at import time).
    """
    from tensors import characters as char_mod

    char_dir = tmp_path / "characters"
    monkeypatch.setattr(char_mod, "CHARACTERS_DIR", char_dir)
    return char_mod, char_dir


class TestParseElements:
    def test_splits_on_commas_and_trims(self):
        from tensors.characters import parse_elements

        assert parse_elements("a, b , c") == ["a", "b", "c"]

    def test_drops_empty_and_duplicates(self):
        from tensors.characters import parse_elements

        assert parse_elements("a, , a, b, b , c") == ["a", "b", "c"]

    def test_preserves_internal_spacing(self):
        from tensors.characters import parse_elements

        assert parse_elements("blond hair, blue eyes") == ["blond hair", "blue eyes"]

    def test_empty_input(self):
        from tensors.characters import parse_elements

        assert parse_elements("") == []
        assert parse_elements("   ,  , ") == []


class TestSaveLoad:
    def test_save_creates_dir_and_file(self, char_env):
        char_mod, char_dir = char_env
        path = char_mod.save_character("cassie", ["blond hair", "broad chin"])

        assert path == char_dir / "cassie.yml"
        assert path.is_file()
        assert path.read_text() == '- "blond hair"\n- "broad chin"\n'

    def test_load_roundtrip(self, char_env):
        char_mod, _ = char_env
        elements = ["a", "b with spaces", "c, with, commas", 'd "quotes"']
        char_mod.save_character("mixed", elements)

        assert char_mod.load_character("mixed") == elements

    def test_load_missing_raises(self, char_env):
        char_mod, _ = char_env
        with pytest.raises(FileNotFoundError):
            char_mod.load_character("nope")

    def test_load_tolerates_hand_edited_single_quoted_yaml(self, char_env):
        char_mod, char_dir = char_env
        char_dir.mkdir(parents=True)
        (char_dir / "manual.yml").write_text("- 'foo'\n- 'it''s'\n- bare\n# comment\n\n")

        assert char_mod.load_character("manual") == ["foo", "it's", "bare"]

    def test_save_overwrites(self, char_env):
        char_mod, _ = char_env
        char_mod.save_character("c", ["a"])
        char_mod.save_character("c", ["b", "c"])

        assert char_mod.load_character("c") == ["b", "c"]

    def test_save_empty_list_writes_empty_file(self, char_env):
        char_mod, _char_dir = char_env
        path = char_mod.save_character("empty", [])

        assert path.is_file()
        assert path.read_text() == ""
        assert char_mod.load_character("empty") == []


class TestListAndDelete:
    def test_list_empty_when_dir_missing(self, char_env):
        char_mod, _ = char_env
        assert char_mod.list_characters() == []

    def test_list_sorted_names(self, char_env):
        char_mod, _ = char_env
        char_mod.save_character("zeta", ["x"])
        char_mod.save_character("alpha", ["y"])
        char_mod.save_character("mu", ["z"])

        assert char_mod.list_characters() == ["alpha", "mu", "zeta"]

    def test_delete_existing_returns_true(self, char_env):
        char_mod, _ = char_env
        char_mod.save_character("doomed", ["x"])

        assert char_mod.delete_character("doomed") is True
        assert char_mod.list_characters() == []

    def test_delete_missing_returns_false(self, char_env):
        char_mod, _ = char_env
        assert char_mod.delete_character("ghost") is False


class TestNameValidation:
    @pytest.mark.parametrize("bad_name", ["", "foo/bar", "../etc", "with space", "name.yml/extra", "a$b"])
    def test_invalid_names_rejected(self, char_env, bad_name):
        char_mod, _ = char_env
        with pytest.raises(ValueError, match=r"Invalid character name"):
            char_mod.save_character(bad_name, ["x"])

    @pytest.mark.parametrize("good_name", ["a", "foo_bar", "foo-bar", "foo.bar", "Mixed123", "v1.2_alpha-beta"])
    def test_valid_names_accepted(self, char_env, good_name):
        char_mod, _ = char_env
        path = char_mod.save_character(good_name, ["x"])
        assert path.is_file()

    def test_load_validates_name(self, char_env):
        char_mod, _ = char_env
        with pytest.raises(ValueError, match=r"Invalid character name"):
            char_mod.load_character("../escape")

    def test_delete_validates_name(self, char_env):
        char_mod, _ = char_env
        with pytest.raises(ValueError, match=r"Invalid character name"):
            char_mod.delete_character("../escape")


class TestResolveCharacter:
    def test_named_only(self, char_env):
        char_mod, _ = char_env
        char_mod.save_character("base", ["x", "y"])

        assert char_mod.resolve_character(character="base") == ["x", "y"]

    def test_inline_only(self, char_env):
        char_mod, _ = char_env

        assert char_mod.resolve_character(character_prompt="a, b, c") == ["a", "b", "c"]

    def test_extra_only(self, char_env):
        char_mod, _ = char_env

        assert char_mod.resolve_character(extra=["a", "b"]) == ["a", "b"]

    def test_named_plus_inline_dedup_preserves_order(self, char_env):
        char_mod, _ = char_env
        char_mod.save_character("base", ["a", "b"])

        # 'b' is shared — should appear once, in named position
        assert char_mod.resolve_character(character="base", character_prompt="b, c") == ["a", "b", "c"]

    def test_all_three_sources_merged(self, char_env):
        char_mod, _ = char_env
        char_mod.save_character("base", ["a", "b"])

        result = char_mod.resolve_character(
            character="base",
            character_prompt="c, d",
            extra=["d", "e"],
        )
        assert result == ["a", "b", "c", "d", "e"]

    def test_no_args_returns_empty(self, char_env):
        char_mod, _ = char_env
        assert char_mod.resolve_character() == []

    def test_missing_named_raises(self, char_env):
        char_mod, _ = char_env
        with pytest.raises(FileNotFoundError):
            char_mod.resolve_character(character="absent")


class TestCharacterPath:
    def test_path_does_not_require_existence(self, char_env):
        char_mod, char_dir = char_env

        # character_path is pure — doesn't touch disk
        assert char_mod.character_path("ghost") == char_dir / "ghost.yml"

    def test_path_validates_name(self, char_env):
        char_mod, _ = char_env
        with pytest.raises(ValueError):
            char_mod.character_path("bad name")
