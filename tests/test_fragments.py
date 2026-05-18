"""Tests for the generic FragmentLibrary in tensors.fragments.

The character/scene-specific modules are thin wrappers over this class; their
behavioral coverage lives in test_characters.py and test_scenes.py. Here we
focus on cross-kind invariants (kind-aware error messages, name validation,
parse helper).
"""

from __future__ import annotations

import pytest


@pytest.fixture
def lib_factory(tmp_path):
    """Return a callable that builds an isolated FragmentLibrary per kind."""
    from tensors.fragments import FragmentLibrary

    def _make(kind: str = "characters"):
        return FragmentLibrary(kind, base_dir=tmp_path / kind)

    return _make


class TestParseElements:
    def test_splits_trims_and_dedupes(self):
        from tensors.fragments import parse_elements

        assert parse_elements("a, b , a, c, b") == ["a", "b", "c"]

    def test_empty_returns_empty(self):
        from tensors.fragments import parse_elements

        assert parse_elements("") == []
        assert parse_elements("  , , ") == []


class TestKindAwareMessages:
    def test_singular_in_error_for_plural_kind(self, lib_factory):
        lib = lib_factory("scenes")
        with pytest.raises(ValueError, match=r"Invalid scene name"):
            lib.save("bad name", ["x"])

    def test_singular_in_error_for_already_singular_kind(self, lib_factory):
        lib = lib_factory("style")  # already singular — should NOT strip the 'e'
        with pytest.raises(ValueError, match=r"Invalid style name"):
            lib.save("bad name", ["x"])

    def test_load_missing_uses_kind_singular(self, lib_factory):
        lib = lib_factory("scenes")
        with pytest.raises(FileNotFoundError, match=r"Scene 'ghost' not found"):
            lib.load("ghost")


class TestLibraryConstructor:
    def test_rejects_invalid_kind(self):
        from tensors.fragments import FragmentLibrary

        with pytest.raises(ValueError, match=r"Invalid library kind"):
            FragmentLibrary("bad kind")

    def test_accepts_dotted_alphanumeric_kind(self):
        from tensors.fragments import FragmentLibrary

        lib = FragmentLibrary("custom_v2.beta")
        assert lib.kind == "custom_v2.beta"


class TestResolve:
    def test_only_named(self, lib_factory):
        lib = lib_factory()
        lib.save("base", ["x", "y"])
        assert lib.resolve(name="base") == ["x", "y"]

    def test_only_inline(self, lib_factory):
        lib = lib_factory()
        assert lib.resolve(inline="a, b, c") == ["a", "b", "c"]

    def test_named_inline_extras_merge_deduped_in_order(self, lib_factory):
        lib = lib_factory()
        lib.save("base", ["a", "b"])
        assert lib.resolve(name="base", inline="b, c", extra=["c", "d"]) == ["a", "b", "c", "d"]

    def test_no_args_empty(self, lib_factory):
        lib = lib_factory()
        assert lib.resolve() == []

    def test_named_missing_raises(self, lib_factory):
        lib = lib_factory()
        with pytest.raises(FileNotFoundError):
            lib.resolve(name="absent")
