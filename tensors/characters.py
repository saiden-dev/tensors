"""Character library: named lists of prompt elements for the *who*.

Characters live in ``~/.local/share/tensors/characters/<name>.yml`` and contain a
flat YAML list of strings describing a subject (appearance, identity, outfit
fragments). They are injected into a generation prompt via ``--character`` or
``--character-prompt`` on ``tsr generate`` and embedded in the ``character``
field of ``tsr template`` JSON output.

This module exposes a function-style API on top of
:class:`tensors.fragments.FragmentLibrary`. Each call instantiates a library
rooted at the current module-level ``CHARACTERS_DIR`` so tests can monkeypatch
the directory without re-importing.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003  # used in runtime return annotations

from tensors.config import DATA_DIR
from tensors.fragments import FragmentLibrary, parse_elements

__all__ = [
    "CHARACTERS_DIR",
    "character_path",
    "delete_character",
    "list_characters",
    "load_character",
    "parse_elements",
    "resolve_character",
    "save_character",
]

# Default storage location. Tests may monkeypatch this attribute; every helper
# below dereferences it via globals() so the override is picked up live.
CHARACTERS_DIR = DATA_DIR / "characters"


def _lib() -> FragmentLibrary:
    """Build a fresh library bound to the current ``CHARACTERS_DIR`` value."""
    return FragmentLibrary("characters", base_dir=globals()["CHARACTERS_DIR"])


def character_path(name: str) -> Path:
    """Return the on-disk path for a character name (without ensuring existence)."""
    return _lib().path(name)


def save_character(name: str, elements: list[str]) -> Path:
    """Persist a character's elements to disk and return the file path."""
    return _lib().save(name, elements)


def load_character(name: str) -> list[str]:
    """Load a character's elements. Raises ``FileNotFoundError`` if missing."""
    return _lib().load(name)


def list_characters() -> list[str]:
    """Return sorted character names."""
    return _lib().list()


def delete_character(name: str) -> bool:
    """Delete a character file. Returns True on success, False if missing."""
    return _lib().delete(name)


def resolve_character(
    *,
    character: str | None = None,
    character_prompt: str | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    """Merge a named character with an inline ``--character-prompt`` and extras."""
    return _lib().resolve(name=character, inline=character_prompt, extra=extra)
