"""Scene library: named lists of prompt elements for the *where*.

Scenes live in ``~/.local/share/tensors/scenes/<name>.yml`` and contain a flat
YAML list of strings describing a setting (location, lighting, camera, etc.).
They are injected into a generation prompt via ``--scene`` or ``--scene-prompt``
on ``tsr generate`` and embedded in the ``scene`` field of ``tsr template``
JSON output.

This module exposes a function-style API on top of
:class:`tensors.fragments.FragmentLibrary`. Each call instantiates a library
rooted at the current module-level ``SCENES_DIR`` so tests can monkeypatch the
directory without re-importing.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003  # used in runtime return annotations

from tensors.config import DATA_DIR
from tensors.fragments import FragmentLibrary, parse_elements

__all__ = [
    "SCENES_DIR",
    "delete_scene",
    "list_scenes",
    "load_scene",
    "parse_elements",
    "resolve_scene",
    "save_scene",
    "scene_path",
]

# Default storage location. Tests may monkeypatch this attribute; every helper
# below dereferences it via globals() so the override is picked up live.
SCENES_DIR = DATA_DIR / "scenes"


def _lib() -> FragmentLibrary:
    """Build a fresh library bound to the current ``SCENES_DIR`` value."""
    return FragmentLibrary("scenes", base_dir=globals()["SCENES_DIR"])


def scene_path(name: str) -> Path:
    """Return the on-disk path for a scene name (without ensuring existence)."""
    return _lib().path(name)


def save_scene(name: str, elements: list[str]) -> Path:
    """Persist a scene's elements to disk and return the file path."""
    return _lib().save(name, elements)


def load_scene(name: str) -> list[str]:
    """Load a scene's elements. Raises ``FileNotFoundError`` if missing."""
    return _lib().load(name)


def list_scenes() -> list[str]:
    """Return sorted scene names."""
    return _lib().list()


def delete_scene(name: str) -> bool:
    """Delete a scene file. Returns True on success, False if missing."""
    return _lib().delete(name)


def resolve_scene(
    *,
    scene: str | None = None,
    scene_prompt: str | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    """Merge a named scene with an inline ``--scene-prompt`` and extras."""
    return _lib().resolve(name=scene, inline=scene_prompt, extra=extra)
