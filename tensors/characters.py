"""Character library: named lists of prompt elements, stored as YAML.

Characters live in ``~/.local/share/tensors/characters/<name>.yml`` and contain a
flat YAML list of strings describing a subject (appearance, identity, outfit
fragments). They are injected into a generation prompt via ``--character`` or
``--character-prompt`` on ``tsr generate``.

Format is JSON-compatible YAML — each line is ``- "value"`` so the files round-trip
through both ``json`` and any YAML parser. Manual hand-editing with plain or
single-quoted scalars is also accepted on read.
"""

from __future__ import annotations

import json
import re
from pathlib import Path  # noqa: TC003  # used in runtime return annotations exposed to typer

from tensors.config import DATA_DIR

# Storage location for character YAML files.
CHARACTERS_DIR = DATA_DIR / "characters"

# Restrict character names to a safe subset so they can't escape CHARACTERS_DIR
# via path traversal and so file listings stay tidy.
_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

# Minimum length for a quoted YAML scalar: opening + closing quote.
_MIN_QUOTED_SCALAR_LEN = 2


def _validate_name(name: str) -> None:
    if not name or not _NAME_RE.match(name):
        raise ValueError(f"Invalid character name {name!r}: only letters, digits, '.', '_', '-' allowed")


def character_path(name: str) -> Path:
    """Return the on-disk path for a character name (without ensuring it exists)."""
    _validate_name(name)
    return CHARACTERS_DIR / f"{name}.yml"


def parse_elements(value: str) -> list[str]:
    """Split a comma-separated prompt fragment into clean, order-preserving elements.

    Empty pieces and duplicates are dropped. Used by ``tsr character save`` and
    by the ``--character-prompt`` CLI flag so both share identical splitting
    semantics.
    """
    parts = [p.strip() for p in value.split(",")]
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def save_character(name: str, elements: list[str]) -> Path:
    """Persist a character's elements to disk and return the file path.

    Overwrites any existing file. Each element is written on its own line as a
    JSON-encoded YAML scalar (``- "value"``), which keeps embedded commas,
    quotes, and unicode safe.
    """
    CHARACTERS_DIR.mkdir(parents=True, exist_ok=True)
    path = character_path(name)
    body = "\n".join(f"- {json.dumps(e, ensure_ascii=False)}" for e in elements)
    path.write_text(body + "\n" if body else "")
    return path


def load_character(name: str) -> list[str]:
    """Load a character's elements. Raises ``FileNotFoundError`` if missing.

    Accepts both JSON-quoted scalars (``- "value"``), single-quoted YAML scalars
    (``- 'value'``) and bare scalars (``- value``). Blank lines and ``#`` comments
    are ignored.
    """
    path = character_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"Character {name!r} not found at {path}")

    elements: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("-"):
            # Skip any non-list-item lines (e.g. a YAML document header users
            # might add manually); we only consume flat list entries.
            continue
        item = line[1:].lstrip()
        if not item:
            continue
        # Prefer strict JSON decode (covers our own writer output and any
        # double-quoted YAML scalar). Fall back to single-quoted YAML, then
        # bare scalar.
        try:
            value = json.loads(item)
            if not isinstance(value, str):
                value = str(value)
        except json.JSONDecodeError:
            value = item[1:-1].replace("''", "'") if len(item) >= _MIN_QUOTED_SCALAR_LEN and item[0] == item[-1] == "'" else item
        elements.append(value)
    return elements


def list_characters() -> list[str]:
    """Return sorted character names. Empty list if no characters dir exists yet."""
    if not CHARACTERS_DIR.is_dir():
        return []
    return sorted(p.stem for p in CHARACTERS_DIR.glob("*.yml") if p.is_file())


def delete_character(name: str) -> bool:
    """Delete a character file. Returns True on success, False if it was missing."""
    path = character_path(name)
    if not path.is_file():
        return False
    path.unlink()
    return True


def resolve_character(
    *,
    character: str | None = None,
    character_prompt: str | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    """Merge a named character with an inline ``--character-prompt`` and optional extras.

    Resolution order (first match wins per duplicate): named character →
    ``--character-prompt`` elements → ``extra`` list. The result preserves order
    and drops duplicates, mirroring ``parse_elements``.
    """
    merged: list[str] = []
    seen: set[str] = set()

    def _push(items: list[str]) -> None:
        for item in items:
            if item and item not in seen:
                seen.add(item)
                merged.append(item)

    if character:
        _push(load_character(character))
    if character_prompt:
        _push(parse_elements(character_prompt))
    if extra:
        _push(extra)
    return merged
