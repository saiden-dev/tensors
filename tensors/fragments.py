"""Generic prompt-fragment library.

A *fragment* is a named, ordered list of comma-style prompt elements
(e.g. ``["blond hair", "broad chin"]``) stored as a flat YAML list on disk.
Different *kinds* of fragments (characters, scenes, …) each get their own
subdirectory under ``~/.local/share/tensors/<kind>/`` and their own
``FragmentLibrary`` instance.

Files are written as JSON-encoded YAML scalars (``- "value"``) so they round-trip
through both ``json`` and any YAML parser. Hand-edited single-quoted or bare
scalars are accepted on read.
"""

from __future__ import annotations

import json
import re
from pathlib import Path  # noqa: TC003  # used in runtime return annotations exposed to typer
from typing import TYPE_CHECKING

from tensors.config import DATA_DIR

if TYPE_CHECKING:
    # Qualified `builtins.list` is referenced in annotations inside FragmentLibrary
    # because the class defines a method named `list` that shadows the builtin
    # at class-scope name resolution. Static-only — not needed at runtime.
    import builtins

# Restrict fragment names to a safe subset so they can't escape the storage dir
# via path traversal and so file listings stay tidy.
_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

# Minimum length for a quoted YAML scalar: opening + closing quote.
_MIN_QUOTED_SCALAR_LEN = 2


class FragmentLibrary:
    """A named collection of prompt-fragment YAML files of a single *kind*.

    Each ``FragmentLibrary`` is rooted at ``<DATA_DIR>/<kind>/`` (overridable for
    tests). Instance methods are stateless wrappers around that directory.
    """

    def __init__(self, kind: str, base_dir: Path | None = None) -> None:
        """Create a library for ``kind`` (e.g. ``"characters"`` or ``"scenes"``).

        ``base_dir`` defaults to ``DATA_DIR / kind`` and is recomputed lazily so
        tests can monkeypatch ``DATA_DIR`` *or* the ``base_dir`` attribute
        directly without re-importing.
        """
        if not kind or not _NAME_RE.match(kind):
            raise ValueError(f"Invalid library kind {kind!r}")
        self.kind = kind
        self.base_dir = base_dir if base_dir is not None else DATA_DIR / kind

    # ---------- internals ----------

    @property
    def _singular(self) -> str:
        """Human-readable singular form of ``kind`` used in error messages."""
        return self.kind[:-1] if self.kind.endswith("s") else self.kind

    def _validate_name(self, name: str) -> None:
        if not name or not _NAME_RE.match(name):
            raise ValueError(f"Invalid {self._singular} name {name!r}: only letters, digits, '.', '_', '-' allowed")

    def path(self, name: str) -> Path:
        """Return the on-disk path for ``name`` (without ensuring it exists)."""
        self._validate_name(name)
        return self.base_dir / f"{name}.yml"

    # ---------- CRUD ----------

    def save(self, name: str, elements: list[str]) -> Path:
        """Persist ``elements`` to disk and return the file path.

        Overwrites any existing file. Each element is written on its own line as
        a JSON-encoded YAML scalar so embedded commas, quotes and unicode are
        safe.
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)
        path = self.path(name)
        body = "\n".join(f"- {json.dumps(e, ensure_ascii=False)}" for e in elements)
        path.write_text(body + "\n" if body else "")
        return path

    def load(self, name: str) -> list[str]:
        """Load a fragment. Raises ``FileNotFoundError`` if missing.

        Accepts JSON-quoted scalars (``- "value"``), single-quoted YAML scalars
        (``- 'value'``) and bare scalars (``- value``). Blank lines and ``#``
        comments are ignored.
        """
        path = self.path(name)
        if not path.is_file():
            raise FileNotFoundError(f"{self._singular.capitalize()} {name!r} not found at {path}")

        elements: list[str] = []
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if not line.startswith("-"):
                # Skip non-list lines (e.g. a YAML document header users might
                # add manually); we only consume flat list entries.
                continue
            item = line[1:].lstrip()
            if not item:
                continue
            try:
                value = json.loads(item)
                if not isinstance(value, str):
                    value = str(value)
            except json.JSONDecodeError:
                value = (
                    item[1:-1].replace("''", "'") if len(item) >= _MIN_QUOTED_SCALAR_LEN and item[0] == item[-1] == "'" else item
                )
            elements.append(value)
        return elements

    def list(self) -> list[str]:
        """Return sorted fragment names. Empty list if the dir doesn't exist yet."""
        if not self.base_dir.is_dir():
            return []
        return sorted(p.stem for p in self.base_dir.glob("*.yml") if p.is_file())

    def delete(self, name: str) -> bool:
        """Delete a fragment. Returns True on success, False if it was missing."""
        path = self.path(name)
        if not path.is_file():
            return False
        path.unlink()
        return True

    # ---------- helpers ----------

    def resolve(
        self,
        *,
        name: str | None = None,
        inline: str | None = None,
        # NOTE: `builtins.list` qualifier needed because this class defines a
        # `list()` method below, which shadows the builtin in class-scope name
        # resolution. Affects mypy/pyright even with `from __future__ import annotations`.
        extra: builtins.list[str] | None = None,
    ) -> builtins.list[str]:
        """Merge a named fragment with an inline CSV string and optional extras.

        Resolution order (first match wins per duplicate): named → inline → extra.
        Result preserves order and drops duplicates and empty pieces.
        """
        merged: list[str] = []
        seen: set[str] = set()

        def _push(items: list[str]) -> None:
            for item in items:
                if item and item not in seen:
                    seen.add(item)
                    merged.append(item)

        if name:
            _push(self.load(name))
        if inline:
            _push(parse_elements(inline))
        if extra:
            _push(extra)
        return merged


def parse_elements(value: str) -> list[str]:
    """Split a comma-separated prompt fragment into clean, order-preserving elements.

    Empty pieces and duplicates are dropped. Shared by ``tsr character|scene save``
    and the ``--character-prompt`` / ``--scene-prompt`` CLI flags so the splitting
    semantics stay identical across surfaces.
    """
    parts = [p.strip() for p in value.split(",")]
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out
