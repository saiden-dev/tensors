"""Template library: full generation configs derived from models + scenes.

A *template* is a complete `tsr generate --input` payload for a specific model
and prompt: dimensions, sampler, scheduler, steps, cfg, guidance, vae, and the
scene/character lists. Templates extend scenes (which are just lists of prompt
elements) with all the family-resolved generation params, so they can be fed
straight to `tsr generate` without re-resolving anything.

Each template lives at
``~/.local/share/tensors/templates/<model_stem>/<name>.json``
and uses the same JSON shape as ``tsr template -m <model>`` standalone output,
so the on-disk format and the ad-hoc one-shot template format are identical.

Module-level :data:`TEMPLATES_DIR` is read via :func:`globals` on every call so
tests can monkeypatch it without re-importing.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path  # noqa: TC003  # used in runtime return annotations exposed to typer
from typing import Any

from tensors.config import DATA_DIR

__all__ = [
    "META_KEY_MAP",
    "SAMPLER_NORMALIZE",
    "SCHEDULER_NORMALIZE",
    "TEMPLATES_DIR",
    "build_template",
    "delete_template",
    "derive_overrides_from_images",
    "list_templates",
    "load_template",
    "param_from_civitai_meta",
    "save_template",
    "template_dir_for",
    "template_path",
]

# Default storage location. Tests may monkeypatch this; every helper below
# dereferences via globals() so overrides are picked up live.
TEMPLATES_DIR = DATA_DIR / "templates"

# Restrict template + model names to the same safe subset used by FragmentLibrary
# so they can't escape the storage dir via path traversal.
_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

# CivitAI A1111-style image meta → tsr template key mapping.
# Each entry maps a source key in the `meta` dict of a CivitAI image to a
# (tsr_key, converter) pair. The converter is either a callable (applied to the
# raw value) or a literal sentinel string ("sampler" / "scheduler") that
# triggers the corresponding normalize-and-translate path below.
META_KEY_MAP: dict[str, tuple[str, Any]] = {
    "sampler": ("sampler", "sampler"),
    "Schedule type": ("scheduler", "scheduler"),
    "steps": ("steps", int),
    "cfgScale": ("cfg", float),
    "Distilled CFG Scale": ("guidance", float),
}

# A1111 / CivitAI sampler labels → ComfyUI canonical sampler names.
# Lookup is case-folded; unknown labels fall through with whitespace replaced
# by underscores (so e.g. "DPM++ 2M Karras" we don't know about still passes
# through as "dpm++_2m_karras" — wrong but loud).
SAMPLER_NORMALIZE: dict[str, str] = {
    "euler": "euler",
    "euler a": "euler_ancestral",
    "euler ancestral": "euler_ancestral",
    "dpm++ 2m": "dpmpp_2m",
    "dpm++ 2m karras": "dpmpp_2m",
    "dpm++ 2m sde": "dpmpp_2m_sde",
    "dpm++ 2m sde karras": "dpmpp_2m_sde",
    "dpm++ 3m sde": "dpmpp_3m_sde",
    "dpm++ 3m sde karras": "dpmpp_3m_sde",
    "dpm++ sde": "dpmpp_sde",
    "dpm++ sde karras": "dpmpp_sde",
    "dpm++ 2s a": "dpmpp_2s_ancestral",
    "dpm++ 2s ancestral": "dpmpp_2s_ancestral",
    "heun": "heun",
    "ddim": "ddim",
    "lms": "lms",
    "unipc": "uni_pc",
    "uni_pc": "uni_pc",
    "lcm": "lcm",
    "dpmpp_2m": "dpmpp_2m",
    "dpmpp_2m_sde": "dpmpp_2m_sde",
}

SCHEDULER_NORMALIZE: dict[str, str] = {
    "simple": "simple",
    "normal": "normal",
    "karras": "karras",
    "sgm uniform": "sgm_uniform",
    "sgm_uniform": "sgm_uniform",
    "beta": "beta",
    "ddim_uniform": "ddim_uniform",
    "exponential": "exponential",
}


def _validate_name(name: str, kind: str = "template") -> None:
    if not name or not _NAME_RE.match(name):
        raise ValueError(f"Invalid {kind} name {name!r}: only letters, digits, '.', '_', '-' allowed")


def _root() -> Path:
    """Return the live TEMPLATES_DIR (allows tests to monkeypatch the module attr)."""
    root: Path = globals()["TEMPLATES_DIR"]
    return root


def template_dir_for(model_stem: str) -> Path:
    """Return the per-model template directory (without ensuring existence)."""
    _validate_name(model_stem, "model")
    return _root() / model_stem


def template_path(model_stem: str, name: str) -> Path:
    """Return the on-disk path for a template (without ensuring existence)."""
    _validate_name(name)
    return template_dir_for(model_stem) / f"{name}.json"


def save_template(model_stem: str, name: str, data: dict[str, Any]) -> Path:
    """Persist a template dict to disk as JSON and return its path."""
    path = template_path(model_stem, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return path


def load_template(model_stem: str, name: str) -> dict[str, Any]:
    """Load a template dict. Raises FileNotFoundError if missing."""
    path = template_path(model_stem, name)
    if not path.is_file():
        raise FileNotFoundError(f"Template {model_stem}/{name!r} not found at {path}")
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def list_templates(model_stem: str | None = None) -> list[tuple[str, str]]:
    """List saved templates as (model_stem, template_name) pairs, sorted.

    With ``model_stem`` set, restrict to that one model's subdirectory.
    """
    root = _root()
    if not root.is_dir():
        return []
    if model_stem is not None:
        _validate_name(model_stem, "model")
        sub = root / model_stem
        if not sub.is_dir():
            return []
        return [(model_stem, p.stem) for p in sorted(sub.glob("*.json")) if p.is_file()]
    out: list[tuple[str, str]] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        for p in sorted(d.glob("*.json")):
            if p.is_file():
                out.append((d.name, p.stem))
    return out


def delete_template(model_stem: str, name: str) -> bool:
    """Delete a template file. Returns True on success, False if missing."""
    path = template_path(model_stem, name)
    if not path.is_file():
        return False
    path.unlink()
    return True


def param_from_civitai_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Extract tsr-canonical generation params from a CivitAI image meta dict.

    Returns only keys that were present in the input and successfully converted.
    Unknown sampler / scheduler labels are still surfaced (with whitespace →
    underscores) rather than silently dropped — the calling layer can decide
    whether to use or ignore them.
    """
    out: dict[str, Any] = {}
    for src_key, (dst_key, converter) in META_KEY_MAP.items():
        if src_key not in meta:
            continue
        raw = meta[src_key]
        try:
            if converter == "sampler":
                normalized = str(raw).strip().lower()
                out[dst_key] = SAMPLER_NORMALIZE.get(normalized, normalized.replace(" ", "_"))
            elif converter == "scheduler":
                normalized = str(raw).strip().lower()
                out[dst_key] = SCHEDULER_NORMALIZE.get(normalized, normalized.replace(" ", "_"))
            else:
                if isinstance(raw, str):
                    # CivitAI emits "Distilled CFG Scale" as a string with comma
                    # or dot decimal separator depending on locale; normalize.
                    raw = raw.strip().replace(",", ".")
                out[dst_key] = converter(raw)
        except (ValueError, TypeError):
            # Conversion failed for this image; skip the key, keep going.
            continue
    return out


def build_template(
    *,
    model_filename: str,
    family: str | None,
    defaults: dict[str, Any],
    base_model_str: str | None,
    width: int,
    height: int,
    orientation: str,
    scene_elements: list[str],
    scene_name: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a complete generation template dict.

    Shape mirrors ``tsr template -m <model>`` output exactly so the result is
    a drop-in for ``tsr generate --input``. Showcase-derived ``overrides`` win
    over family-resolved ``defaults``.
    """
    tpl: dict[str, Any] = {
        "prompt": "",
        "negative_prompt": defaults.get("negative_prompt", ""),
        "model": model_filename,
        "width": width,
        "height": height,
        "steps": defaults.get("steps"),
        "cfg": defaults.get("cfg"),
        "sampler": defaults.get("sampler"),
        "scheduler": defaults.get("scheduler"),
        "vae": defaults.get("vae"),
        "orientation": orientation,
        "seed": -1,
        "count": 1,
    }
    # Flux models carry an explicit guidance dial; default to tsr's own 3.5
    # when no override supplies one.
    if (family or "").startswith("flux"):
        tpl["guidance"] = 3.5
    quality_prefix = defaults.get("quality_prefix", "")
    if quality_prefix:
        tpl["quality_prefix"] = quality_prefix
    # Showcase-derived overrides win over family defaults.
    if overrides:
        for k, v in overrides.items():
            tpl[k] = v
    tpl["scene"] = scene_elements
    tpl["_scene_name"] = scene_name
    tpl["_family"] = family or "unknown"
    if base_model_str:
        tpl["_base_model"] = base_model_str
    return tpl


def derive_overrides_from_images(images: list[dict[str, Any]]) -> dict[str, Any]:
    """Mode-of-each-param across showcase images that carry generation meta.

    Returns a dict of ``{tsr_key: most_common_value}`` suitable for merging
    on top of a base template. Skips images without a ``meta`` dict and any
    image meta whose param-extraction yielded nothing.
    """
    counters: dict[str, Counter[Any]] = {}
    for img in images:
        meta = img.get("meta") or {}
        if not meta:
            continue
        params = param_from_civitai_meta(meta)
        for k, v in params.items():
            counters.setdefault(k, Counter())[v] += 1

    overrides: dict[str, Any] = {}
    for k, ctr in counters.items():
        if ctr:
            overrides[k] = ctr.most_common(1)[0][0]
    return overrides
