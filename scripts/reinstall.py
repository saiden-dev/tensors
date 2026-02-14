#!/usr/bin/env python3
"""Reinstall tensors locally and on junkpile."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
JUNKPILE_HOST = "chi@junkpile"
JUNKPILE_PATH = "~/Projects/tensors"


def get_version() -> str:
    """Get current version from pyproject.toml."""
    content = PYPROJECT.read_text()
    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if not match:
        raise ValueError("Could not find version in pyproject.toml")
    return match.group(1)


def bump_version(current: str) -> str:
    """Bump patch version."""
    parts = current.split(".")
    parts[-1] = str(int(parts[-1]) + 1)
    return ".".join(parts)


def set_version(new_version: str) -> None:
    """Update version in pyproject.toml."""
    content = PYPROJECT.read_text()
    # Only replace the project version (line starts with 'version')
    content = re.sub(r'^version\s*=\s*"[^"]+"', f'version = "{new_version}"', content, count=1, flags=re.MULTILINE)
    PYPROJECT.write_text(content)
    print(f"  Updated pyproject.toml to {new_version}")


def run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    """Run a command."""
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def install_local() -> None:
    """Install locally with uv."""
    print("\n[2/4] Installing locally...")
    run(["uv", "pip", "install", "-e", "."], check=True)


def sync_to_junkpile() -> None:
    """Sync project to junkpile."""
    print("\n[3/4] Syncing to junkpile...")
    excludes = [
        ".git", ".venv", "__pycache__", "*.pyc", ".mypy_cache",
        ".pytest_cache", ".ruff_cache", ".coverage", "*.egg-info",
        "node_modules", ".tmp",
    ]
    cmd = ["rsync", "-avz", "--delete"]
    for exc in excludes:
        cmd.extend(["--exclude", exc])
    cmd.extend([f"{PROJECT_ROOT}/", f"{JUNKPILE_HOST}:{JUNKPILE_PATH}/"])
    run(cmd)


def install_junkpile() -> None:
    """Install on junkpile."""
    print("\n[4/4] Installing on junkpile...")
    run(["ssh", JUNKPILE_HOST, f"cd {JUNKPILE_PATH} && pip install -e '.[server]'"])


def main() -> int:
    """Main entry point."""
    current = get_version()
    new_version = bump_version(current)

    print(f"\n[1/4] Bumping version {current} -> {new_version}...")
    set_version(new_version)

    try:
        install_local()
        sync_to_junkpile()
        install_junkpile()
    except subprocess.CalledProcessError as e:
        print(f"\nError: Command failed with exit code {e.returncode}")
        return 1

    print(f"\n Done! tensors {new_version} installed locally and on junkpile")
    return 0


if __name__ == "__main__":
    sys.exit(main())
