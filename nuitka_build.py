#!/usr/bin/env python3
"""Build script for creating a standalone binary with Nuitka."""

import subprocess
import sys
from pathlib import Path


def main():
    project_dir = Path(__file__).parent
    main_file = project_dir / "tensors.py"
    output_dir = project_dir / "dist"

    output_dir.mkdir(exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "nuitka",
        "--standalone",
        "--onefile",
        f"--output-dir={output_dir}",
        "--output-filename=sft",
        "--assume-yes-for-downloads",
        "--remove-output",
        # Include required packages
        "--include-package=rich",
        "--include-package=httpx",
        "--include-package=safetensors",
        str(main_file),
    ]

    print(f"Building with command:\n{' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)
    print(f"\nBuild complete! Binary at: {output_dir / 'sft'}")


if __name__ == "__main__":
    main()
