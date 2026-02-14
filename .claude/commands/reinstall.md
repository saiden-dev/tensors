# Reinstall tensors

Bump version, reinstall locally and on junkpile.

Run the reinstall script:

```bash
python scripts/reinstall.py
```

This will:
1. Bump the patch version in pyproject.toml
2. Install locally with `uv pip install -e .`
3. Sync the project to junkpile via rsync
4. Install on junkpile with `pip install -e '.[server]'`
