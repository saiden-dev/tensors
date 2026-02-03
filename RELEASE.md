# Release Process

## Publishing a Release

Push a version tag to trigger the publish workflow:

```bash
git tag v0.1.2
git push origin v0.1.2
```

The workflow will:
1. Build the Python package
2. Publish to PyPI
3. Create a GitHub Release

## Building Platform Binaries

Platform-specific binaries can be built locally using [Nuitka](https://nuitka.net/).

### Prerequisites

```bash
pip install nuitka
pip install -e .
```

### Build Commands

**Linux / macOS:**
```bash
python -m nuitka \
  --standalone \
  --onefile \
  --output-dir=dist \
  --output-filename=tsr \
  --assume-yes-for-downloads \
  --remove-output \
  tensors.py
```

**Windows:**
```powershell
python -m nuitka `
  --standalone `
  --onefile `
  --output-dir=dist `
  --output-filename=tsr.exe `
  --assume-yes-for-downloads `
  --remove-output `
  tensors.py
```

### Output Artifacts

| Platform      | Arch  | Filename           |
|---------------|-------|--------------------|
| Linux         | x64   | `tsr-linux-x64`    |
| Linux         | arm64 | `tsr-linux-arm64`  |
| macOS         | arm64 | `tsr-macos-arm64`  |
| macOS         | x64   | `tsr-macos-x64`    |
| Windows       | x64   | `tsr-windows-x64.exe` |

### macOS Code Signing (Optional)

To sign and notarize macOS binaries:

```bash
# Sign the binary
codesign --force --options runtime --sign "Developer ID Application" dist/tsr

# Create zip for notarization
ditto -c -k --keepParent dist/tsr dist/tsr.zip

# Submit for notarization
xcrun notarytool submit dist/tsr.zip \
  --apple-id "$APPLE_ID" \
  --password "$APPLE_ID_PASSWORD" \
  --team-id "$APPLE_TEAM_ID" \
  --wait

# Staple the notarization ticket
xcrun stapler staple dist/tsr
```

Required environment variables:
- `APPLE_ID` - Apple Developer account email
- `APPLE_ID_PASSWORD` - App-specific password
- `APPLE_TEAM_ID` - Developer Team ID
