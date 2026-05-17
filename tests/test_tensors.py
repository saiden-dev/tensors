"""Tests for tensors module."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from rich.console import Console
from typer.testing import CliRunner

from tensors import config
from tensors.api import (
    download_model,
    fetch_civitai_by_hash,
    fetch_civitai_model,
    fetch_civitai_model_version,
    search_civitai,
)
from tensors.cli import app
from tensors.config import (
    BaseModel,
    ModelType,
    SortOrder,
    get_default_output_path,
    get_model_paths,
    load_api_key,
    load_config,
    save_config,
)
from tensors.display import (
    _format_count,
    _format_size,
    display_civitai_data,
    display_file_info,
    display_local_metadata,
    display_model_info,
    display_search_results,
)
from tensors.safetensor import get_base_name, read_safetensor_metadata

runner = CliRunner()


class TestReadSafetensorMetadata:
    """Tests for read_safetensor_metadata function."""

    def test_reads_valid_safetensor(self, temp_safetensor: Path) -> None:
        """Test reading metadata from a valid safetensor file."""
        result = read_safetensor_metadata(temp_safetensor)

        assert "metadata" in result
        assert "tensor_count" in result
        assert "header_size" in result
        assert result["metadata"]["test_key"] == "test_value"
        assert result["tensor_count"] == 0  # No tensors, just metadata

    def test_raises_on_short_file(self, tmp_path: Path) -> None:
        """Test that short files raise ValueError."""
        short_file = tmp_path / "short.safetensors"
        short_file.write_bytes(b"short")

        with pytest.raises(ValueError, match="too short"):
            read_safetensor_metadata(short_file)

    def test_raises_on_truncated_header(self, tmp_path: Path) -> None:
        """Test that truncated headers raise ValueError."""
        truncated = tmp_path / "truncated.safetensors"
        # Write header size that claims 1000 bytes but only provide 10
        with truncated.open("wb") as f:
            f.write(struct.pack("<Q", 1000))
            f.write(b"x" * 10)

        with pytest.raises(ValueError, match="truncated"):
            read_safetensor_metadata(truncated)

    def test_raises_on_huge_header_size(self, tmp_path: Path) -> None:
        """Test that unreasonably large header sizes raise ValueError."""
        huge = tmp_path / "huge.safetensors"
        with huge.open("wb") as f:
            f.write(struct.pack("<Q", 200_000_000))  # 200MB header

        with pytest.raises(ValueError, match="Invalid header size"):
            read_safetensor_metadata(huge)


class TestGetBaseName:
    """Tests for get_base_name function."""

    def test_removes_safetensors_extension(self) -> None:
        """Test that .safetensors extension is removed."""
        assert get_base_name(Path("model.safetensors")) == "model"

    def test_removes_sft_extension(self) -> None:
        """Test that .sft extension is removed."""
        assert get_base_name(Path("model.sft")) == "model"

    def test_handles_uppercase_extension(self) -> None:
        """Test that uppercase extensions are handled."""
        assert get_base_name(Path("model.SAFETENSORS")) == "model"

    def test_preserves_name_without_known_extension(self) -> None:
        """Test that unknown extensions use stem."""
        assert get_base_name(Path("model.bin")) == "model"


class TestGetDefaultOutputPath:
    """Tests for get_default_output_path function."""

    def test_returns_checkpoint_path(self) -> None:
        """Test that Checkpoint type returns checkpoints directory."""
        result = get_default_output_path("Checkpoint")
        assert result is not None
        assert "checkpoints" in str(result)

    def test_returns_lora_path(self) -> None:
        """Test that LORA type returns loras directory."""
        result = get_default_output_path("LORA")
        assert result is not None
        assert "loras" in str(result)

    def test_returns_none_for_unknown_type(self) -> None:
        """Test that unknown types return None."""
        assert get_default_output_path("UnknownType") is None
        assert get_default_output_path(None) is None


class TestGetModelPaths:
    """Tests for get_model_paths function."""

    def test_returns_dict_with_all_types(self) -> None:
        """Test that all model types are included."""
        paths = get_model_paths()
        assert isinstance(paths, dict)
        assert "Checkpoint" in paths
        assert "LORA" in paths
        assert "LoCon" in paths
        assert "TextualInversion" in paths
        assert "VAE" in paths
        assert "Controlnet" in paths

    def test_config_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that config.toml paths override defaults."""
        # Create a config file with custom path
        config_file = tmp_path / "config.toml"
        config_file.write_text('[paths]\ncheckpoints = "/custom/checkpoints"\n')
        monkeypatch.setattr(config, "CONFIG_FILE", config_file)

        paths = get_model_paths()
        assert paths["Checkpoint"] == Path("/custom/checkpoints")
        # Other types should still be defaults
        assert "loras" in str(paths["LORA"])

    def test_get_default_output_path_uses_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that get_default_output_path respects config overrides."""
        config_file = tmp_path / "config.toml"
        config_file.write_text('[paths]\nloras = "/custom/loras"\n')
        monkeypatch.setattr(config, "CONFIG_FILE", config_file)

        result = get_default_output_path("LORA")
        assert result == Path("/custom/loras")

        # LoCon should also use the loras path
        result = get_default_output_path("LoCon")
        assert result == Path("/custom/loras")


class TestLoadApiKey:
    """Tests for load_api_key function."""

    def test_returns_env_var_if_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that environment variable takes precedence."""
        monkeypatch.setenv("CIVITAI_API_KEY", "test-key-from-env")
        assert load_api_key() == "test-key-from-env"

    def test_returns_none_if_no_key(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test that None is returned when no key is available."""
        monkeypatch.delenv("CIVITAI_API_KEY", raising=False)
        # Point config and legacy files to nonexistent paths
        monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "nonexistent" / "config.toml")
        monkeypatch.setattr(config, "LEGACY_RC_FILE", tmp_path / "nonexistent")
        assert load_api_key() is None

    def test_returns_key_from_config_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test that key is loaded from TOML config file."""
        monkeypatch.delenv("CIVITAI_API_KEY", raising=False)
        config_file = tmp_path / "config.toml"
        config_file.write_text('[api]\ncivitai_key = "key-from-config"\n')
        monkeypatch.setattr(config, "CONFIG_FILE", config_file)
        monkeypatch.setattr(config, "LEGACY_RC_FILE", tmp_path / "nonexistent")
        assert load_api_key() == "key-from-config"

    def test_returns_key_from_legacy_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test that key is loaded from legacy RC file when no config exists."""
        monkeypatch.delenv("CIVITAI_API_KEY", raising=False)
        legacy_file = tmp_path / ".sftrc"
        legacy_file.write_text("legacy-key")
        monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "nonexistent" / "config.toml")
        monkeypatch.setattr(config, "LEGACY_RC_FILE", legacy_file)
        assert load_api_key() == "legacy-key"


class TestSaveConfig:
    """Tests for save_config function."""

    def test_saves_simple_config(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test saving a simple config."""
        config_dir = tmp_path / "config"
        config_file = config_dir / "config.toml"
        monkeypatch.setattr(config, "CONFIG_DIR", config_dir)
        monkeypatch.setattr(config, "CONFIG_FILE", config_file)

        save_config({"key": "value"})

        assert config_file.exists()
        content = config_file.read_text()
        assert 'key = "value"' in content

    def test_saves_nested_config(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test saving a nested config with sections."""
        config_dir = tmp_path / "config"
        config_file = config_dir / "config.toml"
        monkeypatch.setattr(config, "CONFIG_DIR", config_dir)
        monkeypatch.setattr(config, "CONFIG_FILE", config_file)

        save_config({"api": {"civitai_key": "test-key"}})

        content = config_file.read_text()
        assert "[api]" in content
        assert 'civitai_key = "test-key"' in content

    def test_saves_numeric_values(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test saving numeric values without quotes."""
        config_dir = tmp_path / "config"
        config_file = config_dir / "config.toml"
        monkeypatch.setattr(config, "CONFIG_DIR", config_dir)
        monkeypatch.setattr(config, "CONFIG_FILE", config_file)

        save_config({"timeout": 30})

        content = config_file.read_text()
        assert "timeout = 30" in content


class TestLoadConfig:
    """Tests for load_config function."""

    def test_returns_empty_dict_if_no_config(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test that empty dict is returned when config file doesn't exist."""
        monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "nonexistent.toml")
        assert load_config() == {}


class TestEnums:
    """Tests for enum to_api methods."""

    def test_model_type_to_api(self) -> None:
        """Test ModelType enum to_api conversion."""
        assert ModelType.checkpoint.to_api() == "Checkpoint"
        assert ModelType.lora.to_api() == "LORA"
        assert ModelType.embedding.to_api() == "TextualInversion"
        assert ModelType.vae.to_api() == "VAE"
        assert ModelType.controlnet.to_api() == "Controlnet"
        assert ModelType.locon.to_api() == "LoCon"

    def test_base_model_to_api(self) -> None:
        """Test BaseModel enum to_api conversion."""
        assert BaseModel.sd15.to_api() == "SD 1.5"
        assert BaseModel.sdxl.to_api() == "SDXL 1.0"
        assert BaseModel.pony.to_api() == "Pony"
        assert BaseModel.flux_dev.to_api() == "Flux.1 D"
        assert BaseModel.illustrious.to_api() == "Illustrious"

    def test_sort_order_to_api(self) -> None:
        """Test SortOrder enum to_api conversion."""
        assert SortOrder.downloads.to_api() == "Most Downloaded"
        assert SortOrder.rating.to_api() == "Highest Rated"
        assert SortOrder.newest.to_api() == "Newest"


class TestModelFamilyDetection:
    """Tests for detect_model_family and get_model_generation_defaults."""

    def test_detect_pony_from_base_model(self) -> None:
        """Test detecting Pony family from base_model field."""
        from tensors.config import detect_model_family

        assert detect_model_family("model.safetensors", "Pony") == "pony"
        assert detect_model_family("anything.safetensors", "PONY") == "pony"

    def test_detect_pony_from_filename(self) -> None:
        """Test detecting Pony family from filename."""
        from tensors.config import detect_model_family

        assert detect_model_family("ponyDiffusionV6XL.safetensors") == "pony"
        assert detect_model_family("autismmixPony_v10.safetensors") == "pony"

    def test_detect_illustrious_from_base_model(self) -> None:
        """Test detecting Illustrious family from base_model field."""
        from tensors.config import detect_model_family

        assert detect_model_family("model.safetensors", "Illustrious") == "illustrious"

    def test_detect_illustrious_from_filename(self) -> None:
        """Test detecting Illustrious family from filename."""
        from tensors.config import detect_model_family

        assert detect_model_family("illustriousXL_v10.safetensors") == "illustrious"
        assert detect_model_family("noobaiXL_v10.safetensors") == "illustrious"

    def test_detect_flux_variants(self) -> None:
        """Test detecting Flux family variants."""
        from tensors.config import detect_model_family

        assert detect_model_family("flux1-dev.safetensors") == "flux"
        assert detect_model_family("flux1-schnell.safetensors") == "flux_schnell"
        assert detect_model_family("model.safetensors", "Flux.1 D") == "flux"
        assert detect_model_family("model.safetensors", "Flux.1 S schnell") == "flux_schnell"

    def test_detect_flux_unet_lust(self) -> None:
        """lust_*.safetensors → flux2_klein (Klein detection wins via filename pattern).

        Originally classified as flux_unet, but lust_v10 is actually Flux.2 Klein 9B
        (per CivitAI base_model). Klein detection runs before flux_unet, so the
        lust_ pattern in FLUX2_KLEIN_PATTERNS takes precedence.
        """
        from tensors.config import detect_model_family

        assert detect_model_family("lust_v10.safetensors") == "flux2_klein"
        assert detect_model_family("LUST_v10.safetensors") == "flux2_klein"

    def test_detect_flux_unet_cyberrealistic(self) -> None:
        """cyberrealisticFlux_*.safetensors → flux_unet (intercepts generic 'flux' match)."""
        from tensors.config import detect_model_family

        assert detect_model_family("cyberrealisticFlux_v25.safetensors") == "flux_unet"

    def test_detect_flux_unet_getphat(self) -> None:
        """getphatFLUXReality_*.safetensors → flux_unet."""
        from tensors.config import detect_model_family

        assert detect_model_family("getphatFLUXReality_v11Softcore.safetensors") == "flux_unet"

    def test_detect_flux_unet_moody(self) -> None:
        """moodyDesireMix_*.safetensors → flux2_klein (Klein, not Flux.1 D).

        Originally classified as flux_unet, but moodyDesireMix is Flux.2 Klein
        9B per CivitAI. Klein detection wins via the moodydesire filename pattern.
        """
        from tensors.config import detect_model_family

        assert detect_model_family("moodyDesireMix_v20PRO.safetensors") == "flux2_klein"

    def test_detect_flux_unet_fcfluxpony(self) -> None:
        """fcFluxPony*.safetensors → flux_unet (intercepts flux + fluxpony)."""
        from tensors.config import detect_model_family

        assert (
            detect_model_family("fcFluxPonyPerfectBase_fcFluxPerfectBase.safetensors")
            == "flux_unet"
        )

    def test_detect_flux_unet_overrides_base_model(self) -> None:
        """Filename UNet-only pattern wins over a (likely wrong) CivitAI base_model tag."""
        from tensors.config import detect_model_family

        # cyberrealisticFlux: filename pattern wins over wrong "Pony" tag → flux_unet.
        assert (
            detect_model_family("cyberrealisticFlux_v25.safetensors", "Pony") == "flux_unet"
        )
        # getphat: filename pattern wins over wrong "SDXL 1.0" tag → flux_unet.
        assert (
            detect_model_family("getphatFLUXReality_v11.safetensors", "SDXL 1.0")
            == "flux_unet"
        )

    def test_flux_unet_family_defaults_has_external_clip(self) -> None:
        """flux_unet preset advertises external_clip + clip filenames."""
        from tensors.config import MODEL_FAMILY_DEFAULTS

        defaults = MODEL_FAMILY_DEFAULTS["flux_unet"]
        assert defaults["external_clip"] is True
        assert defaults["clip_l"] == "clip_l.safetensors"
        assert defaults["clip_t5"] == "t5xxl_fp16.safetensors"
        # Sanity: same sampling profile as flux
        assert defaults["cfg"] == 1.0
        assert defaults["guidance"] == 3.5
        assert defaults["vae"] == "ae.safetensors"

    def test_get_model_generation_defaults_flux_unet(self) -> None:
        """flux_unet model resolves to the flux_unet preset with external_clip set."""
        from tensors.config import get_model_generation_defaults

        # getphat is genuinely Flux.1 D UNet-only (not Klein).
        defaults = get_model_generation_defaults("getphatFLUXReality_v11.safetensors")
        assert defaults["family"] == "flux_unet"
        assert defaults["external_clip"] is True
        assert defaults["sampler"] == "euler"
        assert defaults["scheduler"] == "simple"

    # ---- Flux.2 Klein 9B detection + workflow ----

    def test_detect_flux2_klein_from_base_model(self) -> None:
        """base_model='Flux.2 Klein 9B-base' → flux2_klein."""
        from tensors.config import detect_model_family

        assert detect_model_family("anything.safetensors", "Flux.2 Klein 9B-base") == "flux2_klein"
        assert detect_model_family("anything.safetensors", "Flux.2 Klein 9B") == "flux2_klein"
        # Compact variant ("flux2 klein" without the dot) — also accepted.
        assert detect_model_family("anything.safetensors", "flux2 Klein") == "flux2_klein"

    def test_detect_flux2_klein_from_filename(self) -> None:
        """Filename fallback: lust_ and moodydesire → flux2_klein even without DB metadata."""
        from tensors.config import detect_model_family

        assert detect_model_family("lust_v10.safetensors") == "flux2_klein"
        assert detect_model_family("moodyDesireMix_v20PRO.safetensors") == "flux2_klein"

    def test_detect_flux2_klein_overrides_flux_unet(self) -> None:
        """Klein detection runs BEFORE flux_unet, so Klein patterns win."""
        from tensors.config import detect_model_family

        # lust_ matches both FLUX2_KLEIN_PATTERNS and FLUX_UNET_ONLY_PATTERNS.
        # Klein check runs first → flux2_klein.
        assert detect_model_family("lust_v10.safetensors") == "flux2_klein"
        # Even with wrong base_model, Klein filename wins.
        assert detect_model_family("lust_v10.safetensors", "SDXL 1.0") == "flux2_klein"

    def test_flux2_klein_family_defaults(self) -> None:
        """flux2_klein preset has external_clip + Qwen3 encoder + Flux.2 VAE."""
        from tensors.config import MODEL_FAMILY_DEFAULTS

        defaults = MODEL_FAMILY_DEFAULTS["flux2_klein"]
        assert defaults["external_clip"] is True
        assert defaults["clip_encoder"] == "qwen_3_8b_fp8mixed.safetensors"
        assert defaults["clip_type"] == "flux2"
        assert defaults["vae"] == "flux2-vae.safetensors"
        assert defaults["cfg"] == 1.0
        assert defaults["guidance"] == 3.5

    def test_build_workflow_flux2_klein_uses_cliploader(self) -> None:
        """Flux.2 Klein workflow uses CLIPLoader(type=flux2) + EmptyFlux2LatentImage +
        custom-sampling pipeline (no plain KSampler, no DualCLIPLoader)."""
        from tensors.comfyui import _build_workflow

        wf = _build_workflow(prompt="test", model="lust_v10.safetensors", seed=42)

        class_types = {node["class_type"] for node in wf.values()}
        # Required Flux.2-specific nodes
        assert "CLIPLoader" in class_types
        assert "EmptyFlux2LatentImage" in class_types
        assert "Flux2Scheduler" in class_types
        assert "BasicGuider" in class_types
        assert "SamplerCustomAdvanced" in class_types
        assert "RandomNoise" in class_types
        # Forbidden — these belong to Flux.1 / SDXL paths
        assert "DualCLIPLoader" not in class_types
        assert "KSampler" not in class_types
        assert "EmptySD3LatentImage" not in class_types
        assert "CheckpointLoaderSimple" not in class_types
        assert "ModelSamplingFlux" not in class_types
        # Verify CLIPLoader is configured correctly
        clip_nodes = [n for n in wf.values() if n["class_type"] == "CLIPLoader"]
        assert len(clip_nodes) == 1
        assert clip_nodes[0]["inputs"]["type"] == "flux2"
        assert clip_nodes[0]["inputs"]["clip_name"] == "qwen_3_8b_fp8mixed.safetensors"
        # VAE is the Flux.2 one, not Flux.1's ae.safetensors
        vae_nodes = [n for n in wf.values() if n["class_type"] == "VAELoader"]
        assert vae_nodes[0]["inputs"]["vae_name"] == "flux2-vae.safetensors"

    def test_build_workflow_flux2_klein_with_lora(self) -> None:
        """LoRA injection inserts LoraLoader and reroutes BasicGuider + text encoders."""
        from tensors.comfyui import _build_workflow

        wf = _build_workflow(
            prompt="test",
            model="lust_v10.safetensors",
            seed=42,
            lora_name="some_flux_lora.safetensors",
            lora_strength=0.8,
        )

        # LoRA node added at "110"
        assert "110" in wf
        assert wf["110"]["class_type"] == "LoraLoader"
        assert wf["110"]["inputs"]["lora_name"] == "some_flux_lora.safetensors"
        assert wf["110"]["inputs"]["strength_model"] == 0.8
        # BasicGuider (model consumer) now wired to LoRA output
        assert wf["154"]["inputs"]["model"] == ["110", 0]
        # Both text encoders re-routed to LoRA clip output
        assert wf["130"]["inputs"]["clip"] == ["110", 1]
        assert wf["131"]["inputs"]["clip"] == ["110", 1]

    def test_detect_sdxl_variants(self) -> None:
        """Test detecting SDXL family variants."""
        from tensors.config import detect_model_family

        assert detect_model_family("juggernautXL_v9.safetensors") == "sdxl"
        assert detect_model_family("sdxl_lightning_4step.safetensors") == "sdxl_lightning"
        assert detect_model_family("sdxl_turbo.safetensors") == "sdxl_turbo"
        assert detect_model_family("model.safetensors", "SDXL 1.0") == "sdxl"
        assert detect_model_family("model.safetensors", "SDXL Lightning") == "sdxl_lightning"
        assert detect_model_family("model.safetensors", "SDXL Turbo") == "sdxl_turbo"

    def test_detect_sd15_variants(self) -> None:
        """Test detecting SD 1.5 family variants."""
        from tensors.config import detect_model_family

        assert detect_model_family("dreamshaper_v8.safetensors") == "sd15"
        assert detect_model_family("sd15_lcm.safetensors") == "sd15_lcm"
        assert detect_model_family("model.safetensors", "SD 1.5") == "sd15"
        assert detect_model_family("model.safetensors", "SD 1.5 LCM") == "sd15_lcm"

    def test_detect_unknown_returns_none(self) -> None:
        """Test that unknown models return None."""
        from tensors.config import detect_model_family

        assert detect_model_family("random_model.safetensors") is None
        assert detect_model_family("unknown.safetensors", "Unknown") is None

    def test_get_model_generation_defaults_pony(self) -> None:
        """Test getting generation defaults for Pony models."""
        from tensors.config import get_model_generation_defaults

        defaults = get_model_generation_defaults("ponyDiffusionV6XL.safetensors")
        assert defaults["family"] == "pony"
        assert defaults["sampler"] == "euler_ancestral"
        assert defaults["scheduler"] == "normal"
        assert defaults["steps"] == 25
        assert defaults["cfg"] == 6.5

    def test_get_model_generation_defaults_flux(self) -> None:
        """Test getting generation defaults for Flux models.

        Flux Dev is guidance-distilled: KSampler.cfg MUST be 1.0; the real
        prompt-adherence dial is the FluxGuidance node's ``guidance`` value.
        See https://comfyanonymous.github.io/ComfyUI_examples/flux/
        """
        from tensors.config import get_model_generation_defaults

        defaults = get_model_generation_defaults("flux1-dev-fp8.safetensors")
        assert defaults["family"] == "flux"
        assert defaults["sampler"] == "euler"
        assert defaults["scheduler"] == "simple"
        assert defaults["cfg"] == 1.0
        assert defaults["guidance"] == 3.5

    def test_get_model_generation_defaults_flux_schnell(self) -> None:
        """Test getting generation defaults for Flux Schnell models."""
        from tensors.config import get_model_generation_defaults

        defaults = get_model_generation_defaults("flux1-schnell.safetensors")
        assert defaults["family"] == "flux_schnell"
        assert defaults["steps"] == 4
        assert defaults["cfg"] == 1.0
        assert defaults["guidance"] == 3.5

    def test_detect_zimage(self) -> None:
        """Test detecting ZImageTurbo family."""
        from tensors.config import detect_model_family

        assert detect_model_family("zimageturbo_v1.safetensors") == "zimage"
        assert detect_model_family("ZIMAGE_xl.safetensors") == "zimage"
        assert detect_model_family("model.safetensors", "ZImageTurbo") == "zimage"

    def test_get_model_generation_defaults_zimage(self) -> None:
        """Test getting generation defaults for ZImageTurbo models."""
        from tensors.config import get_model_generation_defaults

        defaults = get_model_generation_defaults("zimageturbo_v1.safetensors")
        assert defaults["family"] == "zimage"
        assert defaults["sampler"] == "euler"
        assert defaults["scheduler"] == "simple"
        assert defaults["steps"] == 4
        assert defaults["cfg"] == 1.0
        assert defaults["vae"] == "ae.safetensors"

    def test_flux_uses_ae_vae(self) -> None:
        """Test that Flux models use ae.safetensors VAE."""
        from tensors.config import get_model_generation_defaults

        defaults = get_model_generation_defaults("flux1-dev-fp8.safetensors")
        assert defaults["vae"] == "ae.safetensors"

        defaults_schnell = get_model_generation_defaults("flux1-schnell.safetensors")
        assert defaults_schnell["vae"] == "ae.safetensors"

    def test_get_model_generation_defaults_sdxl_lightning(self) -> None:
        """Test getting generation defaults for SDXL Lightning models."""
        from tensors.config import get_model_generation_defaults

        defaults = get_model_generation_defaults("sdxl_lightning_4step.safetensors")
        assert defaults["family"] == "sdxl_lightning"
        assert defaults["sampler"] == "euler"
        assert defaults["scheduler"] == "sgm_uniform"
        assert defaults["steps"] == 8
        assert defaults["cfg"] == 2.0

    def test_get_model_generation_defaults_unknown_falls_back_to_sdxl(self) -> None:
        """Test that unknown models fall back to SDXL defaults."""
        from tensors.config import get_model_generation_defaults

        defaults = get_model_generation_defaults("unknown_model.safetensors")
        assert defaults["family"] is None
        assert defaults["sampler"] == "dpmpp_2m"
        assert defaults["scheduler"] == "karras"


class TestFluxWorkflowBuilder:
    """Tests for the Flux-specific branch of _build_workflow."""

    def test_flux_dispatch_uses_flux_template(self) -> None:
        """Building a workflow for a Flux model emits the Flux node graph."""
        from tensors.comfyui import _build_workflow

        wf = _build_workflow(prompt="a cat", model="flux1-dev-fp8.safetensors")

        # Flux template uses node IDs in the 100s; default SDXL template uses single digits.
        assert "100" in wf and wf["100"]["class_type"] == "CheckpointLoaderSimple"
        assert "120" in wf and wf["120"]["class_type"] == "ModelSamplingFlux"
        assert "140" in wf and wf["140"]["class_type"] == "FluxGuidance"
        assert "132" in wf and wf["132"]["class_type"] == "ConditioningZeroOut"
        assert "150" in wf and wf["150"]["class_type"] == "EmptySD3LatentImage"
        assert "3" not in wf  # default SDXL KSampler ID must NOT be present

    def test_flux_ksampler_cfg_locked_to_one(self) -> None:
        """KSampler cfg MUST be 1.0 for Flux Dev — caller cfg must NOT leak through."""
        from tensors.comfyui import _build_workflow

        wf = _build_workflow(prompt="a cat", model="flux1-dev-fp8.safetensors", cfg=7.5)
        assert wf["160"]["inputs"]["cfg"] == 1.0
        # The caller's cfg=7.5 should be re-routed to FluxGuidance
        assert wf["140"]["inputs"]["guidance"] == 7.5

    def test_flux_explicit_guidance_wins_over_cfg(self) -> None:
        """Explicit guidance overrides re-interpreted cfg."""
        from tensors.comfyui import _build_workflow

        wf = _build_workflow(prompt="a cat", model="flux1-dev-fp8.safetensors", cfg=7.5, guidance=4.0)
        assert wf["140"]["inputs"]["guidance"] == 4.0

    def test_flux_default_guidance_from_preset(self) -> None:
        """No caller value -> preset guidance (3.5) wins."""
        from tensors.comfyui import _build_workflow

        wf = _build_workflow(prompt="a cat", model="flux1-dev-fp8.safetensors")
        assert wf["140"]["inputs"]["guidance"] == 3.5

    def test_flux_lora_injection(self) -> None:
        """LoRA injects node 110 and reroutes ModelSamplingFlux + CLIPTextEncodes."""
        from tensors.comfyui import _build_workflow

        wf = _build_workflow(
            prompt="a cat",
            model="flux1-dev-fp8.safetensors",
            lora_name="my_style.safetensors",
            lora_strength=0.7,
        )
        assert "110" in wf and wf["110"]["class_type"] == "LoraLoader"
        assert wf["110"]["inputs"]["lora_name"] == "my_style.safetensors"
        assert wf["110"]["inputs"]["strength_model"] == 0.7
        # Downstream consumers must read from the LoRA node
        assert wf["120"]["inputs"]["model"] == ["110", 0]
        assert wf["130"]["inputs"]["clip"] == ["110", 1]
        assert wf["131"]["inputs"]["clip"] == ["110", 1]

    def test_flux_external_vae_swaps_decoder_input(self) -> None:
        """Providing an external VAE adds node 171 (VAELoader) and rewires VAEDecode."""
        from tensors.comfyui import _build_workflow

        wf = _build_workflow(
            prompt="a cat",
            model="flux1-dev-fp8.safetensors",
            vae="ae.safetensors",
        )
        assert "171" in wf and wf["171"]["class_type"] == "VAELoader"
        assert wf["171"]["inputs"]["vae_name"] == "ae.safetensors"
        assert wf["170"]["inputs"]["vae"] == ["171", 0]

    def test_flux_model_sampling_dimensions_match_latent(self) -> None:
        """ModelSamplingFlux width/height must equal the latent dimensions for correct shift."""
        from tensors.comfyui import _build_workflow

        wf = _build_workflow(
            prompt="a cat",
            model="flux1-dev-fp8.safetensors",
            width=1216,
            height=832,
        )
        assert wf["120"]["inputs"]["width"] == 1216
        assert wf["120"]["inputs"]["height"] == 832
        assert wf["150"]["inputs"]["width"] == 1216
        assert wf["150"]["inputs"]["height"] == 832

    def test_non_flux_model_uses_default_template(self) -> None:
        """SDXL/Pony/etc. checkpoints continue to use the legacy template."""
        from tensors.comfyui import _build_workflow

        wf = _build_workflow(prompt="a cat", model="ponyDiffusionV6XL.safetensors")
        # Default SDXL template has KSampler at node "3"
        assert "3" in wf and wf["3"]["class_type"] == "KSampler"
        # Flux-specific nodes must NOT be present
        assert "140" not in wf
        assert "120" not in wf


class TestFluxUnetWorkflowBuilder:
    """Tests for the UNet-only Flux workflow (split CLIP/T5/VAE loaders)."""

    def test_build_workflow_flux_unet_uses_dual_clip_loader(self) -> None:
        """flux_unet checkpoints emit UNETLoader + DualCLIPLoader + VAELoader and NO CheckpointLoaderSimple."""
        from tensors.comfyui import _build_workflow

        # getphat is genuinely Flux.1 D UNet-only. lust_v10 used to live here
        # but is actually Flux.2 Klein — see TestFamilyDetection.
        wf = _build_workflow(prompt="a cat", model="getphatFLUXReality_v11.safetensors")

        # Three split loaders at the canonical IDs
        assert wf["100"]["class_type"] == "UNETLoader"
        assert wf["101"]["class_type"] == "DualCLIPLoader"
        assert wf["102"]["class_type"] == "VAELoader"

        # The combined checkpoint loader must NOT appear anywhere.
        for node in wf.values():
            assert node["class_type"] != "CheckpointLoaderSimple"

        # UNet filename plumbed through
        assert wf["100"]["inputs"]["unet_name"] == "getphatFLUXReality_v11.safetensors"

        # DualCLIPLoader configured for flux with both encoders
        clip_inputs = wf["101"]["inputs"]
        assert clip_inputs["clip_name1"] == "clip_l.safetensors"
        assert clip_inputs["clip_name2"] == "t5xxl_fp16.safetensors"
        assert clip_inputs["type"] == "flux"

        # VAE defaults to ae.safetensors
        assert wf["102"]["inputs"]["vae_name"] == "ae.safetensors"

        # Downstream wiring: CLIPTextEncode reads from DualCLIPLoader, VAEDecode from VAELoader
        assert wf["130"]["inputs"]["clip"] == ["101", 0]
        assert wf["131"]["inputs"]["clip"] == ["101", 0]
        assert wf["170"]["inputs"]["vae"] == ["102", 0]
        # ModelSamplingFlux still reads MODEL from node 100 (now UNETLoader)
        assert wf["120"]["inputs"]["model"] == ["100", 0]

    def test_flux_unet_inherits_flux_sampling_profile(self) -> None:
        """flux_unet locks KSampler.cfg to 1.0 and exposes the FluxGuidance dial."""
        from tensors.comfyui import _build_workflow

        wf = _build_workflow(
            prompt="a cat", model="getphatFLUXReality_v11.safetensors", cfg=7.5
        )
        assert wf["160"]["inputs"]["cfg"] == 1.0
        # The caller's cfg=7.5 should re-route to FluxGuidance (same precedence as plain flux)
        assert wf["140"]["inputs"]["guidance"] == 7.5

    def test_flux_unet_lora_injection_wires_split_loaders(self) -> None:
        """LoRA on flux_unet injects node 110 reading MODEL from UNETLoader and CLIP from DualCLIPLoader."""
        from tensors.comfyui import _build_workflow

        wf = _build_workflow(
            prompt="a cat",
            model="getphatFLUXReality_v11.safetensors",
            lora_name="my_style.safetensors",
            lora_strength=0.6,
        )
        assert wf["110"]["class_type"] == "LoraLoader"
        assert wf["110"]["inputs"]["lora_name"] == "my_style.safetensors"
        assert wf["110"]["inputs"]["strength_model"] == 0.6
        # LoRA reads model from UNETLoader, clip from DualCLIPLoader
        assert wf["110"]["inputs"]["model"] == ["100", 0]
        assert wf["110"]["inputs"]["clip"] == ["101", 0]
        # Downstream consumers now read from the LoRA outputs
        assert wf["120"]["inputs"]["model"] == ["110", 0]
        assert wf["130"]["inputs"]["clip"] == ["110", 1]
        assert wf["131"]["inputs"]["clip"] == ["110", 1]

    def test_flux_unet_external_vae_overrides_default(self) -> None:
        """Caller-provided VAE replaces ae.safetensors on the VAELoader (no new node)."""
        from tensors.comfyui import _build_workflow

        wf = _build_workflow(
            prompt="a cat",
            model="getphatFLUXReality_v11.safetensors",
            vae="other_vae.safetensors",
        )
        assert wf["102"]["inputs"]["vae_name"] == "other_vae.safetensors"
        # And VAEDecode still wires through node 102 — no shadow node 171.
        assert wf["170"]["inputs"]["vae"] == ["102", 0]
        assert "171" not in wf

    def test_flux_unet_via_explicit_family_override(self) -> None:
        """A non-pattern filename still gets the UNet workflow when -F flux_unet is forced.

        We can't pass --family directly to _build_workflow (it auto-detects from
        the filename), but a checkpoint matching a UNet-only pattern proves the
        family→workflow dispatch end-to-end.
        """
        from tensors.comfyui import _build_workflow

        # fcFluxPony is genuinely Flux.1 D (UNet-only) — moodyDesire was Klein.
        wf = _build_workflow(
            prompt="a cat",
            model="fcFluxPonyPerfectBase_fcFluxPerfectBase.safetensors",
        )
        assert wf["100"]["class_type"] == "UNETLoader"
        assert wf["101"]["class_type"] == "DualCLIPLoader"


class TestDisplayFormatters:
    """Tests for display formatting functions."""

    def test_format_size_kb(self) -> None:
        """Test formatting sizes in KB."""
        assert _format_size(500) == "500 KB"
        assert _format_size(1023) == "1023 KB"

    def test_format_size_mb(self) -> None:
        """Test formatting sizes in MB."""
        assert _format_size(1024) == "1.0 MB"
        assert _format_size(2048) == "2.0 MB"
        assert _format_size(1024 * 500) == "500.0 MB"

    def test_format_size_gb(self) -> None:
        """Test formatting sizes in GB."""
        assert _format_size(1024 * 1024) == "1.00 GB"
        assert _format_size(1024 * 1024 * 2.5) == "2.50 GB"

    def test_format_count_small(self) -> None:
        """Test formatting small counts."""
        assert _format_count(0) == "0"
        assert _format_count(999) == "999"

    def test_format_count_thousands(self) -> None:
        """Test formatting counts in thousands."""
        assert _format_count(1000) == "1.0K"
        assert _format_count(5500) == "5.5K"
        assert _format_count(999999) == "1000.0K"

    def test_format_count_millions(self) -> None:
        """Test formatting counts in millions."""
        assert _format_count(1_000_000) == "1.0M"
        assert _format_count(2_500_000) == "2.5M"


class TestDisplayFunctions:
    """Tests for display functions with console output."""

    def test_display_file_info(self, temp_safetensor: Path) -> None:
        """Test display_file_info renders without error."""
        console = Console(force_terminal=True, width=80)
        metadata = read_safetensor_metadata(temp_safetensor)
        # Should not raise
        display_file_info(temp_safetensor, metadata, "ABC123", console)

    def test_display_local_metadata_with_data(self) -> None:
        """Test display_local_metadata with metadata."""
        console = Console(force_terminal=True, width=80)
        metadata = {"metadata": {"key1": "value1", "key2": "value2"}, "tensor_count": 0, "header_size": 100}
        # Should not raise
        display_local_metadata(metadata, console)

    def test_display_local_metadata_empty(self) -> None:
        """Test display_local_metadata with no metadata."""
        console = Console(force_terminal=True, width=80)
        metadata: dict[str, Any] = {"metadata": {}, "tensor_count": 0, "header_size": 100}
        # Should not raise
        display_local_metadata(metadata, console)

    def test_display_local_metadata_with_filter(self) -> None:
        """Test display_local_metadata with key filter."""
        console = Console(force_terminal=True, width=80)
        metadata = {"metadata": {"key1": "value1", "key2": "value2"}, "tensor_count": 0, "header_size": 100}
        # Should not raise
        display_local_metadata(metadata, console, keys_filter=["key1"])

    def test_display_civitai_data_none(self) -> None:
        """Test display_civitai_data with None."""
        console = Console(force_terminal=True, width=80)
        # Should not raise
        display_civitai_data(None, console)

    def test_display_civitai_data_with_data(self) -> None:
        """Test display_civitai_data with model data."""
        console = Console(force_terminal=True, width=80)
        data = {
            "modelId": 123,
            "id": 456,
            "name": "Test Model v1",
            "baseModel": "SDXL 1.0",
            "createdAt": "2024-01-01",
            "trainedWords": ["word1", "word2"],
            "downloadUrl": "https://example.com/download",
            "files": [
                {
                    "primary": True,
                    "name": "model.safetensors",
                    "sizeKB": 5000,
                    "metadata": {"format": "SafeTensor", "fp": "fp16", "size": "full"},
                }
            ],
        }
        # Should not raise
        display_civitai_data(data, console)

    def test_display_model_info(self) -> None:
        """Test display_model_info with model data."""
        console = Console(force_terminal=True, width=80)
        data = {
            "id": 123,
            "name": "Test Model",
            "type": "LORA",
            "nsfw": False,
            "creator": {"username": "testuser"},
            "tags": ["tag1", "tag2"],
            "stats": {"downloadCount": 1000, "thumbsUpCount": 100},
            "modelVersions": [
                {
                    "id": 456,
                    "name": "v1.0",
                    "baseModel": "SDXL 1.0",
                    "createdAt": "2024-01-01",
                    "files": [{"primary": True, "name": "model.safetensors", "sizeKB": 5000}],
                }
            ],
        }
        # Should not raise
        display_model_info(data, console)

    def test_display_search_results_empty(self) -> None:
        """Test display_search_results with no results."""
        console = Console(force_terminal=True, width=80)
        # Should not raise
        display_search_results({"items": []}, console)

    def test_display_search_results_with_data(self) -> None:
        """Test display_search_results with results."""
        console = Console(force_terminal=True, width=80)
        results = {
            "items": [
                {
                    "id": 123,
                    "name": "Test Model",
                    "type": "LORA",
                    "modelVersions": [{"baseModel": "SDXL 1.0", "files": [{"primary": True, "sizeKB": 5000}]}],
                    "stats": {"downloadCount": 1000, "thumbsUpCount": 100},
                }
            ],
            "metadata": {"totalItems": 1},
        }
        # Should not raise
        display_search_results(results, console)


class TestAPIFunctions:
    """Tests for API functions with mocked HTTP."""

    @respx.mock
    def test_fetch_model_version_success(self) -> None:
        """Test successful model version fetch."""
        console = Console(force_terminal=True, width=80)
        respx.get("https://civitai.com/api/v1/model-versions/123").mock(
            return_value=httpx.Response(200, json={"id": 123, "name": "Test"})
        )

        result = fetch_civitai_model_version(123, None, console)
        assert result == {"id": 123, "name": "Test"}

    @respx.mock
    def test_fetch_model_version_not_found(self) -> None:
        """Test model version not found."""
        console = Console(force_terminal=True, width=80)
        respx.get("https://civitai.com/api/v1/model-versions/999").mock(return_value=httpx.Response(404))

        result = fetch_civitai_model_version(999, None, console)
        assert result is None

    @respx.mock
    def test_fetch_model_success(self) -> None:
        """Test successful model fetch."""
        console = Console(force_terminal=True, width=80)
        respx.get("https://civitai.com/api/v1/models/123").mock(
            return_value=httpx.Response(200, json={"id": 123, "name": "Test Model"})
        )

        result = fetch_civitai_model(123, None, console)
        assert result == {"id": 123, "name": "Test Model"}

    @respx.mock
    def test_fetch_model_not_found(self) -> None:
        """Test model not found."""
        console = Console(force_terminal=True, width=80)
        respx.get("https://civitai.com/api/v1/models/999").mock(return_value=httpx.Response(404))

        result = fetch_civitai_model(999, None, console)
        assert result is None

    @respx.mock
    def test_fetch_by_hash_success(self) -> None:
        """Test successful hash lookup."""
        console = Console(force_terminal=True, width=80)
        respx.get("https://civitai.com/api/v1/model-versions/by-hash/ABC123").mock(
            return_value=httpx.Response(200, json={"id": 456, "name": "Found"})
        )

        result = fetch_civitai_by_hash("ABC123", None, console)
        assert result == {"id": 456, "name": "Found"}

    @respx.mock
    def test_fetch_by_hash_not_found(self) -> None:
        """Test hash not found."""
        console = Console(force_terminal=True, width=80)
        respx.get("https://civitai.com/api/v1/model-versions/by-hash/NOTFOUND").mock(return_value=httpx.Response(404))

        result = fetch_civitai_by_hash("NOTFOUND", None, console)
        assert result is None

    @respx.mock
    def test_search_civitai_success(self) -> None:
        """Test successful search."""
        console = Console(force_terminal=True, width=80)
        respx.get("https://civitai.com/api/v1/models").mock(
            return_value=httpx.Response(200, json={"items": [{"id": 1}], "metadata": {}})
        )

        result = search_civitai("test", None, None, SortOrder.downloads, 20, None, console)
        assert result is not None
        assert len(result["items"]) == 1

    @respx.mock
    def test_search_civitai_with_filters(self) -> None:
        """Test search with type and base model filters."""
        console = Console(force_terminal=True, width=80)
        respx.get("https://civitai.com/api/v1/models").mock(
            return_value=httpx.Response(200, json={"items": [{"id": 1, "name": "Test LORA"}], "metadata": {}})
        )

        result = search_civitai("test", ModelType.lora, BaseModel.sdxl, SortOrder.downloads, 20, None, console)
        assert result is not None

    @respx.mock
    def test_download_model_success(self, tmp_path: Path) -> None:
        """Test successful model download."""
        console = Console(force_terminal=True, width=80)
        dest = tmp_path / "model.safetensors"
        respx.get("https://civitai.com/api/download/models/123").mock(
            return_value=httpx.Response(200, content=b"fake model data")
        )

        result = download_model(123, dest, None, console, resume=False)
        assert result is True
        assert dest.exists()
        assert dest.read_bytes() == b"fake model data"

    @respx.mock
    def test_download_model_unauthorized(self, tmp_path: Path) -> None:
        """Test download with 401 unauthorized."""
        console = Console(force_terminal=True, width=80)
        dest = tmp_path / "model.safetensors"
        respx.get("https://civitai.com/api/download/models/123").mock(return_value=httpx.Response(401))

        result = download_model(123, dest, None, console, resume=False)
        assert result is False


class TestCLI:
    """Tests for CLI commands."""

    def test_help(self) -> None:
        """Test --help works."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "safetensor" in result.stdout.lower()

    def test_info_file_not_found(self, tmp_path: Path) -> None:
        """Test info command with non-existent file."""
        result = runner.invoke(app, ["info", str(tmp_path / "nonexistent.safetensors")])
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    def test_info_with_safetensor(self, temp_safetensor: Path) -> None:
        """Test info command with valid safetensor file."""
        result = runner.invoke(app, ["info", str(temp_safetensor), "--skip-civitai"])
        assert result.exit_code == 0

    def test_info_json_output(self, temp_safetensor: Path) -> None:
        """Test info command with JSON output."""
        result = runner.invoke(app, ["info", str(temp_safetensor), "--skip-civitai", "--json"])
        assert result.exit_code == 0
        assert "sha256" in result.stdout

    def test_info_meta_filter(self, temp_safetensor: Path) -> None:
        """Test info command with metadata filter."""
        result = runner.invoke(app, ["info", str(temp_safetensor), "--meta", "test_key"])
        assert result.exit_code == 0
        assert "test_value" in result.stdout

    @respx.mock
    def test_search_command(self) -> None:
        """Test search command."""
        respx.get("https://civitai.com/api/v1/models").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [{"id": 1, "name": "Test", "type": "LORA", "modelVersions": [], "stats": {}}],
                    "metadata": {"totalItems": 1},
                },
            )
        )

        result = runner.invoke(app, ["search", "test"])
        assert result.exit_code == 0

    @respx.mock
    def test_get_command(self) -> None:
        """Test get command."""
        respx.get("https://civitai.com/api/v1/models/123").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": 123,
                    "name": "Test Model",
                    "type": "LORA",
                    "nsfw": False,
                    "stats": {},
                    "modelVersions": [],
                },
            )
        )

        result = runner.invoke(app, ["get", "123"])
        assert result.exit_code == 0

    @respx.mock
    def test_get_command_not_found(self) -> None:
        """Test get command with non-existent model."""
        respx.get("https://civitai.com/api/v1/models/999").mock(return_value=httpx.Response(404))

        result = runner.invoke(app, ["get", "999"])
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    def test_config_show(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test config --show command."""
        monkeypatch.delenv("CIVITAI_API_KEY", raising=False)
        monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.toml")
        monkeypatch.setattr(config, "LEGACY_RC_FILE", tmp_path / "nonexistent")

        result = runner.invoke(app, ["config", "--show"])
        assert result.exit_code == 0
        assert "config file" in result.stdout.lower()

    def test_download_no_args(self) -> None:
        """Test dl command with no arguments."""
        result = runner.invoke(app, ["dl"])
        assert result.exit_code == 1
        assert "must specify" in result.stdout.lower()
