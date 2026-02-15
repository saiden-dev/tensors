"""Simple ComfyUI client for basic txt2img generation."""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from collections.abc import Callable

from tensors.config import load_config, save_config

DEFAULT_WORKFLOW = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "cfg": 7,
            "denoise": 1,
            "latent_image": ["5", 0],
            "model": ["4", 0],
            "negative": ["7", 0],
            "positive": ["6", 0],
            "sampler_name": "euler_ancestral",
            "scheduler": "normal",
            "seed": -1,
            "steps": 20,
        },
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": ""},
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {"batch_size": 1, "height": 512, "width": 512},
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["4", 1], "text": ""},
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["4", 1], "text": ""},
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "comfy", "images": ["8", 0]},
    },
}

COMFY_CONTAINER = "comfyui"
COMFY_HOST = "junkpile"


def get_last_checkpoint() -> str | None:
    """Get last used checkpoint from config."""
    cfg = load_config()
    value = cfg.get("comfy", {}).get("last_checkpoint")
    return str(value) if value else None


def save_last_checkpoint(checkpoint: str) -> None:
    """Save last used checkpoint to config."""
    cfg = load_config()
    if "comfy" not in cfg:
        cfg["comfy"] = {}
    cfg["comfy"]["last_checkpoint"] = checkpoint
    save_config(cfg)


def restart_comfy_container(on_status: Callable[[str], None] | None = None) -> None:
    """Restart the ComfyUI container on junkpile and wait for it to be ready."""
    def status(msg: str) -> None:
        if on_status:
            on_status(msg)

    status("Restarting ComfyUI container...")
    subprocess.run(
        ["ssh", COMFY_HOST, f"docker restart {COMFY_CONTAINER}"],
        check=True,
        capture_output=True,
    )

    # Wait for ComfyUI to be ready
    status("Waiting for ComfyUI to start...")
    max_wait = 120
    start = time.time()
    while time.time() - start < max_wait:
        try:
            resp = httpx.get(f"http://{COMFY_HOST}:8188/system_stats", timeout=5)
            if resp.is_success:
                status("ComfyUI is ready!")
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)

    raise TimeoutError(f"ComfyUI did not start within {max_wait}s")


class ComfyClient:
    """Simple ComfyUI API client."""

    def __init__(self, base_url: str = "http://127.0.0.1:8188", timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())

    def get_checkpoints(self) -> list[str]:
        """List available checkpoint models."""
        resp = httpx.get(f"{self.base_url}/object_info/CheckpointLoaderSimple", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return list(data.get("CheckpointLoaderSimple", {}).get("input", {}).get("required", {}).get("ckpt_name", [[]])[0])

    def get_loras(self) -> list[str]:
        """List available LoRAs."""
        resp = httpx.get(f"{self.base_url}/object_info/LoraLoader", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return list(data.get("LoraLoader", {}).get("input", {}).get("required", {}).get("lora_name", [[]])[0])

    def get_samplers(self) -> list[str]:
        """List available samplers."""
        resp = httpx.get(f"{self.base_url}/object_info/KSampler", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return list(data.get("KSampler", {}).get("input", {}).get("required", {}).get("sampler_name", [[]])[0])

    def queue_prompt(self, workflow: dict[str, Any]) -> str:
        """Queue a prompt and return the prompt_id."""
        payload = {"prompt": workflow, "client_id": self.client_id}
        resp = httpx.post(f"{self.base_url}/prompt", json=payload, timeout=30)
        resp.raise_for_status()
        return str(resp.json()["prompt_id"])

    def get_history(self, prompt_id: str) -> dict[str, Any] | None:
        """Get history for a prompt_id."""
        resp = httpx.get(f"{self.base_url}/history/{prompt_id}", timeout=10)
        if resp.is_success:
            data = resp.json()
            return dict(data.get(prompt_id, {})) if prompt_id in data else None
        return None

    def wait_for_completion(
        self,
        prompt_id: str,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, Any]:
        """Wait for prompt completion with progress updates via websocket."""
        import websocket  # noqa: PLC0415

        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/ws?clientId={self.client_id}"

        result: dict[str, Any] = {}
        completed = False

        def on_message(ws: Any, message: str) -> None:  # noqa: ARG001
            nonlocal completed, result
            try:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "progress":
                    progress_data = data.get("data", {})
                    current = progress_data.get("value", 0)
                    total = progress_data.get("max", 1)
                    if on_progress:
                        on_progress(current, total, "sampling")

                elif msg_type == "executing":
                    exec_data = data.get("data", {})
                    if exec_data.get("node") is None and exec_data.get("prompt_id") == prompt_id:
                        # Execution finished
                        completed = True

                elif msg_type == "executed":
                    exec_data = data.get("data", {})
                    if exec_data.get("prompt_id") == prompt_id:
                        result = exec_data

            except json.JSONDecodeError:
                pass

        def on_error(ws: Any, error: Exception) -> None:  # noqa: ARG001
            nonlocal completed
            completed = True

        def on_close(ws: Any, close_status_code: int, close_msg: str) -> None:  # noqa: ARG001
            nonlocal completed
            completed = True

        ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )

        # Run websocket in a thread
        import threading  # noqa: PLC0415

        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()

        # Wait for completion
        start = time.time()
        while not completed and time.time() - start < self.timeout:
            time.sleep(0.1)

        ws.close()

        if not completed:
            raise TimeoutError(f"Prompt {prompt_id} did not complete within {self.timeout}s")

        # Get final history
        history = self.get_history(prompt_id)
        if not history:
            raise RuntimeError(f"Could not get history for prompt {prompt_id}")

        return history

    def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """Download an image from ComfyUI."""
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        resp = httpx.get(f"{self.base_url}/view", params=params, timeout=30)
        resp.raise_for_status()
        return resp.content

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        checkpoint: str | None = None,
        lora: str | None = None,
        lora_strength: float = 0.8,
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg: float = 7.0,
        seed: int = -1,
        sampler: str = "euler_ancestral",
        scheduler: str = "normal",
        on_progress: Callable[[int, int, str], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        auto_restart: bool = True,
    ) -> dict[str, Any]:
        """Generate an image with a simple txt2img workflow."""
        # Use first checkpoint if not specified
        if not checkpoint:
            # Try last used checkpoint first
            checkpoint = get_last_checkpoint()
            if not checkpoint:
                checkpoints = self.get_checkpoints()
                if not checkpoints:
                    raise ValueError("No checkpoints available")
                checkpoint = checkpoints[0]

        # Check if we need to restart container for model change
        if auto_restart:
            last_checkpoint = get_last_checkpoint()
            if last_checkpoint and last_checkpoint != checkpoint:
                if on_status:
                    on_status(f"Model changed: {last_checkpoint} -> {checkpoint}")
                restart_comfy_container(on_status)

        # Save checkpoint as last used
        save_last_checkpoint(checkpoint)

        # Build workflow
        workflow = json.loads(json.dumps(DEFAULT_WORKFLOW))
        workflow["4"]["inputs"]["ckpt_name"] = checkpoint
        workflow["5"]["inputs"]["width"] = width
        workflow["5"]["inputs"]["height"] = height
        workflow["6"]["inputs"]["text"] = prompt
        workflow["7"]["inputs"]["text"] = negative_prompt
        workflow["3"]["inputs"]["steps"] = steps
        workflow["3"]["inputs"]["cfg"] = cfg
        workflow["3"]["inputs"]["seed"] = seed if seed >= 0 else int(time.time() * 1000) % (2**32)
        workflow["3"]["inputs"]["sampler_name"] = sampler
        workflow["3"]["inputs"]["scheduler"] = scheduler

        # Add LoRA if specified
        if lora:
            workflow["10"] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": lora,
                    "strength_model": lora_strength,
                    "strength_clip": lora_strength,
                    "model": ["4", 0],
                    "clip": ["4", 1],
                },
            }
            # Rewire: KSampler uses LoRA output instead of checkpoint
            workflow["3"]["inputs"]["model"] = ["10", 0]
            # Rewire: CLIP encoders use LoRA output
            workflow["6"]["inputs"]["clip"] = ["10", 1]
            workflow["7"]["inputs"]["clip"] = ["10", 1]

        if on_status:
            on_status("Queueing prompt...")

        # Queue and wait
        prompt_id = self.queue_prompt(workflow)

        if on_status:
            on_status("Generating...")

        history = self.wait_for_completion(prompt_id, on_progress)

        # Extract output images
        outputs = history.get("outputs", {})
        images = []
        for _node_id, node_output in outputs.items():
            if "images" in node_output:
                for img in node_output["images"]:
                    images.append({
                        "filename": img["filename"],
                        "subfolder": img.get("subfolder", ""),
                        "type": img.get("type", "output"),
                    })

        return {
            "prompt_id": prompt_id,
            "images": images,
            "checkpoint": checkpoint,
            "lora": lora,
            "seed": workflow["3"]["inputs"]["seed"],
        }

    def generate_and_save(
        self,
        prompt: str,
        output_path: str | Path,
        **kwargs: Any,
    ) -> Path:
        """Generate an image and save it locally."""
        result = self.generate(prompt, **kwargs)
        if not result["images"]:
            raise RuntimeError("No images generated")

        img_info = result["images"][0]
        img_data = self.get_image(img_info["filename"], img_info["subfolder"], img_info["type"])

        output = Path(output_path)
        output.write_bytes(img_data)
        return output
