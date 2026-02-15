"""Simple ComfyUI client for basic txt2img generation."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

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

    def wait_for_completion(self, prompt_id: str, poll_interval: float = 0.5) -> dict[str, Any]:
        """Poll until the prompt completes."""
        start = time.time()
        while time.time() - start < self.timeout:
            history = self.get_history(prompt_id)
            if history and history.get("outputs"):
                return history
            time.sleep(poll_interval)
        raise TimeoutError(f"Prompt {prompt_id} did not complete within {self.timeout}s")

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
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg: float = 7.0,
        seed: int = -1,
        sampler: str = "euler_a",
        scheduler: str = "normal",
    ) -> dict[str, Any]:
        """Generate an image with a simple txt2img workflow."""
        # Use first checkpoint if not specified
        if not checkpoint:
            checkpoints = self.get_checkpoints()
            if not checkpoints:
                raise ValueError("No checkpoints available")
            checkpoint = checkpoints[0]

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

        # Queue and wait
        prompt_id = self.queue_prompt(workflow)
        history = self.wait_for_completion(prompt_id)

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
