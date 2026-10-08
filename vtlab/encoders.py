"""Frozen encoders. The color encoder is ONLY an offline synthetic control."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


CAPTIONS = (
    "A red dot moves from left to right.",
    "A red dot moves from right to left.",
    "A red dot moves from top to bottom.",
    "A red dot moves from bottom to top.",
    "A blue dot moves from left to right.",
    "A blue dot moves from right to left.",
    "A blue dot moves from top to bottom.",
    "A blue dot moves from bottom to top.",
)
SEMANTIC_IDS = (
    "red_right", "red_left", "red_down", "red_up",
    "blue_right", "blue_left", "blue_down", "blue_up",
)


class ColorEncoder:
    """Handcrafted dot positions plus one-hot text, not a pretrained model.

    Frame vectors are [red_present, red_x, red_y, blue_present, blue_x,
    blue_y]. Coordinates are divided by the image dimensions. This makes
    integer centroids in 64px synthetic images exactly representable and
    prevents floating-point summation order from distinguishing reverse pairs.
    """

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        result = np.zeros((len(images), 6), dtype=np.float32)
        for index, image in enumerate(images):
            rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
            height, width = rgb.shape[:2]
            red = (rgb[:, :, 0] - rgb[:, :, 1] > 80) & (rgb[:, :, 0] - rgb[:, :, 2] > 80)
            blue = (rgb[:, :, 2] - rgb[:, :, 0] > 80) & (rgb[:, :, 2] - rgb[:, :, 1] > 80)
            for offset, mask in ((0, red), (3, blue)):
                ys, xs = np.nonzero(mask)
                if len(xs):
                    result[index, offset:offset + 3] = (1.0, xs.mean() / width, ys.mean() / height)
        return result

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        result = np.zeros((len(texts), len(CAPTIONS)), dtype=np.float32)
        for index, caption in enumerate(texts):
            if caption not in CAPTIONS:
                raise ValueError(
                    "The color encoder only accepts the eight synthetic captions. "
                    "Use encoder='clip' for real video/text. Supported captions: "
                    + " | ".join(CAPTIONS)
                )
            result[index, CAPTIONS.index(caption)] = 1.0
        return result


class ClipEncoder:
    """CLIP image/text features; model parameters remain frozen."""

    def __init__(self, device: str, model_name: str, cache_dir: Path | None = None):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError(
                "CLIP extraction needs torch and transformers in your existing Python "
                "environment. Nothing has been installed. Use encoder='color' for the "
                "offline synthetic experiment, or select an existing environment with "
                "these packages before running real-data extraction."
            ) from exc
        if device == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError("PyTorch XPU is unavailable in this environment; retry with device='cpu'.")
        if device != "cpu" and device != "xpu":
            raise ValueError("This CPU/Intel project supports device='cpu' or device='xpu'.")
        self.torch = torch
        self.device = torch.device(device)
        cache_dir = cache_dir or Path(__file__).resolve().parents[1] / ".cache" / "huggingface"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.processor = CLIPProcessor.from_pretrained(model_name, cache_dir=str(cache_dir), use_fast=False)
        # This official checkpoint ships PyTorch weights. Explicit selection
        # avoids transformers starting a second background conversion/download.
        self.model = CLIPModel.from_pretrained(model_name, cache_dir=str(cache_dir), use_safetensors=False)
        self.model.to(self.device).eval()
        self.model.requires_grad_(False)

    def _numpy(self, features) -> np.ndarray:
        # Recent transformers versions may return a model output object.
        if not self.torch.is_tensor(features):
            for attribute in ("image_embeds", "text_embeds", "pooler_output"):
                candidate = getattr(features, attribute, None)
                if candidate is not None:
                    features = candidate
                    break
        features = features.float()
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features.detach().cpu().numpy().astype(np.float32, copy=False)

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        with self.torch.inference_mode():
            inputs = self.processor(images=list(images), return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            return self._numpy(self.model.get_image_features(**inputs))

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        with self.torch.inference_mode():
            inputs = self.processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            return self._numpy(self.model.get_text_features(**inputs))


def get_encoder(encoder: str = "clip", device: str = "cpu",
                model_name: str = "openai/clip-vit-base-patch32",
                cache_dir: Path | None = None):
    if encoder == "color":
        return ColorEncoder()
    if encoder == "clip":
        return ClipEncoder(device=device, model_name=model_name, cache_dir=cache_dir)
    raise ValueError(f"Unknown encoder: {encoder!r}; choose 'color' or 'clip'.")


def encode_text_queries(texts: list[str], encoder: str = "clip", device: str = "cpu",
                        model_name: str = "openai/clip-vit-base-patch32",
                        cache_dir: Path | None = None) -> np.ndarray:
    if not texts:
        raise ValueError("At least one query is required.")
    return get_encoder(encoder, device, model_name, cache_dir).encode_texts(texts)


def extract_features(manifest_path: Path, cache_path: Path, encoder: str = "color",
                     frames: int = 8, device: str = "cpu",
                     model_name: str = "openai/clip-vit-base-patch32", batch_size: int = 16) -> dict:
    """Public convenience export; import lazily to avoid a data/encoder cycle."""
    from .data import extract_features as extract
    return extract(manifest_path, cache_path, encoder, frames, device, model_name, batch_size)
