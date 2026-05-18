"""
Generate images from already-computed OpenCLIP ViT-H/14 embeddings plus depth.

Expected inputs:
- semantic_embeddings: shape (257, 1024) for one OpenCLIP ViT-H/14 image
  embedding, or (batch, 257, 1024). This file does not tokenize, run 4M, or
  compute CLIP features; pass the semantic embedding in directly.
- depth_map: a PIL image, numpy array, or torch tensor containing a grayscale
  or RGB depth map. It is normalized if needed and resized to output_size.

Implementation: SDXL base + SDXL depth ControlNet + IP-Adapter-Plus. Model
weights are downloaded by Hugging Face/diffusers the first time the pipeline is
loaded on whichever machine runs inference.

Tunable params:
- semantic_scale: IP-Adapter strength for the OpenCLIP embedding. Raise it when
  semantics are reliable; lower it when predicted embeddings are noisy.
- depth_scale: ControlNet strength for the depth map. Raise it for stronger
  geometry adherence; lower it when depth predictions are noisy.
- output_size: generated image size. Defaults to 1024 for SDXL quality; try
  768 or 512 to reduce VRAM/runtime. Must be divisible by 8.
- num_inference_steps/guidance_scale/seed: other per-call generation controls
  exposed by latent_to_image(...).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline


semantic_scale = 0.7
depth_scale = 0.55

SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNET_MODEL_ID = "diffusers/controlnet-depth-sdxl-1.0-small"
IP_ADAPTER_MODEL_ID = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHT_NAME = "ip-adapter-plus_sdxl_vit-h.safetensors"
OUTPUT_SIZE = 1024


def latent_to_image(
    semantic_embeddings: torch.Tensor | np.ndarray,
    depth_map: Image.Image | torch.Tensor | np.ndarray,
    *,
    seed: int | None = None,
    num_inference_steps: int = 30,
    guidance_scale: float = 5.0,
    output_size: int = OUTPUT_SIZE,
) -> Image.Image:
    """Generate an image from OpenCLIP ViT-H/14 semantic embeddings and a depth map."""
    if output_size <= 0 or output_size % 8 != 0:
        raise ValueError("output_size must be a positive integer divisible by 8.")

    device = _default_device()
    pipe = _load_pipeline(device)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    result = pipe(
        prompt="",
        negative_prompt="",
        image=_prepare_depth_map(depth_map, output_size=output_size),
        ip_adapter_image_embeds=_prepare_ip_adapter_embeds(
            semantic_embeddings,
            device=device,
            dtype=pipe.unet.dtype,
            include_negative=guidance_scale > 1.0,
        ),
        controlnet_conditioning_scale=depth_scale,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        height=output_size,
        width=output_size,
    )
    return result.images[0]


@lru_cache(maxsize=1)
def _load_pipeline(device: str) -> StableDiffusionXLControlNetPipeline:
    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32

    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL_ID,
        torch_dtype=dtype,
    )
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        SDXL_MODEL_ID,
        controlnet=controlnet,
        torch_dtype=dtype,
    )
    pipe.load_ip_adapter(
        IP_ADAPTER_MODEL_ID,
        subfolder=IP_ADAPTER_SUBFOLDER,
        weight_name=IP_ADAPTER_WEIGHT_NAME,
        image_encoder_folder=None,
    )
    pipe.set_ip_adapter_scale(semantic_scale)
    return pipe.to(device)


def _prepare_ip_adapter_embeds(
    semantic_embeddings: torch.Tensor | np.ndarray,
    *,
    device: str,
    dtype: torch.dtype,
    include_negative: bool,
) -> list[torch.Tensor]:
    embeds = torch.as_tensor(semantic_embeddings, dtype=dtype, device=device)

    if embeds.ndim == 2:
        embeds = embeds.unsqueeze(0)
    if embeds.ndim != 3:
        raise ValueError(
            "semantic_embeddings must have shape (tokens, dim) or "
            "(batch, tokens, dim)."
        )

    positive_embeds = embeds.unsqueeze(0)
    if not include_negative:
        return [positive_embeds]

    negative_embeds = torch.zeros_like(positive_embeds)
    return [torch.cat([negative_embeds, positive_embeds], dim=0)]


def _prepare_depth_map(
    depth_map: Image.Image | torch.Tensor | np.ndarray,
    *,
    output_size: int,
) -> Image.Image:
    if isinstance(depth_map, Image.Image):
        image = depth_map
    else:
        depth_array = _to_numpy(depth_map)
        if depth_array.ndim == 3 and depth_array.shape[0] in {1, 3}:
            depth_array = np.moveaxis(depth_array, 0, -1)
        if depth_array.ndim == 3 and depth_array.shape[-1] == 1:
            depth_array = depth_array[..., 0]

        if depth_array.dtype != np.uint8:
            depth_array = depth_array.astype(np.float32)
            depth_min = float(np.nanmin(depth_array))
            depth_max = float(np.nanmax(depth_array))
            if depth_max > depth_min:
                depth_array = (depth_array - depth_min) / (depth_max - depth_min)
            depth_array = (np.clip(depth_array, 0.0, 1.0) * 255).astype(np.uint8)

        image = Image.fromarray(depth_array)

    return image.convert("RGB").resize((output_size, output_size), Image.LANCZOS)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
