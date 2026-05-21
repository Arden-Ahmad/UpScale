from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Callable

from PIL import Image
import torch


DiffusionProgressCallback = Callable[[float, str], None]
DEFAULT_DIFFUSION_MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEFAULT_DIFFUSION_PROMPT = ""
DEFAULT_DIFFUSION_NEGATIVE_PROMPT = ""
DEFAULT_DIFFUSION_STEPS = 20
DEFAULT_DIFFUSION_GUIDANCE_SCALE = 1.0


@dataclass(frozen=True)
class DiffusionSettings:
    model_id: str
    prompt: str
    negative_prompt: str
    num_inference_steps: int
    guidance_scale: float
    seed: int | None


def load_diffusion_settings_from_env() -> DiffusionSettings:
    model_id = os.environ.get("UPSCALE_DIFFUSION_MODEL", DEFAULT_DIFFUSION_MODEL_ID).strip()
    if not model_id:
        model_id = DEFAULT_DIFFUSION_MODEL_ID

    prompt = os.environ.get("UPSCALE_DIFFUSION_PROMPT", DEFAULT_DIFFUSION_PROMPT)
    negative_prompt = os.environ.get("UPSCALE_DIFFUSION_NEGATIVE_PROMPT", DEFAULT_DIFFUSION_NEGATIVE_PROMPT)

    try:
        num_inference_steps = max(
            1,
            int(os.environ.get("UPSCALE_DIFFUSION_STEPS", str(DEFAULT_DIFFUSION_STEPS))),
        )
    except ValueError as exc:
        raise ValueError("UPSCALE_DIFFUSION_STEPS must be an integer.") from exc

    try:
        guidance_scale = float(
            os.environ.get("UPSCALE_DIFFUSION_GUIDANCE", str(DEFAULT_DIFFUSION_GUIDANCE_SCALE))
        )
    except ValueError as exc:
        raise ValueError("UPSCALE_DIFFUSION_GUIDANCE must be numeric.") from exc

    seed_value = os.environ.get("UPSCALE_DIFFUSION_SEED")
    if seed_value is None or not seed_value.strip():
        seed = None
    else:
        try:
            seed = int(seed_value)
        except ValueError as exc:
            raise ValueError("UPSCALE_DIFFUSION_SEED must be an integer.") from exc

    return DiffusionSettings(
        model_id=model_id,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )


class DiffusionRefiner:
    def __init__(self, settings: DiffusionSettings, device: torch.device) -> None:
        self.settings = settings
        self.device = device
        self.pipeline = None

    def _load_pipeline(self):
        if self.pipeline is not None:
            return self.pipeline

        try:
            from diffusers import StableDiffusionImg2ImgPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Diffusion denoise requires diffusers, transformers, accelerate, and safetensors. "
                "Install the updated requirements before using denoise strength above 0."
            ) from exc

        torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.settings.model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        pipeline.set_progress_bar_config(disable=True)
        if hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing()
        if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_slicing"):
            pipeline.vae.enable_slicing()
        pipeline.to(self.device)
        self.pipeline = pipeline
        return pipeline

    def refine(
        self,
        image: Image.Image,
        denoise_strength: float,
        progress_callback: DiffusionProgressCallback | None = None,
    ) -> Image.Image:
        if denoise_strength <= 0:
            return image

        if progress_callback is not None:
            progress_callback(0.05, "Preparing diffusion init image")

        pipeline = self._load_pipeline()
        alpha_channel = image.getchannel("A") if "A" in image.getbands() else None
        init_image = image.convert("RGB")
        target_size = init_image.size
        prepared_size = (
            max(8, math.ceil(target_size[0] / 8) * 8),
            max(8, math.ceil(target_size[1] / 8) * 8),
        )
        if prepared_size != target_size:
            init_image = init_image.resize(prepared_size, Image.Resampling.LANCZOS)

        if progress_callback is not None:
            progress_callback(0.2, f"Running diffusion refinement at {denoise_strength:g}")

        pipeline_kwargs: dict[str, object] = {
            "prompt": self.settings.prompt,
            "image": init_image,
            "strength": denoise_strength,
            "num_inference_steps": self.settings.num_inference_steps,
            "guidance_scale": self.settings.guidance_scale,
        }
        if self.settings.negative_prompt:
            pipeline_kwargs["negative_prompt"] = self.settings.negative_prompt
        if self.settings.seed is not None:
            generator_device = "cuda" if self.device.type == "cuda" else "cpu"
            pipeline_kwargs["generator"] = torch.Generator(device=generator_device).manual_seed(self.settings.seed)

        try:
            result = pipeline(**pipeline_kwargs).images[0]
        finally:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        if result.size != target_size:
            result = result.resize(target_size, Image.Resampling.LANCZOS)

        if alpha_channel is not None:
            rgba_result = result.convert("RGBA")
            rgba_result.putalpha(alpha_channel.resize(target_size, Image.Resampling.LANCZOS))
            result = rgba_result

        if progress_callback is not None:
            progress_callback(1.0, "Diffusion refinement finished")

        return result