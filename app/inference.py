from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
import math
import re
import threading
from typing import Callable
from uuid import uuid4

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn


ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class ModelInfo:
    key: str
    label: str
    path: Path
    native_scale: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "key": self.key,
            "label": self.label,
            "native_scale": self.native_scale,
        }


@dataclass
class UpscaleJob:
    id: str
    status: str
    progress: int
    message: str
    input_filename: str
    model_key: str
    upscale_factor: float
    batch_id: str | None = None
    item_index: int | None = None
    original_width: int | None = None
    original_height: int | None = None
    result_width: int | None = None
    result_height: int | None = None
    output_url: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, int | float | str | None]:
        return {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "input_filename": self.input_filename,
            "model_key": self.model_key,
            "upscale_factor": self.upscale_factor,
            "batch_id": self.batch_id,
            "item_index": self.item_index,
            "original_width": self.original_width,
            "original_height": self.original_height,
            "result_width": self.result_width,
            "result_height": self.result_height,
            "output_url": self.output_url,
            "error": self.error,
        }


@dataclass(frozen=True)
class BatchJob:
    id: str
    model_key: str
    upscale_factor: float
    tile_size: int
    job_ids: list[str]

    def to_dict(self, items: list[UpscaleJob]) -> dict[str, object]:
        total_items = len(items)
        completed_count = sum(1 for item in items if item.status == "completed")
        failed_count = sum(1 for item in items if item.status == "failed")
        running_items = [item for item in items if item.status == "running"]
        queued_count = sum(1 for item in items if item.status == "queued")
        progress = round(sum(item.progress for item in items) / total_items) if total_items else 0

        if running_items:
            running_item = min(
                running_items,
                key=lambda item: item.item_index if item.item_index is not None else total_items,
            )
            active_position = (running_item.item_index or 0) + 1
            status = "running"
            message = f"Processing {active_position} of {total_items}: {running_item.input_filename}"
        elif queued_count == total_items:
            status = "queued"
            message = f"Queued {total_items} image{'s' if total_items != 1 else ''}"
        elif completed_count == total_items:
            status = "completed"
            message = f"Finished {completed_count} image{'s' if completed_count != 1 else ''}"
        elif completed_count + failed_count == total_items:
            status = "completed_with_errors"
            message = (
                f"Finished {completed_count} image{'s' if completed_count != 1 else ''}; "
                f"{failed_count} failed"
            )
            progress = 100
        else:
            status = "running"
            message = (
                f"Queued {queued_count} image{'s' if queued_count != 1 else ''}; "
                f"{completed_count} done"
            )

        return {
            "id": self.id,
            "status": status,
            "progress": progress,
            "message": message,
            "model_key": self.model_key,
            "upscale_factor": self.upscale_factor,
            "tile_size": self.tile_size,
            "total_items": total_items,
            "completed_count": completed_count,
            "failed_count": failed_count,
            "queued_count": queued_count,
            "items": [item.to_dict() for item in items],
        }


def sequential(*modules: nn.Module | None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for module in modules:
        if module is None:
            continue
        if isinstance(module, nn.Sequential):
            layers.extend(list(module.children()))
            continue
        layers.append(module)
    return nn.Sequential(*layers)


def conv_block(in_channels: int, out_channels: int, activation: bool) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)]
    if activation:
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
    return sequential(*layers)


def upconv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        conv_block(in_channels, out_channels, activation=True),
    )


class ResidualDenseBlock5C(nn.Module):
    def __init__(self, channels: int = 64, growth_channels: int = 32) -> None:
        super().__init__()
        self.conv1 = conv_block(channels, growth_channels, activation=True)
        self.conv2 = conv_block(channels + growth_channels, growth_channels, activation=True)
        self.conv3 = conv_block(channels + growth_channels * 2, growth_channels, activation=True)
        self.conv4 = conv_block(channels + growth_channels * 3, growth_channels, activation=True)
        self.conv5 = conv_block(channels + growth_channels * 4, channels, activation=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), dim=1))
        x3 = self.conv3(torch.cat((x, x1, x2), dim=1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), dim=1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        return x5.mul(0.2) + x


class RRDB(nn.Module):
    def __init__(self, channels: int, growth_channels: int = 32) -> None:
        super().__init__()
        self.RDB1 = ResidualDenseBlock5C(channels, growth_channels)
        self.RDB2 = ResidualDenseBlock5C(channels, growth_channels)
        self.RDB3 = ResidualDenseBlock5C(channels, growth_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(0.2) + x


class ShortcutBlock(nn.Module):
    def __init__(self, submodule: nn.Module) -> None:
        super().__init__()
        self.sub = submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.sub(x)


class RRDBNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        feature_channels: int = 64,
        rrdb_blocks: int = 23,
        growth_channels: int = 32,
        scale: int = 4,
    ) -> None:
        super().__init__()
        fea_conv = conv_block(in_channels, feature_channels, activation=False)
        rrdb_trunk = [RRDB(feature_channels, growth_channels) for _ in range(rrdb_blocks)]
        lr_conv = conv_block(feature_channels, feature_channels, activation=False)
        hr_conv = conv_block(feature_channels, feature_channels, activation=True)
        last_conv = conv_block(feature_channels, out_channels, activation=False)
        upsamplers = [upconv_block(feature_channels, feature_channels) for _ in range(int(math.log2(scale)))]

        self.model = sequential(
            fea_conv,
            ShortcutBlock(sequential(*rrdb_trunk, lr_conv)),
            *upsamplers,
            hr_conv,
            last_conv,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load_state_dict(model_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(model_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format for {model_path.name}.")

    state_dict = checkpoint
    for key in ("params_ema", "params", "state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            state_dict = candidate
            break

    cleaned_state: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        normalized_key = key[7:] if key.startswith("module.") else key
        cleaned_state[normalized_key] = value

    return cleaned_state


def infer_rrdb_blocks(state_dict: dict[str, torch.Tensor]) -> int:
    pattern = re.compile(r"^model\.1\.sub\.(\d+)\.RDB1\.conv1\.0\.weight$")
    block_indexes = [int(match.group(1)) for key in state_dict if (match := pattern.match(key))]
    if not block_indexes:
        raise ValueError("The selected model is not a classic ESRGAN RRDB checkpoint.")
    return max(block_indexes) + 1


def infer_native_scale(model_path: Path) -> int:
    match = re.search(r"(?i)(\d+)x", model_path.stem)
    if match:
        return max(1, int(match.group(1)))
    return 4


def sanitize_stem(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")
    return sanitized or "image"


def format_model_label(value: str) -> str:
    return re.sub(r"[_-]+", " ", value).strip() or value


def resolve_device_label(device: torch.device) -> str:
    if device.type != "cuda":
        return "CPU"

    try:
        return f"GPU (CUDA: {torch.cuda.get_device_name(0)})"
    except Exception:
        return "GPU (CUDA)"


def resolve_torch_device(device_preference: str | None = None) -> tuple[torch.device, str]:
    preference = (device_preference or os.getenv("UPSCALE_DEVICE") or "auto").strip().lower()
    valid_preferences = {"auto", "cuda", "cpu"}
    if preference not in valid_preferences:
        raise ValueError("UPSCALE_DEVICE must be one of: auto, cuda, cpu.")

    cuda_available = torch.cuda.is_available()
    if preference == "cpu":
        device = torch.device("cpu")
        return device, resolve_device_label(device)

    if preference == "cuda" and not cuda_available:
        if torch.version.cuda is None:
            raise RuntimeError(
                "CUDA was requested, but the installed PyTorch build has no CUDA support. "
                "Install a CUDA-enabled torch build or switch UPSCALE_DEVICE back to auto."
            )
        raise RuntimeError(
            "CUDA was requested, but no CUDA device is available. "
            "In Colab, enable a GPU runtime before starting UpScale."
        )

    if cuda_available:
        device = torch.device("cuda")
        return device, resolve_device_label(device)

    device = torch.device("cpu")
    return device, resolve_device_label(device)


class ESRGANUpscaler:
    def __init__(self, model_info: ModelInfo, device: torch.device) -> None:
        self.model_info = model_info
        self.device = device
        self.native_scale = model_info.native_scale
        state_dict = load_state_dict(model_info.path)
        rrdb_blocks = infer_rrdb_blocks(state_dict)
        self.model = RRDBNet(rrdb_blocks=rrdb_blocks, scale=self.native_scale)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.model.to(self.device)

    def upscale(
        self,
        image: Image.Image,
        outscale: float,
        tile_size: int,
        progress_callback: ProgressCallback | None = None,
    ) -> Image.Image:
        source = ImageOps.exif_transpose(image)
        alpha_channel = source.getchannel("A") if "A" in source.getbands() else None
        rgb_image = source.convert("RGB")
        width, height = rgb_image.size

        source_array = np.asarray(rgb_image, dtype=np.float32) / 255.0
        source_tensor = torch.from_numpy(source_array).permute(2, 0, 1).unsqueeze(0)
        effective_tile = self._resolve_tile_size(width, height, tile_size)

        if progress_callback is not None:
            progress_callback(0.0, "Preparing image tensor")

        output_tensor = self._run_model(source_tensor, effective_tile, progress_callback)
        output_tensor = output_tensor.clamp(0.0, 1.0)
        output_array = (
            output_tensor.squeeze(0)
            .permute(1, 2, 0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        result = Image.fromarray(output_array, mode="RGB")

        target_size = (
            max(1, round(width * outscale)),
            max(1, round(height * outscale)),
        )
        if result.size != target_size:
            if progress_callback is not None:
                progress_callback(0.94, "Resizing to the requested scale")
            result = result.resize(target_size, Image.Resampling.LANCZOS)

        if alpha_channel is not None:
            resized_alpha = alpha_channel.resize(target_size, Image.Resampling.LANCZOS)
            rgba_result = result.convert("RGBA")
            rgba_result.putalpha(resized_alpha)
            result = rgba_result

        if progress_callback is not None:
            progress_callback(1.0, "Rendering preview")

        return result

    def _resolve_tile_size(self, width: int, height: int, tile_size: int) -> int:
        if tile_size > 0:
            return tile_size

        longest_edge = max(width, height)
        if self.device.type == "cuda":
            if longest_edge <= 1400:
                return 0
            return 512

        if longest_edge <= 800:
            return 0
        if longest_edge <= 1800:
            return 384
        return 256

    def _run_model(
        self,
        source_tensor: torch.Tensor,
        tile_size: int,
        progress_callback: ProgressCallback | None,
    ) -> torch.Tensor:
        _, _, height, width = source_tensor.shape
        if tile_size <= 0 or tile_size >= max(height, width):
            with torch.inference_mode():
                result = self.model(source_tensor.to(self.device)).cpu()
            if progress_callback is not None:
                progress_callback(0.9, "Rendered 1 of 1 tiles")
            return result

        tile_pad = 16
        output = torch.empty(
            (1, 3, height * self.native_scale, width * self.native_scale),
            dtype=torch.float32,
        )
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)
        total_tiles = tiles_x * tiles_y
        processed_tiles = 0

        for tile_y in range(tiles_y):
            for tile_x in range(tiles_x):
                start_x = tile_x * tile_size
                end_x = min(start_x + tile_size, width)
                start_y = tile_y * tile_size
                end_y = min(start_y + tile_size, height)

                padded_start_x = max(start_x - tile_pad, 0)
                padded_end_x = min(end_x + tile_pad, width)
                padded_start_y = max(start_y - tile_pad, 0)
                padded_end_y = min(end_y + tile_pad, height)

                tile = source_tensor[
                    :,
                    :,
                    padded_start_y:padded_end_y,
                    padded_start_x:padded_end_x,
                ].to(self.device)

                with torch.inference_mode():
                    tile_output = self.model(tile).cpu()

                out_start_x = start_x * self.native_scale
                out_end_x = end_x * self.native_scale
                out_start_y = start_y * self.native_scale
                out_end_y = end_y * self.native_scale

                crop_start_x = (start_x - padded_start_x) * self.native_scale
                crop_end_x = crop_start_x + (end_x - start_x) * self.native_scale
                crop_start_y = (start_y - padded_start_y) * self.native_scale
                crop_end_y = crop_start_y + (end_y - start_y) * self.native_scale

                output[:, :, out_start_y:out_end_y, out_start_x:out_end_x] = tile_output[
                    :,
                    :,
                    crop_start_y:crop_end_y,
                    crop_start_x:crop_end_x,
                ]

                processed_tiles += 1
                if progress_callback is not None:
                    progress_callback(
                        processed_tiles / total_tiles * 0.9,
                        f"Rendered tile {processed_tiles} of {total_tiles}",
                    )

        return output


class UpscaleService:
    def __init__(self, model_dir: Path, output_dir: Path, upload_dir: Path) -> None:
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.upload_dir = upload_dir
        self.output_dir.mkdir(exist_ok=True)
        self.upload_dir.mkdir(exist_ok=True)
        self.device, self.device_label = resolve_torch_device()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="upscale")
        self.jobs: dict[str, UpscaleJob] = {}
        self.job_lock = threading.Lock()
        self.batches: dict[str, BatchJob] = {}
        self.batch_lock = threading.Lock()
        self.model_cache: dict[str, ESRGANUpscaler] = {}
        self.model_lock = threading.Lock()

    def list_models(self) -> list[ModelInfo]:
        models: list[ModelInfo] = []
        for model_path in sorted(self.model_dir.glob("*.pth")):
            models.append(
                ModelInfo(
                    key=model_path.name,
                    label=format_model_label(model_path.stem),
                    path=model_path,
                    native_scale=infer_native_scale(model_path),
                )
            )
        return models

    def get_job(self, job_id: str) -> dict[str, int | float | str | None]:
        with self.job_lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return job.to_dict()

    def get_batch(self, batch_id: str) -> dict[str, object]:
        with self.batch_lock:
            batch = self.batches.get(batch_id)
            if batch is None:
                raise KeyError(batch_id)

        with self.job_lock:
            items = [self.jobs[job_id] for job_id in batch.job_ids if job_id in self.jobs]

        return batch.to_dict(items)

    def submit_job(
        self,
        image_bytes: bytes,
        filename: str,
        model_key: str,
        upscale_factor: float,
        tile_size: int,
    ) -> str:
        model_info = self._get_model_info(model_key)
        job_id, input_path = self._queue_job(
            image_bytes=image_bytes,
            filename=filename,
            model_info=model_info,
            upscale_factor=upscale_factor,
        )

        self.executor.submit(
            self._process_job,
            job_id,
            input_path,
            filename,
            model_info,
            upscale_factor,
            tile_size,
        )
        return job_id

    def submit_batch(
        self,
        images: list[tuple[bytes, str]],
        model_key: str,
        upscale_factor: float,
        tile_size: int,
    ) -> dict[str, object]:
        if not images:
            raise ValueError("Please select at least one image.")

        model_info = self._get_model_info(model_key)
        batch_id = uuid4().hex
        queued_items: list[dict[str, str]] = []
        job_ids: list[str] = []

        for item_index, (image_bytes, filename) in enumerate(images):
            job_id, input_path = self._queue_job(
                image_bytes=image_bytes,
                filename=filename,
                model_info=model_info,
                upscale_factor=upscale_factor,
                batch_id=batch_id,
                item_index=item_index,
            )
            job_ids.append(job_id)
            queued_items.append({"job_id": job_id, "input_filename": filename})
            self.executor.submit(
                self._process_job,
                job_id,
                input_path,
                filename,
                model_info,
                upscale_factor,
                tile_size,
            )

        batch = BatchJob(
            id=batch_id,
            model_key=model_info.key,
            upscale_factor=upscale_factor,
            tile_size=tile_size,
            job_ids=job_ids,
        )
        with self.batch_lock:
            self.batches[batch_id] = batch

        return {
            "batch_id": batch_id,
            "items": queued_items,
        }

    def _queue_job(
        self,
        image_bytes: bytes,
        filename: str,
        model_info: ModelInfo,
        upscale_factor: float,
        batch_id: str | None = None,
        item_index: int | None = None,
    ) -> tuple[str, Path]:
        job_id = uuid4().hex
        input_stem = sanitize_stem(Path(filename).stem)
        input_name = f"{job_id[:8]}-{input_stem}{Path(filename).suffix or '.png'}"
        input_path = self.upload_dir / input_name
        input_path.write_bytes(image_bytes)

        job = UpscaleJob(
            id=job_id,
            status="queued",
            progress=0,
            message="Queued for processing",
            input_filename=filename,
            model_key=model_info.key,
            upscale_factor=upscale_factor,
            batch_id=batch_id,
            item_index=item_index,
        )
        with self.job_lock:
            self.jobs[job_id] = job

        return job_id, input_path

    def _get_model_info(self, model_key: str) -> ModelInfo:
        for model in self.list_models():
            if model.key == model_key:
                return model
        raise ValueError(f"Model '{model_key}' was not found in the model folder.")

    def _get_or_load_model(self, model_info: ModelInfo) -> ESRGANUpscaler:
        with self.model_lock:
            cached_model = self.model_cache.get(model_info.key)
            if cached_model is not None:
                return cached_model

            model_runner = ESRGANUpscaler(model_info, self.device)
            self.model_cache[model_info.key] = model_runner
            return model_runner

    def _update_job(self, job_id: str, **changes: object) -> None:
        with self.job_lock:
            job = self.jobs[job_id]
            for key, value in changes.items():
                setattr(job, key, value)

    def _process_job(
        self,
        job_id: str,
        input_path: Path,
        input_filename: str,
        model_info: ModelInfo,
        upscale_factor: float,
        tile_size: int,
    ) -> None:
        try:
            self._update_job(job_id, status="running", progress=4, message="Opening image")
            with Image.open(input_path) as uploaded_image:
                source_image = ImageOps.exif_transpose(uploaded_image).copy()

            self._update_job(
                job_id,
                progress=10,
                message="Inspecting image",
                original_width=source_image.width,
                original_height=source_image.height,
            )

            self._update_job(job_id, progress=16, message=f"Loading {model_info.label}")
            model_runner = self._get_or_load_model(model_info)

            def on_progress(progress_fraction: float, message: str) -> None:
                mapped_progress = 16 + round(progress_fraction * 78)
                self._update_job(job_id, progress=min(mapped_progress, 94), message=message)

            result_image = model_runner.upscale(source_image, upscale_factor, tile_size, on_progress)

            factor_label = f"{upscale_factor:g}x".replace(".", "p")
            source_stem = sanitize_stem(Path(input_filename).stem)
            output_name = (
                f"{job_id[:8]}-"
                f"{source_stem}-"
                f"{sanitize_stem(model_info.path.stem)}-"
                f"{factor_label}.png"
            )
            output_path = self.output_dir / output_name
            self._update_job(job_id, progress=96, message="Saving output image")
            result_image.save(output_path, format="PNG")

            self._update_job(
                job_id,
                status="completed",
                progress=100,
                message="Upscale finished",
                result_width=result_image.width,
                result_height=result_image.height,
                output_url=f"/outputs/{output_name}",
            )
        except Exception as exc:
            self._update_job(
                job_id,
                status="failed",
                progress=100,
                message="Upscale failed",
                error=str(exc),
            )
        finally:
            input_path.unlink(missing_ok=True)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()