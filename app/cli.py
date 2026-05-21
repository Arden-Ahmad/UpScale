from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

from app.inference import ModelInfo, UpscaleService, build_output_name


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = BASE_DIR / "uploads"
SUPPORTED_DEVICE_PREFERENCES = ("auto", "cuda", "cpu")
DEFAULT_UPSCALE_FACTOR = 4.0
DEFAULT_TILE_SIZE = 0


class TerminalProgressBar:
    def __init__(self, width: int = 32) -> None:
        self.width = width
        self._last_line_length = 0
        self._completed = False

    def update(self, progress_fraction: float, message: str) -> None:
        bounded_progress = max(0.0, min(progress_fraction, 1.0))
        filled_width = round(self.width * bounded_progress)
        bar = "#" * filled_width + "-" * (self.width - filled_width)
        line = f"[{bar}] {bounded_progress * 100:6.2f}% {message}"
        padding = max(0, self._last_line_length - len(line))
        print(f"\r{line}{' ' * padding}", end="", flush=True)
        self._last_line_length = len(line)
        if bounded_progress >= 1.0:
            print()
            self._completed = True

    def finish(self) -> None:
        if self._completed or self._last_line_length == 0:
            return
        print()
        self._completed = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upscale one image from the terminal.")
    parser.add_argument("--input", "-i", help="Input image path")
    parser.add_argument("--model", "-m", help="Model key or 1-based model index")
    parser.add_argument("--factor", "-f", type=float, help="Upscale factor between 1 and 8")
    parser.add_argument("--denoise", type=float, help="Diffusion denoise strength between 0 and 1")
    parser.add_argument("--output", "-o", help="Output image path")
    parser.add_argument(
        "--tile-size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help="Tile size override, or 0 to auto-select",
    )
    parser.add_argument(
        "--device",
        choices=SUPPORTED_DEVICE_PREFERENCES,
        help="Compute device preference",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List discovered models and exit",
    )
    return parser


def resolve_user_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def prompt_with_default(prompt: str, default: str) -> str:
    response = input(f"{prompt} [{default}]: ").strip()
    return response or default


def resolve_input_path(provided_value: str | None) -> Path:
    if provided_value is not None:
        input_path = resolve_user_path(provided_value)
        if input_path.is_file():
            return input_path
        raise ValueError(f"Input image was not found: {input_path}")

    if not sys.stdin.isatty():
        raise ValueError("Provide --input when running non-interactively.")

    while True:
        input_path = resolve_user_path(input("Input image path: ").strip())
        if input_path.is_file():
            return input_path
        print(f"Image not found: {input_path}", file=sys.stderr)


def print_models(models: list[ModelInfo]) -> None:
    print("Available models:")
    for index, model in enumerate(models, start=1):
        print(f"  {index}. {model.label} ({model.key}, native {model.native_scale}x)")


def resolve_model_choice(models: list[ModelInfo], provided_value: str | None) -> ModelInfo:
    if not models:
        raise RuntimeError("No .pth model files were found in the model folder.")

    def select(choice: str) -> ModelInfo:
        normalized_choice = choice.strip()
        if not normalized_choice:
            raise ValueError("Model selection cannot be empty.")
        if normalized_choice.isdigit():
            model_index = int(normalized_choice) - 1
            if 0 <= model_index < len(models):
                return models[model_index]
            raise ValueError(f"Model index must be between 1 and {len(models)}.")

        lowered_choice = normalized_choice.lower()
        for model in models:
            if model.key.lower() == lowered_choice:
                return model
            if model.label.lower() == lowered_choice:
                return model
            if model.path.stem.lower() == lowered_choice:
                return model
        raise ValueError(f"Model '{choice}' was not found.")

    if provided_value is not None:
        return select(provided_value)

    if not sys.stdin.isatty():
        raise ValueError("Provide --model when running non-interactively.")

    print_models(models)
    while True:
        try:
            return select(prompt_with_default("Select a model", "1"))
        except ValueError as exc:
            print(exc, file=sys.stderr)


def validate_upscale_factor(value: float) -> float:
    if value < 1 or value > 8:
        raise ValueError("Upscale factor must be between 1x and 8x.")
    return value


def resolve_upscale_factor(provided_value: float | None) -> float:
    if provided_value is not None:
        return validate_upscale_factor(provided_value)

    if not sys.stdin.isatty():
        return DEFAULT_UPSCALE_FACTOR

    while True:
        raw_value = prompt_with_default("Upscale factor", f"{DEFAULT_UPSCALE_FACTOR:g}")
        try:
            return validate_upscale_factor(float(raw_value))
        except ValueError as exc:
            print(exc, file=sys.stderr)


def validate_tile_size(value: int) -> int:
    if value < 0 or value > 2048:
        raise ValueError("Tile size must be between 0 and 2048.")
    return value


def validate_denoise_strength(value: float) -> float:
    if value < 0 or value > 1:
        raise ValueError("Denoise strength must be between 0 and 1.")
    return value


def resolve_denoise_strength(provided_value: float | None) -> float:
    if provided_value is not None:
        return validate_denoise_strength(provided_value)

    if not sys.stdin.isatty():
        return 0.0

    while True:
        raw_value = prompt_with_default("Denoise strength", "0")
        try:
            return validate_denoise_strength(float(raw_value))
        except ValueError as exc:
            print(exc, file=sys.stderr)


def resolve_output_path(
    service: UpscaleService,
    input_path: Path,
    model_info: ModelInfo,
    upscale_factor: float,
    denoise_strength: float,
    provided_value: str | None,
) -> Path:
    default_output_path = service.output_dir / build_output_name(
        input_path.name,
        model_info,
        upscale_factor,
        denoise_strength=denoise_strength,
    )

    if provided_value is not None:
        output_path = resolve_user_path(provided_value)
    elif sys.stdin.isatty():
        output_path = resolve_user_path(
            prompt_with_default("Output image path", str(default_output_path))
        )
    else:
        output_path = default_output_path

    if output_path.exists() and output_path.is_dir():
        return output_path / default_output_path.name
    return output_path


def describe_device(service: UpscaleService) -> str:
    if service.device.type != "cuda":
        return service.device_label
    return f"{service.device_label}: {torch.cuda.get_device_name(0)}"


def run_cli(args: argparse.Namespace) -> int:
    service = UpscaleService(
        model_dir=MODEL_DIR,
        output_dir=OUTPUT_DIR,
        upload_dir=UPLOAD_DIR,
        device_preference=args.device,
    )
    models = service.list_models()
    if args.list_models:
        print(f"Device: {describe_device(service)}")
        print_models(models)
        return 0

    input_path = resolve_input_path(args.input)
    model_info = resolve_model_choice(models, args.model)
    upscale_factor = resolve_upscale_factor(args.factor)
    denoise_strength = resolve_denoise_strength(args.denoise)
    tile_size = validate_tile_size(args.tile_size)
    output_path = resolve_output_path(
        service,
        input_path,
        model_info,
        upscale_factor,
        denoise_strength,
        args.output,
    )

    print(f"Device: {describe_device(service)}")
    print(f"Input image: {input_path}")
    print(f"Model: {model_info.label} ({model_info.key})")
    print(f"Upscale factor: {upscale_factor:g}x")
    print(f"Denoise strength: {denoise_strength:g}")
    print(f"Output image: {output_path}")

    progress_bar = TerminalProgressBar()
    try:
        result_path = service.upscale_image_file(
            input_path=input_path,
            model_key=model_info.key,
            upscale_factor=upscale_factor,
            denoise_strength=denoise_strength,
            tile_size=tile_size,
            output_path=output_path,
            progress_callback=progress_bar.update,
        )
    except Exception:
        progress_bar.finish()
        raise

    print(f"Saved upscaled image to {result_path}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run_cli(args)
    except Exception as exc:  # pragma: no cover - terminal failure path
        print(f"Unable to upscale image: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())