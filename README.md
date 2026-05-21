# UpScale

UpScale is an ESRGAN image upscaler with two entry points:

- a local FastAPI browser app for single images and batches
- a terminal CLI that works well in Google Colab, including CUDA selection and a live progress bar

It discovers `.pth` weights from the `model/` folder, lets you queue one image or a whole batch in the browser, compare before and after in a split viewer, and export each finished render. For Colab or headless use, the terminal mode can prompt for an image, let you pick an available model, choose an upscale factor, and write the output file directly.

## Features

- Browser-based UI served locally with FastAPI
- Terminal CLI for one-off upscales with interactive prompts
- Model picker wired to the existing `model/` folder
- Batch processing with a selectable image queue
- Original and estimated output resolution panels for the focused item
- Drag-and-drop input preview, output preview, and per-item downloads
- Before/after split view with zoom, pan, and reset controls
- Background upscale jobs with aggregate progress polling and per-item status cards
- Classic ESRGAN RRDB checkpoint support, matching the included UltraSharp and UltraYandere weights
- Optional Stable Diffusion img2img refinement with `denoise strength` from `0` to `1`
- Explicit device selection with `auto`, `cuda`, or `cpu`

## One-click launcher

Double-click [start-upscale.bat](start-upscale.bat) in Windows Explorer.

The launcher will:

- create `.venv` if it does not exist yet
- install dependencies the first time, or whenever `requirements.txt` changes
- start the FastAPI server in its own console window
- open `http://127.0.0.1:8000` in your browser

This launcher is for local desktop use. In Google Colab, use the terminal commands below and do not create a virtualenv.

## Run locally

```powershell
python launch_upscale.py
```

That command creates or reuses `.venv`, installs dependencies when needed, starts the FastAPI app, and opens the browser.

If you prefer the manual path:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python launch_upscale.py serve --reload
```

Then open `http://127.0.0.1:8000` in your browser.

## Terminal CLI

The terminal mode is useful for headless machines and Colab. If you omit arguments, it prompts for the missing values, including denoise strength.

```powershell
python launch_upscale.py cli
```

Useful commands:

```powershell
python launch_upscale.py cli --list-models
python launch_upscale.py cli --input uploads\example.png --model 1 --factor 4 --output outputs\example-4x.png
python launch_upscale.py cli --input uploads\example.png --model 4x-UltraSharp.pth --factor 2 --denoise 0.3 --device cuda
```

Notes for terminal mode:

- `--model` accepts either the model filename or the 1-based index shown by `--list-models`
- output defaults to the `outputs/` folder if you do not supply `--output`
- `--tile-size 0` keeps automatic tiling enabled
- `--denoise 0` keeps pure ESRGAN output; values above `0` add a Stable Diffusion img2img refinement pass after ESRGAN
- `--device cuda` fails fast if PyTorch cannot access a CUDA device, instead of silently falling back to CPU
- the first denoise-enabled run downloads the diffusion checkpoint defined by `UPSCALE_DIFFUSION_MODEL` unless it is already cached

## Denoise Strength

The GUI exposes denoise strength inside **Advanced options**, and the CLI exposes it through `--denoise`.

- `0` means ESRGAN only
- values between `0` and `1` run ESRGAN first, then a Stable Diffusion img2img refinement pass with the same numeric `strength` value
- `1` gives the diffusion stage maximum freedom to move away from the ESRGAN output

This is the same kind of `strength` parameter used by Automatic1111 img2img, but matching the exact same output requires more than the number alone. To get close to an Automatic1111 setup, you also need to match the same diffusion checkpoint, prompt, negative prompt, step count, guidance scale, and seed.

These environment variables control the diffusion stage:

- `UPSCALE_DIFFUSION_MODEL` defaults to `runwayml/stable-diffusion-v1-5`
- `UPSCALE_DIFFUSION_PROMPT` defaults to an empty prompt
- `UPSCALE_DIFFUSION_NEGATIVE_PROMPT` defaults to an empty negative prompt
- `UPSCALE_DIFFUSION_STEPS` defaults to `20`
- `UPSCALE_DIFFUSION_GUIDANCE` defaults to `1.0`
- `UPSCALE_DIFFUSION_SEED` is optional; set it if you want repeatable denoise runs

## Google Colab

Use the terminal workflow in Colab and install dependencies into the current runtime, not into a virtualenv.

1. Switch the notebook runtime to `T4 GPU`.
2. Open a terminal in the repository checkout.
3. Install dependencies into the active Colab Python:

```bash
pip install -r requirements-colab.txt
```

4. Optionally choose the Stable Diffusion checkpoint used for denoise refinement:

```bash
export UPSCALE_DIFFUSION_MODEL=runwayml/stable-diffusion-v1-5
```

5. Run the CLI in CUDA mode:

```bash
python launch_upscale.py cli --device cuda --denoise 0.3
```

For a fully non-interactive run:

```bash
python launch_upscale.py cli \
	--input /content/input.png \
	--model 1 \
	--factor 4 \
	--denoise 0.3 \
	--output /content/output.png \
	--device cuda
```

If you want to start the web server from Colab instead of the CLI:

```bash
python launch_upscale.py serve --foreground --host 0.0.0.0 --device cuda
```

The server binds to `0.0.0.0` in Colab. You still need a Colab notebook proxy or tunnel if you want to open that browser UI from outside the runtime.

## Notes

- The included models are native `4x` ESRGAN weights. If you choose another factor in the UI, the app performs the ESRGAN pass first and then resizes to the exact requested factor.
- The batch queue runs images sequentially to keep GPU and CPU memory use predictable, and you can click any queue item to inspect or download it while the rest continue processing.
- The app automatically tiles larger images to reduce memory pressure, and you can override tile size from the Advanced options panel.
- The compare workspace uses the finished output size for the split view, so zoom and pan line up the original and upscaled detail at the same visual scale.
- Denoise-enabled runs use a lazily loaded Stable Diffusion img2img pipeline; the first run can take longer because it has to load or download the diffusion checkpoint.
- `UPSCALE_DEVICE` also supports `auto`, `cuda`, or `cpu` if you want to control the device through the environment instead of CLI flags.
- CPU mode works, but it will be much slower than CUDA if a compatible GPU is available.