# UpScale

UpScale is a local browser app for ESRGAN image upscaling. It discovers `.pth` weights from the `model/` folder, lets you queue one image or a whole batch, compare before and after in a split viewer, and export each finished render from the browser.

## Features

- Browser-based UI served locally with FastAPI
- Model picker wired to the existing `model/` folder
- Batch processing with a selectable image queue
- Original and estimated output resolution panels for the focused item
- Drag-and-drop input preview, output preview, and per-item downloads
- Before/after split view with zoom, pan, and reset controls
- Background upscale jobs with aggregate progress polling and per-item status cards
- Classic ESRGAN RRDB checkpoint support, matching the included UltraSharp and UltraYandere weights
- Explicit device selection with automatic CUDA detection and fail-fast `cuda` mode

## One-click launcher

Double-click [start-upscale.bat](start-upscale.bat) in Windows Explorer.

The launcher will:

- create `.venv` if it does not exist yet
- install dependencies the first time, or whenever `requirements.txt` changes
- start the FastAPI server in its own console window
- open `http://127.0.0.1:8000` in your browser

## Terminal launcher

`launch_upscale.py` now works from a regular terminal on Windows, Linux, and Google Colab.

Common options:

- `--foreground` keeps Uvicorn attached to the current terminal
- `--background` starts the server and returns after the health check passes
- `--host` and `--port` control the bind address
- `--device auto|cuda|cpu` controls the `UPSCALE_DEVICE` setting passed into the app

## Run it locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python launch_upscale.py --foreground --reload
```

Then open `http://127.0.0.1:8000` in your browser.

If you prefer to run Uvicorn directly, this still works:

```powershell
$env:UPSCALE_DEVICE = "auto"
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## Run it in Google Colab

1. Switch the Colab runtime to `T4 GPU`.
2. Open a Colab terminal and clone or copy this repository into the runtime.
3. Start UpScale from the repo root.

```bash
cd /content/UpScale
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python launch_upscale.py --foreground --no-browser
```

What the launcher does in Colab:

- installs from `requirements-colab.txt` so it does not replace Colab's preinstalled PyTorch build
- binds the server to `0.0.0.0` by default
- disables automatic browser launch by default
- requests `UPSCALE_DEVICE=cuda` by default, so startup fails immediately if the runtime is not actually on CUDA

To verify CUDA before opening the app:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
curl http://127.0.0.1:8000/api/health
```

The health response includes the active device label. On a T4 runtime it should report a CUDA GPU, for example `GPU (CUDA: Tesla T4)`.

If you want to run Uvicorn directly in Colab instead of the launcher:

```bash
source .venv/bin/activate
python -m pip install -r requirements-colab.txt
UPSCALE_DEVICE=cuda uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Use Colab's port preview or proxy feature to open port `8000` from your browser.

## Notes

- The included models are native `4x` ESRGAN weights. If you choose another factor in the UI, the app performs the ESRGAN pass first and then resizes to the exact requested factor.
- The batch queue runs images sequentially to keep GPU and CPU memory use predictable, and you can click any queue item to inspect or download it while the rest continue processing.
- The app automatically tiles larger images to reduce memory pressure, and you can override tile size from the Advanced options panel.
- The compare workspace uses the finished output size for the split view, so zoom and pan line up the original and upscaled detail at the same visual scale.
- CPU mode still works in `auto`, but it will be much slower than CUDA.
- Use `--device cuda` or set `UPSCALE_DEVICE=cuda` when you want the app to fail fast instead of silently falling back to CPU.