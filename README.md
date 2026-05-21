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

## One-click launcher

Double-click [start-upscale.bat](start-upscale.bat) in Windows Explorer.

The launcher will:

- create `.venv` if it does not exist yet
- install dependencies the first time, or whenever `requirements.txt` changes
- start the FastAPI server in its own console window
- open `http://127.0.0.1:8000` in your browser

## Run it

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000` in your browser.

## Notes

- The included models are native `4x` ESRGAN weights. If you choose another factor in the UI, the app performs the ESRGAN pass first and then resizes to the exact requested factor.
- The batch queue runs images sequentially to keep GPU and CPU memory use predictable, and you can click any queue item to inspect or download it while the rest continue processing.
- The app automatically tiles larger images to reduce memory pressure, and you can override tile size from the Advanced options panel.
- The compare workspace uses the finished output size for the split view, so zoom and pan line up the original and upscaled detail at the same visual scale.
- CPU mode works, but it will be much slower than CUDA if a compatible GPU is available.