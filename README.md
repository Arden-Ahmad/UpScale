# UpScale

UpScale is a local browser app for ESRGAN image upscaling. It discovers `.pth` weights from the `model/` folder, lets you preview the input image, choose a model, set the upscale factor, and monitor progress while the render runs.

## Features

- Browser-based UI served locally with FastAPI
- Model picker wired to the existing `model/` folder
- Original and estimated output resolution panels
- Drag-and-drop input preview and output preview
- Background upscale jobs with progress polling and download link
- Classic ESRGAN RRDB checkpoint support, matching the included UltraSharp and UltraYandere weights

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
- The app automatically tiles larger images to reduce memory pressure, and you can override tile size from the Advanced options panel.
- CPU mode works, but it will be much slower than CUDA if a compatible GPU is available.