from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.inference import UpscaleService


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
STATIC_DIR = BASE_DIR / "app" / "static"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = BASE_DIR / "uploads"

OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

service = UpscaleService(model_dir=MODEL_DIR, output_dir=OUTPUT_DIR, upload_dir=UPLOAD_DIR)

app = FastAPI(title="UpScale", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health_check() -> dict[str, bool | str]:
    return {"ok": True, "device": service.device_label}


@app.get("/api/models")
def list_models() -> dict[str, list[dict[str, int | str]] | str]:
    return {
        "device": service.device_label,
        "models": [model.to_dict() for model in service.list_models()],
    }


async def read_uploaded_image(image: UploadFile) -> bytes:
    if not image.filename:
        raise HTTPException(status_code=400, detail="Please select an input image.")
    if not (image.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Only image uploads are supported: {image.filename}")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail=f"The selected image is empty: {image.filename}")
    return image_bytes


def validate_upscale_inputs(upscale_factor: float, tile_size: int, denoise_strength: float) -> None:
    if upscale_factor < 1 or upscale_factor > 8:
        raise HTTPException(status_code=400, detail="Upscale factor must be between 1x and 8x.")
    if tile_size < 0 or tile_size > 2048:
        raise HTTPException(status_code=400, detail="Tile size must be between 0 and 2048.")
    if denoise_strength < 0 or denoise_strength > 1:
        raise HTTPException(status_code=400, detail="Denoise strength must be between 0 and 1.")


@app.post("/api/upscale", status_code=202)
async def upscale_image(
    image: UploadFile = File(...),
    model: str = Form(...),
    upscale_factor: float = Form(...),
    tile_size: int = Form(0),
    denoise_strength: float = Form(0),
) -> dict[str, str]:
    validate_upscale_inputs(upscale_factor, tile_size, denoise_strength)
    image_bytes = await read_uploaded_image(image)

    try:
        job_id = service.submit_job(
            image_bytes=image_bytes,
            filename=image.filename,
            model_key=model,
            upscale_factor=upscale_factor,
            denoise_strength=denoise_strength,
            tile_size=tile_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"job_id": job_id}


@app.post("/api/upscale/batch", status_code=202)
async def upscale_batch(
    images: list[UploadFile] = File(...),
    model: str = Form(...),
    upscale_factor: float = Form(...),
    tile_size: int = Form(0),
    denoise_strength: float = Form(0),
) -> dict[str, object]:
    validate_upscale_inputs(upscale_factor, tile_size, denoise_strength)

    if not images:
        raise HTTPException(status_code=400, detail="Please select at least one image.")

    payloads: list[tuple[bytes, str]] = []
    for image in images:
        image_bytes = await read_uploaded_image(image)
        if image.filename is None:
            raise HTTPException(status_code=400, detail="One of the selected images has no filename.")
        payloads.append((image_bytes, image.filename))

    try:
        return service.submit_batch(
            images=payloads,
            model_key=model,
            upscale_factor=upscale_factor,
            denoise_strength=denoise_strength,
            tile_size=tile_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, int | float | str | None]:
    try:
        return service.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc


@app.get("/api/batches/{batch_id}")
def get_batch(batch_id: str) -> dict[str, object]:
    try:
        return service.get_batch(batch_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Batch not found.") from exc