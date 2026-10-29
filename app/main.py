"""Local demonstration UI. Run with: python -m uvicorn app.main:app --host 127.0.0.1."""
from app.network import configure_tls

configure_tls()

from contextlib import asynccontextmanager
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.database import MongoDatabase
from app.ui_jobs import JobStore

ROOT = Path(__file__).resolve().parents[1]
MAX_UPLOAD = 12 * 1024 * 1024
MAX_PIXELS = 16_000_000
FORMATS = {"PNG": ".png", "JPEG": ".jpg"}
EXTENSION_FORMATS = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG"}
MIME_FORMATS = {"image/png": "PNG", "image/jpeg": "JPEG"}
SAMPLES = {
    "sinhala": {"path": ROOT / "uploads/e8ac730b_agbho.png", "label": "Sinhala news", "language": "සිංහල"},
    "english": {"path": ROOT / "uploads/7efc26c2_Rumesh.png", "label": "English news", "language": "English"},
}


def validate_image(data, filename=None, content_type=None):
    if not data:
        raise HTTPException(400, "Choose an image first.")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Image is too large. Maximum file size is 12 MB.")
    expected_format = None
    if filename is not None:
        expected_format = EXTENSION_FORMATS.get(Path(filename).suffix.lower())
        if expected_format is None:
            raise HTTPException(415, "Choose a PNG, JPG or JPEG image only.")
    # Some clients cannot infer a MIME type; decoded content is authoritative.
    mime_format = MIME_FORMATS.get(content_type)
    if content_type and content_type != "application/octet-stream" and mime_format is None:
        raise HTTPException(415, "Choose a PNG, JPG or JPEG image only.")
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format not in FORMATS:
                raise HTTPException(415, "Choose a PNG, JPG or JPEG image only.")
            if ((expected_format and image.format != expected_format)
                    or (mime_format and image.format != mime_format)):
                raise HTTPException(415, "The image contents do not match its file type. Export it as PNG or JPEG and try again.")
            if image.width * image.height > MAX_PIXELS:
                raise HTTPException(413, "Image dimensions are too large. Maximum size is 16 megapixels.")
            info = {"width": image.width, "height": image.height, "format": image.format, "bytes": len(data)}
            image.verify()
        with Image.open(BytesIO(data)) as source:
            source.load()
            preview = ImageOps.exif_transpose(source).convert("RGB")
            preview.thumbnail((1400, 1400))
        return info, preview
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise HTTPException(400, "This file is not a readable image.") from error


def create_app(upload_dir=None, runner=None, database=None):
    destination = Path(upload_dir) if upload_dir else ROOT / "uploads/ui"
    database = database or MongoDatabase()
    jobs = JobStore(runner, repository=database) if runner else JobStore(repository=database)

    @asynccontextmanager
    async def lifespan(application):
        database.connect()
        yield
        jobs.close()
        database.close()

    application = FastAPI(title="Verity · Image Verification", lifespan=lifespan)
    application.state.jobs = jobs
    application.state.database = database
    application.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])

    @application.middleware("http")
    async def local_safety(request: Request, call_next):
        if request.method == "POST":
            origin = request.headers.get("origin")
            if origin and urlparse(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": "Cross-origin submissions are not allowed."}, status_code=403)
            length = request.headers.get("content-length", "0")
            if not length.isdigit() or int(length) > MAX_UPLOAD + 1024 * 1024:
                return JSONResponse({"detail": "Upload exceeds the 12 MB image limit."}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    application.mount("/static", StaticFiles(directory=ROOT / "app/web"), name="static")

    @application.get("/")
    def index():
        return FileResponse(ROOT / "app/web/index.html")

    @application.get("/api/health")
    def health():
        return {"status": "ready", "mode": "local_demo", "max_upload_mb": 12,
                "allowed_extensions": list(EXTENSION_FORMATS), "max_pixels": MAX_PIXELS,
                "mongodb": database.health(),
                "note": "Models load on first analysis. MongoDB retains history when configured."}

    @application.get("/api/samples")
    def samples():
        return [{"id": key, "label": row["label"], "language": row["language"], "image_url": f"/api/samples/{key}/image"}
                for key, row in SAMPLES.items() if row["path"].is_file()]

    @application.get("/api/samples/{sample_id}/image")
    def sample_image(sample_id: str):
        row = SAMPLES.get(sample_id)
        if not row or not row["path"].is_file():
            raise HTTPException(404, "Sample not available.")
        return FileResponse(row["path"])

    @application.post("/api/analyses", status_code=202)
    async def analyze(file: UploadFile | None = File(None), sample_id: str = Form(""),
                      headline: str = Form("", max_length=2000), comparison_text: str = Form("", max_length=2000),
                      include_details: bool = Form(False)):
        if bool(file) == bool(sample_id):
            raise HTTPException(400, "Choose one uploaded image or one sample.")
        if comparison_text.strip():
            from app.services.language_service import detect_language
            if detect_language(comparison_text) != "en" or len(comparison_text.split()) < 2:
                raise HTTPException(400, "Comparison text must be an English sentence or headline.")
        if file:
            try:
                data = await file.read(MAX_UPLOAD + 1)
                filename = (file.filename or "").replace("\\", "/").split("/")[-1]
                content_type = file.content_type
            finally:
                await file.close()
        else:
            row = SAMPLES.get(sample_id)
            if not row or not row["path"].is_file():
                raise HTTPException(404, "Sample not available.")
            data, filename = row["path"].read_bytes(), row["path"].name
            content_type = None
        info, preview = validate_image(data, filename, content_type)
        job_id = uuid4().hex
        folder = destination / job_id
        folder.mkdir(parents=True)
        image_path = folder / ("input" + FORMATS[info["format"]])
        image_path.write_bytes(data)  # Preserve the original; preview is a separate image.
        preview.save(folder / "preview.png")
        try:
            return jobs.create(job_id, image_path, filename, info, {
                "headline": headline.strip(), "comparison_text": comparison_text.strip(), "include_details": include_details,
            })
        except ValueError as error:
            # Remove only files created by this rejected request, never user originals.
            image_path.unlink(missing_ok=True)
            (folder / "preview.png").unlink(missing_ok=True)
            folder.rmdir()
            raise HTTPException(429, str(error)) from error

    def get_job(job_id):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Analysis not found in this session. Start a new analysis.")
        return job

    @application.get("/api/analyses")
    def history():
        return jobs.recent()

    @application.get("/api/analyses/{job_id}")
    def result(job_id: str):
        return get_job(job_id)

    @application.get("/api/analyses/{job_id}/image")
    def image(job_id: str):
        get_job(job_id)
        return FileResponse(destination / job_id / "preview.png")

    @application.get("/api/analyses/{job_id}/report")
    def report(job_id: str):
        job = get_job(job_id)
        if job["status"] in ("queued", "running"):
            raise HTTPException(409, "The analysis is still running.")
        return Response(json.dumps(job, ensure_ascii=False, indent=2, allow_nan=False),
                        media_type="application/json; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="verification-{job_id[:8]}.json"'})

    return application


app = create_app()
