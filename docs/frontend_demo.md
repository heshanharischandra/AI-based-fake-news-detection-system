# Frontend demo

The local browser UI is served by the existing Python project. It accepts an uploaded PNG, JPEG, or WebP image and runs the existing pipeline through the following browser-visible stages:

1. Sinhala/English headline extraction
2. Headline and claim preparation
3. BLIP visual description
4. CLIP image-text consistency
5. Image-generation indicator
6. Local image reuse detection
7. Context verification
8. Evidence-weighted scoring

The UI keeps uncertainty visible: a model score is not presented as proof that a news story is true, and verification scoring stays pending when verified context evidence is missing. See `evidence_coverage.md` to register trusted reference evidence.

## Start

From the project root in PowerShell:

```powershell
.\run_ui.ps1
```

Open `http://127.0.0.1:8000` in a browser. Models load lazily when an analysis begins, so the first run may take longer.

## MongoDB history

Copy `.env.example` to `.env` and set `MONGODB_URI` to a local MongoDB or MongoDB Atlas connection string. For Atlas, create a database user and allow this machine's IP address in Network Access first. Keep the real URI only in `.env`; that file is ignored by Git.

With MongoDB configured, analysis state and results are stored in the `image_verification.analyses` collection by default and remain available after an application restart. Change `MONGODB_DATABASE` or `MONGODB_COLLECTION` if needed. The `/api/health` response reports MongoDB as `connected`, `unavailable`, or `disabled`. A database outage does not stop local analysis or report-file creation.

The interactive `run_demo.py` command stores its completed records separately in the `demo_analyses` collection. Override that name with `MONGODB_DEMO_COLLECTION` when needed.

## Model download certificates

The application uses `truststore` to validate HTTPS with the operating system's
trusted certificates. It is included in the project requirements and initialized
by the UI and model services. Certificate and hostname verification stay enabled.
Restart the UI after installing dependencies or changing this configuration.

If `CERTIFICATE_VERIFY_FAILED` still occurs on another machine, check its clock
and trusted certificate configuration (including any organization-managed HTTPS
proxy). Do not disable TLS verification. Model loading and model accuracy are
separate concerns; a successful download does not establish accuracy.
