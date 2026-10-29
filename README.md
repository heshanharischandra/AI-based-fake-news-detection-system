# AI-Based Fake News Detection System

## Image Content Verification and Authenticity Analysis

This repository contains the **Image Content Verification and Authenticity Analysis** component of an AI-based fake news detection system. It processes Sinhala and English news images, extracts embedded text and headlines, checks image–text consistency, detects AI-generated visuals, identifies image reuse from a reference collection, and produces a verified context score for claim assessment.

## Current Status

The project is currently at the **PP1 prototype stage**.

### Implemented

- Image upload through a FastAPI web interface
- Image validation and preprocessing
- Sinhala and English OCR headline extraction
- Multi-pass image preprocessing for OCR quality improvement
- Language detection and text normalisation
- Headline translation to English for cross-language analysis
- Image–text semantic similarity scoring using CLIP
- AI-generated image detection using a pretrained binary classifier
- Multi-view augmentation for classifier robustness
- Local image reuse detection using perceptual hash ensemble and SIFT feature matching
- News context verification against a verified reference database
- Evidence-gated verification scoring across four context factors
- Natural language explanation generation
- Evidence and reference retrieval
- MongoDB persistence for analysis history
- REST API for programmatic access
- End-to-end evaluation script with ground-truth support

### In Progress

- Calibrating AI-generation and similarity thresholds against labelled image pairs
- Improving Sinhala OCR accuracy with custom-trained tessdata
- Expanding the verified news context reference database
- Final verdict generation:
  - `VERIFIED`
  - `LIKELY_MANIPULATED`
  - `INSUFFICIENT_EVIDENCE`
- Model evaluation using suitable held-out datasets
- End-to-end performance evaluation

## Processing Pipeline

```text
Uploaded Image
      ↓
Image Validation and Preprocessing
      ↓
OCR Headline Extraction (Sinhala / English)
      ↓
Language Detection and Translation
      ↓
Image–Text Consistency Check (CLIP)
      ↓
AI-Generation Detection
      ↓
Image Reuse Detection (Hash Ensemble + SIFT)
      ↓
News Context Verification
      ↓
Evidence-Gated Verification Scoring
      ↓
Explanation and Evidence Retrieval
```

## Project Structure

```text
ImageVerificationSystem/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── database.py
│   ├── ui_jobs.py
│   ├── network.py
│   ├── reference_registry.py
│   └── services/
│       ├── pipeline_service.py
│       ├── ocr_service.py
│       ├── authenticity_service.py
│       ├── reuse_service.py
│       ├── similarity_service.py
│       ├── scoring_service.py
│       ├── context_service.py
│       ├── claim_service.py
│       ├── visual_service.py
│       ├── explanation_service.py
│       ├── evidence_service.py
│       ├── language_service.py
│       ├── news_context_classifier.py
│       └── translation_service.py
├── app/web/
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── scripts/
│   ├── train_sinhala_ocr.py
│   ├── train_news_context.py
│   ├── evaluate_ocr_runtime.py
│   ├── register_verified_reference.py
│   └── download_sinhala_translation.py
├── tests/
│   ├── test_authenticity_service.py
│   ├── test_database.py
│   ├── test_evaluate_system.py
│   ├── test_news_context_classifier.py
│   ├── test_ocr_headline_layout.py
│   ├── test_ocr_training_safety.py
│   ├── test_pipeline_service.py
│   ├── test_reference_registry.py
│   ├── test_reuse_service.py
│   ├── test_run_demo_database.py
│   ├── test_sinhala_ocr.py
│   └── test_web_ui.py
├── docs/
├── evaluate_system.py
├── run_ui.ps1
├── requirements.txt
├── .gitignore
└── README.md
```

## Main Technologies

- Python
- FastAPI
- Tesseract OCR (English + Sinhala)
- OpenCV
- PyTorch
- Transformers (HuggingFace)
- CLIP (image–text similarity)
- Sentence Transformers
- ImageHash (perceptual hashing)
- MongoDB

## Dataset Plan

Different datasets support different stages of the component:

| Dataset | Purpose |
|---|---|
| Sinhala Misinformation Corpus | Sinhala OCR accuracy evaluation and training |
| ISOT Fake News Dataset | English news context reference and claim evaluation |
| Custom verified news context data | Context verification and scoring calibration |
| Custom Sinhala OCR training data | Fine-tuning Sinhala tessdata for news images |
| Labelled image pairs (real vs. AI) | Calibrating the AI-generation detection threshold |

Datasets are stored locally and excluded from GitHub because of their size and licensing requirements.

## Environment Configuration

Create a `.env` file in the project root:

```env
# MongoDB (optional — omit to run without persistence)
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=image_verification
MONGODB_COLLECTION=analyses

# Tesseract (if not on PATH)
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe

# Custom Sinhala tessdata directory (optional)
# SINHALA_TESSDATA_DIR=C:\path\to\tessdata
```

Never commit the real `.env` file or any API keys. The repository's `.gitignore` excludes this file.

## Running the Application

Activate the Python virtual environment:

```powershell
.venv311\Scripts\Activate.ps1
```

Run the web application:

```powershell
.\run_ui.ps1
```

Or start directly with uvicorn:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The application should open at **http://localhost:8000** in the web browser.

## Testing

The repository contains test scripts for individual services and the integrated pipeline:

```powershell
python -m pytest tests/ -v
```

Or run individual test files:

```powershell
python -m pytest tests/test_authenticity_service.py
python -m pytest tests/test_reuse_service.py
python -m pytest tests/test_pipeline_service.py
python -m pytest tests/test_sinhala_ocr.py
```

Some tests require Tesseract installation, model downloads, or local reference image data.

## Security and Repository Rules

The following content is intentionally excluded from GitHub:

- `.env` files and API keys
- Python virtual environments
- Datasets
- Uploaded images
- ML model weights
- Reference image collection
- Generated reports
- Python cache files

## Future Work

- Fine-tune Sinhala tessdata specifically for news image typography
- Calibrate AI-generation and CLIP thresholds using labelled image pairs
- Develop multilingual claim-evidence verification
- Expand verified news context reference database
- Evaluate reuse detection accuracy using controlled near-duplicate pairs
- Add final verdict labels with confidence scores
- Perform complete Sinhala and English end-to-end evaluation
- Integrate with the Video Content Verification component for unified analysis

## Branch

Development for this component is maintained in:

```text
Image-Content-Verification-and-Authenticity-Analysis
```

## Author

**[Your Name]**

AI-Based Fake News Detection System  
Image Content Verification and Authenticity Analysis Component
