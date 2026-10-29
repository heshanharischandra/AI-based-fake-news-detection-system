# AI-Based Fake News Detection System

## Video Content Verification and Claim Analysis

This repository contains the **Video Content Verification and Claim Analysis** component of an AI-based fake news detection system. It processes Sinhala and English videos, extracts spoken content, identifies factual claims, retrieves relevant evidence, and prepares the information required for claim verification.

## Current Status

The project is currently at the **PP1 prototype stage**.

### Implemented

- Video upload through a Streamlit interface
- Video validation and processing
- Audio extraction from uploaded videos
- Frame extraction for visual analysis
- Sinhala and English speech-to-text processing
- Transcript generation
- Transcript summarization
- Candidate claim identification
- Named Entity Recognition
- Evidence-search query generation
- News evidence retrieval
- Evidence relevance ranking
- Upload pipeline integration
- Individual test scripts for the main services

### In Progress

- Fine-tuning a multilingual claim-classification model
- Improving Sinhala claim detection
- Claim-evidence verification
- Final verdict generation:
  - `SUPPORTED`
  - `REFUTED`
  - `INSUFFICIENT_EVIDENCE`
- Model evaluation using suitable datasets
- End-to-end performance evaluation

## Processing Pipeline

```text
Uploaded Video
      ↓
Video Validation
      ↓
Audio and Frame Extraction
      ↓
Sinhala/English Speech-to-Text
      ↓
Transcript Summarization
      ↓
Candidate Claim Detection
      ↓
Named Entity Recognition
      ↓
Evidence Query Generation
      ↓
News Evidence Retrieval
      ↓
Evidence Relevance Ranking
      ↓
Claim Verification
      ↓
Final Verdict and Explanation
```

## Project Structure

```text
VideoVerification/
├── app/
│   └── services/
│       ├── claim_analyzer.py
│       ├── evidence_ranker.py
│       ├── evidence_retriever.py
│       ├── ner_extractor.py
│       ├── query_generator.py
│       ├── sinhala_speech_to_text.py
│       ├── speech_to_text.py
│       ├── summarizer.py
│       ├── upload_pipeline.py
│       └── video_processor.py
├── streamlit_app.py
├── inspect_datasets.py
├── test_claim_analyzer.py
├── test_evidence_retriever.py
├── test_ner.py
├── test_pipeline.py
├── test_query_generator.py
├── test_sinhala_evidence.py
├── test_summarizer.py
├── test_summarizer_regression.py
├── test_upload_pipeline.py
├── .gitignore
└── README.md
```

## Main Technologies

- Python
- Streamlit
- OpenCV
- FFmpeg
- Faster-Whisper
- Multilingual transformer models
- Named Entity Recognition
- Sentence embeddings
- NewsAPI

## Dataset Plan

Different datasets support different stages of the component:

| Dataset | Purpose |
|---|---|
| ViClaim | Claim detection from video transcripts |
| Sinhala Misinformation Corpus | Sinhala misinformation research and evaluation |
| ISOT Fake News Dataset | English article-level fake and real news analysis |
| TRUE / 3MFact | Video claim and evidence verification |
| Custom Sinhala claim-evidence data | Sinhala claim-verification evaluation |

Datasets are stored locally and excluded from GitHub because of their size and licensing requirements.

## Environment Configuration

Create a `.env` file in the project root:

```env
NEWS_API_KEY=your_newsapi_key_here
```

Never commit the real `.env` file or API key. The repository’s `.gitignore` excludes this file.

## Running the Application

Activate the Python virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

Run the Streamlit application:

```powershell
streamlit run streamlit_app.py
```

The application should open in the web browser.

## Testing

The repository contains test scripts for individual services and the integrated pipeline. For example:

```powershell
python test_claim_analyzer.py
python test_evidence_retriever.py
python test_pipeline.py
python test_upload_pipeline.py
```

Some tests require configured API access, model downloads, or local media files.

## Security and Repository Rules

The following content is intentionally excluded from GitHub:

- `.env` files and API keys
- Python virtual environments
- Datasets
- Uploaded videos
- Extracted audio
- Extracted frames
- Generated transcripts
- Upload-processing results
- Python cache files

## Future Work

- Fine-tune XLM-RoBERTa using ViClaim
- Create and annotate a Sinhala claim dataset
- Develop multilingual claim-evidence verification
- Evaluate speech recognition using Sinhala audio
- Evaluate evidence relevance ranking
- Add confidence scores and explanations
- Improve the final Streamlit results interface
- Perform complete Sinhala and English end-to-end evaluation

## Branch

Development for this component is maintained in:

```text
Video-Content-Verification-and-Claim-Analysis
```

## Author

**Heshan Harischandra**

AI-Based Fake News Detection System  
Video Content Verification and Claim Analysis Component