# Cross-Platform Source Credibility Verification System

This branch contains my contribution to the AI-based fake-news detection
project. It focuses on checking the credibility and identity signals of news
publishers across websites and social platforms.

## Main features

- Publisher website and source-profile assessment
- Publisher identity and metadata analysis
- Discovery of linked Facebook pages and YouTube channels
- Cross-platform source association checks
- Related news-coverage discovery
- English and Sinhala response support
- FastAPI backend and Streamlit user interface
- Identity-matching pilot data, model, and evaluation results

## Project structure

```text
main.py                    FastAPI backend
streamlit_app.py           Streamlit interface
services/                  Analysis, scoring, and verification services
data/                      Publisher and identity datasets
models/                    Model artifacts
results/                   Evaluation results
test_*.py                  Project tests
```

## Requirements

- Python 3.11 or newer
- Internet access for public website analysis
- A configured database when assessment history is needed

Install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the project

Start the backend:

```powershell
uvicorn main:app --host 127.0.0.1 --port 8002
```

In another terminal, start the interface:

```powershell
streamlit run streamlit_app.py
```

The application will display the local web address in the terminal.

## Important note

This system provides source-credibility signals and does not automatically
decide whether an article is true or false. A social-media link found on a
publisher website shows an observed association; it does not independently
prove account ownership or article publication.

## Testing

```powershell
python -m pytest test_source_assessment.py test_identity_model.py
```

This README describes my work on the
`cross-platform-source-credibility-verification-system` branch.
