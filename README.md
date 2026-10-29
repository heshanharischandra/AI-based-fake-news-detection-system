# Cross-Platform Source Credibility Verification System

This branch contains my contribution to the AI-based fake-news detection
project: a source-credibility and cross-platform verification system for
publisher websites and news articles.

The system assesses credibility signals without treating those signals as
proof that an individual article is true or false. It provides explainable
results in English and Sinhala and keeps article truth detection as a separate
component.

## What this branch provides

- Publisher/source profile assessment from a website URL
- Source identity, metadata, contact information, and publisher signals
- Cross-platform discovery for linked Facebook pages and YouTube channels
- A distinction between an associated account and independently verified
  account ownership
- Related-coverage discovery for an article
- Assessment history stored in MongoDB when database access is available
- FastAPI endpoints for backend integration
- Streamlit `SourceLens` interface for interactive assessment
- Identity-matching pilot data, model, validation scripts, and result reports
- English (`en`) and Sinhala (`si`) response support

## Project structure

```text
.
├── main.py                         # FastAPI application
├── streamlit_app.py                # SourceLens Streamlit interface
├── unified_api.py                  # Additional source assessment API logic
├── database.py                     # MongoDB connection and persistence
├── services/
│   ├── analysis.py                 # Website and article analysis
│   ├── source_assessment.py        # Source profile and credibility output
│   ├── cross_platform_verifier.py # Cross-platform association checks
│   ├── discovery.py                # Related coverage discovery
│   ├── identity_verification.py    # Publisher identity verification
│   └── ...                         # Fetching, scoring, and platform services
├── data/                           # Publisher and identity datasets
├── models/                         # Trained pilot model artifacts
├── results/                        # Validation and evaluation outputs
└── test_*.py                       # Focused tests
```

## Requirements

- Python 3.11 or newer
- MongoDB connection for persistent assessment history
- Internet access for fetching public publisher and article pages

Install the Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

Create a local `.env` file in the project root. Do not commit this file.

```env
MONGODB_URI=mongodb+srv://<username>:<password>@<cluster>/<database>
MONGODB_DB=source_credibility_db
SOURCE_API_URL=http://127.0.0.1:8002
```

`MONGODB_URI` is required when the application needs to read or save
assessment history. `MONGODB_DB` defaults to `source_credibility_db`.
`SOURCE_API_URL` is used by the Streamlit interface and defaults to
`http://127.0.0.1:8002`.

## Run the application

Start the FastAPI backend:

```powershell
uvicorn main:app --host 127.0.0.1 --port 8002 --reload
```

In a second terminal, start the Streamlit interface:

```powershell
streamlit run streamlit_app.py
```

Open the Streamlit URL shown in the terminal, usually
`http://localhost:8501`.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Service metadata and supported languages |
| `GET` | `/health` | API and database health status |
| `POST` | `/assess` | Build a credibility assessment for a publisher URL |
| `POST` | `/coverage` | Discover related coverage for an article URL |
| `GET` | `/history/{domain}` | Retrieve previous assessments for a domain |

Example source assessment:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8002/assess `
  -ContentType "application/json" `
  -Body '{"url":"https://www.newsfirst.lk/","output_language":"en"}'
```

The interactive API documentation is available at
`http://127.0.0.1:8002/docs` while the backend is running.

## Verification model boundaries

The result should be interpreted as a credibility-oriented source assessment,
not as an automatic verdict that an article is real or fake.

- A link from a publisher website to a social account establishes an observed
  association only.
- It does not independently verify account ownership.
- It does not prove that the linked account published a particular story.
- Network failures, blocked pages, missing metadata, and unavailable MongoDB
  history are reported as unavailable signals rather than silently treated as
  negative evidence.

## Testing and evaluation

Run the focused tests from the project root:

```powershell
python -m pytest test_source_assessment.py test_identity_model.py
```

The identity pilot utilities can be run separately when updating the datasets
or model:

```powershell
python check_ml_readiness.py
python validate_identity_pilot.py
```

Generated evaluation files are stored under `results/`. Dataset and model
updates should be reviewed before replacing the current artifacts.

## Branch scope

This README documents the work on the
`cross-platform-source-credibility-verification-system` branch. Changes here
are intended to be integrated into the wider fake-news detection system through
the project team's normal review and merge process.
