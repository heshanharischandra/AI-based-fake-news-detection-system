# SourceLens

## Cross-Platform Source Credibility Verification System

SourceLens is a research-oriented system for evaluating the credibility signals
of online news publishers. It analyzes a publisher's website, extracts
transparency and identity indicators, discovers related social profiles and
coverage, and returns an evidence-based assessment in English or Sinhala.

> **Scope:** SourceLens evaluates source credibility signals. It does not
> determine whether an article is factually true or false, and an observed
> social-media association does not independently prove account ownership.

## Highlights

- Website security, identity, ownership, contact, and editorial-policy checks
- Publisher metadata and domain analysis
- Discovery of linked Facebook pages and YouTube channels
- Cross-platform publisher identity matching
- Related coverage discovery for an article or publisher
- Historical assessment storage and retrieval with MongoDB
- English (`en`) and Sinhala (`si`) output support
- FastAPI service with an interactive Streamlit interface
- Identity-matching pilot data, model artifacts, and evaluation results

## Architecture

```text
Streamlit UI
     |
     v
FastAPI API
     |
     +--> Website and article analysis
     +--> Source credibility assessment
     +--> Social-profile discovery and scoring
     +--> Related coverage discovery
     |
     +--> MongoDB assessment history (optional)
```

### Repository layout

| Path | Purpose |
| --- | --- |
| `main.py` | Primary FastAPI application |
| `streamlit_app.py` | Streamlit user interface |
| `services/` | Analysis, discovery, scoring, and verification services |
| `data/` | Publisher, identity, and evaluation datasets |
| `models/` | Trained model artifacts |
| `results/` | Evaluation outputs and reports |
| `test_*.py` | Automated tests |
| `requirements.txt` | Python dependency constraints |

## Requirements

- Python 3.11 or newer
- Internet access for public website and coverage analysis
- MongoDB connection string for assessment history
- Optional third-party credentials for integrations that require them

## Installation

```powershell
git clone https://github.com/heshanharischandra/AI-based-fake-news-detection-system.git
cd AI-based-fake-news-detection-system

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a local `.env` file when database-backed history or optional
integrations are required. Never commit this file.

```dotenv
MONGODB_URI=mongodb+srv://<username>:<password>@<cluster>/<database>
MONGODB_DB=source_credibility_db
SOURCE_API_URL=http://127.0.0.1:8002
```

`MONGODB_DB` defaults to `source_credibility_db`, and the Streamlit interface
uses `http://127.0.0.1:8002` when `SOURCE_API_URL` is not set.

## Running locally

Start the API in one terminal:

```powershell
uvicorn main:app --host 127.0.0.1 --port 8002 --reload
```

Start the interface in a second terminal:

```powershell
streamlit run streamlit_app.py
```

The API is available at `http://127.0.0.1:8002`. Interactive API
documentation is available at `/docs`.

## API quick start

Check service status:

```powershell
curl http://127.0.0.1:8002/health
```

Assess a publisher source:

```powershell
curl -X POST http://127.0.0.1:8002/assess `
  -H "Content-Type: application/json" `
  -d '{\"url\":\"https://example.com/news/article\",\"output_language\":\"en\"}'
```

The main API endpoints are:

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Service metadata and supported languages |
| `GET` | `/health` | API and database health status |
| `POST` | `/assess` | Analyze a publisher source |
| `POST` | `/coverage` | Discover related news coverage |
| `GET` | `/history/{domain}` | Retrieve recent assessments for a domain |

OpenAPI documentation is generated automatically by FastAPI at
`http://127.0.0.1:8002/docs`.

## Testing

Run the focused assessment and identity test suites:

```powershell
python -m pytest test_source_assessment.py test_identity_model.py
```

Run all repository tests:

```powershell
python -m pytest
```

## Responsible use and limitations

- Results are indicators intended to support human review, not final
  fact-checking decisions.
- Website content can change, become unavailable, or block automated requests.
- Public profile links and matching signals should be independently verified.
- Do not use the system as the sole basis for reputational, legal, employment,
  safety, or other high-impact decisions.
- Respect the terms of service, robots directives, rate limits, and applicable
  privacy laws for every data source.

## Development notes

The project is currently a research prototype. Changes to scoring logic,
identity matching, datasets, or model artifacts should be accompanied by
updated tests and evaluation results where applicable.

## License

No license file is currently included in this repository. Add an appropriate
license before distributing the project or accepting external contributions.
