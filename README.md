# 🍷 Wine Label Recognition API

A production-ready **FastAPI** backend that accepts wine-bottle images (by file upload or URL) and returns AI-powered label recognition results via the **api4ai wine-recognition2** model on RapidAPI.

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Prerequisites](#prerequisites)
3. [Local Setup](#local-setup)
4. [Running the Server](#running-the-server)
5. [API Endpoints](#api-endpoints)
6. [curl Examples](#curl-examples)
7. [Expected Response](#expected-response)
8. [Running Tests](#running-tests)
9. [Configuration Reference](#configuration-reference)
10. [Architecture Notes](#architecture-notes)

---

## Project Structure

```
wine0clock_backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, CORS, global handlers, probes
│   ├── api.py               # APIRouter with /file and /url endpoints
│   ├── schemas.py           # Pydantic request/response models
│   ├── config.py            # Settings (pydantic-settings, .env support)
│   └── services/
│       ├── __init__.py
│       └── rapidapi_client.py  # httpx client, retry, response parsing
├── tests/
│   ├── __init__.py
│   └── test_api.py          # pytest suite (validation, mocks, errors)
├── .env.example             # Template for environment variables
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.11+ |
| pip | latest |
| RapidAPI account | [Sign up](https://rapidapi.com/) |
| wine-recognition2 subscription | [Subscribe](https://rapidapi.com/api4ai-api4ai-default/api/wine-recognition2) |

---

## Local Setup

```bash
# 1. Clone / enter the project
cd wine0clock_backend

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Open .env and fill in RAPIDAPI_KEY and RAPIDAPI_HOST
```

Your `.env` file (never commit this):
```dotenv
RAPIDAPI_KEY=abc123yourKeyHere
RAPIDAPI_HOST=wine-recognition2.p.rapidapi.com
FRONTEND_ORIGIN=http://localhost:3000
```

---

## Running the Server

```bash
# Development (auto-reload on code changes)
uvicorn app.main:app --reload --port 8000

# Production-style (multiple workers)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Interactive API docs are available at:
- Swagger UI → http://localhost:8000/docs
- ReDoc      → http://localhost:8000/redoc

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness probe (checks credentials) |
| `POST` | `/api/v1/recognize/file` | Upload image file (multipart/form-data) |
| `POST` | `/api/v1/recognize/url` | Recognize from a public image URL |

### Query Parameters (both recognition endpoints)

| Parameter | Type | Default | Max | Description |
|-----------|------|---------|-----|-------------|
| `top_k` | `int` | `5` | `10` | Number of candidates to return |
| `include_raw` | `bool` | `false` | — | Include full upstream JSON |

---

## curl Examples

### File Upload

```bash
curl -X POST "http://localhost:8000/api/v1/recognize/file?top_k=5" \
  -H "accept: application/json" \
  -F "file=@/path/to/wine_label.jpg"
```

#### With raw response included

```bash
curl -X POST "http://localhost:8000/api/v1/recognize/file?top_k=3&include_raw=true" \
  -H "accept: application/json" \
  -F "file=@/path/to/wine_label.png"
```

---

### URL Mode

```bash
curl -X POST "http://localhost:8000/api/v1/recognize/url?top_k=5" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Bottle_of_Chateau_Margaux.jpg/220px-Bottle_of_Chateau_Margaux.jpg"}'
```

---

### Health / Readiness Probes

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

---

## Expected Response

### Successful recognition (`200 OK`)

```json
{
  "request_id": "d4f2a1b3-7c8e-4f9d-a0b1-c2d3e4f5a6b7",
  "top_candidates": [
    {
      "label": "Château Margaux 2015",
      "confidence": 0.92
    },
    {
      "label": "Château Latour 2016",
      "confidence": 0.75
    },
    {
      "label": "Penfolds Grange 2018",
      "confidence": 0.61
    }
  ],
  "candidate_count": 3,
  "elapsed_ms": 843.17,
  "raw_response": null
}
```

### Validation error (`422 Unprocessable Entity`)

```json
{
  "detail": "Unsupported file extension '.gif'. Allowed: .jpeg, .jpg, .png, .webp"
}
```

### Upstream API error (`502 Bad Gateway`)

```json
{
  "detail": "Upstream recognition API error: Upstream returned 500"
}
```

### Service not configured (`503 Service Unavailable`)

```json
{
  "detail": "RapidAPI credentials are not configured on this server."
}
```

---

## Running Tests

```bash
pytest -v
```

The test suite covers:
- **File validation** – wrong extension, unsupported content-type, oversized file, empty file
- **Successful mock calls** – file upload, URL recognition, `top_k` slicing, `include_raw` flag
- **Error handling** – upstream timeout → 502, upstream 500 → 502, missing credentials → 503
- **Response parsing** – `_parse_candidates` unit tests (sorting, empty results, bad status)
- **Probes** – `/health` and `/ready` endpoints

---

## Configuration Reference

All settings can be provided as environment variables or in a `.env` file.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `RAPIDAPI_KEY` | ✅ | — | Your RapidAPI subscription key |
| `RAPIDAPI_HOST` | ✅ | — | `wine-recognition2.p.rapidapi.com` |
| `FRONTEND_ORIGIN` | ❌ | `*` | Comma-separated CORS allowed origins |
| `HTTP_TIMEOUT_SECONDS` | ❌ | `10.0` | Per-request timeout in seconds |
| `HTTP_MAX_RETRIES` | ❌ | `1` | Retries on 429/5xx responses |
| `MAX_FILE_SIZE_BYTES` | ❌ | `10485760` | 10 MB upload limit |
| `DEFAULT_TOP_K` | ❌ | `5` | Default candidates returned |
| `MAX_TOP_K` | ❌ | `10` | Hard cap on `top_k` |

---

## Architecture Notes

- **No image persistence** – images are held in memory for the duration of a single request and then garbage-collected. Nothing is written to disk.
- **Async throughout** – `httpx.AsyncClient` is used for non-blocking I/O; the FastAPI event loop is never blocked.
- **Retry logic** – transient failures (HTTP 429, 500, 502, 503, 504) trigger up to `HTTP_MAX_RETRIES` additional attempts.
- **Secret safety** – credentials are read from environment variables and never appear in logs or responses.
- **Structured logging** – every recognition request emits `request_id`, `file_size`, `elapsed_ms`, and upstream status without leaking secrets.
- **Unified response schema** – both `/file` and `/url` endpoints return identical JSON so the frontend needs only one response handler.
