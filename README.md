# ApplicationOps API

Standalone FastAPI backend for the ApplicationOps manual job application tracker.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment

Copy `.env.example` values into your shell or local environment as needed:

```text
DATABASE_URL=sqlite:///./job_tracker.db
FRONTEND_ORIGIN=http://localhost:3000,http://127.0.0.1:3000
```

## Run

```bash
uvicorn app.main:app --reload
```

## Test

```bash
pytest
```

## Browser Context Endpoints

The Phase 2 browser-context endpoints store only the current tab URL and title captured by the separate Chrome extension. They do not fetch URLs, scrape page content, infer application fields, or modify job application rows.

### Capture Current Page

```bash
curl -X POST http://127.0.0.1:8000/browser-context \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/job",
    "page_title": "AI Engineer - Example Company"
  }'
```

Response:

```json
{
  "context": {
    "id": 1,
    "url": "https://example.com/job",
    "page_title": "AI Engineer - Example Company",
    "captured_at": "2026-06-04T12:00:00"
  }
}
```

### Latest Captured Page

```bash
curl http://127.0.0.1:8000/browser-context/latest
```

Response when a context exists:

```json
{
  "context": {
    "id": 1,
    "url": "https://example.com/job",
    "page_title": "AI Engineer - Example Company",
    "captured_at": "2026-06-04T12:00:00"
  }
}
```

Response when none exists:

```json
{
  "context": null
}
```

## CORS

The backend supports local frontend origins from `FRONTEND_ORIGIN`, plus `http://localhost:3000` and `http://127.0.0.1:3000` by default.

For local unpacked Chrome extension development, the backend allows origins matching `chrome-extension://...`. Chrome generates the extension ID when loaded unpacked, so the exact origin cannot be known ahead of time. This exception is limited to the Chrome extension scheme and does not allow arbitrary web origins.

## Scope

This API exposes the Phase 1 tracker CRUD endpoints and the Phase 2 browser-context capture endpoints. It does not include AI, voice, CSV import/export, reminders, analytics, Docker, PostgreSQL, timelines, event sourcing, scraping, or metadata inference.
