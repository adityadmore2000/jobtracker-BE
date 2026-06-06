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

## Transcript Parsing Endpoints

Phase 3 adds deterministic English transcript parsing. These endpoints return draft patches only. They do not create, update, or persist tracker rows.

### Parse Transcript

```bash
curl -X POST http://127.0.0.1:8000/transcript/parse \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "Add a Bootcoding AI Engineer internship. Use the current link. Set priority to medium."
  }'
```

### Parse Correction

```bash
curl -X POST http://127.0.0.1:8000/transcript/parse-correction \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "Remove Agentic AI Engineer tag. Add Networked stage. Append comment saying one request is pending."
  }'
```

The parser is rule-based and local. It does not call Ollama, external LLM APIs, speech-to-text, or browser scraping. It only fills fields that are explicitly present in the transcript. `Current Stage`, `NEXT ACTION`, `COMMENTS`, and `ENGAGED (# OF DAYS)` are never inferred.

## Company Confirmation Flow

New application creation now uses a small two-step backend flow so the backend stays authoritative about whether a company is already known.

### Create Application Candidate

```bash
curl -X POST http://127.0.0.1:8000/applications/create-candidate \
  -H "Content-Type: application/json" \
  -d '{
    "company": "Crew Trim Labs",
    "roles_json": ["AI Engineer"],
    "employment_types_json": ["Full Time"],
    "job_link": "",
    "location": "",
    "status": "",
    "current_stages_json": [],
    "priority": "",
    "engaged_days": 0,
    "next_action": "",
    "comments": "",
    "raw_transcript": "Add Crew Trim Labs for an AI Engineer role.",
    "original_extracted_company_name": "Crew Trim Labs"
  }'
```

Response for a genuinely new company:

```json
{
  "status": "confirmation_required",
  "requires_confirmation": true,
  "candidate": {
    "company": "Crew Trim Labs",
    "roles_json": ["AI Engineer"],
    "employment_types_json": ["Full Time"],
    "job_link": "",
    "location": "",
    "status": "",
    "current_stages_json": [],
    "priority": "",
    "engaged_days": 0,
    "next_action": "",
    "comments": "",
    "raw_transcript": "Add Crew Trim Labs for an AI Engineer role.",
    "original_extracted_company_name": "Crew Trim Labs",
    "audio_reference": null
  }
}
```

Response for an existing company or alias match:

```json
{
  "status": "created",
  "requires_confirmation": false,
  "application": {
    "id": 1,
    "company": "Analytics Vidhya",
    "...": "existing application fields"
  }
}
```

### Confirm Company Name And Create

```bash
curl -X POST http://127.0.0.1:8000/applications/confirm-company \
  -H "Content-Type: application/json" \
  -d '{
    "company": "Crew Trim Labs",
    "confirmed_company_name": "Krutrim Labs",
    "roles_json": ["AI Engineer"],
    "employment_types_json": ["Full Time"],
    "job_link": "",
    "location": "",
    "status": "",
    "current_stages_json": [],
    "priority": "",
    "engaged_days": 0,
    "next_action": "",
    "comments": "",
    "raw_transcript": "Add Crew Trim Labs for an AI Engineer role.",
    "original_extracted_company_name": "Crew Trim Labs",
    "audio_reference": null
  }'
```

This creates the application using the confirmed company name, stores a canonical company record, stores an alias when the original extracted value differs meaningfully, and stores an ASR correction event when transcript metadata is present.

### ASR Hotwords

```bash
curl http://127.0.0.1:8000/asr/hotwords
```

Response:

```json
{
  "hotwords": ["Krutrim Labs", "Crew Trim Labs", "Analytics Vidhya", "AI Engineer"],
  "limit": 100
}
```

The hotword list is deterministic, deduplicated after normalization, and bounded to 100 items. It includes canonical companies first, then aliases, then existing tracker company values, then static job-tracker vocabulary.

## CORS

The backend supports local frontend origins from `FRONTEND_ORIGIN`, plus `http://localhost:3000` and `http://127.0.0.1:3000` by default.

For local unpacked Chrome extension development, the backend allows origins matching `chrome-extension://...`. Chrome generates the extension ID when loaded unpacked, so the exact origin cannot be known ahead of time. This exception is limited to the Chrome extension scheme and does not allow arbitrary web origins.

## Scope

This API exposes the Phase 1 tracker CRUD endpoints, the Phase 2 browser-context capture endpoints, and the Phase 3 deterministic transcript parsing endpoints. It does not include AI, voice recording, speech-to-text, CSV import/export, reminders, analytics, Docker, PostgreSQL, timelines, event sourcing, scraping, or metadata inference.
