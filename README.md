# Job Tracker API

Standalone FastAPI backend for the Job Tracker manual job application tracker.

This backend is now PostgreSQL-only. SQLite is no longer supported, and old local `.db` files are not migrated automatically.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment

Create `jobtracker-BE/.env` from `.env.example` and fill in your local PostgreSQL credentials:

```text
DATABASE_URL=postgresql+psycopg://<user>:<password>@localhost:5432/job_tracker
TEST_DATABASE_URL=postgresql+psycopg://<user>:<password>@localhost:5432/job_tracker_test
FRONTEND_ORIGIN=http://localhost:3000,http://127.0.0.1:3000
AUTO_MIGRATE=false
```

The backend automatically loads `jobtracker-BE/.env` relative to the backend root.

Variable precedence is:

```text
OS environment variable
    >
jobtracker-BE/.env
    >
startup error
```

Manual `source .env` and manual `DATABASE_URL` exports are not required.

`DATABASE_URL` is required at startup. The backend fails clearly when it is missing or does not point to PostgreSQL.

## Local PostgreSQL Bootstrap

Reuse the existing local PostgreSQL server. For the current local setup, that is the existing `resume_tailor` Docker container with PostgreSQL published on local port `5432`.

Create the separate `job_tracker` databases without touching the existing Resume Tailor database:

```bash
cd jobtracker-BE
source .venv/bin/activate
python scripts/bootstrap_postgres.py
```

This script creates:

- `job_tracker`
- `job_tracker_test`

if they do not already exist.

## Alembic Migration

Schema management uses Alembic. You can either enable automatic migrations at startup or run them manually.

### Automatic Migration On Startup

Set this in `jobtracker-BE/.env`:

```text
AUTO_MIGRATE=true
```

Then use the normal local startup flow:

```bash
docker start resume_tailor

cd /home/aditya/dev-work/job_tracker_assistant/jobtracker-BE
source .venv/bin/activate

python scripts/bootstrap_postgres.py
uvicorn app.main:app --reload
```

When `AUTO_MIGRATE=true`, the backend runs `alembic upgrade head` once during startup before serving requests. If the migration fails, startup fails clearly.

### Manual Migration Alternative

Keep this in `jobtracker-BE/.env`:

```text
AUTO_MIGRATE=false
```

Then run:

```bash
cd jobtracker-BE
source .venv/bin/activate
alembic upgrade head
```

## Run

```bash
cd /home/aditya/dev-work/job_tracker_assistant/jobtracker-BE
source .venv/bin/activate
uvicorn app.main:app --reload
```

## Test

```bash
cd /home/aditya/dev-work/job_tracker_assistant/jobtracker-BE
source .venv/bin/activate
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

This API exposes the Phase 1 tracker CRUD endpoints, the Phase 2 browser-context capture endpoints, and the Phase 3 deterministic transcript parsing endpoints. It does not include AI, voice recording, speech-to-text, CSV import/export, reminders, analytics, timelines, event sourcing, scraping, or metadata inference.

## Immediate Adaptation Loop

The current local ASR adaptation loop is:

```text
transcript input
    -> /transcript/parse
    -> structured draft in the frontend
    -> /applications/create-candidate
    -> confirmation popup only when the company is genuinely new
    -> /applications/confirm-company
    -> application row + canonical company + optional alias + correction event
    -> /asr/hotwords for later transcription requests
```

- Existing-company creates and edits keep the low-friction path.
- New-company creates require backend-confirmed manual company-name confirmation before persistence.
- Alias creation is exact-and-normalized, not fuzzy.
- Periodic fine-tuning is not automatically triggered in this phase.

## PostgreSQL Schema

Alembic manages the current required tables:

- `job_applications`
- `browser_context`
- `canonical_companies`
- `company_aliases`
- `asr_company_correction_events`

## Current Limitations

- SQLite is no longer supported by the backend runtime or tests.
- Existing SQLite files can remain on disk, but they are not read, written, or migrated automatically.
- `audio_reference` is nullable metadata only; this backend does not retain audio clips.
- Training/export review is manual. Correction capture supports later curation, not automatic fine-tuning.
