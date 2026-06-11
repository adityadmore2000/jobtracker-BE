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
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2:3b
OLLAMA_TIMEOUT_SECONDS=20
OLLAMA_KEEP_ALIVE=10m
OLLAMA_MAX_TOOL_TURNS=2
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
docker start resume_tailor
docker start ollama

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

Phase 3 now uses a local Ollama-backed semantic interpreter. Transcript requests stay non-persistent: they prepare create or update previews, but they do not create, update, or persist tracker rows by themselves.

Architecture:

```text
transcript
    -> Ollama /api/chat with llama3.2:3b, two passes per interpret() call:
         pass 1: field extraction (authoritative explicitly-stated fields)
         pass 2: tool selection (one tool call from message.tool_calls)
    -> Pydantic tool-argument validation
    -> deterministic backend validation, merge, and resolution
    -> bounded retries: interpret() may run again on a missing-company
       clarification or a schema-repair, capped by OLLAMA_MAX_TOOL_TURNS
       (default 2 interpret() calls per transcript request); once the cap is
       reached the backend returns a clarification instead of retrying further
    -> preview / clarification / confirmation response
    -> database mutation only through the normal create/update endpoints
```

Each `interpret()` call therefore issues two Ollama `/api/chat` requests
(extraction then selection). `OLLAMA_MAX_TOOL_TURNS` caps how many times
`interpret()` runs for a single transcript, not the number of Ollama requests.

The backend remains authoritative for:

- allowed enum validation
- exact normalized company lookup
- canonical company alias lookup
- free-form role acceptance (roles are non-blank strings; there is no role enum)
- ambiguity detection
- partial unsaved-draft acceptance
- clarification templates for common missing-target cases
- new-company confirmation
- preview-before-save behavior
- database writes

The interpreter is never allowed to write to PostgreSQL directly.
Regex transcript parsing remains removed. LiveKit has not been added.

### Semantic Interpreter Health

```bash
curl http://127.0.0.1:8000/semantic-interpreter/health
```

Example response:

```json
{
  "status": "ok",
  "provider": "ollama",
  "model": "llama3.2:3b",
  "mode": "tool_calling"
}
```

### Parse Transcript

```bash
curl -X POST http://127.0.0.1:8000/transcript/parse \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "Add AI Engineer role for Neilsoft"
  }'
```

### Supported Transcript Examples

Create commands:

- `I have a requirement. I want to add an application neilsoft`
- `Add a Neilsoft application`
- `Neilsoft sathi application add kar`
- `Add AI Engineer role for Neilsoft`
- `Add Neilsoft for AI Engineer role`
- `Track an AI Engineer opening at Neilsoft`
- `I applied to Neilsoft as an AI Engineer`
- `Add Neilsoft for RAG`
- `Add Neilsoft for AI Engineer and RAG roles`
- `AI Engineer role` when an active unsaved draft already exists
- `fulltime ani onsite` when an active unsaved draft already exists
- `Applied stage thev` when an active unsaved draft already exists

Update commands:

- `Mark Neilsoft as rejected`
- `Neilsoft high priority kar`
- `Set the next action for Neilsoft to follow up with HR`
- `Add note saying recruiter la udya ping karaycha`
- `Make it high priority` only when the frontend provides an explicitly selected persisted `active_application.application_id`

Context policy:

- `active_draft` is only for unsaved-draft enrichment.
- `active_application` is only for an explicitly selected persisted tracker row.
- `recent_actions` are prompt context only and do not authorize persisted-row mutation.
- Saved-row updates require explicit company in the utterance or an explicitly selected persisted row id.
- `request_draft_save` may prepare a draft for the normal save flow, but it must not persist directly from the LLM tool call.
- Read/delete transcript tools have not been added yet.

Unsupported or incomplete commands:

- `Add application` without a company and without an active draft
- `Make it high priority` without an active draft and without an explicitly selected persisted row
- Unsupported priority values such as `Set Neilsoft priority to urgent`
- Unsupported status values such as `Mark Neilsoft as interviewing`
- Unknown-company updates
- Broad narration that does not clearly map to one of the supported intents

### Semantic Response Behavior

- Single create proposals return one editable preview draft.
- Multi-role create proposals return multiple drafts and set `needs_confirmation=true`.
- Existing-application updates return a resolved preview draft only when exactly one row matches.
- `request_draft_save` remains preview-only and must route through the normal explicit save path.
- Context-based follow-ups can use a bounded session context payload from the frontend.
- If Ollama is unavailable, times out, returns malformed JSON, or returns schema-invalid output, the backend returns a recoverable error and no tracker changes are saved.
- Regex-based transcript interpretation has been removed completely.
- LiveKit has not been added yet.

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

Deleting an application preserves ASR correction history. Related correction records retain their metadata and set `application_id` to `NULL`.

## CORS

The backend supports local frontend origins from `FRONTEND_ORIGIN`, plus `http://localhost:3000` and `http://127.0.0.1:3000` by default.

For local unpacked Chrome extension development, the backend allows origins matching `chrome-extension://...`. Chrome generates the extension ID when loaded unpacked, so the exact origin cannot be known ahead of time. This exception is limited to the Chrome extension scheme and does not allow arbitrary web origins.

## Scope

This API exposes the Phase 1 tracker CRUD endpoints, the Phase 2 browser-context capture endpoints, and the Phase 3 Ollama-backed semantic transcript endpoints. It does not include LiveKit, voice recording, speech-to-text inside this service, CSV import/export, reminders, analytics, timelines, event sourcing, scraping, or metadata inference beyond the validated transcript proposal flow.

## Immediate Adaptation Loop

The current local ASR adaptation loop is:

```text
transcript input
    -> /transcript/parse
    -> structured create or update preview in the frontend
    -> create preview: /applications/create-candidate
    -> confirmation popup only when the company is genuinely new
    -> create preview: /applications/confirm-company
    -> update preview: PATCH /applications/{id}
    -> application row + canonical company + optional alias + correction event
    -> /asr/hotwords for later transcription requests
```

- Existing-company creates and edits keep the low-friction path.
- New-company creates require backend-confirmed manual company-name confirmation before persistence.
- Existing-company transcript updates reuse exact normalized matching against canonical names, aliases, and existing tracker companies.
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
- Transcript updates require exactly one resolved application for the requested company. If multiple rows share the same company, the backend refuses to guess and asks for a manual edit instead.
