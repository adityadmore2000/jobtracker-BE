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
FRONTEND_ORIGIN=http://localhost:3000
```

## Run

```bash
uvicorn app.main:app --reload
```

## Test

```bash
pytest
```

## Scope

This API exposes only the Phase 1 tracker CRUD endpoints for the single `job_applications` table. It does not include AI, voice, CSV import/export, browser extension code, reminders, analytics, Docker, PostgreSQL, timelines, or event sourcing.
