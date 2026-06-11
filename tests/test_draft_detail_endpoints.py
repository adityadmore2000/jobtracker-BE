"""Tests for route-addressable draft detail endpoints.

Covers GET /drafts/{id} and DELETE /drafts/{id}, added so the UI can directly
address and discard a persisted draft by id. DELETE reuses the discard_draft
dispatcher operation, which cascade-deletes draft-linked notes.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from app.company_resolution import get_or_create_company
from app.database import SessionLocal
from app.main import app
from app.models import ApplicationNote, JobApplication
from app.role_resolution import normalize_role_name


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def db():
    with SessionLocal() as session:
        yield session


def _row(db, company, role, *, is_draft=False, archived=False, status=""):
    company_obj = get_or_create_company(db, company)
    row = JobApplication(
        company_id=company_obj.id,
        role=role,
        normalized_role=normalize_role_name(role),
        employment_types_json=[],
        job_link="",
        location="",
        status=status,
        current_stages_json=[],
        priority="",
        engaged_days=0,
        next_action="",
        comments="",
        is_draft=is_draft,
        draft_created_at=datetime.now(timezone.utc) if is_draft else None,
        archived_at=datetime.now(timezone.utc) if archived else None,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


# ── GET /drafts/{id} ────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_get_draft_returns_persisted_draft(client, db):
    d = _row(db, "Aiden AI", "AI Engineer", is_draft=True)
    resp = await client.get(f"/drafts/{d.id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == d.id
    assert body["is_draft"] is True
    assert body["company"] == "Aiden AI"
    assert body["role"] == "AI Engineer"
    # Scalar public schema.
    assert "employment_types" in body and "current_stages" in body


@pytest.mark.anyio
async def test_get_draft_missing_returns_404(client, db):
    resp = await client.get("/drafts/999999")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_get_draft_rejects_saved_application(client, db):
    saved = _row(db, "Neilsoft", "ML Engineer", is_draft=False, status="applied")
    resp = await client.get(f"/drafts/{saved.id}")
    assert resp.status_code == 404
    assert "not a draft" in resp.json()["detail"].lower()


@pytest.mark.anyio
async def test_get_draft_rejects_archived_application(client, db):
    archived = _row(db, "Google", "ML Engineer", is_draft=False, archived=True)
    resp = await client.get(f"/drafts/{archived.id}")
    assert resp.status_code == 404


# ── DELETE /drafts/{id} ─────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_delete_draft_removes_persisted_draft(client, db):
    d = _row(db, "Aiden AI", "AI Engineer", is_draft=True)
    draft_id = d.id

    resp = await client.delete(f"/drafts/{draft_id}")
    assert resp.status_code == 204

    db.expire_all()
    assert db.get(JobApplication, draft_id) is None


@pytest.mark.anyio
async def test_delete_draft_cascades_linked_notes(client, db):
    d = _row(db, "Aiden AI", "AI Engineer", is_draft=True)
    draft_id = d.id
    note = ApplicationNote(
        application_id=draft_id,
        text="follow up next week",
        created_at=datetime.now(timezone.utc),
    )
    db.add(note)
    db.commit()
    note_id = note.id

    # Sanity: the note exists before delete.
    assert db.get(ApplicationNote, note_id) is not None

    resp = await client.delete(f"/drafts/{draft_id}")
    assert resp.status_code == 204

    db.expire_all()
    assert db.get(JobApplication, draft_id) is None
    assert db.get(ApplicationNote, note_id) is None


@pytest.mark.anyio
async def test_delete_draft_rejects_saved_application(client, db):
    saved = _row(db, "Neilsoft", "ML Engineer", is_draft=False, status="applied")
    saved_id = saved.id

    resp = await client.delete(f"/drafts/{saved_id}")
    assert resp.status_code == 404
    assert "not a draft" in resp.json()["detail"].lower()

    # The saved row must survive a rejected draft-delete.
    db.expire_all()
    assert db.get(JobApplication, saved_id) is not None


@pytest.mark.anyio
async def test_delete_draft_missing_returns_404(client, db):
    resp = await client.delete("/drafts/999999")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_deleted_draft_no_longer_listed(client, db):
    d = _row(db, "Aiden AI", "AI Engineer", is_draft=True)
    draft_id = d.id

    listed = await client.get("/drafts")
    assert draft_id in [r["id"] for r in listed.json()]

    resp = await client.delete(f"/drafts/{draft_id}")
    assert resp.status_code == 204

    listed_after = await client.get("/drafts")
    assert draft_id not in [r["id"] for r in listed_after.json()]
