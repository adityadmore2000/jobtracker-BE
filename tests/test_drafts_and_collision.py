"""Tests for the GET /drafts endpoint and structured create-collision metadata."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from app.company_resolution import get_or_create_company
from app.database import SessionLocal
from app.main import app
from app.models import JobApplication
from app.mutation_dispatcher import dispatch
from app.mutation_schemas import ApplicationChanges, MutationPayload, MutationTarget
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


# ── GET /drafts ───────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_drafts_endpoint_returns_only_drafts(client, db):
    d = _row(db, "Aiden AI", "AI Engineer", is_draft=True)
    _row(db, "Neilsoft", "AI Engineer", is_draft=False)  # saved → excluded
    _row(db, "Google", "ML Engineer", is_draft=False, archived=True)  # archived saved → excluded

    resp = await client.get("/drafts")
    assert resp.status_code == 200
    body = resp.json()
    assert [r["id"] for r in body] == [d.id]
    assert body[0]["is_draft"] is True
    assert body[0]["company"] == "Aiden AI"
    assert body[0]["role"] == "AI Engineer"
    # Scalar schema, no stale *_json public fields.
    assert "roles_json" not in body[0]
    assert "employment_types" in body[0] and "current_stages" in body[0]


@pytest.mark.anyio
async def test_drafts_endpoint_excludes_saved_and_archived(client, db):
    _row(db, "Neilsoft", "AI Engineer", is_draft=False)
    _row(db, "Google", "ML Engineer", is_draft=False, archived=True)
    resp = await client.get("/drafts")
    assert resp.status_code == 200
    assert resp.json() == []


# ── Collision metadata ────────────────────────────────────────────────────────

def _create_payload(company, role, status=""):
    return MutationPayload(
        operation="create_draft",
        target=MutationTarget(),
        changes=ApplicationChanges(company=company, role=role, status=status),
    )


def test_collision_with_existing_draft(db):
    existing = _row(db, "Aiden AI", "AI Engineer", is_draft=True)
    result = dispatch(_create_payload("Aiden AI", "AI Engineer"), db)
    assert result.collision is not None
    assert result.collision.kind == "draft"
    assert result.collision.draft_id == existing.id
    assert result.collision.application_id is None
    assert result.collision.company == "Aiden AI"
    assert result.collision.role == "AI Engineer"
    assert result.collision.archived is False


def test_collision_with_active_saved_application(db):
    existing = _row(db, "Aiden AI", "AI Engineer", is_draft=False, status="applied")
    result = dispatch(_create_payload("Aiden AI", "AI Engineer", status="applied"), db)
    assert result.collision is not None
    assert result.collision.kind == "active_application"
    assert result.collision.application_id == existing.id
    assert result.collision.draft_id is None
    assert result.collision.archived is False


def test_collision_with_archived_application(db):
    existing = _row(db, "Aiden AI", "AI Engineer", is_draft=False, archived=True, status="rejected")
    # Reapply will restore + mark applied, but the collision metadata must still
    # identify the archived row by id with archived=True.
    result = dispatch(_create_payload("Aiden AI", "AI Engineer"), db)
    assert result.collision is not None
    assert result.collision.kind == "archived_application"
    assert result.collision.application_id == existing.id
    assert result.collision.archived is True


@pytest.mark.anyio
async def test_collision_surfaced_in_public_transcript_response(client, db):
    existing = _row(db, "Aiden AI", "AI Engineer", is_draft=True)
    # Drive a create through the fast path (deterministic, no extractor needed).
    resp = await client.post(
        "/transcript/parse",
        json={"transcript": "add application for AI Engineer at Aiden AI", "context": {}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["collision"] is not None
    assert body["collision"]["kind"] == "draft"
    assert body["collision"]["draft_id"] == existing.id
