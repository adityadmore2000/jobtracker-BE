"""
Tests for the single-call semantic extractor architecture.

Covers spec sections A–K:
  A. Deterministic fast-path regression (no extractor call)
  B. Natural create statements
  C. Natural multi-field create
  D. Natural multi-field draft update
  E. Natural multi-field saved-row update
  F. Explicit natural saved-row target
  G. Note safety
  H. Mixed-intent rejection
  I. Clarification continuation
  J. Suggestion-only UX
  K. Failure safety (no mutation, legacy LLM never called)

The Ollama call is always stubbed — tests never hit a live model. The single
extractor is feature-flagged ON per-test; conftest defaults it OFF.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

import app.main as main_module
from app.company_resolution import get_or_create_company
from app.database import SessionLocal
from app.main import app
from app.models import ApplicationChangeDraft, ApplicationNote, JobApplication
from app.role_resolution import normalize_role_name
from app.semantic_command_extractor import (
    SemanticExtractorInvalidResponseError,
    SemanticExtractorUnavailableError,
)
from app.semantic_command_schemas import SemanticChanges, SemanticCommand, SemanticTarget
from app.semantic_interpreter import get_semantic_interpreter


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

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


@pytest.fixture(autouse=True)
def enable_extractor(monkeypatch):
    """Re-enable the single extractor for this module (conftest disables it)."""
    monkeypatch.setenv("USE_SINGLE_SEMANTIC_EXTRACTOR", "1")
    yield


class _RaisingLegacyInterpreter:
    """Asserts the legacy dual-output pipeline is NEVER invoked."""

    def interpret(self, transcript, context=None):
        raise AssertionError("legacy interpreter must not be called")

    def extract_fields(self, transcript, context=None):
        raise AssertionError("legacy extract_fields must not be called")

    def select_tool(self, transcript, context=None):
        raise AssertionError("legacy select_tool must not be called")

    def health_check(self):
        return {}


@pytest.fixture(autouse=True)
def block_legacy_interpreter():
    app.dependency_overrides[get_semantic_interpreter] = lambda: _RaisingLegacyInterpreter()
    yield
    app.dependency_overrides.pop(get_semantic_interpreter, None)


def stub_extractor(monkeypatch, command: SemanticCommand):
    calls = {"count": 0, "transcripts": []}

    def fake(transcript, read_only_context=None, **kwargs):
        calls["count"] += 1
        calls["transcripts"].append(transcript)
        from app.semantic_command_extractor import ExtractionMetrics
        return command, ExtractionMetrics(latency_ms=1)

    monkeypatch.setattr(main_module, "extract_semantic_command_once", fake)
    return calls


def stub_extractor_raises(monkeypatch, exc: Exception):
    calls = {"count": 0}

    def fake(transcript, read_only_context=None, **kwargs):
        calls["count"] += 1
        raise exc

    monkeypatch.setattr(main_module, "extract_semantic_command_once", fake)
    return calls


def forbid_extractor(monkeypatch):
    """Assert the extractor is never reached (deterministic fast path expected)."""
    calls = {"count": 0}

    def fake(transcript, read_only_context=None, **kwargs):
        calls["count"] += 1
        raise AssertionError("extractor must NOT be called for deterministic fast paths")

    monkeypatch.setattr(main_module, "extract_semantic_command_once", fake)
    return calls


def seed_saved_application(db, company: str, role: str, **fields) -> JobApplication:
    company_obj = get_or_create_company(db, company)
    app_row = JobApplication(
        company_id=company_obj.id,
        role=role,
        normalized_role=normalize_role_name(role),
        employment_types_json=fields.get("employment_types_json", []),
        job_link=fields.get("job_link", ""),
        location=fields.get("location", ""),
        status=fields.get("status", ""),
        current_stages_json=fields.get("current_stages_json", []),
        priority=fields.get("priority", ""),
        engaged_days=0,
        next_action=fields.get("next_action", ""),
        comments=fields.get("comments", ""),
        is_draft=False,
    )
    db.add(app_row)
    db.commit()
    db.refresh(app_row)
    return app_row


def seed_draft(db, company: str, role: str) -> JobApplication:
    company_obj = get_or_create_company(db, company)
    app_row = JobApplication(
        company_id=company_obj.id,
        role=role,
        normalized_role=normalize_role_name(role),
        employment_types_json=[],
        job_link="",
        location="",
        status="",
        current_stages_json=[],
        priority="",
        engaged_days=0,
        next_action="",
        comments="",
        is_draft=True,
        draft_created_at=datetime.now(timezone.utc),
    )
    db.add(app_row)
    db.commit()
    db.refresh(app_row)
    return app_row


# ─────────────────────────────────────────────────────────────────────────────
# A. Deterministic fast-path regression — extractor must NOT be called
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
@pytest.mark.parametrize(
    "transcript,context",
    [
        ("save it", {"draft_id": "1"}),
        ("discard draft", {"draft_id": "1"}),
        ("set priority as medium", {"draft_id": "1"}),
        ("set priority to medium", {"draft_id": "1"}),
        ("update location to on-site", {"draft_id": "1"}),
        ("add application for AI Engineer at Neilsoft", {}),
        ("add a note saying recruiter replied", {"draft_id": "1"}),
    ],
)
async def test_fast_path_does_not_call_extractor(client, db, monkeypatch, transcript, context):
    calls = forbid_extractor(monkeypatch)
    # Seed a draft when the context references draft_id=1.
    if context.get("draft_id") == "1":
        seed_draft(db, "Neilsoft", "AI Engineer")
    resp = await client.post("/transcript/parse", json={"transcript": transcript, "context": context})
    assert resp.status_code == 200
    assert calls["count"] == 0


@pytest.mark.anyio
async def test_apply_and_discard_changes_fast_path(client, db, monkeypatch):
    calls = forbid_extractor(monkeypatch)
    saved = seed_saved_application(db, "Neilsoft", "AI Engineer")
    cd = ApplicationChangeDraft(kind="update", target_application_id=saved.id, changes_json={"priority": "HIGH"})
    db.add(cd)
    db.commit()
    db.refresh(cd)
    resp = await client.post(
        "/transcript/parse",
        json={"transcript": "apply changes", "context": {"change_draft_id": cd.id}},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "changes_applied"
    assert calls["count"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# B. Natural create statements
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_natural_create_applied(client, db, monkeypatch):
    cmd = SemanticCommand(
        intent="create_application",
        target=SemanticTarget(company="Aiden AI", role="AI Engineer"),
        changes=SemanticChanges(status="applied"),
    )
    calls = stub_extractor(monkeypatch, cmd)
    resp = await client.post(
        "/transcript/parse",
        json={"transcript": "I applied for AI Engineer role at Aiden AI", "context": {}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert calls["count"] == 1
    assert body["status"] == "draft_created"
    assert body["draft"]["company"] == "Aiden AI"
    assert body["draft"]["role"] == "AI Engineer"
    assert body["draft"]["status"] == "applied"
    assert body["draft"]["is_draft"] is True


# ─────────────────────────────────────────────────────────────────────────────
# C. Natural multi-field create
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_natural_multi_field_create(client, db, monkeypatch):
    cmd = SemanticCommand(
        intent="create_application",
        target=SemanticTarget(company="Aiden AI", role="AI Engineer"),
        changes=SemanticChanges(
            status="applied",
            location_mode="on-site",
            employment_types=["Full Time"],
            priority="HIGH",
            current_stages=["Tailored", "Applied"],
        ),
    )
    stub_extractor(monkeypatch, cmd)
    resp = await client.post("/transcript/parse", json={"transcript": "I applied ...", "context": {}})
    body = resp.json()
    assert body["status"] == "draft_created"
    d = body["draft"]
    assert d["status"] == "applied"
    assert d["location"] == "on-site"
    assert d["employment_types"] == ["Full Time"]
    assert d["priority"] == "HIGH"
    assert d["current_stages"] == ["Tailored", "Applied"]


# ─────────────────────────────────────────────────────────────────────────────
# D. Natural multi-field draft update
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_natural_multi_field_draft_update(client, db, monkeypatch):
    draft = seed_draft(db, "Neilsoft", "AI Engineer")
    cmd = SemanticCommand(
        intent="update_application",
        target=SemanticTarget(),
        changes=SemanticChanges(location_mode="on-site", employment_types=["Full Time"], priority="HIGH"),
    )
    stub_extractor(monkeypatch, cmd)
    resp = await client.post(
        "/transcript/parse",
        json={"transcript": "make it onsite, full-time, and priority high", "context": {"draft_id": str(draft.id)}},
    )
    body = resp.json()
    assert body["status"] == "draft_updated"
    d = body["draft"]
    assert d["location"] == "on-site"
    assert d["employment_types"] == ["Full Time"]
    assert d["priority"] == "HIGH"


# ─────────────────────────────────────────────────────────────────────────────
# E. Natural multi-field saved-row update (context selected)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_natural_multi_field_saved_update(client, db, monkeypatch):
    saved = seed_saved_application(db, "Neilsoft", "AI Engineer")
    cmd = SemanticCommand(
        intent="update_application",
        target=SemanticTarget(),
        changes=SemanticChanges(location_mode="hybrid", priority="MEDIUM", current_stages=["Networked", "Engaged"]),
    )
    stub_extractor(monkeypatch, cmd)
    resp = await client.post(
        "/transcript/parse",
        json={
            "transcript": "make it hybrid, priority medium, and stages networked and engaged",
            "context": {"active_application_id": saved.id},
        },
    )
    body = resp.json()
    assert body["status"] == "pending_changes_created"
    pc = body["pending_changes"]
    assert set(pc["changed_fields"]) == {"location", "priority", "current_stages"}
    # Saved row unchanged until Apply.
    db.refresh(saved)
    assert saved.location == ""
    assert saved.priority == ""


# ─────────────────────────────────────────────────────────────────────────────
# F. Explicit natural saved-row target
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_explicit_saved_row_target(client, db, monkeypatch):
    saved = seed_saved_application(db, "Neilsoft", "AI Engineer")
    cmd = SemanticCommand(
        intent="update_application",
        target=SemanticTarget(company="Neilsoft", role="AI Engineer"),
        changes=SemanticChanges(priority="MEDIUM", location_mode="hybrid", current_stages=["Networked", "Engaged"]),
    )
    stub_extractor(monkeypatch, cmd)
    resp = await client.post(
        "/transcript/parse",
        json={"transcript": "For Neilsoft AI Engineer application, set ...", "context": {}},
    )
    body = resp.json()
    assert body["status"] == "pending_changes_created"
    assert body["pending_changes"]["target_application_id"] == saved.id
    assert len(body["pending_changes"]["changed_fields"]) == 3


# ─────────────────────────────────────────────────────────────────────────────
# G. Note safety
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_note_only_does_not_touch_fields(client, db, monkeypatch):
    saved = seed_saved_application(db, "Neilsoft", "AI Engineer", comments="orig", next_action="orig-na")
    cmd = SemanticCommand(
        intent="append_note",
        target=SemanticTarget(company="Neilsoft", role="AI Engineer"),
        note="recruiter replied",
    )
    stub_extractor(monkeypatch, cmd)
    resp = await client.post(
        "/transcript/parse",
        json={"transcript": "I connected with their recruiter, add that as a note", "context": {}},
    )
    body = resp.json()
    assert body["status"] == "note_added"
    db.refresh(saved)
    assert saved.role == "AI Engineer"
    assert saved.comments == "orig"
    assert saved.next_action == "orig-na"
    notes = db.query(ApplicationNote).filter(ApplicationNote.application_id == saved.id).all()
    assert len(notes) == 1
    assert notes[0].text == "recruiter replied"
    assert db.query(ApplicationChangeDraft).count() == 0


@pytest.mark.anyio
async def test_note_on_draft_allowed(client, db, monkeypatch):
    draft = seed_draft(db, "Neilsoft", "AI Engineer")
    cmd = SemanticCommand(intent="append_note", target=SemanticTarget(), note="recruiter replied")
    stub_extractor(monkeypatch, cmd)
    resp = await client.post(
        "/transcript/parse",
        json={"transcript": "add a note saying recruiter replied", "context": {"draft_id": str(draft.id)}},
    )
    # Fast path actually handles this exact phrasing — but with no explicit
    # 'saying' anchor variations the extractor path is used. Either way a note lands.
    assert resp.status_code == 200
    notes = db.query(ApplicationNote).filter(ApplicationNote.application_id == draft.id).all()
    assert len(notes) == 1


# ─────────────────────────────────────────────────────────────────────────────
# H. Mixed-intent rejection
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_mixed_intent_rejected(client, db, monkeypatch):
    saved = seed_saved_application(db, "Neilsoft", "AI Engineer")
    cmd = SemanticCommand(
        intent="update_application",
        target=SemanticTarget(company="Neilsoft", role="AI Engineer"),
        changes=SemanticChanges(priority="MEDIUM"),
        note="recruiter replied",
    )
    stub_extractor(monkeypatch, cmd)
    resp = await client.post(
        "/transcript/parse",
        json={"transcript": "set priority to medium and add a note saying recruiter replied", "context": {}},
    )
    body = resp.json()
    assert body["status"] == "unsupported"
    assert "separate" in body["message"].lower()
    # No mutation of any kind.
    db.refresh(saved)
    assert saved.priority == ""
    assert db.query(ApplicationNote).count() == 0
    assert db.query(ApplicationChangeDraft).count() == 0


# ─────────────────────────────────────────────────────────────────────────────
# I. Clarification continuation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_clarification_continuation_update(client, db, monkeypatch):
    # Two companies share no role conflict; ambiguity comes from no context.
    seed_saved_application(db, "Neilsoft", "AI Engineer")

    # Turn 1: stages with no target → clarification (handled by fast path).
    forbid_extractor(monkeypatch)
    resp1 = await client.post(
        "/transcript/parse",
        json={"transcript": "set current stages as tailored, networked", "context": {}},
    )
    body1 = resp1.json()
    assert body1["status"] == "clarification"
    pending = body1["pending_command"]
    assert pending["missing_field"] == "company"

    # Turn 2: reply "Neilsoft" → unique role → pending changes created.
    resp2 = await client.post(
        "/transcript/parse",
        json={"transcript": "Neilsoft", "context": {"pending_command": pending}},
    )
    body2 = resp2.json()
    assert body2["status"] == "pending_changes_created"
    assert "current_stages" in body2["pending_changes"]["changed_fields"]


@pytest.mark.anyio
async def test_clarification_continuation_role_disambiguation(client, db, monkeypatch):
    seed_saved_application(db, "Neilsoft", "AI Engineer")
    seed_saved_application(db, "Neilsoft", "ML Engineer")
    forbid_extractor(monkeypatch)

    resp1 = await client.post(
        "/transcript/parse",
        json={"transcript": "set current stages as tailored", "context": {}},
    )
    pending = resp1.json()["pending_command"]

    # Reply company → multiple roles → role clarification.
    resp2 = await client.post(
        "/transcript/parse",
        json={"transcript": "Neilsoft", "context": {"pending_command": pending}},
    )
    body2 = resp2.json()
    assert body2["status"] == "clarification"
    assert body2["pending_command"]["missing_field"] == "role"

    # Reply role → resolved.
    resp3 = await client.post(
        "/transcript/parse",
        json={"transcript": "AI Engineer", "context": {"pending_command": body2["pending_command"]}},
    )
    assert resp3.json()["status"] == "pending_changes_created"


@pytest.mark.anyio
async def test_clarification_continuation_note(client, db, monkeypatch):
    seed_saved_application(db, "Neilsoft", "AI Engineer")
    forbid_extractor(monkeypatch)

    resp1 = await client.post(
        "/transcript/parse",
        json={"transcript": "add a note saying recruiter replied", "context": {}},
    )
    pending = resp1.json()["pending_command"]
    assert pending["operation"] == "append_note"

    resp2 = await client.post(
        "/transcript/parse",
        json={"transcript": "Neilsoft", "context": {"pending_command": pending}},
    )
    body2 = resp2.json()
    assert body2["status"] == "note_added"
    assert db.query(ApplicationNote).count() == 1


# ─────────────────────────────────────────────────────────────────────────────
# J. Suggestion-only UX
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_suggestion_only_for_ambiguous_field(client, db, monkeypatch):
    seed_saved_application(db, "Neilsoft", "AI Engineer")
    cmd = SemanticCommand(
        intent="unsupported",
        target=SemanticTarget(company="Neilsoft"),
        changes=SemanticChanges(priority="MEDIUM"),
    )
    stub_extractor(monkeypatch, cmd)
    resp = await client.post(
        "/transcript/parse",
        json={"transcript": "make Neilsoft medium", "context": {}},
    )
    body = resp.json()
    assert body["status"] == "unsupported"
    assert "set priority of Neilsoft to medium" in body["suggested_phrasings"]
    # No mutation.
    assert db.query(ApplicationChangeDraft).count() == 0


@pytest.mark.anyio
async def test_suggestion_generic_examples(client, db, monkeypatch):
    cmd = SemanticCommand(intent="unsupported", target=SemanticTarget(), changes=SemanticChanges())
    stub_extractor(monkeypatch, cmd)
    resp = await client.post("/transcript/parse", json={"transcript": "make it better", "context": {}})
    body = resp.json()
    assert body["status"] == "unsupported"
    assert "set priority as medium" in body["suggested_phrasings"]


# ─────────────────────────────────────────────────────────────────────────────
# K. Failure safety
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
@pytest.mark.parametrize(
    "exc",
    [
        SemanticExtractorUnavailableError("down"),
        SemanticExtractorInvalidResponseError("bad json"),
    ],
)
async def test_extractor_failure_is_safe(client, db, monkeypatch, exc):
    calls = stub_extractor_raises(monkeypatch, exc)
    resp = await client.post("/transcript/parse", json={"transcript": "something weird", "context": {}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "unsupported"
    assert calls["count"] == 1
    assert db.query(JobApplication).count() == 0
    assert db.query(ApplicationChangeDraft).count() == 0


@pytest.mark.anyio
async def test_unverified_application_id_is_not_trusted(client, db, monkeypatch):
    # Extractor supplies an application_id that does not exist; pipeline ignores it
    # and resolves via context/company. With no resolvable target → clarification.
    cmd = SemanticCommand(
        intent="update_application",
        target=SemanticTarget(application_id=99999),
        changes=SemanticChanges(priority="HIGH"),
    )
    stub_extractor(monkeypatch, cmd)
    resp = await client.post("/transcript/parse", json={"transcript": "make it high", "context": {}})
    body = resp.json()
    assert body["status"] == "clarification"
    assert db.query(ApplicationChangeDraft).count() == 0
