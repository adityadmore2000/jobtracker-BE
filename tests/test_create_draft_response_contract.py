"""
Regression tests for the deterministic create-draft response contract.

Verified invariants:
- top-level draft_id == nested draft.id (never 0)
- status is canonical draft_created/draft_updated on success
- no synthetic draft object when a truthful no-op occurs
- Ollama is never called for explicit create commands
"""

import pytest
from fastapi.testclient import TestClient

from app.database import SessionLocal
from app.main import app
from app.models import JobApplication
from app.mutation_dispatcher import dispatch
from app.mutation_schemas import ApplicationChanges, MutationPayload, MutationTarget
from app.semantic_validation import build_transcript_response_from_mutation
from app.schemas import SemanticTranscriptResponse
from app.semantic_schemas import SemanticToolCallProposal
from app.schemas import TranscriptParseRequest
from app.transcript_response_adapter import to_public_transcript_response


_COMPANY = "Runtime Test Labs"
_ROLE = "Response Contract Engineer"


@pytest.fixture
def db():
    with SessionLocal() as session:
        yield session


@pytest.fixture
def client():
    return TestClient(app)


def _parse(client, transcript: str, context: dict | None = None) -> dict:
    body = {"transcript": transcript}
    if context:
        body["context"] = context
    resp = client.post("/transcript/parse", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _cleanup(db, company: str, role: str) -> None:
    from app.company_resolution import get_or_create_company
    from app.role_resolution import normalize_role_name
    company_obj = db.query(__import__("app.models", fromlist=["Company"]).Company).filter_by(name=company).first()
    if company_obj is None:
        return
    rows = (
        db.query(JobApplication)
        .filter(
            JobApplication.company_id == company_obj.id,
            JobApplication.normalized_role == normalize_role_name(role),
        )
        .all()
    )
    for r in rows:
        db.delete(r)
    db.commit()


# ---------------------------------------------------------------------------
# 1. New draft
# ---------------------------------------------------------------------------

def test_new_draft_response_contract(client, db):
    _cleanup(db, _COMPANY, _ROLE)
    transcript = f"add application for {_ROLE} at {_COMPANY}"

    data = _parse(client, transcript)

    assert data["status"] == "draft_created", f"Got status={data['status']!r}, message={data['message']!r}"
    assert data["draft"] is not None
    assert data["draft_id"] is not None

    nested_id = data["draft"]["id"]
    top_level_id = int(data["draft_id"])

    assert nested_id != 0, "nested draft.id must not be 0 for a persisted draft"
    assert nested_id == top_level_id, (
        f"draft_id mismatch: top-level={top_level_id}, nested={nested_id}"
    )

    # Verify row exists in DB
    row = db.get(JobApplication, nested_id)
    assert row is not None
    assert row.is_draft is True

    _cleanup(db, _COMPANY, _ROLE)


# ---------------------------------------------------------------------------
# 2. Existing matching draft — reused, no duplicate row
# ---------------------------------------------------------------------------

def test_existing_draft_reused_response_contract(client, db):
    _cleanup(db, _COMPANY, _ROLE)
    transcript = f"add application for {_ROLE} at {_COMPANY}"

    first = _parse(client, transcript)
    assert first["status"] == "draft_created"
    first_draft_id = int(first["draft_id"])

    second = _parse(client, transcript)

    assert second["draft"] is not None
    assert second["draft_id"] is not None

    nested_id = second["draft"]["id"]
    top_level_id = int(second["draft_id"])

    assert nested_id != 0
    assert nested_id == top_level_id
    assert top_level_id == first_draft_id, "second call must reuse the existing draft, not create a new one"

    # Verify exactly one row exists
    from app.models import Company
    from app.role_resolution import normalize_role_name
    company_obj = db.query(Company).filter_by(name=_COMPANY).first()
    assert company_obj is not None
    rows = (
        db.query(JobApplication)
        .filter(
            JobApplication.company_id == company_obj.id,
            JobApplication.normalized_role == normalize_role_name(_ROLE),
        )
        .all()
    )
    assert len(rows) == 1, f"Expected 1 draft row, found {len(rows)}"

    _cleanup(db, _COMPANY, _ROLE)


# ---------------------------------------------------------------------------
# 3. Existing saved applied row — truthful no-op, no synthetic draft
# ---------------------------------------------------------------------------

def test_existing_applied_row_truthful_no_op(client, db):
    _cleanup(db, _COMPANY, _ROLE)

    # Create draft with applied status, then save it
    create_result = dispatch(
        MutationPayload(
            operation="create_draft",
            target=MutationTarget(),
            changes=ApplicationChanges(company=_COMPANY, role=_ROLE, status="applied"),
        ),
        db,
    )
    assert create_result.success
    draft_id = create_result.draft["id"]

    save_result = dispatch(
        MutationPayload(
            operation="save_draft",
            target=MutationTarget(draft_id=str(draft_id)),
            changes=ApplicationChanges(),
        ),
        db,
    )
    assert save_result.success

    # Now try to create again via transcript/parse
    transcript = f"add application for {_ROLE} at {_COMPANY}"
    data = _parse(client, transcript)

    # Must be a truthful no-op with no synthetic draft object
    # (the reapply matrix returns operation=no_change for applied+non-archived)
    assert data["status"] == "no_change", f"Expected no_change, got {data['status']!r}"
    # No synthetic draft should be returned
    assert data["draft"] is None, "No draft DTO should be returned for a truthful no-op"

    _cleanup(db, _COMPANY, _ROLE)


# ---------------------------------------------------------------------------
# 4. Adapter contract: when draft_id present and draft present, ids must match
# ---------------------------------------------------------------------------

def test_adapter_draft_id_consistency():
    """Unit test: build_transcript_response_from_mutation + adapter round-trip."""
    from app.mutation_schemas import MutationResult

    persisted_draft_dict = {
        "id": 42,
        "company": "Adapter Co",
        "company_id": 1,
        "role": "Contract Tester",
        "employment_types_json": [],
        "job_link": "",
        "location": "",
        "status": "",
        "current_stages_json": [],
        "priority": "",
        "engaged_days": 0,
        "next_action": "",
        "comments": "",
        "is_draft": True,
        "draft_created_at": "2026-06-11T00:00:00+00:00",
        "archived_at": None,
        "created_at": "2026-06-11T00:00:00+00:00",
        "updated_at": "2026-06-11T00:00:00+00:00",
    }
    mutation_result = MutationResult(
        success=True,
        operation="create_draft",
        message="Draft created.",
        draft=persisted_draft_dict,
    )
    payload = TranscriptParseRequest(transcript="add application for Contract Tester at Adapter Co")
    proposal = SemanticToolCallProposal()

    internal = build_transcript_response_from_mutation(mutation_result, payload, proposal)
    public = to_public_transcript_response(internal)

    assert public.draft is not None
    assert public.draft_id is not None
    assert public.draft.id != 0, "nested draft.id must not be 0"
    assert public.draft.id == int(public.draft_id), (
        f"draft id mismatch: top-level={public.draft_id}, nested={public.draft.id}"
    )
    assert public.status == "draft_created"


# ---------------------------------------------------------------------------
# 5. Ollama bypass: explicit create command never calls interpreter
# ---------------------------------------------------------------------------

def test_explicit_create_bypasses_ollama(client, db, monkeypatch):
    _cleanup(db, _COMPANY, _ROLE)

    interpreter_calls = []

    from app import semantic_interpreter as si
    original_get = si.get_semantic_interpreter

    def mock_get():
        class MockInterpreter:
            def interpret(self, *args, **kwargs):
                interpreter_calls.append(("interpret", args))
                raise AssertionError("Ollama must not be called for explicit create commands")

            def extract_fields(self, *args, **kwargs):
                interpreter_calls.append(("extract_fields", args))
                raise AssertionError("Ollama must not be called for explicit create commands")

            @property
            def settings(self):
                return type("S", (), {"max_tool_turns": 2})()

        return MockInterpreter()

    monkeypatch.setattr(si, "get_semantic_interpreter", mock_get)

    transcript = f"add application for {_ROLE} at {_COMPANY}"
    data = _parse(client, transcript)

    assert interpreter_calls == [], f"Ollama was called: {interpreter_calls}"
    assert data["status"] == "draft_created"
    assert data["draft"] is not None

    _cleanup(db, _COMPANY, _ROLE)
