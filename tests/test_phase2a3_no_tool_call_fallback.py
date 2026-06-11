"""Phase 2A.3 — No-tool-call recovery and structured few-shot guidance.

Tests cover:
- No-tool-call noun phrase → draft created (company + role, no active draft)
- No-tool-call active-draft patch → draft updated (non-identity field)
- No-tool-call short status patch → draft updated
- No-tool-call multi-field patch → draft updated
- Lifecycle exclusion — discard_draft transcript must not trigger patch_active_draft
- Lifecycle exclusion — save transcript must not trigger patch_active_draft
- Saved-update exclusion — explicit saved-row update intent must not create draft
- Clarification — role only, no company, no active draft
- Clarification — company only, no active draft, no role
- Empty fields — no draft created
- Helper unit tests: _has_explicit_saved_update_intent, _fields_can_create_or_patch_draft
"""


from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skip(reason="Legacy LLM semantic mutation path disabled. USE_LEGACY_SEMANTIC_MUTATIONS=0.")

from app.semantic_interpreter import (
    SemanticInterpreterInvalidResponseError,
    SemanticInterpreterMetrics,
)
from app.semantic_interpreter import SemanticInterpretationResult
from app.semantic_schemas import (
    SemanticExtractedFields,
    SemanticFieldPatch,
    SemanticInterpreterMetrics,
    SemanticToolCallProposal,
)
from app.semantic_validation import (
    _fields_can_create_or_patch_draft,
    _has_explicit_saved_update_intent,
    _has_lifecycle_intent,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    from httpx import ASGITransport, AsyncClient
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def db():
    from app.database import SessionLocal

    with SessionLocal() as session:
        yield session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _metrics() -> SemanticInterpreterMetrics:
    return SemanticInterpreterMetrics(latency_ms=5)


def _extracted(**kwargs) -> SemanticExtractedFields:
    return SemanticExtractedFields.model_validate(kwargs)


def _fields(**kwargs) -> SemanticFieldPatch:
    return SemanticFieldPatch.model_validate(kwargs)


class NoToolCallInterpreter:
    """Interpreter that raises SemanticInterpreterInvalidResponseError on select_tool
    (simulating LLM returning no tool call) but succeeds on extract_fields."""

    def __init__(self, *, extracted_fields: dict, max_tool_turns: int = 2):
        self._extracted = SemanticExtractedFields.model_validate(extracted_fields)
        self.settings = SimpleNamespace(max_tool_turns=max_tool_turns)

    def extract_fields(self, transcript, context=None):
        return self._extracted, _metrics()

    def select_tool(self, transcript, context=None):
        raise SemanticInterpreterInvalidResponseError(
            "Local language interpreter returned no tool call. No tracker changes were saved."
        )

    def interpret(self, transcript, context=None):
        # interpret() calls extract_fields then select_tool — reproduce that flow
        extracted, metrics = self.extract_fields(transcript, context)
        self.select_tool(transcript, context)  # raises

    def health_check(self):
        return {"status": "ok", "provider": "ollama", "model": "llama3.2:3b", "mode": "tool_calling"}


async def _parse(client, transcript, interpreter, context=None):
    from app.main import app
    from app.semantic_interpreter import get_semantic_interpreter

    app.dependency_overrides[get_semantic_interpreter] = lambda: interpreter
    try:
        resp = await client.post(
            "/transcript/parse", json={"transcript": transcript, "context": context or {}}
        )
    finally:
        app.dependency_overrides.pop(get_semantic_interpreter, None)
    assert resp.status_code == 200
    return resp.json()


def _create_draft(db, *, company="Aiden AI", role="founding engineer"):
    from app.company_resolution import get_or_create_company
    from app.models import JobApplication

    company_obj = get_or_create_company(db, company)
    a = JobApplication(
        company_id=company_obj.id,
        role=role,
        normalized_role=role.lower() if role else "",
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
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    return a


def _create_app(db, *, company="Google", role="AI Engineer", status="applied"):
    from app.company_resolution import get_or_create_company
    from app.models import JobApplication
    from app.role_resolution import normalize_role_name

    company_obj = get_or_create_company(db, company)
    a = JobApplication(
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
        is_draft=False,
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    return a


# ===========================================================================
# Part 1 — Unit tests for new helpers
# ===========================================================================


@pytest.mark.parametrize("transcript", [
    "update status of Google AI Engineer application to rejected",
    "change status of Neilsoft application to applied",
    "set priority of Google AI Engineer to high",
    "set status of Neilsoft application to in-touch",
    "update priority of Neilsoft application to high",
])
def test_has_explicit_saved_update_intent_positive(transcript):
    assert _has_explicit_saved_update_intent(transcript)


@pytest.mark.parametrize("transcript", [
    "ai engineer role neilsoft company",
    "founding engineer at aiden ai",
    "change status to in-touch",
    "set location to onsite",
    "priority low",
    "discard draft",
    "save it",
    "archive Google AI Engineer",
    # Explicit create intent must override saved-update detection
    "add application for ai engineer role at neilsoft",
    "add application for ai engineer at neilsoft",
    "applied for rag engineer at bootcoding",
])
def test_has_explicit_saved_update_intent_negative(transcript):
    assert not _has_explicit_saved_update_intent(transcript)


def test_fields_can_create_or_patch_draft_with_company_and_role():
    assert _fields_can_create_or_patch_draft(_fields(company="Neilsoft", role="AI Engineer"))


def test_fields_can_create_or_patch_draft_with_status_only():
    assert _fields_can_create_or_patch_draft(_fields(status="in_touch"))


def test_fields_can_create_or_patch_draft_empty():
    assert not _fields_can_create_or_patch_draft(_fields())


def test_fields_can_create_or_patch_draft_company_only():
    assert _fields_can_create_or_patch_draft(_fields(company="Neilsoft"))


# ===========================================================================
# Part 2 — No-tool-call noun phrase → draft created
# ===========================================================================


@pytest.mark.anyio
async def test_no_tool_call_noun_phrase_creates_draft(client, db):
    """'ai engineer role neilsoft company' with no tool call → draft created."""
    interpreter = NoToolCallInterpreter(
        extracted_fields={"company": "Neilsoft", "role": "AI Engineer"},
    )
    result = await _parse(client, "ai engineer role neilsoft company", interpreter)
    assert result["status"] in {"draft_created", "draft_updated"}, (
        f"Expected draft_created/draft_updated, got: {result['status']} / {result}"
    )
    draft_data = result.get("draft", {})
    assert draft_data.get("company") == "Neilsoft", draft_data
    assert draft_data.get("role") == "AI Engineer", draft_data


@pytest.mark.anyio
async def test_no_tool_call_founding_engineer_aiden_ai(client, db):
    """'founding engineer at aiden ai' with no tool call → draft created."""
    interpreter = NoToolCallInterpreter(
        extracted_fields={"company": "Aiden AI", "role": "Founding Engineer"},
    )
    result = await _parse(client, "founding engineer at aiden ai", interpreter)
    assert result["status"] in {"draft_created", "draft_updated"}, result
    draft_data = result.get("draft", {})
    assert draft_data.get("company") == "Aiden AI", draft_data
    assert draft_data.get("role") == "Founding Engineer", draft_data


# ===========================================================================
# Part 3 — No-tool-call active-draft patch
# ===========================================================================


@pytest.mark.anyio
async def test_no_tool_call_patches_active_draft_location(client, db):
    """'set location to onsite' with no tool call + active draft → location patched."""
    draft = _create_draft(db, company="Aiden AI", role="founding engineer")

    interpreter = NoToolCallInterpreter(
        extracted_fields={"location": "onsite"},
    )
    result = await _parse(
        client,
        "set location to onsite",
        interpreter,
        context={
            "draft_id": str(draft.id),
            "active_draft": {"company": "Aiden AI", "role": "founding engineer"},
        },
    )
    assert result["status"] in {"draft_created", "draft_updated"}, result
    draft_data = result.get("draft", {})
    assert draft_data.get("location") == "on-site", draft_data


@pytest.mark.anyio
async def test_no_tool_call_patches_active_draft_status(client, db):
    """'change status to in-touch' with no tool call + active draft → status patched."""
    draft = _create_draft(db, company="Aiden AI", role="founding engineer")

    interpreter = NoToolCallInterpreter(
        extracted_fields={"status": "in-touch"},
    )
    result = await _parse(
        client,
        "change status to in-touch",
        interpreter,
        context={
            "draft_id": str(draft.id),
            "active_draft": {"company": "Aiden AI", "role": "founding engineer"},
        },
    )
    assert result["status"] in {"draft_created", "draft_updated"}, result
    draft_data = result.get("draft", {})
    assert draft_data.get("status") == "in_touch", draft_data


@pytest.mark.anyio
async def test_no_tool_call_multi_field_patch(client, db):
    """'role is AI Engineer, change employment type to fulltime' → multi-field draft patch."""
    draft = _create_draft(db, company="Neilsoft", role="")

    interpreter = NoToolCallInterpreter(
        extracted_fields={"role": "AI Engineer", "employment_types": ["fulltime"]},
    )
    result = await _parse(
        client,
        "role is AI Engineer, change employment type to fulltime",
        interpreter,
        context={
            "draft_id": str(draft.id),
            "active_draft": {"company": "Neilsoft", "role": ""},
        },
    )
    assert result["status"] in {"draft_created", "draft_updated"}, result
    draft_data = result.get("draft", {})
    assert draft_data.get("role") == "AI Engineer", draft_data
    assert "Full Time" in (draft_data.get("employment_types") or []), draft_data


# ===========================================================================
# Part 4 — Lifecycle exclusion
# ===========================================================================


@pytest.mark.anyio
async def test_no_tool_call_lifecycle_discard_not_absorbed(client, db):
    """'discard draft of Aiden AI' with no tool call must not patch the draft."""
    draft = _create_draft(db, company="Aiden AI", role="founding engineer")

    interpreter = NoToolCallInterpreter(
        extracted_fields={"company": "Aiden AI"},
    )
    result = await _parse(
        client,
        "discard draft of Aiden AI",
        interpreter,
        context={
            "draft_id": str(draft.id),
            "active_draft": {"company": "Aiden AI", "role": "founding engineer"},
        },
    )
    # Must NOT be draft_created or draft_updated
    assert result["status"] not in {"draft_created", "draft_updated"}, (
        f"Lifecycle command must not be absorbed into draft patch: {result}"
    )


@pytest.mark.anyio
async def test_no_tool_call_lifecycle_save_not_absorbed(client, db):
    """'save it' with no tool call must not patch the draft."""
    draft = _create_draft(db, company="Aiden AI", role="founding engineer")

    interpreter = NoToolCallInterpreter(
        extracted_fields={},
    )
    result = await _parse(
        client,
        "save it",
        interpreter,
        context={
            "draft_id": str(draft.id),
            "active_draft": {"company": "Aiden AI", "role": "founding engineer"},
        },
    )
    assert result["status"] not in {"draft_created", "draft_updated"}, (
        f"Lifecycle save must not be absorbed into draft patch: {result}"
    )


@pytest.mark.anyio
async def test_no_tool_call_lifecycle_cancel_not_absorbed(client, db):
    """'cancel this draft' with no tool call must not patch the draft."""
    draft = _create_draft(db, company="Aiden AI", role="founding engineer")

    interpreter = NoToolCallInterpreter(
        extracted_fields={},
    )
    result = await _parse(
        client,
        "cancel this draft",
        interpreter,
        context={
            "draft_id": str(draft.id),
            "active_draft": {"company": "Aiden AI", "role": "founding engineer"},
        },
    )
    assert result["status"] not in {"draft_created", "draft_updated"}, (
        f"Lifecycle cancel must not be absorbed into draft patch: {result}"
    )


# ===========================================================================
# Part 5 — Saved-update exclusion
# ===========================================================================


@pytest.mark.anyio
async def test_no_tool_call_saved_update_routes_to_pending_changes(client, db):
    """'update status of Google AI Engineer application to rejected' → Pending Changes, not draft."""
    app = _create_app(db, company="Google", role="AI Engineer", status="applied")

    interpreter = NoToolCallInterpreter(
        extracted_fields={"company": "Google", "role": "AI Engineer", "status": "rejected"},
    )
    result = await _parse(
        client,
        "update status of Google AI Engineer application to rejected",
        interpreter,
        context={"active_application_id": app.id},
    )
    # Should route to pending changes or clarification — must not create a new draft
    assert result["status"] not in {"draft_created"}, (
        f"Saved-row update must not create a new draft: {result}"
    )
    # Saved row must be unchanged
    from app.models import JobApplication
    from app.database import SessionLocal
    with SessionLocal() as s:
        refreshed = s.get(JobApplication, app.id)
        assert refreshed.status == "applied", (
            f"Saved row status must not change without applying pending changes: {refreshed.status}"
        )


# ===========================================================================
# Part 6 — Clarification cases
# ===========================================================================


@pytest.mark.anyio
async def test_no_tool_call_role_only_asks_for_company(client, db):
    """Role only, no company, no active draft → clarification asking for company."""
    interpreter = NoToolCallInterpreter(
        extracted_fields={"role": "AI Engineer"},
    )
    result = await _parse(client, "AI Engineer role", interpreter)
    assert result["status"] in {"clarification", "clarification_required"}, result
    question = result.get("clarification_question", "")
    assert "company" in question.lower(), (
        f"Expected clarification asking for company, got: {question!r}"
    )


@pytest.mark.anyio
async def test_no_tool_call_company_only_asks_for_role(client, db):
    """Company only, no role, no active draft → clarification asking for role."""
    interpreter = NoToolCallInterpreter(
        extracted_fields={"company": "Neilsoft"},
    )
    result = await _parse(client, "Neilsoft company", interpreter)
    assert result["status"] in {"clarification", "clarification_required"}, result
    question = result.get("clarification_question", "")
    assert "neilsoft" in question.lower() or "role" in question.lower(), (
        f"Expected clarification asking for role, got: {question!r}"
    )


# ===========================================================================
# Part 7 — Empty fields → no draft created
# ===========================================================================


@pytest.mark.anyio
async def test_no_tool_call_empty_fields_no_draft(client, db):
    """Empty extracted fields with no tool call → no draft created."""
    interpreter = NoToolCallInterpreter(
        extracted_fields={},
    )
    result = await _parse(client, "okay", interpreter)
    assert result["status"] not in {"draft_created", "draft_updated"}, (
        f"Empty fields must not create a draft: {result}"
    )


# ===========================================================================
# Part 8 — Noun phrase with active draft → patch (not create new)
# ===========================================================================


@pytest.mark.anyio
async def test_no_tool_call_noun_phrase_patches_existing_draft(client, db):
    """Noun phrase 'ai engineer role neilsoft' with active draft → patch, not duplicate."""
    draft = _create_draft(db, company="Neilsoft", role="")

    interpreter = NoToolCallInterpreter(
        extracted_fields={"company": "Neilsoft", "role": "AI Engineer"},
    )
    result = await _parse(
        client,
        "ai engineer role neilsoft company",
        interpreter,
        context={
            "draft_id": str(draft.id),
            "active_draft": {"company": "Neilsoft", "role": ""},
        },
    )
    assert result["status"] in {"draft_created", "draft_updated"}, result
    # Verify draft_id matches the existing draft (patch, not new create)
    result_draft_id = result.get("draft_id")
    assert result_draft_id == str(draft.id), (
        f"Expected patch of existing draft {draft.id}, got draft_id={result_draft_id!r}"
    )
    draft_data = result.get("draft", {})
    assert draft_data.get("role") == "AI Engineer", draft_data
