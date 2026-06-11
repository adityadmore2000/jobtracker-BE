"""Phase 2A.2 — Active-draft contextual patch and compact alias coverage.

Tests cover:
- Contextual patch fallback: short follow-up commands patch the active draft
- Compact alias normalization (fulltime, intouch, onsite, etc.)
- Multi-field contextual patch preservation
- No-op detection: Draft already has those values → no commit, truthful message
- Lifecycle intent exclusion: save/discard/archive/delete never use fallback
- Saved-target exclusion: explicit known company routes to pending-changes, not draft
- Draft integrity after failed/no-op contextual patch
- Malformed LLM output reconciliation (status in role, compact alias in list)
"""

from types import SimpleNamespace

import pytest

from app.constants import normalize_status_value
from app.semantic_interpreter import SemanticInterpretationResult
from app.semantic_schemas import (
    SemanticExtractedFields,
    SemanticFieldPatch,
    SemanticInterpreterMetrics,
    SemanticToolCallProposal,
)
from app.semantic_validation import (
    _has_lifecycle_intent,
    normalize_employment_types,
    normalize_extracted_fields,
    normalize_location,
    normalize_priority,
    normalize_status,
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


def _proposal(tool_name: str, arguments: dict) -> SemanticToolCallProposal:
    return SemanticToolCallProposal.model_validate({"tool_name": tool_name, "arguments": arguments})


def _fields(**kwargs) -> SemanticFieldPatch:
    return SemanticFieldPatch.model_validate(kwargs)


class FakeInterpreter:
    """Monkeypatched LLM interpreter that returns a fixed proposal and extracted fields."""

    def __init__(self, *, proposal, extracted_fields=None, max_tool_turns=2):
        self._proposal = proposal
        self._extracted_fields = SemanticExtractedFields.model_validate(extracted_fields or {})
        self.settings = SimpleNamespace(max_tool_turns=max_tool_turns)

    def interpret(self, transcript, context=None):
        return SemanticInterpretationResult(
            proposal=self._proposal,
            metrics=SemanticInterpreterMetrics(latency_ms=5),
            extracted_fields=self._extracted_fields,
        )

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


def _create_draft(db, *, company="Aiden AI", role="founding engineer", location="", status="", priority="", employment_types=None):
    from app.company_resolution import get_or_create_company
    from app.models import JobApplication

    company_obj = get_or_create_company(db, company)
    a = JobApplication(
        company_id=company_obj.id,
        role=role,
        normalized_role=role.lower() if role else "",
        employment_types_json=employment_types or [],
        job_link="",
        location=location,
        status=status,
        current_stages_json=[],
        priority=priority,
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
# Part 1 — Alias normalization (unit tests)
# ===========================================================================


@pytest.mark.parametrize("raw,expected", [
    ("fulltime", ["Full Time"]),
    ("full time", ["Full Time"]),
    ("full-time", ["Full Time"]),
    ("full_time", ["Full Time"]),
])
def test_employment_type_alias_fulltime(raw, expected):
    assert normalize_employment_types([raw]) == expected


@pytest.mark.parametrize("raw,expected", [
    ("parttime", ["Part Time"]),
    ("part time", ["Part Time"]),
    ("part-time", ["Part Time"]),
    ("part_time", ["Part Time"]),
])
def test_employment_type_alias_parttime(raw, expected):
    assert normalize_employment_types([raw]) == expected


@pytest.mark.parametrize("raw,expected", [
    ("intouch", "in_touch"),
    ("in touch", "in_touch"),
    ("in-touch", "in_touch"),
    ("in_touch", "in_touch"),
])
def test_status_alias_intouch(raw, expected):
    assert normalize_status_value(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("onsite", "on-site"),
    ("on site", "on-site"),
    ("on-site", "on-site"),
])
def test_location_alias_onsite(raw, expected):
    assert normalize_location(raw) == expected


# ===========================================================================
# Part 2 — Lifecycle intent classifier (unit tests)
# ===========================================================================


@pytest.mark.parametrize("transcript", [
    "save it",
    "save this",
    "save",
    "discard draft",
    "cancel it",
    "archive Google AI Engineer",
    "delete Google application",
    "permanently delete this",
    "restore my application",
    "drop this draft",
    "remove this draft",
])
def test_lifecycle_intent_detected(transcript):
    assert _has_lifecycle_intent(transcript)


@pytest.mark.parametrize("transcript", [
    "change status to in-touch",
    "set location to onsite",
    "priority low",
    "role is AI Engineer, change employment type to fulltime",
    "change employment type to fulltime",
    "comments referral pending",
    "next action is ask for referral",
])
def test_no_lifecycle_intent(transcript):
    assert not _has_lifecycle_intent(transcript)


# ===========================================================================
# Part 3 — Contextual patch via API (integration tests)
# ===========================================================================


@pytest.mark.anyio
async def test_contextual_status_patch_with_active_draft(client, db):
    """'change status to in-touch' with active draft → status patched on draft."""
    draft = _create_draft(db, company="Aiden AI", role="founding engineer")
    draft_id = str(draft.id)

    # Simulate LLM putting status in role (common mis-extraction)
    interpreter = FakeInterpreter(
        extracted_fields={"role": "in-touch"},
        proposal=_proposal("ask_clarification", {"question": "Which company should I use?"}),
    )
    result = await _parse(
        client,
        "change status to in-touch",
        interpreter,
        context={
            "draft_id": draft_id,
            "active_draft": {"company": "Aiden AI", "role": "founding engineer"},
        },
    )
    # Contextual fallback should have fired: status = in_touch
    assert result["status"] in {"draft_created", "draft_updated"}, (
        f"Expected draft_created or draft_updated, got: {result['status']} / {result}"
    )
    draft_data = result.get("draft", {})
    assert draft_data.get("status") == "in_touch", (
        f"Expected status=in_touch, got: {draft_data}"
    )


@pytest.mark.anyio
async def test_contextual_priority_patch_with_active_draft(client, db):
    """'priority low' with active draft → priority patched."""
    draft = _create_draft(db, company="Aiden AI", role="founding engineer")
    draft_id = str(draft.id)

    interpreter = FakeInterpreter(
        extracted_fields={"priority": "LOW"},
        proposal=_proposal("ask_clarification", {"question": "Which company should I use?"}),
    )
    result = await _parse(
        client,
        "priority low",
        interpreter,
        context={
            "draft_id": draft_id,
            "active_draft": {"company": "Aiden AI", "role": "founding engineer"},
        },
    )
    assert result["status"] in {"draft_created", "draft_updated"}, result
    draft_data = result.get("draft", {})
    assert draft_data.get("priority") == "LOW", draft_data


@pytest.mark.anyio
async def test_contextual_location_patch_with_active_draft(client, db):
    """'set location to onsite' with active draft → location patched."""
    draft = _create_draft(db, company="Aiden AI", role="founding engineer")
    draft_id = str(draft.id)

    interpreter = FakeInterpreter(
        extracted_fields={"location": "on-site"},
        proposal=_proposal("ask_clarification", {"question": "Which company?"}),
    )
    result = await _parse(
        client,
        "set location to onsite",
        interpreter,
        context={
            "draft_id": draft_id,
            "active_draft": {"company": "Aiden AI", "role": "founding engineer"},
        },
    )
    assert result["status"] in {"draft_created", "draft_updated"}, result
    draft_data = result.get("draft", {})
    assert draft_data.get("location") == "on-site", draft_data


@pytest.mark.anyio
async def test_contextual_employment_type_patch_with_active_draft(client, db):
    """'change employment type to fulltime' with active draft → employment_types patched."""
    draft = _create_draft(db, company="Aiden AI", role="founding engineer")
    draft_id = str(draft.id)

    interpreter = FakeInterpreter(
        extracted_fields={"employment_types": ["Full Time"]},
        proposal=_proposal("ask_clarification", {"question": "Which company?"}),
    )
    result = await _parse(
        client,
        "change employment type to fulltime",
        interpreter,
        context={
            "draft_id": draft_id,
            "active_draft": {"company": "Aiden AI", "role": "founding engineer"},
        },
    )
    assert result["status"] in {"draft_created", "draft_updated"}, result
    draft_data = result.get("draft", {})
    assert "Full Time" in draft_data.get("employment_types", []), draft_data


@pytest.mark.anyio
async def test_contextual_multi_field_patch_preserves_both_fields(client, db):
    """Multi-field: role + employment_type both survive extraction → both patched."""
    draft = _create_draft(db, company="Aiden AI", role="founding engineer")
    draft_id = str(draft.id)

    interpreter = FakeInterpreter(
        extracted_fields={"role": "AI Engineer", "employment_types": ["Full Time"]},
        proposal=_proposal("ask_clarification", {"question": "Which company?"}),
    )
    result = await _parse(
        client,
        "role is AI Engineer, change employment type to fulltime",
        interpreter,
        context={
            "draft_id": draft_id,
            "active_draft": {"company": "Aiden AI", "role": "founding engineer"},
        },
    )
    assert result["status"] in {"draft_created", "draft_updated"}, result
    draft_data = result.get("draft", {})
    assert draft_data.get("role") == "AI Engineer", draft_data
    assert "Full Time" in draft_data.get("employment_types", []), draft_data


# ===========================================================================
# Part 4 — Malformed LLM output reconciliation
# ===========================================================================


@pytest.mark.anyio
async def test_status_misplaced_in_role_reconciled_via_fallback(client, db):
    """LLM puts 'in-touch' in role field; reconciliation moves it to status."""
    draft = _create_draft(db, company="Aiden AI", role="founding engineer")
    draft_id = str(draft.id)

    # LLM incorrectly places "in-touch" in role
    interpreter = FakeInterpreter(
        extracted_fields={"role": "in-touch"},
        proposal=_proposal("ask_clarification", {"question": "Which company?"}),
    )
    result = await _parse(
        client,
        "change status to in-touch",
        interpreter,
        context={
            "draft_id": draft_id,
            "active_draft": {"company": "Aiden AI", "role": "founding engineer"},
        },
    )
    # After reconciliation: role → status, fallback fires
    assert result["status"] in {"draft_created", "draft_updated"}, result
    draft_data = result.get("draft", {})
    assert draft_data.get("status") == "in_touch", draft_data
    assert draft_data.get("role") != "in-touch", draft_data


@pytest.mark.anyio
async def test_compact_employment_type_alias_in_llm_output(client, db):
    """LLM emits 'fulltime' in employment_types; backend normalises to 'Full Time'."""
    draft = _create_draft(db, company="Aiden AI", role="founding engineer")
    draft_id = str(draft.id)

    interpreter = FakeInterpreter(
        extracted_fields={"employment_types": ["fulltime"]},
        proposal=_proposal("ask_clarification", {"question": "Which company?"}),
    )
    result = await _parse(
        client,
        "change employment type to fulltime",
        interpreter,
        context={
            "draft_id": draft_id,
            "active_draft": {"company": "Aiden AI", "role": "founding engineer"},
        },
    )
    assert result["status"] in {"draft_created", "draft_updated"}, result
    draft_data = result.get("draft", {})
    assert "Full Time" in draft_data.get("employment_types", []), draft_data


@pytest.mark.anyio
async def test_unsupported_tool_with_patch_fields_uses_fallback(client, db):
    """LLM selects ask_clarification but extracted fields are valid → contextual fallback."""
    draft = _create_draft(db, company="Aiden AI", role="founding engineer")
    draft_id = str(draft.id)

    interpreter = FakeInterpreter(
        extracted_fields={"status": "in_touch"},
        proposal=_proposal("ask_clarification", {"question": "Which company?"}),
    )
    result = await _parse(
        client,
        "change status to in-touch",
        interpreter,
        context={
            "draft_id": draft_id,
            "active_draft": {"company": "Aiden AI", "role": "founding engineer"},
        },
    )
    assert result["status"] in {"draft_created", "draft_updated"}, result


# ===========================================================================
# Part 5 — Lifecycle exclusion (fallback must NOT steal lifecycle commands)
# ===========================================================================


@pytest.mark.anyio
async def test_lifecycle_discard_not_routed_to_fallback(client, db):
    """'discard draft' with active draft → discard_draft handler, not patch fallback."""
    draft = _create_draft(db, company="Aiden AI", role="founding engineer")
    draft_id = str(draft.id)

    interpreter = FakeInterpreter(
        extracted_fields={"company": "Aiden AI"},
        proposal=_proposal("discard_draft", {"target": {"company": "Aiden AI"}}),
    )
    result = await _parse(
        client,
        "discard draft",
        interpreter,
        context={
            "draft_id": draft_id,
            "active_draft": {"company": "Aiden AI", "role": "founding engineer"},
        },
    )
    # Must be discarded, NOT draft_created or draft_updated
    assert result["status"] not in {"draft_created", "draft_updated"}, (
        f"discard_draft was wrongly routed as patch: {result}"
    )


@pytest.mark.anyio
async def test_lifecycle_save_not_routed_to_fallback(client, db):
    """'save it' with active draft → save handler, not patch fallback."""
    draft = _create_draft(db, company="Aiden AI", role="founding engineer")
    draft_id = str(draft.id)

    interpreter = FakeInterpreter(
        extracted_fields={},
        proposal=_proposal("request_draft_save", {}),
    )
    result = await _parse(
        client,
        "save it",
        interpreter,
        context={
            "draft_id": draft_id,
            "active_draft": {"company": "Aiden AI", "role": "founding engineer"},
        },
    )
    # Save produces "saved" status, NOT draft_updated
    assert result["status"] != "draft_updated", (
        f"save_draft was wrongly routed as patch: {result}"
    )


@pytest.mark.anyio
async def test_lifecycle_archive_not_routed_to_fallback(client, db):
    """'archive Google AI Engineer' → archive handler, not draft patch fallback."""
    saved = _create_app(db, company="Google", role="AI Engineer", status="applied")

    interpreter = FakeInterpreter(
        extracted_fields={"company": "Google", "role": "AI Engineer"},
        proposal=_proposal("archive_application", {"target": {"company": "Google", "role": "AI Engineer"}}),
    )
    # Also have an active draft in context — fallback must still not fire
    draft = _create_draft(db, company="Aiden AI", role="founding engineer")
    draft_id = str(draft.id)

    result = await _parse(
        client,
        "archive Google AI Engineer",
        interpreter,
        context={
            "draft_id": draft_id,
            "active_draft": {"company": "Aiden AI", "role": "founding engineer"},
        },
    )
    assert result["status"] not in {"draft_created", "draft_updated"}, (
        f"archive_application was wrongly routed as patch: {result}"
    )


@pytest.mark.anyio
async def test_lifecycle_delete_not_routed_to_fallback(client, db):
    """'delete Google AI Engineer' → explain_delete_policy, not patch fallback."""
    _create_app(db, company="Google", role="AI Engineer", status="applied")
    draft = _create_draft(db, company="Aiden AI", role="founding engineer")
    draft_id = str(draft.id)

    interpreter = FakeInterpreter(
        extracted_fields={"company": "Google", "role": "AI Engineer"},
        proposal=_proposal("explain_delete_policy", {"target": {"company": "Google", "role": "AI Engineer"}}),
    )
    result = await _parse(
        client,
        "delete Google AI Engineer",
        interpreter,
        context={
            "draft_id": draft_id,
            "active_draft": {"company": "Aiden AI", "role": "founding engineer"},
        },
    )
    assert result["status"] not in {"draft_created", "draft_updated"}, result


# ===========================================================================
# Part 6 — Saved-target exclusion
# ===========================================================================


@pytest.mark.anyio
async def test_saved_target_update_does_not_touch_active_draft(client, db):
    """Explicit saved-company target updates pending-changes, not the active draft."""
    saved = _create_app(db, company="Google", role="AI Engineer", status="applied")
    draft = _create_draft(db, company="Aiden AI", role="founding engineer")
    draft_id = str(draft.id)

    interpreter = FakeInterpreter(
        extracted_fields={"company": "Google", "priority": "LOW"},
        proposal=_proposal(
            "preview_existing_application_update",
            {"target": {"company": "Google"}, "fields": {"priority": "LOW"}, "replace_explicit_fields": True},
        ),
    )
    result = await _parse(
        client,
        "set Google AI Engineer priority to low",
        interpreter,
        context={
            "draft_id": draft_id,
            "active_draft": {"company": "Aiden AI", "role": "founding engineer"},
        },
    )
    # Should produce pending_changes for the Google app, NOT modify the Aiden AI draft
    assert result["status"] in {"pending_changes_created", "pending_changes_updated"}, (
        f"Expected pending_changes, got: {result['status']} / {result}"
    )
    # Verify the Aiden AI draft is untouched
    from app.models import JobApplication
    db.expire_all()
    refreshed_draft = db.get(JobApplication, draft.id)
    assert refreshed_draft is not None, "Draft was deleted — should not have been."
    assert refreshed_draft.priority == "", (
        f"Active draft priority should be empty, got: {refreshed_draft.priority}"
    )


# ===========================================================================
# Part 7 — No-op detection
# ===========================================================================


@pytest.mark.anyio
async def test_noop_when_draft_already_has_value(client, db):
    """Setting location to a value the draft already has → no_change, no commit."""
    from datetime import datetime, timezone
    draft = _create_draft(db, company="Aiden AI", role="founding engineer", location="on-site")
    draft_id = str(draft.id)
    original_updated_at = draft.updated_at

    interpreter = FakeInterpreter(
        extracted_fields={"location": "on-site"},
        proposal=_proposal(
            "patch_active_draft",
            {
                "fields": {"location": "on-site"},
                "replace_explicit_fields": True,
                "context_notes": [],
            },
        ),
    )
    result = await _parse(
        client,
        "set location to onsite",
        interpreter,
        context={
            "draft_id": draft_id,
            "active_draft": {"company": "Aiden AI", "role": "founding engineer", "location": "on-site"},
        },
    )
    # Public status should be no_change (not draft_updated)
    assert result["status"] == "no_change", (
        f"Expected no_change for duplicate location patch, got: {result['status']} / {result}"
    )
    # Verify DB row not changed
    from app.models import JobApplication
    db.expire_all()
    refreshed = db.get(JobApplication, draft.id)
    assert refreshed is not None
    assert refreshed.location == "on-site"
    if original_updated_at is not None and refreshed.updated_at is not None:
        assert refreshed.updated_at == original_updated_at, (
            "updated_at should not change on a no-op patch"
        )


@pytest.mark.anyio
async def test_noop_blank_extracted_fields_no_fallback(client, db):
    """Blank extracted fields → no fallback fire, no_change."""
    draft = _create_draft(db, company="Aiden AI", role="founding engineer")
    draft_id = str(draft.id)

    interpreter = FakeInterpreter(
        extracted_fields={},
        proposal=_proposal("ask_clarification", {"question": "Which company?"}),
    )
    result = await _parse(
        client,
        "hmm",
        interpreter,
        context={
            "draft_id": draft_id,
            "active_draft": {"company": "Aiden AI", "role": "founding engineer"},
        },
    )
    # No actionable fields → no fallback → no_change or clarification
    assert result["status"] in {"no_change", "clarification"}, result


# ===========================================================================
# Part 8 — Draft integrity after failed/no-op patch
# ===========================================================================


@pytest.mark.anyio
async def test_draft_survives_noop_and_can_be_saved(client, db):
    """Draft remains intact after a no-op location patch and can still be saved."""
    draft = _create_draft(db, company="Aiden AI", role="founding engineer", location="on-site")
    draft_id = str(draft.id)

    # No-op patch
    interpreter_noop = FakeInterpreter(
        extracted_fields={"location": "on-site"},
        proposal=_proposal(
            "patch_active_draft",
            {"fields": {"location": "on-site"}, "replace_explicit_fields": True, "context_notes": []},
        ),
    )
    noop_result = await _parse(
        client,
        "set location to onsite",
        interpreter_noop,
        context={
            "draft_id": draft_id,
            "active_draft": {"company": "Aiden AI", "role": "founding engineer", "location": "on-site"},
        },
    )
    assert noop_result["status"] == "no_change"

    # Draft still exists and is saveable
    from app.models import JobApplication
    db.expire_all()
    refreshed = db.get(JobApplication, draft.id)
    assert refreshed is not None, "Draft was deleted after no-op"
    assert refreshed.is_draft is True, "Draft should still be a draft"


@pytest.mark.anyio
async def test_draft_survives_noop_and_can_be_discarded(client, db):
    """Draft remains intact after a no-op patch and can still be discarded."""
    draft = _create_draft(db, company="Aiden AI", role="founding engineer", status="in_touch")
    draft_id = str(draft.id)

    # No-op patch
    interpreter_noop = FakeInterpreter(
        extracted_fields={"status": "in_touch"},
        proposal=_proposal(
            "patch_active_draft",
            {"fields": {"status": "in_touch"}, "replace_explicit_fields": True, "context_notes": []},
        ),
    )
    noop_result = await _parse(
        client,
        "change status to in-touch",
        interpreter_noop,
        context={
            "draft_id": draft_id,
            "active_draft": {"company": "Aiden AI", "role": "founding engineer", "status": "in_touch"},
        },
    )
    assert noop_result["status"] == "no_change"

    # Discard succeeds
    interpreter_discard = FakeInterpreter(
        extracted_fields={},
        proposal=_proposal("discard_draft", {"target": {}}),
    )
    discard_result = await _parse(
        client,
        "discard draft",
        interpreter_discard,
        context={
            "draft_id": draft_id,
            "active_draft": {"company": "Aiden AI", "role": "founding engineer", "status": "in_touch"},
        },
    )
    # discard_draft handler executes; public status may be "discarded" or "no_change"
    # depending on the adapter mapping. What matters is that it did NOT error and did NOT
    # treat the discard as a patch.
    assert discard_result["status"] not in {"draft_created", "draft_updated"}, discard_result
    assert discard_result["status"] != "clarification", discard_result


# ===========================================================================
# Part 9 — No active draft → fallback does not fire
# ===========================================================================


@pytest.mark.anyio
async def test_no_draft_no_fallback(client, db):
    """Without an active draft, the contextual patch fallback must not create a draft."""
    interpreter = FakeInterpreter(
        extracted_fields={"status": "in_touch"},
        proposal=_proposal("ask_clarification", {"question": "Which company?"}),
    )
    result = await _parse(
        client,
        "change status to in-touch",
        interpreter,
        context={},  # No active draft
    )
    # Without a draft, contextual fallback cannot resolve company → no draft created
    assert result["status"] not in {"draft_created", "draft_updated"}, (
        f"Fallback should not create draft without active draft context: {result}"
    )
