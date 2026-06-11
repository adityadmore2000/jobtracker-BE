"""Phase 2A regression tests — hardening LLM-first semantic interpretation.

Tests cover:
- Part A: Open-ended role extraction
- Part B: Separator-tolerant controlled-value normalization
- Part C: Generic controlled-field reconciliation
- Part D: Company-only target resolution for updates
- Part E: Archive intent via chat
- Part F: Delete-policy guidance
- Part G: Truthful messages for recognized-but-unsupported patterns
"""


from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skip(reason="Legacy LLM semantic mutation path disabled. USE_LEGACY_SEMANTIC_MUTATIONS=0.")
from httpx import ASGITransport, AsyncClient

from app.constants import LOCATION_ALIASES, normalize_status_value
from app.database import SessionLocal
from app.main import app
from app.models import JobApplication
from app.mutation_dispatcher import dispatch
from app.mutation_schemas import ApplicationChanges, MutationPayload, MutationTarget
from app.semantic_interpreter import get_semantic_interpreter
from app.semantic_schemas import (
    ArchiveApplicationArguments,
    ExplainDeletePolicyArguments,
    PreviewExistingApplicationTarget,
    SemanticExtractedFields,
    SemanticInterpreterMetrics,
    SemanticToolCallProposal,
)
from app.semantic_validation import (
    _reconcile_controlled_field_misclassification,
    normalize_extracted_fields,
    normalize_location,
)
from app.semantic_schemas import SemanticFieldPatch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _proposal(tool_name: str, arguments: dict) -> SemanticToolCallProposal:
    return SemanticToolCallProposal.model_validate({"tool_name": tool_name, "arguments": arguments})


class FakeInterpreter:
    def __init__(self, *, proposal, extracted_fields=None, max_tool_turns=2):
        self._proposal = proposal
        self._extracted_fields = SemanticExtractedFields.model_validate(extracted_fields or {})
        self.settings = SimpleNamespace(max_tool_turns=max_tool_turns)

    def interpret(self, transcript, context=None):
        from app.semantic_interpreter import SemanticInterpretationResult
        return SemanticInterpretationResult(
            proposal=self._proposal,
            metrics=SemanticInterpreterMetrics(latency_ms=5),
            extracted_fields=self._extracted_fields,
        )

    def health_check(self):
        return {"status": "ok", "provider": "ollama", "model": "llama3.2:3b", "mode": "tool_calling"}


async def _parse(client, transcript, interpreter, context=None):
    app.dependency_overrides[get_semantic_interpreter] = lambda: interpreter
    try:
        resp = await client.post("/transcript/parse", json={"transcript": transcript, "context": context or {}})
    finally:
        app.dependency_overrides.pop(get_semantic_interpreter, None)
    assert resp.status_code == 200
    return resp.json()


def _create_app(db, *, company="Google", role="AI Engineer", status="applied", archived=False):
    from app.company_resolution import get_or_create_company
    company_obj = get_or_create_company(db, company)
    a = JobApplication(
        company_id=company_obj.id,
        role=role,
        normalized_role=role.lower(),
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
    if archived:
        from datetime import datetime, timezone
        a.archived_at = datetime.now(timezone.utc)
    db.add(a)
    db.commit()
    db.refresh(a)
    return a


# ===========================================================================
# Part A — Open-ended role extraction
# ===========================================================================


@pytest.mark.anyio
async def test_open_role_founding_engineer_creates_draft(client):
    """'add aiden ai application for founding engineer' → draft_created with open-ended role."""
    interpreter = FakeInterpreter(
        extracted_fields={"company": "aiden ai", "role": "founding engineer"},
        proposal=_proposal(
            "patch_active_draft",
            {"fields": {"company": "aiden ai", "role": "founding engineer"}, "replace_explicit_fields": True, "context_notes": []},
        ),
    )
    result = await _parse(client, "add aiden ai application for founding engineer", interpreter)

    assert result["status"] in {"draft_created", "draft_updated"}, f"Unexpected status: {result['status']} / {result}"
    assert result["draft"]["company"] == "aiden ai"
    assert result["draft"]["role"] == "founding engineer"


@pytest.mark.anyio
async def test_open_role_llm_inference_optimization_engineer(client):
    """Unknown long role title is preserved as-is."""
    interpreter = FakeInterpreter(
        extracted_fields={"company": "Acme", "role": "LLM Inference Optimization Engineer"},
        proposal=_proposal(
            "patch_active_draft",
            {"fields": {"company": "Acme", "role": "LLM Inference Optimization Engineer"}, "replace_explicit_fields": True, "context_notes": []},
        ),
    )
    result = await _parse(client, "add Acme application for LLM Inference Optimization Engineer", interpreter)

    assert result["status"] in {"draft_created", "draft_updated"}, result
    assert result["draft"]["role"] == "LLM Inference Optimization Engineer"


# ===========================================================================
# Part B — Separator-tolerant normalization
# ===========================================================================


@pytest.mark.parametrize("raw_status,expected", [
    ("in-touch", "in_touch"),
    ("in_touch", "in_touch"),
    ("in touch", "in_touch"),
    ("IN-TOUCH", "in_touch"),
    ("In-Touch", "in_touch"),
    ("applied", "applied"),
    ("rejected", "rejected"),
])
def test_status_separator_variants_normalize(raw_status, expected):
    assert normalize_status_value(raw_status) == expected


@pytest.mark.parametrize("raw_location,expected", [
    ("onsite", "on-site"),
    ("on site", "on-site"),
    ("on-site", "on-site"),
    ("remote", "remote"),
    ("hybrid", "hybrid"),
])
def test_location_alias_variants_normalize(raw_location, expected):
    from app.semantic_validation import _normalize_lookup_text
    token = _normalize_lookup_text(raw_location)
    assert LOCATION_ALIASES.get(token) == expected


def test_normalize_location_onsite_returns_on_site():
    assert normalize_location("onsite") == "on-site"


def test_normalize_location_on_site_hyphen_returns_on_site():
    assert normalize_location("on-site") == "on-site"


def test_normalize_location_on_site_space_returns_on_site():
    assert normalize_location("on site") == "on-site"


@pytest.mark.anyio
async def test_status_in_touch_hyphen_creates_pending_changes(client, db):
    """'update status to in-touch' → pending_changes_created with status=in_touch."""
    google_app = _create_app(db, company="Google", role="AI Engineer")

    interpreter = FakeInterpreter(
        extracted_fields={"company": "Google", "role": "AI Engineer", "status": "in-touch"},
        proposal=_proposal(
            "preview_existing_application_update",
            {
                "target": {"company": "Google", "role": "AI Engineer"},
                "fields": {"status": "in-touch"},
                "replace_explicit_fields": True,
            },
        ),
    )
    result = await _parse(client, "update status of ai engineer application at google to in-touch", interpreter)

    assert result["status"] in {"pending_changes_created", "pending_changes_updated"}, result
    assert result.get("pending_changes") is not None
    assert result["pending_changes"]["preview"]["status"] == "in_touch"


# ===========================================================================
# Part C — Generic controlled-field reconciliation
# ===========================================================================


def test_reconcile_onsite_from_employment_types_to_location():
    """'onsite' in employment_types is unambiguously a location — move it."""
    patch = SemanticFieldPatch(employment_types=["onsite"], location=None)
    reconciled = _reconcile_controlled_field_misclassification(patch)

    assert reconciled.employment_types is None
    assert reconciled.location == "on-site"


def test_reconcile_does_not_move_when_location_already_set():
    """If location is already populated, do not overwrite it."""
    patch = SemanticFieldPatch(employment_types=["onsite"], location="remote")
    reconciled = _reconcile_controlled_field_misclassification(patch)

    assert reconciled.employment_types == ["onsite"]
    assert reconciled.location == "remote"


def test_reconcile_valid_employment_type_not_moved():
    """A valid employment_type like 'fulltime' must not be moved."""
    patch = SemanticFieldPatch(employment_types=["fulltime"], location=None)
    reconciled = _reconcile_controlled_field_misclassification(patch)

    assert reconciled.employment_types == ["fulltime"]
    assert reconciled.location is None


def test_normalize_extracted_fields_onsite_as_employment_type_reconciles():
    """normalize_extracted_fields applies reconciliation before validation."""
    raw = SemanticExtractedFields(employment_types=["onsite"])
    result, warnings = normalize_extracted_fields(raw)

    assert result is not None, f"Validation failed with: {warnings}"
    assert result.location == "on-site"
    assert result.employment_types is None


@pytest.mark.anyio
async def test_onsite_misclassified_as_employment_type_creates_pending_changes(client, db):
    """LLM extracts employment_types=['onsite']; reconciliation moves it to location."""
    google_app = _create_app(db, company="Google", role="AI Engineer")

    interpreter = FakeInterpreter(
        extracted_fields={"company": "Google", "role": "AI Engineer", "employment_types": ["onsite"]},
        proposal=_proposal(
            "preview_existing_application_update",
            {
                "target": {"company": "Google", "role": "AI Engineer"},
                "fields": {"employment_types": ["onsite"]},
                "replace_explicit_fields": True,
            },
        ),
    )
    result = await _parse(client, "Google AI engineer is onsite location", interpreter)

    assert result["status"] in {"pending_changes_created", "pending_changes_updated"}, result
    assert result["pending_changes"]["preview"]["location"] == "on-site"


@pytest.mark.anyio
async def test_onsite_in_location_field_creates_pending_changes(client, db):
    """LLM extracts location='onsite' correctly; normalization produces 'on-site'."""
    google_app = _create_app(db, company="Google", role="AI Engineer")

    interpreter = FakeInterpreter(
        extracted_fields={"company": "Google", "role": "AI Engineer", "location": "onsite"},
        proposal=_proposal(
            "preview_existing_application_update",
            {
                "target": {"company": "Google", "role": "AI Engineer"},
                "fields": {"location": "onsite"},
                "replace_explicit_fields": True,
            },
        ),
    )
    result = await _parse(client, "Google AI engineer has location onsite", interpreter)

    assert result["status"] in {"pending_changes_created", "pending_changes_updated"}, result
    assert result["pending_changes"]["preview"]["location"] == "on-site"


# ===========================================================================
# Part D — Company-only target resolution
# ===========================================================================


@pytest.mark.anyio
async def test_priority_update_company_only_one_match_creates_pending_changes(client, db):
    """Company-only priority command with one matching row → pending_changes_created."""
    _create_app(db, company="Google", role="AI Engineer")

    interpreter = FakeInterpreter(
        extracted_fields={"company": "Google", "priority": "low"},
        proposal=_proposal(
            "preview_existing_application_update",
            {
                "target": {"company": "Google"},
                "fields": {"priority": "low"},
                "replace_explicit_fields": True,
            },
        ),
    )
    result = await _parse(client, "set the priority of google application to low", interpreter)

    assert result["status"] in {"pending_changes_created", "pending_changes_updated"}, result
    assert result["pending_changes"]["preview"]["priority"] == "LOW"


@pytest.mark.anyio
async def test_priority_update_company_only_multiple_matches_asks_clarification(client, db):
    """Company-only priority command with multiple matching rows → clarification."""
    _create_app(db, company="Google", role="AI Engineer")
    _create_app(db, company="Google", role="ML Engineer")

    interpreter = FakeInterpreter(
        extracted_fields={"company": "Google", "priority": "low"},
        proposal=_proposal(
            "preview_existing_application_update",
            {
                "target": {"company": "Google"},
                "fields": {"priority": "low"},
                "replace_explicit_fields": True,
            },
        ),
    )
    result = await _parse(client, "set the priority of google application to low", interpreter)

    assert result["status"] == "clarification", result
    assert result["clarification_question"] is not None


@pytest.mark.anyio
async def test_priority_update_no_match_returns_no_change(client, db):
    """Priority command for a company with no rows → informative no_change."""
    interpreter = FakeInterpreter(
        extracted_fields={"company": "NoSuchCo", "priority": "low"},
        proposal=_proposal(
            "preview_existing_application_update",
            {
                "target": {"company": "NoSuchCo"},
                "fields": {"priority": "low"},
                "replace_explicit_fields": True,
            },
        ),
    )
    result = await _parse(client, "set the priority of nosuchco application to low", interpreter)

    assert result["status"] == "no_change", result


# ===========================================================================
# Part E — Archive via chat
# ===========================================================================


@pytest.mark.anyio
async def test_archive_application_via_semantic_tool(client, db):
    """archive_application semantic tool archives a uniquely matched active row."""
    _create_app(db, company="Google", role="AI Engineer")

    interpreter = FakeInterpreter(
        extracted_fields={"company": "Google", "role": "AI Engineer"},
        proposal=_proposal(
            "archive_application",
            {"target": {"company": "Google", "role": "AI Engineer"}},
        ),
    )
    result = await _parse(client, "archieve Google application having AI Engineer role", interpreter)

    assert result["status"] == "updated", result
    with SessionLocal() as verify_db:
        apps = verify_db.query(JobApplication).all()
        assert any(a.archived_at is not None for a in apps), "Expected at least one archived application"


@pytest.mark.anyio
async def test_archive_already_archived_returns_no_change(client, db):
    """Archiving an already-archived application returns a truthful no-op."""
    _create_app(db, company="Google", role="AI Engineer", archived=True)

    interpreter = FakeInterpreter(
        extracted_fields={"company": "Google", "role": "AI Engineer"},
        proposal=_proposal(
            "archive_application",
            {"target": {"company": "Google", "role": "AI Engineer"}},
        ),
    )
    result = await _parse(client, "archive google ai engineer", interpreter)

    assert result["status"] == "no_change", result
    assert "already archived" in (result.get("message") or "").lower() or any(
        "already archived" in w.lower() for w in result.get("warnings", [])
    )


@pytest.mark.anyio
async def test_archive_no_match_returns_no_change(client, db):
    """Archive of non-existent application returns no_change with informative message."""
    interpreter = FakeInterpreter(
        extracted_fields={"company": "NoSuchCo", "role": "Engineer"},
        proposal=_proposal(
            "archive_application",
            {"target": {"company": "NoSuchCo", "role": "Engineer"}},
        ),
    )
    result = await _parse(client, "archive nosuchco engineer", interpreter)

    assert result["status"] == "no_change", result


@pytest.mark.anyio
async def test_archive_multiple_matches_asks_clarification(client, db):
    """Archive with ambiguous company+role match returns clarification."""
    _create_app(db, company="Google", role="AI Engineer")
    _create_app(db, company="Google", role="ML Engineer")

    interpreter = FakeInterpreter(
        extracted_fields={"company": "Google"},
        proposal=_proposal(
            "archive_application",
            {"target": {"company": "Google"}},
        ),
    )
    result = await _parse(client, "archive google application", interpreter)

    assert result["status"] == "clarification", result


# ===========================================================================
# Part F — Delete-policy guidance
# ===========================================================================


@pytest.mark.anyio
async def test_delete_active_application_returns_archive_first_guidance(client, db):
    """Deleting an active application returns guidance to archive first."""
    _create_app(db, company="Google", role="AI Engineer")

    interpreter = FakeInterpreter(
        extracted_fields={"company": "Google", "role": "AI Engineer"},
        proposal=_proposal(
            "explain_delete_policy",
            {"target": {"company": "Google", "role": "AI Engineer"}},
        ),
    )
    result = await _parse(client, "delete google application for AI Engineer role", interpreter)

    assert result["status"] == "clarification", result
    msg = (result.get("clarification_question") or "").lower()
    assert "archive" in msg or "archived view" in msg, f"Expected archive guidance, got: {msg}"


@pytest.mark.anyio
async def test_delete_archived_application_returns_ui_guidance(client, db):
    """Deleting an archived application returns guidance to use archived view."""
    _create_app(db, company="Google", role="AI Engineer", archived=True)

    interpreter = FakeInterpreter(
        extracted_fields={"company": "Google", "role": "AI Engineer"},
        proposal=_proposal(
            "explain_delete_policy",
            {"target": {"company": "Google", "role": "AI Engineer"}},
        ),
    )
    result = await _parse(client, "delete google ai engineer permanently", interpreter)

    assert result["status"] == "clarification", result
    msg = (result.get("clarification_question") or "").lower()
    assert "archived view" in msg or "permanently" in msg.lower(), f"Expected archived-view guidance, got: {msg}"


@pytest.mark.anyio
async def test_delete_no_match_returns_no_change(client, db):
    """Delete-policy for non-existent application returns no_change."""
    interpreter = FakeInterpreter(
        extracted_fields={"company": "NoSuchCo", "role": "Engineer"},
        proposal=_proposal(
            "explain_delete_policy",
            {"target": {"company": "NoSuchCo", "role": "Engineer"}},
        ),
    )
    result = await _parse(client, "delete nosuchco engineer", interpreter)

    assert result["status"] == "no_change", result


@pytest.mark.anyio
async def test_delete_multiple_matches_asks_clarification(client, db):
    """Delete-policy with ambiguous match returns clarification with role choices."""
    _create_app(db, company="Google", role="AI Engineer")
    _create_app(db, company="Google", role="ML Engineer")

    interpreter = FakeInterpreter(
        extracted_fields={"company": "Google"},
        proposal=_proposal(
            "explain_delete_policy",
            {"target": {"company": "Google"}},
        ),
    )
    result = await _parse(client, "delete google application", interpreter)

    assert result["status"] == "clarification", result


# ===========================================================================
# Existing behavior preservation
# ===========================================================================


@pytest.mark.anyio
async def test_draft_create_and_save_still_works(client, db):
    """Basic draft create + save is unaffected."""
    interpreter = FakeInterpreter(
        extracted_fields={"company": "TestCo", "role": "SWE"},
        proposal=_proposal(
            "patch_active_draft",
            {"fields": {"company": "TestCo", "role": "SWE"}, "replace_explicit_fields": True, "context_notes": []},
        ),
    )
    result = await _parse(client, "add TestCo application for SWE", interpreter)

    assert result["status"] in {"draft_created", "draft_updated"}, result
    assert result["draft"]["company"] == "TestCo"
    assert result["draft"]["role"] == "SWE"


@pytest.mark.anyio
async def test_pending_changes_create_still_works(client, db):
    """preview_existing_application_update still creates pending changes."""
    _create_app(db, company="Neilsoft", role="AI Engineer")

    interpreter = FakeInterpreter(
        extracted_fields={"company": "Neilsoft", "role": "AI Engineer", "status": "applied"},
        proposal=_proposal(
            "preview_existing_application_update",
            {
                "target": {"company": "Neilsoft", "role": "AI Engineer"},
                "fields": {"status": "applied"},
                "replace_explicit_fields": True,
            },
        ),
    )
    result = await _parse(client, "update neilsoft ai engineer status to applied", interpreter)

    assert result["status"] in {"pending_changes_created", "pending_changes_updated"}, result
    assert result["pending_changes"]["preview"]["status"] == "applied"
