import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.constants import EMPLOYMENT_TYPE_ALIASES, LOCATION_ALIASES, PRIORITY_ALIASES, STATUS_ALIASES, STATUS_OPTIONS, ALLOWED_EMPLOYMENT_TYPES, ALLOWED_LOCATIONS, ALLOWED_PRIORITIES, normalize_status_value
from app.mutation_dispatcher import _OPERATION_TO_TRANSCRIPT_OP, dispatch
from app.mutation_schemas import ALLOWED_OPERATIONS, ApplicationChanges, MutationPayload, MutationTarget
from app.semantic_interpreter import get_semantic_interpreter
from app.semantic_schemas import SemanticExtractedFields, SemanticInterpreterMetrics, SemanticToolCallProposal
from app.database import SessionLocal
from app.models import JobApplication


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client


@pytest.fixture
def db():
    with SessionLocal() as session:
        yield session


def make_payload(operation: str, changes: dict | None = None, target: dict | None = None, notes: list | None = None) -> MutationPayload:
    return MutationPayload(
        operation=operation,
        target=MutationTarget(**(target or {})),
        changes=ApplicationChanges(**(changes or {})),
        notes_to_append=notes or [],
    )


def test_dispatch_rejects_unknown_operation(db):
    payload = make_payload("delete_everything")
    result = dispatch(payload, db)
    assert result.success is False
    assert "Unknown operation" in result.message


def test_dispatch_rejects_invalid_priority_enum(db):
    payload = make_payload("create_draft", changes={"company": "Neilsoft", "priority": "ultra"})
    result = dispatch(payload, db)
    assert result.success is False
    assert "priority" in result.message.lower()


def test_dispatch_rejects_invalid_location_mode_enum(db):
    payload = make_payload("create_draft", changes={"company": "Neilsoft", "location_mode": "in-person"})
    result = dispatch(payload, db)
    assert result.success is False
    assert "location_mode" in result.message.lower()


def test_dispatch_rejects_patch_application_without_application_id(db):
    payload = make_payload("patch_application", changes={"priority": "HIGH"})
    result = dispatch(payload, db)
    assert result.success is False
    assert "application_id" in result.message.lower()


def test_dispatch_create_draft_returns_preview(db):
    payload = make_payload("create_draft", changes={"company": "Neilsoft", "roles": ["AI Engineer"]})
    result = dispatch(payload, db)
    assert result.success is True
    assert result.draft is not None
    assert result.draft["company"] == "Neilsoft"


def test_dispatch_ask_clarification_returns_question(db):
    payload = make_payload("ask_clarification", notes=["Which company should I use?"])
    result = dispatch(payload, db)
    assert result.success is True
    assert result.clarification_question == "Which company should I use?"


@pytest.mark.anyio
async def test_existing_transcript_parse_behavior_unchanged(client):
    from types import SimpleNamespace
    from app.semantic_schemas import SemanticInterpreterMetrics, SemanticToolCallProposal
    from app.semantic_interpreter import SemanticInterpretationResult

    class FakeInterpreter:
        settings = SimpleNamespace(max_tool_turns=2)

        def interpret(self, transcript, context=None):
            return SemanticInterpretationResult(
                proposal=SemanticToolCallProposal(
                    tool_name="patch_active_draft",
                    arguments={
                        "fields": {"company": "Neilsoft", "roles": ["AI Engineer"]},
                        "replace_explicit_fields": True,
                        "context_notes": [],
                    },
                ),
                metrics=SemanticInterpreterMetrics(latency_ms=10),
                extracted_fields=SemanticExtractedFields(company="Neilsoft", roles=["AI Engineer"]),
            )

        def health_check(self):
            return {}

    app.dependency_overrides[get_semantic_interpreter] = lambda: FakeInterpreter()
    try:
        response = await client.post(
            "/transcript/parse",
            json={"transcript": "Add Neilsoft AI Engineer application"},
        )
    finally:
        app.dependency_overrides.pop(get_semantic_interpreter, None)

    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "draft" in body
    assert body["status"] in {"draft_created", "draft_updated", "saved", "updated"}
    assert body["draft"]["company"] == "Neilsoft"
    assert body["draft"]["roles"] == ["AI Engineer"]


# ---------------------------------------------------------------------------
# Multi-value field preservation
# ---------------------------------------------------------------------------

def test_create_draft_preserves_multiple_roles(db):
    payload = make_payload("create_draft", changes={
        "company": "Acme",
        "roles": ["AI Engineer", "RAG Engineer"],
    })
    result = dispatch(payload, db)
    assert result.success is True
    assert result.draft["roles_json"] == ["AI Engineer", "RAG Engineer"]

    row = db.get(JobApplication, result.draft["id"])
    assert row.roles_json == ["AI Engineer", "RAG Engineer"]


def test_patch_draft_preserves_multiple_employment_types(db):
    create = dispatch(make_payload("create_draft", changes={"company": "Acme"}), db)
    draft_id = str(create.draft["id"])

    patch = dispatch(make_payload("patch_draft", changes={
        "employment_types": ["Full Time", "Part Time"],
    }, target={"draft_id": draft_id}), db)

    assert patch.success is True
    row = db.get(JobApplication, int(draft_id))
    db.refresh(row)
    assert row.employment_types_json == ["Full Time", "Part Time"]


def test_patch_application_preserves_multiple_current_stages(db):
    create = dispatch(make_payload("create_draft", changes={"company": "Acme", "roles": ["AI Engineer"]}), db)
    save = dispatch(make_payload("save_draft", target={"draft_id": str(create.draft["id"])}), db)
    app_id = save.application["id"]

    patch = dispatch(make_payload("patch_application", changes={
        "current_stages": ["Tailored", "Applied", "Networked"],
    }, target={"application_id": app_id}), db)

    assert patch.success is True
    row = db.get(JobApplication, app_id)
    db.refresh(row)
    assert row.current_stages_json == ["Tailored", "Applied", "Networked"]


# ---------------------------------------------------------------------------
# Unknown role acceptance
# ---------------------------------------------------------------------------

def test_unknown_role_is_accepted_without_whitelist_rejection(db):
    payload = make_payload("create_draft", changes={
        "company": "DeepMind",
        "roles": ["LLM Inference Optimization Engineer"],
    })
    result = dispatch(payload, db)
    assert result.success is True
    row = db.get(JobApplication, result.draft["id"])
    assert row.roles_json == ["LLM Inference Optimization Engineer"]


# ---------------------------------------------------------------------------
# Status validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("canonical", ["in_touch", "applied", "accepted", "rejected"])
def test_dispatch_accepts_canonical_status(db, canonical):
    payload = make_payload("create_draft", changes={"company": "Acme", "status": canonical})
    result = dispatch(payload, db)
    assert result.success is True
    row = db.get(JobApplication, result.draft["id"])
    assert row.status == canonical


@pytest.mark.parametrize("alias,expected", [
    ("in touch", "in_touch"),
    ("submitted application", "applied"),
    ("application sent", "applied"),
    ("selected", "accepted"),
    ("declined", "rejected"),
    ("got rejected", "rejected"),
])
def test_dispatch_normalizes_status_alias(db, alias, expected):
    payload = make_payload("create_draft", changes={"company": "Acme", "status": alias})
    result = dispatch(payload, db)
    assert result.success is True
    row = db.get(JobApplication, result.draft["id"])
    assert row.status == expected


def test_dispatch_rejects_unknown_status(db):
    payload = make_payload("create_draft", changes={"company": "Acme", "status": "recruiter screening"})
    result = dispatch(payload, db)
    assert result.success is False
    assert "status" in result.message.lower()


# ---------------------------------------------------------------------------
# Save-flow truthfulness
# ---------------------------------------------------------------------------

def test_save_draft_with_draft_id_saves_once_and_returns_application(db):
    create = dispatch(make_payload("create_draft", changes={"company": "SaveCo", "roles": ["AI Engineer"]}), db)
    draft_id = str(create.draft["id"])

    save = dispatch(make_payload("save_draft", target={"draft_id": draft_id}), db)

    assert save.success is True
    assert save.application is not None
    assert save.application["is_draft"] is False
    # Draft key should not be set on a saved application
    assert save.draft is None

    row = db.get(JobApplication, int(draft_id))
    db.refresh(row)
    assert row.is_draft is False
    assert row.draft_created_at is None


def test_save_draft_without_draft_id_returns_error(db):
    result = dispatch(make_payload("save_draft"), db)
    assert result.success is False
    assert "draft" in result.message.lower()


def test_second_save_attempt_on_already_saved_application_fails(db):
    create = dispatch(make_payload("create_draft", changes={"company": "SaveCo"}), db)
    draft_id = str(create.draft["id"])
    dispatch(make_payload("save_draft", target={"draft_id": draft_id}), db)

    # Try to save again with the same draft_id — row is no longer a draft
    second_save = dispatch(make_payload("save_draft", target={"draft_id": draft_id}), db)
    assert second_save.success is False


# ---------------------------------------------------------------------------
# Archived-row patch guard
# ---------------------------------------------------------------------------

def test_patch_application_rejects_archived_row(db):
    create = dispatch(make_payload("create_draft", changes={"company": "ArchivedCo"}), db)
    save = dispatch(make_payload("save_draft", target={"draft_id": str(create.draft["id"])}), db)
    app_id = save.application["id"]

    archive = dispatch(make_payload("archive_application", target={"application_id": app_id}), db)
    assert archive.success is True

    patch = dispatch(make_payload("patch_application", changes={"priority": "HIGH"}, target={"application_id": app_id}), db)
    assert patch.success is False
    assert "archived" in patch.message.lower()

    row = db.get(JobApplication, app_id)
    db.refresh(row)
    assert row.priority != "HIGH"


# ---------------------------------------------------------------------------
# Discard-without-draft behavior
# ---------------------------------------------------------------------------

def test_discard_without_draft_id_is_explicit_noop(db):
    result = dispatch(make_payload("discard_draft"), db)
    assert result.success is True
    assert "no active draft" in result.message.lower()


def test_discard_with_nonexistent_draft_id_returns_error(db):
    result = dispatch(make_payload("discard_draft", target={"draft_id": "999999"}), db)
    assert result.success is False


# ---------------------------------------------------------------------------
# Operation mapping completeness
# ---------------------------------------------------------------------------

def test_all_allowed_operations_have_explicit_transcript_op_mapping():
    assert set(_OPERATION_TO_TRANSCRIPT_OP.keys()) == ALLOWED_OPERATIONS


# ---------------------------------------------------------------------------
# Alias-map consistency (parametrized)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alias,canonical", list(STATUS_ALIASES.items()))
def test_status_alias_target_is_canonical(alias, canonical):
    assert canonical in STATUS_OPTIONS, f"STATUS_ALIASES[{alias!r}] = {canonical!r} is not in STATUS_OPTIONS"


@pytest.mark.parametrize("alias,canonical", list(EMPLOYMENT_TYPE_ALIASES.items()))
def test_employment_type_alias_target_is_canonical(alias, canonical):
    assert canonical in ALLOWED_EMPLOYMENT_TYPES


@pytest.mark.parametrize("alias,canonical", list(LOCATION_ALIASES.items()))
def test_location_alias_target_is_canonical(alias, canonical):
    assert canonical in ALLOWED_LOCATIONS


@pytest.mark.parametrize("alias,canonical", list(PRIORITY_ALIASES.items()))
def test_priority_alias_target_is_canonical(alias, canonical):
    assert canonical in ALLOWED_PRIORITIES


# ---------------------------------------------------------------------------
# normalize_status_value
# ---------------------------------------------------------------------------

def test_normalize_status_value_canonical_passthrough():
    for value in STATUS_OPTIONS:
        assert normalize_status_value(value) == value


def test_normalize_status_value_returns_none_for_unknown():
    assert normalize_status_value("recruiter screening") is None
    assert normalize_status_value("interviewing") is None
    assert normalize_status_value("offer pending") is None
