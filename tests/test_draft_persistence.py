from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import SessionLocal
from app.main import app
from app.models import JobApplication
from app.mutation_dispatcher import dispatch
from app.mutation_schemas import ApplicationChanges, MutationPayload, MutationTarget
from app.semantic_interpreter import SemanticInterpretationResult, get_semantic_interpreter
from app.semantic_schemas import SemanticExtractedFields, SemanticInterpreterMetrics, SemanticToolCallProposal


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


def test_create_draft_creates_db_row(db):
    payload = make_payload("create_draft", changes={"company": "Neilsoft", "role": "AI Engineer"})
    result = dispatch(payload, db)

    assert result.success is True
    assert result.draft is not None
    draft_id = result.draft["id"]
    app = db.get(JobApplication, draft_id)
    assert app is not None
    assert app.is_draft is True
    assert app.company == "Neilsoft"
    assert app.draft_created_at is not None


@pytest.mark.anyio
async def test_draft_row_not_visible_in_applications_list(client, db):
    payload = make_payload("create_draft", changes={"company": "Neilsoft"})
    result = dispatch(payload, db)
    assert result.success is True

    response = await client.get("/applications")
    assert response.status_code == 200
    listed = response.json()
    assert all(rec["company"] != "Neilsoft" or rec.get("id") != result.draft["id"] for rec in listed)
    ids = [rec["id"] for rec in listed]
    assert result.draft["id"] not in ids


def test_patch_draft_updates_db_row(db):
    create_payload = make_payload("create_draft", changes={"company": "Neilsoft"})
    create_result = dispatch(create_payload, db)
    assert create_result.success is True
    draft_id = str(create_result.draft["id"])

    patch_payload = make_payload("patch_draft", changes={"priority": "HIGH"}, target={"draft_id": draft_id})
    patch_result = dispatch(patch_payload, db)
    assert patch_result.success is True

    app = db.get(JobApplication, int(draft_id))
    db.refresh(app)
    assert app.priority == "HIGH"
    assert app.is_draft is True


def test_save_draft_sets_is_draft_false(db):
    create_payload = make_payload("create_draft", changes={"company": "Neilsoft", "role": "AI Engineer"})
    create_result = dispatch(create_payload, db)
    assert create_result.success is True
    draft_id = str(create_result.draft["id"])

    save_payload = make_payload("save_draft", target={"draft_id": draft_id})
    save_result = dispatch(save_payload, db)
    assert save_result.success is True

    app = db.get(JobApplication, int(draft_id))
    db.refresh(app)
    assert app.is_draft is False
    assert app.draft_created_at is None


@pytest.mark.anyio
async def test_save_draft_row_visible_in_applications_list(client, db):
    create_payload = make_payload("create_draft", changes={"company": "DraftCo", "role": "Engineer"})
    create_result = dispatch(create_payload, db)
    draft_id = str(create_result.draft["id"])

    save_payload = make_payload("save_draft", target={"draft_id": draft_id})
    dispatch(save_payload, db)

    response = await client.get("/applications")
    assert response.status_code == 200
    ids = [rec["id"] for rec in response.json()]
    assert int(draft_id) in ids


def test_discard_draft_deletes_db_row(db):
    create_payload = make_payload("create_draft", changes={"company": "Neilsoft"})
    create_result = dispatch(create_payload, db)
    draft_id = str(create_result.draft["id"])

    discard_payload = make_payload("discard_draft", target={"draft_id": draft_id})
    discard_result = dispatch(discard_payload, db)
    assert discard_result.success is True

    app = db.get(JobApplication, int(draft_id))
    assert app is None


@pytest.mark.anyio
async def test_patch_application_rejects_draft_row(client, db):
    create_payload = make_payload("create_draft", changes={"company": "Neilsoft"})
    create_result = dispatch(create_payload, db)
    draft_id = create_result.draft["id"]

    response = await client.patch(f"/applications/{draft_id}", json={"priority": "HIGH"})
    assert response.status_code == 400
    assert "draft" in response.json()["detail"].lower()


@pytest.mark.anyio
async def test_draft_id_returned_in_transcript_response(client):
    class FakeInterpreter:
        settings = SimpleNamespace(max_tool_turns=2)

        def interpret(self, transcript, context=None):
            return SemanticInterpretationResult(
                proposal=SemanticToolCallProposal(
                    tool_name="patch_active_draft",
                    arguments={
                        "fields": {"company": "NewCo", "role": "AI Engineer"},
                        "replace_explicit_fields": True,
                        "context_notes": [],
                    },
                ),
                metrics=SemanticInterpreterMetrics(latency_ms=10),
                extracted_fields=SemanticExtractedFields(company="NewCo", role="AI Engineer"),
            )

        def health_check(self):
            return {}

    # Use controlled command syntax; LLM path is intentionally disabled.
    response = await client.post(
        "/transcript/parse",
        json={"transcript": "add application for AI Engineer at NewCo"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"draft_created", "draft_updated"}
    assert body["draft_id"] is not None
    assert isinstance(body["draft_id"], str)

    # Verify the DB row was created
    with SessionLocal() as db:
        draft_id = int(body["draft_id"])
        row = db.get(JobApplication, draft_id)
        assert row is not None
        assert row.is_draft is True
        assert row.company == "NewCo"


# ---------------------------------------------------------------------------
# Save-flow truthfulness via transcript fast-path
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_fast_path_save_returns_saved_state_not_preview_with_confirmation(client, db):
    """Fast-path 'save' dispatches once, row becomes is_draft=False, response reflects saved state."""
    # Create a draft row via dispatcher
    create_payload = make_payload("create_draft", changes={"company": "SaveFlowCo", "role": "AI Engineer"})
    create_result = dispatch(create_payload, db)
    assert create_result.success is True
    draft_id = str(create_result.draft["id"])

    # Send 'save it' through the transcript endpoint with draft_id in context
    response = await client.post(
        "/transcript/parse",
        json={
            "transcript": "save it",
            "context": {"draft_id": draft_id},
        },
    )
    assert response.status_code == 200
    body = response.json()

    # Row should be saved (is_draft=False)
    with SessionLocal() as check_db:
        row = check_db.get(JobApplication, int(draft_id))
        assert row.is_draft is False
        assert row.draft_created_at is None

    # Response must reflect saved state (no longer a draft)
    assert body["draft_id"] is None  # draft_id cleared after save
    assert body["status"] == "saved"
