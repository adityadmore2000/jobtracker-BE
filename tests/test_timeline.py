import pytest
from httpx import ASGITransport, AsyncClient

from app.database import SessionLocal
from app.main import app
from app.models import ApplicationEvent, JobApplication
from app.mutation_dispatcher import dispatch
from app.mutation_schemas import ApplicationChanges, MutationPayload, MutationTarget


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


def create_saved_application(db, company: str = "TestCo", role: str = "AI Engineer") -> int:
    create_result = dispatch(
        make_payload("create_draft", changes={"company": company, "role": role}),
        db,
    )
    assert create_result.success
    draft_id = create_result.draft["id"]
    save_result = dispatch(
        make_payload("save_draft", target={"draft_id": str(draft_id)}),
        db,
    )
    assert save_result.success
    return save_result.application["id"]


@pytest.mark.anyio
async def test_save_draft_produces_application_saved_event(client, db):
    create_result = dispatch(
        make_payload("create_draft", changes={"company": "EventCo", "role": "ML Engineer"}),
        db,
    )
    draft_id = create_result.draft["id"]
    save_result = dispatch(
        make_payload("save_draft", target={"draft_id": str(draft_id)}),
        db,
    )
    app_id = save_result.application["id"]

    response = await client.get(f"/applications/{app_id}/timeline")
    assert response.status_code == 200
    timeline = response.json()["timeline"]
    event_types = [e["event_type"] for e in timeline]
    assert "application_saved" in event_types


@pytest.mark.anyio
async def test_patch_application_produces_field_changed_event(client, db):
    app_id = create_saved_application(db, company="FieldCo")
    dispatch(
        make_payload("patch_application", target={"application_id": app_id}, changes={"priority": "HIGH"}),
        db,
    )

    response = await client.get(f"/applications/{app_id}/timeline")
    assert response.status_code == 200
    timeline = response.json()["timeline"]
    field_changed = [e for e in timeline if e["event_type"] == "field_changed" and e["payload"]["field"] == "priority"]
    assert len(field_changed) == 1
    assert field_changed[0]["payload"]["to"] == "HIGH"


@pytest.mark.anyio
async def test_patch_application_status_produces_status_changed_event(client, db):
    app_id = create_saved_application(db, company="StatusCo")
    dispatch(
        make_payload("patch_application", target={"application_id": app_id}, changes={"status": "Applied"}),
        db,
    )

    response = await client.get(f"/applications/{app_id}/timeline")
    assert response.status_code == 200
    timeline = response.json()["timeline"]
    status_events = [e for e in timeline if e["event_type"] == "status_changed"]
    field_changed_status = [e for e in timeline if e["event_type"] == "field_changed" and e["payload"]["field"] == "status"]
    assert len(status_events) == 1
    assert len(field_changed_status) == 0


@pytest.mark.anyio
async def test_unchanged_field_produces_no_event(client, db):
    app_id = create_saved_application(db, company="NoCo")
    # patch with same priority (default is "")
    dispatch(
        make_payload("patch_application", target={"application_id": app_id}, changes={"priority": ""}),
        db,
    )

    response = await client.get(f"/applications/{app_id}/timeline")
    assert response.status_code == 200
    timeline = response.json()["timeline"]
    priority_events = [e for e in timeline if e.get("payload", {}).get("field") == "priority"]
    assert len(priority_events) == 0


@pytest.mark.anyio
async def test_append_note_produces_note_added_event(client, db):
    app_id = create_saved_application(db, company="NoteCo")
    dispatch(
        make_payload("append_note", target={"application_id": app_id}, notes=["Great recruiter"]),
        db,
    )

    response = await client.get(f"/applications/{app_id}/timeline")
    assert response.status_code == 200
    timeline = response.json()["timeline"]
    note_events = [e for e in timeline if e["event_type"] == "note_added"]
    assert len(note_events) == 1
    assert note_events[0]["payload"]["text"] == "Great recruiter"


@pytest.mark.anyio
async def test_multiple_field_changes_produce_multiple_events(client, db):
    app_id = create_saved_application(db, company="MultiCo")
    dispatch(
        make_payload(
            "patch_application",
            target={"application_id": app_id},
            changes={"priority": "HIGH", "status": "Applied"},
        ),
        db,
    )

    response = await client.get(f"/applications/{app_id}/timeline")
    assert response.status_code == 200
    timeline = response.json()["timeline"]
    data_events = [e for e in timeline if e["event_type"] in {"field_changed", "status_changed"}]
    assert len(data_events) == 2


@pytest.mark.anyio
async def test_get_timeline_returns_404_for_missing_application(client):
    response = await client.get("/applications/99999/timeline")
    assert response.status_code == 404


@pytest.mark.anyio
async def test_get_timeline_returns_400_for_draft(client, db):
    create_result = dispatch(
        make_payload("create_draft", changes={"company": "DraftCo", "role": "ML Engineer"}),
        db,
    )
    draft_id = create_result.draft["id"]
    response = await client.get(f"/applications/{draft_id}/timeline")
    assert response.status_code == 400


@pytest.mark.anyio
async def test_timeline_ordered_chronologically(client, db):
    app_id = create_saved_application(db, company="OrderCo")
    dispatch(
        make_payload("patch_application", target={"application_id": app_id}, changes={"priority": "LOW"}),
        db,
    )
    dispatch(
        make_payload("append_note", target={"application_id": app_id}, notes=["First note"]),
        db,
    )
    dispatch(
        make_payload("patch_application", target={"application_id": app_id}, changes={"priority": "HIGH"}),
        db,
    )

    response = await client.get(f"/applications/{app_id}/timeline")
    assert response.status_code == 200
    timeline = response.json()["timeline"]
    timestamps = [e["created_at"] for e in timeline]
    assert timestamps == sorted(timestamps)


@pytest.mark.anyio
async def test_field_change_and_note_in_same_payload_both_produce_events(client, db):
    app_id = create_saved_application(db, company="BothCo")
    dispatch(
        make_payload(
            "patch_application",
            target={"application_id": app_id},
            changes={"priority": "MEDIUM"},
            notes=["Attached note"],
        ),
        db,
    )

    response = await client.get(f"/applications/{app_id}/timeline")
    assert response.status_code == 200
    timeline = response.json()["timeline"]
    event_types = [e["event_type"] for e in timeline]
    assert "field_changed" in event_types
    assert "note_added" in event_types


@pytest.mark.anyio
async def test_events_cascade_delete_with_application(client, db):
    app_id = create_saved_application(db, company="DeleteCo")
    dispatch(
        make_payload("append_note", target={"application_id": app_id}, notes=["will be gone"]),
        db,
    )

    event_count = db.query(ApplicationEvent).filter(ApplicationEvent.application_id == app_id).count()
    assert event_count > 0

    saved_app = db.get(JobApplication, app_id)
    db.delete(saved_app)
    db.commit()

    remaining = db.query(ApplicationEvent).filter(ApplicationEvent.application_id == app_id).count()
    assert remaining == 0
