import pytest
from httpx import ASGITransport, AsyncClient

from app.database import SessionLocal
from app.main import app
from app.models import ApplicationEvent, JobApplication
from app.mutation_dispatcher import dispatch
from app.mutation_schemas import ApplicationChanges, MutationPayload, MutationTarget
from app.fast_path_parser import try_parse


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


def test_archive_application_sets_archived_at(db):
    app_id = create_saved_application(db)
    result = dispatch(make_payload("archive_application", target={"application_id": app_id}), db)
    assert result.success
    saved_app = db.get(JobApplication, app_id)
    assert saved_app.archived_at is not None


@pytest.mark.anyio
async def test_archived_application_not_in_default_list(client, db):
    app_id = create_saved_application(db, company="ArchivedCo")
    dispatch(make_payload("archive_application", target={"application_id": app_id}), db)

    response = await client.get("/applications")
    assert response.status_code == 200
    ids = [a["id"] for a in response.json()]
    assert app_id not in ids


@pytest.mark.anyio
async def test_archived_application_in_archived_list(client, db):
    app_id = create_saved_application(db, company="InArchivedListCo")
    dispatch(make_payload("archive_application", target={"application_id": app_id}), db)

    response = await client.get("/applications/archived")
    assert response.status_code == 200
    ids = [a["id"] for a in response.json()]
    assert app_id in ids


def test_restore_application_clears_archived_at(db):
    app_id = create_saved_application(db)
    dispatch(make_payload("archive_application", target={"application_id": app_id}), db)
    result = dispatch(make_payload("restore_application", target={"application_id": app_id}), db)
    assert result.success
    saved_app = db.get(JobApplication, app_id)
    assert saved_app.archived_at is None


@pytest.mark.anyio
async def test_restored_application_in_default_list(client, db):
    app_id = create_saved_application(db, company="RestoredCo")
    dispatch(make_payload("archive_application", target={"application_id": app_id}), db)
    dispatch(make_payload("restore_application", target={"application_id": app_id}), db)

    response = await client.get("/applications")
    assert response.status_code == 200
    ids = [a["id"] for a in response.json()]
    assert app_id in ids


def test_archive_already_archived_returns_error(db):
    app_id = create_saved_application(db)
    dispatch(make_payload("archive_application", target={"application_id": app_id}), db)
    result = dispatch(make_payload("archive_application", target={"application_id": app_id}), db)
    assert result.success is False
    assert "already archived" in result.message.lower()


def test_restore_non_archived_returns_error(db):
    app_id = create_saved_application(db)
    result = dispatch(make_payload("restore_application", target={"application_id": app_id}), db)
    assert result.success is False
    assert "not archived" in result.message.lower()


def test_archive_draft_returns_error(db):
    create_result = dispatch(
        make_payload("create_draft", changes={"company": "DraftCo", "role": "ML Engineer"}),
        db,
    )
    draft_id = create_result.draft["id"]
    result = dispatch(make_payload("archive_application", target={"application_id": draft_id}), db)
    assert result.success is False


@pytest.mark.anyio
async def test_delete_endpoint_returns_confirmation_prompt(client, db):
    app_id = create_saved_application(db, company="PromptCo")

    response = await client.delete(f"/applications/{app_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["requires_confirmation"] is True
    assert body["confirmation_kind"] == "archive"
    assert body["application_id"] == app_id

    # Row must NOT be deleted
    saved_app = db.get(JobApplication, app_id)
    assert saved_app is not None


@pytest.mark.anyio
async def test_post_archive_endpoint_archives_application(client, db):
    app_id = create_saved_application(db, company="ArchiveEndpointCo")

    response = await client.post(f"/applications/{app_id}/archive")
    assert response.status_code == 200

    saved_app = db.get(JobApplication, app_id)
    db.refresh(saved_app)
    assert saved_app.archived_at is not None


@pytest.mark.anyio
async def test_post_restore_endpoint_restores_application(client, db):
    app_id = create_saved_application(db, company="RestoreEndpointCo")
    await client.post(f"/applications/{app_id}/archive")

    response = await client.post(f"/applications/{app_id}/restore")
    assert response.status_code == 200

    saved_app = db.get(JobApplication, app_id)
    db.refresh(saved_app)
    assert saved_app.archived_at is None


@pytest.mark.anyio
async def test_archive_produces_archived_event(client, db):
    app_id = create_saved_application(db, company="ArchiveEventCo")
    dispatch(make_payload("archive_application", target={"application_id": app_id}), db)

    response = await client.get(f"/applications/{app_id}/timeline")
    assert response.status_code == 200
    event_types = [e["event_type"] for e in response.json()["timeline"]]
    assert "application_archived" in event_types


@pytest.mark.anyio
async def test_restore_produces_restored_event(client, db):
    app_id = create_saved_application(db, company="RestoreEventCo")
    dispatch(make_payload("archive_application", target={"application_id": app_id}), db)
    dispatch(make_payload("restore_application", target={"application_id": app_id}), db)

    response = await client.get(f"/applications/{app_id}/timeline")
    assert response.status_code == 200
    event_types = [e["event_type"] for e in response.json()["timeline"]]
    assert "application_restored" in event_types


def test_fast_path_archive_by_company_name(db):
    app_id = create_saved_application(db, company="Neilsoft")
    context = {
        "applications": [
            {"id": app_id, "company": "Neilsoft", "archived_at": None}
        ]
    }
    result = try_parse("archive Neilsoft", context)
    assert result is not None
    assert result.operation == "archive_application"
    assert result.target.application_id == app_id


def test_fast_path_archive_ambiguous_company_returns_none(db):
    app_id_1 = create_saved_application(db, company="Rockwell", role="AI Engineer")
    app_id_2 = create_saved_application(db, company="Rockwell", role="ML Engineer")
    context = {
        "applications": [
            {"id": app_id_1, "company": "Rockwell", "archived_at": None},
            {"id": app_id_2, "company": "Rockwell", "archived_at": None},
        ]
    }
    result = try_parse("archive Rockwell", context)
    assert result is None
