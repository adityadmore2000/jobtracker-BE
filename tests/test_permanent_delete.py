"""Tests for Phase 1.5B: permanent deletion of archived saved applications."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import SessionLocal
from app.main import app
from app.models import ApplicationEvent, ApplicationNote, Company, JobApplication
from app.mutation_dispatcher import dispatch
from app.mutation_schemas import ApplicationChanges, MutationPayload, MutationTarget


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


def _create_and_save(db, company: str, role: str) -> int:
    r = dispatch(
        MutationPayload(
            operation="create_draft",
            target=MutationTarget(),
            changes=ApplicationChanges(company=company, role=role),
        ),
        db,
    )
    assert r.success
    rs = dispatch(
        MutationPayload(
            operation="save_draft",
            target=MutationTarget(draft_id=str(r.draft["id"])),
            changes=ApplicationChanges(),
        ),
        db,
    )
    assert rs.success
    return rs.application["id"]


def _archive(db, app_id: int) -> None:
    r = dispatch(
        MutationPayload(
            operation="archive_application",
            target=MutationTarget(application_id=app_id),
            changes=ApplicationChanges(),
        ),
        db,
    )
    assert r.success


# ---------------------------------------------------------------------------
# Dispatcher unit tests
# ---------------------------------------------------------------------------


def test_delete_archived_row_succeeds(db):
    app_id = _create_and_save(db, "Rockwell", "AI Engineer")
    _archive(db, app_id)

    result = dispatch(
        MutationPayload(
            operation="delete_application_permanently",
            target=MutationTarget(application_id=app_id),
            changes=ApplicationChanges(),
        ),
        db,
    )
    assert result.success
    assert result.message == "Application permanently deleted."
    assert db.get(JobApplication, app_id) is None


def test_delete_active_row_rejected(db):
    app_id = _create_and_save(db, "Rockwell", "AI Engineer")

    result = dispatch(
        MutationPayload(
            operation="delete_application_permanently",
            target=MutationTarget(application_id=app_id),
            changes=ApplicationChanges(),
        ),
        db,
    )
    assert not result.success
    assert "Only archived" in result.message
    assert db.get(JobApplication, app_id) is not None


def test_delete_draft_rejected(db):
    r = dispatch(
        MutationPayload(
            operation="create_draft",
            target=MutationTarget(),
            changes=ApplicationChanges(company="Rockwell", role="AI Engineer"),
        ),
        db,
    )
    assert r.success
    draft_id = r.draft["id"]

    result = dispatch(
        MutationPayload(
            operation="delete_application_permanently",
            target=MutationTarget(application_id=draft_id),
            changes=ApplicationChanges(),
        ),
        db,
    )
    assert not result.success
    assert "Drafts" in result.message
    assert db.get(JobApplication, draft_id) is not None


def test_delete_missing_id_returns_error(db):
    result = dispatch(
        MutationPayload(
            operation="delete_application_permanently",
            target=MutationTarget(application_id=999999),
            changes=ApplicationChanges(),
        ),
        db,
    )
    assert not result.success
    assert "not found" in result.message.lower()


def test_delete_removes_notes(db):
    app_id = _create_and_save(db, "Rockwell", "AI Engineer")

    # Append a note.
    dispatch(
        MutationPayload(
            operation="append_note",
            target=MutationTarget(application_id=app_id),
            changes=ApplicationChanges(),
            notes_to_append=["Referral from John"],
        ),
        db,
    )
    assert db.query(ApplicationNote).filter(ApplicationNote.application_id == app_id).count() == 1

    _archive(db, app_id)
    dispatch(
        MutationPayload(
            operation="delete_application_permanently",
            target=MutationTarget(application_id=app_id),
            changes=ApplicationChanges(),
        ),
        db,
    )

    assert db.query(ApplicationNote).filter(ApplicationNote.application_id == app_id).count() == 0


def test_delete_removes_events(db):
    app_id = _create_and_save(db, "Rockwell", "AI Engineer")
    _archive(db, app_id)

    events_before = db.query(ApplicationEvent).filter(ApplicationEvent.application_id == app_id).count()
    assert events_before > 0

    dispatch(
        MutationPayload(
            operation="delete_application_permanently",
            target=MutationTarget(application_id=app_id),
            changes=ApplicationChanges(),
        ),
        db,
    )

    assert db.query(ApplicationEvent).filter(ApplicationEvent.application_id == app_id).count() == 0


def test_delete_preserves_company_row(db):
    app_id = _create_and_save(db, "Rockwell", "AI Engineer")
    row = db.get(JobApplication, app_id)
    company_id = row.company_id
    _archive(db, app_id)

    dispatch(
        MutationPayload(
            operation="delete_application_permanently",
            target=MutationTarget(application_id=app_id),
            changes=ApplicationChanges(),
        ),
        db,
    )

    assert db.get(Company, company_id) is not None


def test_delete_one_application_leaves_sibling_intact(db):
    ai_id = _create_and_save(db, "Rockwell", "AI Engineer")
    get_id = _create_and_save(db, "Rockwell", "GET")
    _archive(db, ai_id)

    dispatch(
        MutationPayload(
            operation="delete_application_permanently",
            target=MutationTarget(application_id=ai_id),
            changes=ApplicationChanges(),
        ),
        db,
    )

    assert db.get(JobApplication, ai_id) is None
    assert db.get(JobApplication, get_id) is not None

    # Company row still exists.
    row = db.get(JobApplication, get_id)
    assert db.get(Company, row.company_id) is not None


# ---------------------------------------------------------------------------
# HTTP endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_delete_endpoint_archived_row_returns_204(client, db):
    app_id = _create_and_save(db, "Rockwell", "AI Engineer")
    _archive(db, app_id)

    response = await client.delete(f"/applications/{app_id}")
    assert response.status_code == 204


@pytest.mark.anyio
async def test_delete_endpoint_row_removed_from_db(client, db):
    app_id = _create_and_save(db, "Rockwell", "AI Engineer")
    _archive(db, app_id)

    await client.delete(f"/applications/{app_id}")

    row = db.get(JobApplication, app_id)
    assert row is None


@pytest.mark.anyio
async def test_delete_endpoint_active_row_returns_400(client, db):
    app_id = _create_and_save(db, "Rockwell", "AI Engineer")

    response = await client.delete(f"/applications/{app_id}")
    assert response.status_code == 400
    assert "Only archived" in response.json()["detail"]


@pytest.mark.anyio
async def test_delete_endpoint_draft_returns_400(client, db):
    r = dispatch(
        MutationPayload(
            operation="create_draft",
            target=MutationTarget(),
            changes=ApplicationChanges(company="Rockwell", role="AI Engineer"),
        ),
        db,
    )
    assert r.success
    draft_id = r.draft["id"]

    response = await client.delete(f"/applications/{draft_id}")
    assert response.status_code == 400
    assert "Drafts" in response.json()["detail"]


@pytest.mark.anyio
async def test_delete_endpoint_missing_returns_404(client):
    response = await client.delete("/applications/999999")
    assert response.status_code == 404
