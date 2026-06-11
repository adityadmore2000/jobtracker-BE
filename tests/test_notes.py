import pytest
from httpx import ASGITransport, AsyncClient

from app.database import SessionLocal
from app.main import app
from app.models import ApplicationNote, JobApplication
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


def create_saved_application(db) -> int:
    """Create a saved (non-draft) application and return its id."""
    create_result = dispatch(
        make_payload("create_draft", changes={"company": "TestCo", "role": "AI Engineer"}),
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


def test_append_note_creates_db_row(db):
    app_id = create_saved_application(db)
    result = dispatch(
        make_payload("append_note", target={"application_id": app_id}, notes=["Recruiter called"]),
        db,
    )
    assert result.success
    note = db.query(ApplicationNote).filter(ApplicationNote.application_id == app_id).first()
    assert note is not None
    assert note.text == "Recruiter called"


def test_append_multiple_notes_in_one_payload(db):
    app_id = create_saved_application(db)
    result = dispatch(
        make_payload("append_note", target={"application_id": app_id}, notes=["Note A", "Note B", "Note C"]),
        db,
    )
    assert result.success
    notes = db.query(ApplicationNote).filter(ApplicationNote.application_id == app_id).all()
    assert len(notes) == 3
    texts = {n.text for n in notes}
    assert texts == {"Note A", "Note B", "Note C"}


@pytest.mark.anyio
async def test_notes_returned_in_chronological_order(client, db):
    app_id = create_saved_application(db)
    dispatch(make_payload("append_note", target={"application_id": app_id}, notes=["First"]), db)
    dispatch(make_payload("append_note", target={"application_id": app_id}, notes=["Second"]), db)
    dispatch(make_payload("append_note", target={"application_id": app_id}, notes=["Third"]), db)

    response = await client.get(f"/applications/{app_id}/notes")
    assert response.status_code == 200
    data = response.json()
    texts = [n["text"] for n in data["notes"]]
    assert texts == ["First", "Second", "Third"]


@pytest.mark.anyio
async def test_get_notes_returns_404_for_missing_application(client):
    response = await client.get("/applications/99999/notes")
    assert response.status_code == 404


@pytest.mark.anyio
async def test_get_notes_returns_400_for_draft(client, db):
    create_result = dispatch(
        make_payload("create_draft", changes={"company": "DraftCo", "role": "ML Engineer"}),
        db,
    )
    draft_id = create_result.draft["id"]
    response = await client.get(f"/applications/{draft_id}/notes")
    assert response.status_code == 400


def test_append_note_rejects_empty_notes_list(db):
    app_id = create_saved_application(db)
    result = dispatch(
        make_payload("append_note", target={"application_id": app_id}, notes=[]),
        db,
    )
    assert result.success is False


def test_append_note_on_draft_is_ignored(db):
    """patch_draft with notes_to_append logs a warning and ignores the notes."""
    create_result = dispatch(
        make_payload("create_draft", changes={"company": "DraftCo", "role": "ML Engineer"}),
        db,
    )
    draft_id = create_result.draft["id"]
    patch_result = dispatch(
        make_payload("patch_draft", target={"draft_id": str(draft_id)}, notes=["Should be ignored"]),
        db,
    )
    assert patch_result.success
    count = db.query(ApplicationNote).filter(ApplicationNote.application_id == draft_id).count()
    assert count == 0


def test_save_draft_with_notes_appends_notes_atomically(db):
    create_result = dispatch(
        make_payload("create_draft", changes={"company": "NotesCo", "role": "AI Engineer"}),
        db,
    )
    draft_id = create_result.draft["id"]
    save_result = dispatch(
        make_payload("save_draft", target={"draft_id": str(draft_id)}, notes=["Saved with note"]),
        db,
    )
    assert save_result.success
    saved_app = db.get(JobApplication, int(draft_id))
    assert saved_app.is_draft is False
    note_count = db.query(ApplicationNote).filter(ApplicationNote.application_id == draft_id).count()
    assert note_count == 1


def test_patch_application_with_notes_is_atomic(db):
    app_id = create_saved_application(db)
    result = dispatch(
        make_payload(
            "patch_application",
            target={"application_id": app_id},
            changes={"priority": "HIGH"},
            notes=["Priority bumped"],
        ),
        db,
    )
    assert result.success
    saved_app = db.get(JobApplication, app_id)
    assert saved_app.priority == "HIGH"
    note_count = db.query(ApplicationNote).filter(ApplicationNote.application_id == app_id).count()
    assert note_count == 1


@pytest.mark.anyio
async def test_notes_cascade_delete_with_application(client, db):
    app_id = create_saved_application(db)
    dispatch(make_payload("append_note", target={"application_id": app_id}, notes=["Will be deleted"]), db)

    note_count = db.query(ApplicationNote).filter(ApplicationNote.application_id == app_id).count()
    assert note_count == 1

    saved_app = db.get(JobApplication, app_id)
    db.delete(saved_app)
    db.commit()

    remaining = db.query(ApplicationNote).filter(ApplicationNote.application_id == app_id).count()
    assert remaining == 0
