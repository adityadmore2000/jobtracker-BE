import pytest
from httpx import ASGITransport, AsyncClient

from app.database import SessionLocal
from app.main import app
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


def _create_draft(db, company: str = "PatchCo", role: str | None = None) -> int:
    payload = MutationPayload(
        operation="create_draft",
        target=MutationTarget(),
        changes=ApplicationChanges(company=company, role=role or "AI Engineer"),
    )
    result = dispatch(payload, db)
    assert result.success
    return result.draft["id"]


@pytest.mark.anyio
async def test_patch_draft_returns_public_dto(client, db):
    draft_id = _create_draft(db)
    response = await client.patch(f"/drafts/{draft_id}", json={"priority": "HIGH"})
    assert response.status_code == 200
    data = response.json()
    assert data["priority"] == "HIGH"
    assert data["is_draft"] is True
    assert data["id"] == draft_id


@pytest.mark.anyio
async def test_patch_draft_persists_to_db(client, db):
    from app.models import JobApplication
    draft_id = _create_draft(db, company="Neilsoft")
    response = await client.patch(f"/drafts/{draft_id}", json={"priority": "MEDIUM", "location": "remote"})
    assert response.status_code == 200
    row = db.get(JobApplication, draft_id)
    db.refresh(row)
    assert row.priority == "MEDIUM"
    assert row.location == "remote"
    assert row.is_draft is True


@pytest.mark.anyio
async def test_patch_draft_role_preserved(client, db):
    draft_id = _create_draft(db)
    response = await client.patch(f"/drafts/{draft_id}", json={"role": "RAG Engineer"})
    assert response.status_code == 200
    data = response.json()
    assert data["role"] == "RAG Engineer"


@pytest.mark.anyio
async def test_patch_draft_unknown_role_accepted(client, db):
    draft_id = _create_draft(db)
    response = await client.patch(f"/drafts/{draft_id}", json={"role": "LLM Inference Optimization Engineer"})
    assert response.status_code == 200
    assert response.json()["role"] == "LLM Inference Optimization Engineer"


@pytest.mark.anyio
async def test_patch_draft_invalid_status_rejected(client, db):
    draft_id = _create_draft(db)
    response = await client.patch(f"/drafts/{draft_id}", json={"status": "nonexistent_status"})
    assert response.status_code == 422


@pytest.mark.anyio
async def test_patch_draft_open_fields(client, db):
    draft_id = _create_draft(db)
    response = await client.patch(
        f"/drafts/{draft_id}",
        json={"next_action": "Follow up Monday", "comments": "Referral pending", "engaged_days": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["next_action"] == "Follow up Monday"
    assert data["comments"] == "Referral pending"
    assert data["engaged_days"] == 3


@pytest.mark.anyio
async def test_patch_missing_draft_returns_404(client, db):
    response = await client.patch("/drafts/999999", json={"priority": "HIGH"})
    assert response.status_code == 404


@pytest.mark.anyio
async def test_patch_draft_endpoint_rejects_saved_application(client, db):
    # Create a draft then save it — patching via /drafts/{id} should reject
    from app.models import JobApplication
    draft_id = _create_draft(db, company="SavedCo")
    save_payload = MutationPayload(
        operation="save_draft",
        target=MutationTarget(draft_id=str(draft_id)),
        changes=ApplicationChanges(),
    )
    dispatch(save_payload, db)
    row = db.get(JobApplication, draft_id)
    db.refresh(row)
    assert row.is_draft is False

    response = await client.patch(f"/drafts/{draft_id}", json={"priority": "HIGH"})
    assert response.status_code == 404


@pytest.mark.anyio
async def test_save_draft_via_endpoint_promotes_to_application(client, db):
    draft_id = _create_draft(db, company="SaveEndpointCo")
    response = await client.post(f"/drafts/{draft_id}/save")
    assert response.status_code == 200
    data = response.json()
    assert data["is_draft"] is False
    assert data["company"] == "SaveEndpointCo"

    # Verify visible in applications list
    list_response = await client.get("/applications")
    ids = [r["id"] for r in list_response.json()]
    assert draft_id in ids


@pytest.mark.anyio
async def test_discard_draft_via_endpoint_deletes_row(client, db):
    from app.models import JobApplication
    draft_id = _create_draft(db, company="DiscardEndpointCo")
    response = await client.post(f"/drafts/{draft_id}/discard")
    assert response.status_code == 204

    row = db.get(JobApplication, draft_id)
    assert row is None


@pytest.mark.anyio
async def test_save_missing_draft_returns_404(client, db):
    response = await client.post("/drafts/999999/save")
    assert response.status_code == 404


@pytest.mark.anyio
async def test_discard_missing_draft_returns_404(client, db):
    response = await client.post("/drafts/999999/discard")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Draft PATCH collision tests (Phase 1.5B)
# ---------------------------------------------------------------------------


def _create_saved_application(db, company: str, role: str) -> int:
    """Create and promote a draft → saved application, returning its id."""
    create = MutationPayload(
        operation="create_draft",
        target=MutationTarget(),
        changes=ApplicationChanges(company=company, role=role),
    )
    r = dispatch(create, db)
    assert r.success
    draft_id = str(r.draft["id"])
    save = MutationPayload(
        operation="save_draft",
        target=MutationTarget(draft_id=draft_id),
        changes=ApplicationChanges(),
    )
    rs = dispatch(save, db)
    assert rs.success
    return rs.application["id"]


@pytest.mark.anyio
async def test_patch_draft_role_collision_returns_409(client, db):
    """Draft role changed to match an existing saved row → HTTP 409."""
    _create_saved_application(db, "Rockwell", "AI Engineer")
    draft_id = _create_draft(db, company="Rockwell", role="GET")

    response = await client.patch(f"/drafts/{draft_id}", json={"role": "AI Engineer"})
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "Rockwell" in detail
    assert "AI Engineer" in detail


@pytest.mark.anyio
async def test_patch_draft_role_collision_draft_unchanged(client, db):
    """After a 409 the draft row is not mutated."""
    from app.models import JobApplication

    _create_saved_application(db, "Rockwell", "AI Engineer")
    draft_id = _create_draft(db, company="Rockwell", role="GET")

    await client.patch(f"/drafts/{draft_id}", json={"role": "AI Engineer"})

    row = db.get(JobApplication, draft_id)
    db.refresh(row)
    assert row.role == "GET"


@pytest.mark.anyio
async def test_patch_draft_company_collision_returns_409(client, db):
    """Draft company changed so that company+role matches an existing saved row → HTTP 409."""
    _create_saved_application(db, "Neilsoft", "AI Engineer")
    draft_id = _create_draft(db, company="Rockwell", role="AI Engineer")

    response = await client.patch(f"/drafts/{draft_id}", json={"company": "Neilsoft"})
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "Neilsoft" in detail


@pytest.mark.anyio
async def test_patch_draft_role_no_collision_succeeds(client, db):
    """Changing a draft role to one that does not collide succeeds."""
    _create_saved_application(db, "Rockwell", "AI Engineer")
    draft_id = _create_draft(db, company="Rockwell", role="GET")

    response = await client.patch(f"/drafts/{draft_id}", json={"role": "Graduate Engineer Trainee"})
    assert response.status_code == 200
    assert response.json()["role"] == "Graduate Engineer Trainee"


@pytest.mark.anyio
async def test_patch_draft_self_collision_not_triggered(client, db):
    """Patching a draft with the same (or normalized-same) role it already has never collides with itself."""
    draft_id = _create_draft(db, company="Rockwell", role="AI Engineer")

    response = await client.patch(f"/drafts/{draft_id}", json={"role": "ai engineer"})
    assert response.status_code == 200
