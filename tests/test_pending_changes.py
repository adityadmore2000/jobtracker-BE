"""Tests for Part B: Pending Changes workflow."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import SessionLocal
from app.main import app
from app.models import ApplicationChangeDraft, JobApplication
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

BASE_APP = {
    "company": "Neilsoft",
    "role": "AI Engineer",
    "employment_types_json": [],
    "job_link": "",
    "location": "",
    "status": "applied",
    "current_stages_json": [],
    "priority": "MEDIUM",
    "engaged_days": 0,
    "next_action": "",
    "comments": "",
}


async def _create_saved(client, company="Neilsoft", role="AI Engineer", status="applied", priority="MEDIUM") -> dict:
    resp = await client.post("/applications", json={
        **BASE_APP,
        "company": company,
        "role": role,
        "status": status,
        "priority": priority,
    })
    assert resp.status_code == 201, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# Dispatcher: create_application_update_draft
# ---------------------------------------------------------------------------


def test_create_update_draft_creates_pending_change(db):
    from app.company_resolution import get_or_create_company
    from app.role_resolution import normalize_role_name
    from datetime import datetime, timezone

    company = get_or_create_company(db, "Neilsoft")
    db.commit()
    app_row = JobApplication(
        company_id=company.id,
        role="AI Engineer",
        normalized_role=normalize_role_name("AI Engineer"),
        employment_types_json=[],
        job_link="",
        location="",
        status="applied",
        current_stages_json=[],
        priority="MEDIUM",
        engaged_days=0,
        next_action="",
        comments="",
        is_draft=False,
    )
    db.add(app_row)
    db.commit()
    db.refresh(app_row)

    result = dispatch(MutationPayload(
        operation="create_application_update_draft",
        target=MutationTarget(application_id=app_row.id),
        changes=ApplicationChanges(priority="HIGH"),
    ), db)

    assert result.success
    assert result.operation == "create_application_update_draft"
    assert result.change_draft is not None
    assert result.change_draft["changes_json"]["priority"] == "HIGH"
    assert result.change_draft["changed_fields"] == ["priority"]
    # Original saved row must be unchanged
    db.refresh(app_row)
    assert app_row.priority == "MEDIUM"


def test_second_update_same_app_patches_existing_draft(db):
    from app.company_resolution import get_or_create_company
    from app.role_resolution import normalize_role_name

    company = get_or_create_company(db, "Neilsoft")
    db.commit()
    app_row = JobApplication(
        company_id=company.id,
        role="AI Engineer",
        normalized_role=normalize_role_name("AI Engineer"),
        employment_types_json=[],
        job_link="",
        location="",
        status="applied",
        current_stages_json=[],
        priority="MEDIUM",
        engaged_days=0,
        next_action="",
        comments="",
        is_draft=False,
    )
    db.add(app_row)
    db.commit()
    db.refresh(app_row)

    # First update: priority HIGH
    r1 = dispatch(MutationPayload(
        operation="create_application_update_draft",
        target=MutationTarget(application_id=app_row.id),
        changes=ApplicationChanges(priority="HIGH"),
    ), db)
    assert r1.success
    cd_id = r1.change_draft["id"]

    # Second update: status rejected
    r2 = dispatch(MutationPayload(
        operation="create_application_update_draft",
        target=MutationTarget(application_id=app_row.id),
        changes=ApplicationChanges(status="rejected"),
    ), db)
    assert r2.success
    assert r2.operation == "patch_application_update_draft"
    assert r2.change_draft["id"] == cd_id  # same draft
    assert "priority" in r2.change_draft["changes_json"]
    assert "status" in r2.change_draft["changes_json"]
    assert sorted(r2.change_draft["changed_fields"]) == ["priority", "status"]

    # Saved row still unchanged
    db.refresh(app_row)
    assert app_row.priority == "MEDIUM"
    assert app_row.status == "applied"


def test_apply_update_draft_mutates_saved_row(db):
    from app.company_resolution import get_or_create_company
    from app.role_resolution import normalize_role_name

    company = get_or_create_company(db, "Neilsoft")
    db.commit()
    app_row = JobApplication(
        company_id=company.id,
        role="AI Engineer",
        normalized_role=normalize_role_name("AI Engineer"),
        employment_types_json=[],
        job_link="",
        location="",
        status="applied",
        current_stages_json=[],
        priority="MEDIUM",
        engaged_days=0,
        next_action="",
        comments="",
        is_draft=False,
    )
    db.add(app_row)
    db.commit()
    db.refresh(app_row)

    r_create = dispatch(MutationPayload(
        operation="create_application_update_draft",
        target=MutationTarget(application_id=app_row.id),
        changes=ApplicationChanges(priority="HIGH"),
    ), db)
    cd_id = r_create.change_draft["id"]

    r_apply = dispatch(MutationPayload(
        operation="apply_application_update_draft",
        target=MutationTarget(change_draft_id=cd_id),
        changes=ApplicationChanges(),
    ), db)

    assert r_apply.success
    assert r_apply.operation == "apply_application_update_draft"
    db.refresh(app_row)
    assert app_row.priority == "HIGH"

    # Change draft must be deleted
    cd = db.get(ApplicationChangeDraft, cd_id)
    assert cd is None


def test_discard_update_draft_leaves_row_unchanged(db):
    from app.company_resolution import get_or_create_company
    from app.role_resolution import normalize_role_name

    company = get_or_create_company(db, "Neilsoft")
    db.commit()
    app_row = JobApplication(
        company_id=company.id,
        role="AI Engineer",
        normalized_role=normalize_role_name("AI Engineer"),
        employment_types_json=[],
        job_link="",
        location="",
        status="applied",
        current_stages_json=[],
        priority="MEDIUM",
        engaged_days=0,
        next_action="",
        comments="",
        is_draft=False,
    )
    db.add(app_row)
    db.commit()
    db.refresh(app_row)

    r_create = dispatch(MutationPayload(
        operation="create_application_update_draft",
        target=MutationTarget(application_id=app_row.id),
        changes=ApplicationChanges(priority="HIGH"),
    ), db)
    cd_id = r_create.change_draft["id"]

    r_discard = dispatch(MutationPayload(
        operation="discard_application_update_draft",
        target=MutationTarget(change_draft_id=cd_id),
        changes=ApplicationChanges(),
    ), db)

    assert r_discard.success
    db.refresh(app_row)
    assert app_row.priority == "MEDIUM"
    assert db.get(ApplicationChangeDraft, cd_id) is None


def test_apply_collision_returns_409(db):
    from app.company_resolution import get_or_create_company
    from app.role_resolution import normalize_role_name

    company = get_or_create_company(db, "Rockwell")
    db.commit()

    app1 = JobApplication(
        company_id=company.id,
        role="AI Engineer",
        normalized_role=normalize_role_name("AI Engineer"),
        employment_types_json=[],
        job_link="",
        location="",
        status="applied",
        current_stages_json=[],
        priority="",
        engaged_days=0,
        next_action="",
        comments="",
        is_draft=False,
    )
    app2 = JobApplication(
        company_id=company.id,
        role="GET",
        normalized_role=normalize_role_name("GET"),
        employment_types_json=[],
        job_link="",
        location="",
        status="applied",
        current_stages_json=[],
        priority="",
        engaged_days=0,
        next_action="",
        comments="",
        is_draft=False,
    )
    db.add_all([app1, app2])
    db.commit()
    db.refresh(app1)
    db.refresh(app2)

    r_create = dispatch(MutationPayload(
        operation="create_application_update_draft",
        target=MutationTarget(application_id=app2.id),
        changes=ApplicationChanges(role="AI Engineer"),
    ), db)
    cd_id = r_create.change_draft["id"]

    r_apply = dispatch(MutationPayload(
        operation="apply_application_update_draft",
        target=MutationTarget(change_draft_id=cd_id),
        changes=ApplicationChanges(),
    ), db)

    assert not r_apply.success
    assert r_apply.conflict
    # Draft must still exist
    assert db.get(ApplicationChangeDraft, cd_id) is not None
    # Rows unchanged
    db.refresh(app2)
    assert app2.role == "GET"


def test_apply_archived_target_rejected(db):
    from app.company_resolution import get_or_create_company
    from app.role_resolution import normalize_role_name
    from datetime import datetime, timezone

    company = get_or_create_company(db, "Neilsoft")
    db.commit()
    app_row = JobApplication(
        company_id=company.id,
        role="AI Engineer",
        normalized_role=normalize_role_name("AI Engineer"),
        employment_types_json=[],
        job_link="",
        location="",
        status="applied",
        current_stages_json=[],
        priority="MEDIUM",
        engaged_days=0,
        next_action="",
        comments="",
        is_draft=False,
    )
    db.add(app_row)
    db.commit()
    db.refresh(app_row)

    r_create = dispatch(MutationPayload(
        operation="create_application_update_draft",
        target=MutationTarget(application_id=app_row.id),
        changes=ApplicationChanges(priority="HIGH"),
    ), db)
    cd_id = r_create.change_draft["id"]

    # Archive the application
    app_row.archived_at = datetime.now(timezone.utc)
    db.commit()

    r_apply = dispatch(MutationPayload(
        operation="apply_application_update_draft",
        target=MutationTarget(change_draft_id=cd_id),
        changes=ApplicationChanges(),
    ), db)

    assert not r_apply.success
    assert r_apply.conflict
    assert db.get(ApplicationChangeDraft, cd_id) is not None


def test_unknown_status_rejected_in_update_draft(db):
    from app.company_resolution import get_or_create_company
    from app.role_resolution import normalize_role_name

    company = get_or_create_company(db, "Neilsoft")
    db.commit()
    app_row = JobApplication(
        company_id=company.id,
        role="AI Engineer",
        normalized_role=normalize_role_name("AI Engineer"),
        employment_types_json=[],
        job_link="",
        location="",
        status="applied",
        current_stages_json=[],
        priority="MEDIUM",
        engaged_days=0,
        next_action="",
        comments="",
        is_draft=False,
    )
    db.add(app_row)
    db.commit()
    db.refresh(app_row)

    result = dispatch(MutationPayload(
        operation="create_application_update_draft",
        target=MutationTarget(application_id=app_row.id),
        changes=ApplicationChanges(status="definitely_not_a_status"),
    ), db)

    assert not result.success


def test_unknown_custom_role_accepted_in_update_draft(db):
    from app.company_resolution import get_or_create_company
    from app.role_resolution import normalize_role_name

    company = get_or_create_company(db, "Neilsoft")
    db.commit()
    app_row = JobApplication(
        company_id=company.id,
        role="AI Engineer",
        normalized_role=normalize_role_name("AI Engineer"),
        employment_types_json=[],
        job_link="",
        location="",
        status="applied",
        current_stages_json=[],
        priority="MEDIUM",
        engaged_days=0,
        next_action="",
        comments="",
        is_draft=False,
    )
    db.add(app_row)
    db.commit()
    db.refresh(app_row)

    result = dispatch(MutationPayload(
        operation="create_application_update_draft",
        target=MutationTarget(application_id=app_row.id),
        changes=ApplicationChanges(role="LLM Inference Optimization Engineer"),
    ), db)

    assert result.success
    assert result.change_draft["changes_json"]["role"] == "LLM Inference Optimization Engineer"


# ---------------------------------------------------------------------------
# API: apply endpoint
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_api_apply_change_draft(client):
    app_data = await _create_saved(client, priority="MEDIUM")
    app_id = app_data["id"]

    # Create pending change via dispatcher directly
    db = SessionLocal()
    try:
        from app.mutation_dispatcher import dispatch as _dispatch
        r = _dispatch(MutationPayload(
            operation="create_application_update_draft",
            target=MutationTarget(application_id=app_id),
            changes=ApplicationChanges(priority="HIGH"),
        ), db)
        cd_id = r.change_draft["id"]
    finally:
        db.close()

    resp = await client.post(f"/application-change-drafts/{cd_id}/apply")
    assert resp.status_code == 200
    data = resp.json()
    assert data["priority"] == "HIGH"

    # GET the application to confirm saved
    resp2 = await client.get(f"/applications/{app_id}")
    assert resp2.status_code == 200
    assert resp2.json()["priority"] == "HIGH"


@pytest.mark.anyio
async def test_api_discard_change_draft(client):
    app_data = await _create_saved(client, priority="MEDIUM")
    app_id = app_data["id"]

    db = SessionLocal()
    try:
        from app.mutation_dispatcher import dispatch as _dispatch
        r = _dispatch(MutationPayload(
            operation="create_application_update_draft",
            target=MutationTarget(application_id=app_id),
            changes=ApplicationChanges(priority="HIGH"),
        ), db)
        cd_id = r.change_draft["id"]
    finally:
        db.close()

    resp = await client.post(f"/application-change-drafts/{cd_id}/discard")
    assert resp.status_code == 200
    assert "discarded" in resp.json()["message"].lower()

    # Application priority still MEDIUM
    resp2 = await client.get(f"/applications/{app_id}")
    assert resp2.json()["priority"] == "MEDIUM"


@pytest.mark.anyio
async def test_api_apply_conflict_returns_409(client):
    # Create two applications at same company
    app1 = await _create_saved(client, company="Rockwell", role="AI Engineer")
    app2 = await _create_saved(client, company="Rockwell", role="GET")

    db = SessionLocal()
    try:
        from app.mutation_dispatcher import dispatch as _dispatch
        r = _dispatch(MutationPayload(
            operation="create_application_update_draft",
            target=MutationTarget(application_id=app2["id"]),
            changes=ApplicationChanges(role="AI Engineer"),
        ), db)
        cd_id = r.change_draft["id"]
    finally:
        db.close()

    resp = await client.post(f"/application-change-drafts/{cd_id}/apply")
    assert resp.status_code == 409

    # app2 role unchanged
    resp2 = await client.get(f"/applications/{app2['id']}")
    assert resp2.json()["role"] == "GET"


@pytest.mark.anyio
async def test_api_get_change_draft(client):
    app_data = await _create_saved(client, priority="MEDIUM")
    app_id = app_data["id"]

    db = SessionLocal()
    try:
        from app.mutation_dispatcher import dispatch as _dispatch
        r = _dispatch(MutationPayload(
            operation="create_application_update_draft",
            target=MutationTarget(application_id=app_id),
            changes=ApplicationChanges(priority="HIGH"),
        ), db)
        cd_id = r.change_draft["id"]
    finally:
        db.close()

    resp = await client.get(f"/application-change-drafts/{cd_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_application_id"] == app_id
    assert data["original"]["priority"] == "MEDIUM"
    assert data["preview"]["priority"] == "HIGH"
    assert "priority" in data["changed_fields"]


# ---------------------------------------------------------------------------
# Transcript/semantic: pending changes not direct patch
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_transcript_update_creates_pending_not_direct_patch(client):
    """Simulate the dispatcher path: sending update changes creates a pending-change draft."""
    app_data = await _create_saved(client, priority="MEDIUM")
    app_id = app_data["id"]

    db = SessionLocal()
    try:
        from app.mutation_dispatcher import dispatch as _dispatch
        # This is what semantic_validation now calls (instead of patch_application)
        result = _dispatch(MutationPayload(
            operation="create_application_update_draft",
            target=MutationTarget(application_id=app_id),
            changes=ApplicationChanges(priority="HIGH"),
        ), db)
        assert result.success
        assert result.change_draft is not None
        assert result.operation in ("create_application_update_draft", "patch_application_update_draft")
    finally:
        db.close()

    # Application must be unchanged
    resp = await client.get(f"/applications/{app_id}")
    assert resp.json()["priority"] == "MEDIUM"


@pytest.mark.anyio
async def test_draft_creation_unaffected_by_pending_changes(client):
    """A new-application draft workflow is independent of pending-changes workflow."""
    # Create saved application
    await _create_saved(client)

    # Create a pending change for it
    db = SessionLocal()
    try:
        from app.mutation_dispatcher import dispatch as _dispatch
        apps = await client.get("/applications")
        app_id = apps.json()[0]["id"]
        _dispatch(MutationPayload(
            operation="create_application_update_draft",
            target=MutationTarget(application_id=app_id),
            changes=ApplicationChanges(priority="HIGH"),
        ), db)
    finally:
        db.close()

    # Create a new application via direct POST (different company/role)
    resp = await client.post("/applications", json={
        **BASE_APP,
        "company": "Acme",
        "role": "Backend Engineer",
    })
    assert resp.status_code == 201


# ---------------------------------------------------------------------------
# Cascade delete: deleting application removes its change draft
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_cascade_delete_removes_change_draft(client):
    app_data = await _create_saved(client)
    app_id = app_data["id"]

    db = SessionLocal()
    try:
        from app.mutation_dispatcher import dispatch as _dispatch
        r = _dispatch(MutationPayload(
            operation="create_application_update_draft",
            target=MutationTarget(application_id=app_id),
            changes=ApplicationChanges(priority="HIGH"),
        ), db)
        cd_id = r.change_draft["id"]
    finally:
        db.close()

    # Archive then permanently delete
    await client.post(f"/applications/{app_id}/archive")
    del_resp = await client.delete(f"/applications/{app_id}")
    assert del_resp.status_code == 204

    db2 = SessionLocal()
    try:
        assert db2.get(ApplicationChangeDraft, cd_id) is None
    finally:
        db2.close()


# ---------------------------------------------------------------------------
# Regression: existing draft/save/discard still works
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_regression_draft_save_discard(client):
    """Verify new-application draft flow is unbroken."""
    resp = await client.post("/applications", json=BASE_APP)
    assert resp.status_code == 201
    app_id = resp.json()["id"]
    assert app_id > 0

    list_resp = await client.get("/applications")
    ids = [a["id"] for a in list_resp.json()]
    assert app_id in ids


@pytest.mark.anyio
async def test_regression_archive_restore(client):
    app_data = await _create_saved(client)
    app_id = app_data["id"]

    await client.post(f"/applications/{app_id}/archive")
    archived = await client.get("/applications/archived")
    assert any(a["id"] == app_id for a in archived.json())

    await client.post(f"/applications/{app_id}/restore")
    active = await client.get("/applications")
    assert any(a["id"] == app_id for a in active.json())
