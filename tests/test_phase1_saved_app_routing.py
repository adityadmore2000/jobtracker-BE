"""Phase 1 fast-path routing fix: saved-application → pending-changes.

Verifies that deterministic single-field commands route correctly for each
context state: active draft, saved application, and neither.
"""
from __future__ import annotations

import pytest

from app.company_resolution import get_or_create_company
from app.database import SessionLocal
from app.fast_path_parser import try_parse
from app.models import ApplicationChangeDraft, JobApplication
from app.mutation_dispatcher import dispatch
from app.mutation_schemas import ApplicationChanges, MutationPayload, MutationTarget
from app.role_resolution import normalize_role_name


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db():
    with SessionLocal() as session:
        yield session


def _saved_app(db, company="Acme", role="AI Engineer", status="applied", priority="MEDIUM") -> JobApplication:
    co = get_or_create_company(db, company)
    db.commit()
    row = JobApplication(
        company_id=co.id,
        role=role,
        normalized_role=normalize_role_name(role),
        employment_types_json=[],
        job_link="",
        location="",
        status=status,
        current_stages_json=[],
        priority=priority,
        engaged_days=0,
        next_action="",
        comments="",
        is_draft=False,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _draft_app(db, company="Acme", role="AI Engineer") -> JobApplication:
    co = get_or_create_company(db, company)
    db.commit()
    row = JobApplication(
        company_id=co.id,
        role=role,
        normalized_role=normalize_role_name(role),
        employment_types_json=[],
        job_link="",
        location="",
        status="in_touch",
        current_stages_json=[],
        priority="MEDIUM",
        engaged_days=0,
        next_action="",
        comments="",
        is_draft=True,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


# ---------------------------------------------------------------------------
# Case 1 — Active draft: patch_draft directly
# ---------------------------------------------------------------------------


class TestActiveDraftRouting:
    def test_set_status_produces_patch_draft(self):
        result = try_parse("set status to in-touch", {"draft_id": "7"})
        assert result is not None
        assert result.operation == "patch_draft"
        assert result.target.draft_id == "7"
        assert result.target.application_id is None
        assert result.changes.status == "in_touch"

    def test_set_priority_produces_patch_draft(self):
        result = try_parse("set priority to high", {"draft_id": "7"})
        assert result is not None
        assert result.operation == "patch_draft"
        assert result.changes.priority == "HIGH"

    def test_set_location_produces_patch_draft(self):
        result = try_parse("set location to onsite", {"draft_id": "7"})
        assert result is not None
        assert result.operation == "patch_draft"
        assert result.changes.location_mode == "on-site"

    def test_set_employment_type_produces_patch_draft(self):
        result = try_parse("set employment type to fulltime", {"draft_id": "7"})
        assert result is not None
        assert result.operation == "patch_draft"
        assert result.changes.employment_types == ["Full Time"]

    def test_draft_dispatches_to_patch_draft_handler(self, db):
        draft = _draft_app(db)
        result = try_parse("set status to applied", {"draft_id": str(draft.id)})
        assert result is not None
        assert result.operation == "patch_draft"
        mr = dispatch(result, db)
        assert mr.success
        assert mr.operation == "patch_draft"
        # Verify no pending-changes record was created
        cd = db.query(ApplicationChangeDraft).filter(
            ApplicationChangeDraft.target_application_id == draft.id
        ).first()
        assert cd is None


# ---------------------------------------------------------------------------
# Case 2 — Saved application: create_application_update_draft
# ---------------------------------------------------------------------------


class TestSavedAppRouting:
    def test_set_status_produces_create_application_update_draft(self):
        result = try_parse("set status to rejected", {"active_application_id": 42})
        assert result is not None
        assert result.operation == "create_application_update_draft"
        assert result.target.application_id == 42
        assert result.target.draft_id is None
        assert result.changes.status == "rejected"

    def test_set_priority_produces_create_application_update_draft(self):
        result = try_parse("set priority to high", {"active_application_id": 42})
        assert result is not None
        assert result.operation == "create_application_update_draft"
        assert result.changes.priority == "HIGH"

    def test_set_location_produces_create_application_update_draft(self):
        result = try_parse("set location to onsite", {"active_application_id": 42})
        assert result is not None
        assert result.operation == "create_application_update_draft"
        assert result.changes.location_mode == "on-site"

    def test_set_employment_type_produces_create_application_update_draft(self):
        result = try_parse("set employment type to fulltime", {"active_application_id": 42})
        assert result is not None
        assert result.operation == "create_application_update_draft"
        assert result.changes.employment_types == ["Full Time"]

    def test_patch_application_not_used(self):
        for transcript in [
            "set status to rejected",
            "set priority to high",
            "set location to onsite",
            "set employment type to fulltime",
        ]:
            result = try_parse(transcript, {"active_application_id": 99})
            assert result is not None
            assert result.operation != "patch_application", (
                f"'{transcript}' produced patch_application — should use create_application_update_draft"
            )

    def test_status_saved_row_unchanged_after_transcript_command(self, db):
        app_row = _saved_app(db, status="applied")
        result = try_parse("set status to rejected", {"active_application_id": app_row.id})
        assert result is not None
        mr = dispatch(result, db)
        assert mr.success
        # Saved row must be unchanged
        db.refresh(app_row)
        assert app_row.status == "applied"
        # Pending changes record must exist
        cd = db.query(ApplicationChangeDraft).filter(
            ApplicationChangeDraft.target_application_id == app_row.id
        ).first()
        assert cd is not None
        assert cd.changes_json.get("status") == "rejected"

    def test_priority_saved_row_unchanged_after_transcript_command(self, db):
        app_row = _saved_app(db, priority="MEDIUM")
        result = try_parse("set priority to high", {"active_application_id": app_row.id})
        assert result is not None
        mr = dispatch(result, db)
        assert mr.success
        db.refresh(app_row)
        assert app_row.priority == "MEDIUM"
        cd = db.query(ApplicationChangeDraft).filter(
            ApplicationChangeDraft.target_application_id == app_row.id
        ).first()
        assert cd is not None
        assert cd.changes_json.get("priority") == "HIGH"

    def test_location_saved_row_unchanged_after_transcript_command(self, db):
        app_row = _saved_app(db)
        result = try_parse("set location to onsite", {"active_application_id": app_row.id})
        assert result is not None
        mr = dispatch(result, db)
        assert mr.success
        db.refresh(app_row)
        assert app_row.location != "on-site"
        cd = db.query(ApplicationChangeDraft).filter(
            ApplicationChangeDraft.target_application_id == app_row.id
        ).first()
        assert cd is not None
        # Dispatcher stores location_mode as "location" in changes_json
        assert cd.changes_json.get("location") == "on-site"

    def test_employment_type_saved_row_unchanged_after_transcript_command(self, db):
        app_row = _saved_app(db)
        result = try_parse("set employment type to fulltime", {"active_application_id": app_row.id})
        assert result is not None
        mr = dispatch(result, db)
        assert mr.success
        db.refresh(app_row)
        assert app_row.employment_types_json == []
        cd = db.query(ApplicationChangeDraft).filter(
            ApplicationChangeDraft.target_application_id == app_row.id
        ).first()
        assert cd is not None
        assert cd.changes_json.get("employment_types") == ["Full Time"]

    def test_pending_changes_result_has_change_draft(self, db):
        app_row = _saved_app(db, status="applied")
        result = try_parse("set status to rejected", {"active_application_id": app_row.id})
        mr = dispatch(result, db)
        assert mr.change_draft is not None
        assert mr.change_draft["changes_json"]["status"] == "rejected"
        assert "status" in mr.change_draft["changed_fields"]


# ---------------------------------------------------------------------------
# Apply behavior: staged changes → apply → saved row mutated
# ---------------------------------------------------------------------------


class TestApplyBehavior:
    def test_apply_changes_mutates_saved_row(self, db):
        app_row = _saved_app(db, status="applied")
        # Stage the change via fast path
        payload = try_parse("set status to rejected", {"active_application_id": app_row.id})
        mr_stage = dispatch(payload, db)
        assert mr_stage.success
        cd_id = mr_stage.change_draft["id"]
        # Saved row still unchanged
        db.refresh(app_row)
        assert app_row.status == "applied"
        # Apply
        mr_apply = dispatch(MutationPayload(
            operation="apply_application_update_draft",
            target=MutationTarget(change_draft_id=cd_id),
            changes=ApplicationChanges(),
        ), db)
        assert mr_apply.success
        assert mr_apply.operation == "apply_application_update_draft"
        db.refresh(app_row)
        assert app_row.status == "rejected"
        # Change draft consumed
        assert db.get(ApplicationChangeDraft, cd_id) is None

    def test_apply_priority_change_mutates_saved_row(self, db):
        app_row = _saved_app(db, priority="MEDIUM")
        payload = try_parse("set priority to high", {"active_application_id": app_row.id})
        mr_stage = dispatch(payload, db)
        cd_id = mr_stage.change_draft["id"]
        mr_apply = dispatch(MutationPayload(
            operation="apply_application_update_draft",
            target=MutationTarget(change_draft_id=cd_id),
            changes=ApplicationChanges(),
        ), db)
        assert mr_apply.success
        db.refresh(app_row)
        assert app_row.priority == "HIGH"


# ---------------------------------------------------------------------------
# Discard behavior: staged changes → discard → saved row unchanged
# ---------------------------------------------------------------------------


class TestDiscardBehavior:
    def test_discard_leaves_saved_row_unchanged(self, db):
        app_row = _saved_app(db, status="applied")
        payload = try_parse("set status to rejected", {"active_application_id": app_row.id})
        mr_stage = dispatch(payload, db)
        cd_id = mr_stage.change_draft["id"]
        mr_discard = dispatch(MutationPayload(
            operation="discard_application_update_draft",
            target=MutationTarget(change_draft_id=cd_id),
            changes=ApplicationChanges(),
        ), db)
        assert mr_discard.success
        assert mr_discard.operation == "discard_application_update_draft"
        db.refresh(app_row)
        assert app_row.status == "applied"
        assert db.get(ApplicationChangeDraft, cd_id) is None


# ---------------------------------------------------------------------------
# Context precedence: draft_id wins over active_application_id
# ---------------------------------------------------------------------------


class TestContextPrecedence:
    def test_draft_id_wins_over_active_application_id(self):
        context = {"draft_id": "5", "active_application_id": 99}
        result = try_parse("set status to applied", context)
        assert result is not None
        assert result.operation == "patch_draft"
        assert result.target.draft_id == "5"
        assert result.target.application_id is None

    def test_draft_id_wins_saved_app_untouched(self, db):
        saved = _saved_app(db, company="Globex", role="Data Engineer", status="applied")
        draft = _draft_app(db, company="Initech", role="Platform Engineer")
        context = {"draft_id": str(draft.id), "active_application_id": saved.id}
        result = try_parse("set status to rejected", context)
        assert result is not None
        assert result.operation == "patch_draft"
        mr = dispatch(result, db)
        assert mr.success
        # Saved row must be completely untouched
        db.refresh(saved)
        assert saved.status == "applied"
        cd = db.query(ApplicationChangeDraft).filter(
            ApplicationChangeDraft.target_application_id == saved.id
        ).first()
        assert cd is None


# ---------------------------------------------------------------------------
# Case 3 — No context: try_parse returns None
# ---------------------------------------------------------------------------


class TestNoContext:
    def test_set_status_no_context_returns_none(self):
        assert try_parse("set status to rejected", {}) is None

    def test_set_priority_no_context_returns_none(self):
        assert try_parse("set priority to high", {}) is None

    def test_set_location_no_context_returns_none(self):
        assert try_parse("set location to onsite", {}) is None

    def test_set_employment_type_no_context_returns_none(self):
        assert try_parse("set employment type to fulltime", {}) is None
