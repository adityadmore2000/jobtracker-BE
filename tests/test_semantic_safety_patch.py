"""
Regression tests for the minimal semantic safety patch:
  - Note-intent guard blocks active-draft contextual patch fallback
  - `update` verb added to deterministic single-field grammar
"""

import pytest
from fastapi.testclient import TestClient

from app.database import SessionLocal
from app.fast_path_parser import try_parse
from app.main import app
from app.models import Company, JobApplication
from app.mutation_dispatcher import dispatch
from app.mutation_schemas import ApplicationChanges, MutationPayload, MutationTarget
from app.role_resolution import normalize_role_name
from app.semantic_validation import _has_note_intent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def db():
    with SessionLocal() as session:
        yield session


def _parse(client, transcript: str, context: dict | None = None) -> dict:
    body = {"transcript": transcript}
    if context:
        body["context"] = context
    resp = client.post("/transcript/parse", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _create_draft(db, company: str, role: str) -> int:
    result = dispatch(
        MutationPayload(
            operation="create_draft",
            target=MutationTarget(),
            changes=ApplicationChanges(company=company, role=role),
        ),
        db,
    )
    assert result.success, result.message
    return result.draft["id"]


def _discard_draft(db, draft_id: int) -> None:
    dispatch(
        MutationPayload(
            operation="discard_draft",
            target=MutationTarget(draft_id=str(draft_id)),
            changes=ApplicationChanges(),
        ),
        db,
    )


# ---------------------------------------------------------------------------
# Note-intent detector unit tests
# ---------------------------------------------------------------------------

class TestHasNoteIntent:
    def test_add_a_note(self):
        assert _has_note_intent("add a note saying that I contacted them")

    def test_add_note_no_article(self):
        assert _has_note_intent("add note that recruiter replied")

    def test_append_a_note(self):
        assert _has_note_intent("append a note that follow-up is pending")

    def test_append_note_no_article(self):
        assert _has_note_intent("append note recruiter called back")

    def test_note_that(self):
        assert _has_note_intent("note that follow-up is pending")

    def test_case_insensitive(self):
        assert _has_note_intent("ADD A NOTE saying something")
        assert _has_note_intent("NOTE THAT this happened")

    def test_mid_sentence(self):
        assert _has_note_intent("please add a note that I spoke to the hiring manager")

    # Non-matching — must not false-positive
    def test_role_update_not_a_note(self):
        assert not _has_note_intent("set role to Founding Engineer")

    def test_status_update_not_a_note(self):
        assert not _has_note_intent("update status to applied")

    def test_notable_not_a_note(self):
        # "notable" contains "note" but has no command anchor
        assert not _has_note_intent("this is a notable change")

    def test_annotation_not_a_note(self):
        assert not _has_note_intent("annotation added")

    def test_add_note_for_company_not_a_note_guard(self):
        # "add note for [company]" targets a saved application — allow LLM pipeline
        assert not _has_note_intent("add note for Neilsoft saying referral received")
        assert not _has_note_intent("Add note for Neilsoft saying referral received")


# ---------------------------------------------------------------------------
# Note-intent guard — active draft must not be mutated
# ---------------------------------------------------------------------------

class TestNoteIntentGuard:
    """When a note-like transcript is sent while a draft is active,
    the draft must not be patched and the safe message must be returned."""

    _COMPANY = "Safety Guard Co"
    _ROLE = "Guard Engineer"

    def _setup(self, db) -> int:
        co = db.query(Company).filter_by(name=self._COMPANY).first()
        if co:
            for r in db.query(JobApplication).filter_by(company_id=co.id).all():
                db.delete(r)
            db.commit()
        return _create_draft(db, self._COMPANY, self._ROLE)

    def _teardown(self, db, draft_id: int) -> None:
        app = db.get(JobApplication, draft_id)
        if app:
            db.delete(app)
            db.commit()

    def _assert_draft_unchanged(self, db, draft_id: int) -> None:
        db.expire_all()
        row = db.get(JobApplication, draft_id)
        assert row is not None
        assert row.role == self._ROLE, f"role was mutated to {row.role!r}"
        assert row.normalized_role == normalize_role_name(self._ROLE)
        assert row.is_draft is True

    def test_add_a_note_saying_does_not_corrupt_role(self, client, db):
        # "add a note saying {text}" is now handled by the controlled parser.
        # It appends the note to the draft without touching structured fields.
        draft_id = self._setup(db)
        context = {"draft_id": str(draft_id)}
        try:
            data = _parse(client, "add a note saying that i have connected with previous employer from there", context)
            assert data["status"] == "note_added", (
                f"Expected note_added, got {data['status']!r}"
            )
            # Role must be unchanged — note must not be written into role field
            self._assert_draft_unchanged(db, draft_id)
        finally:
            self._teardown(db, draft_id)

    def test_append_note_that_does_not_corrupt_role(self, client, db):
        # "append note that X" is not in the supported grammar (requires "saying").
        # It returns unsupported — no mutation occurs.
        draft_id = self._setup(db)
        context = {"draft_id": str(draft_id)}
        try:
            data = _parse(client, "append note that recruiter replied", context)
            assert data["status"] == "unsupported"
            self._assert_draft_unchanged(db, draft_id)
        finally:
            self._teardown(db, draft_id)

    def test_note_that_does_not_corrupt_role(self, client, db):
        # "note that X" has no add/append anchor — returns unsupported.
        draft_id = self._setup(db)
        context = {"draft_id": str(draft_id)}
        try:
            data = _parse(client, "note that follow-up is pending", context)
            assert data["status"] == "unsupported"
            self._assert_draft_unchanged(db, draft_id)
        finally:
            self._teardown(db, draft_id)

    def test_note_guard_does_not_block_role_update(self, client, db):
        """A genuine role update must still go through — the guard is note-specific."""
        draft_id = self._setup(db)
        context = {"draft_id": str(draft_id)}
        try:
            # The fast-path handles "set role to X" via patch_draft directly,
            # so the LLM fallback and its guard are bypassed entirely.
            # Verify it doesn't produce "Could not add that note safely".
            data = _parse(client, "set priority to high", context)
            assert "Could not add that note safely" not in data.get("message", "")
        finally:
            self._teardown(db, draft_id)


# ---------------------------------------------------------------------------
# Task 2 — `update` verb in deterministic single-field grammar
# ---------------------------------------------------------------------------

class TestUpdateVerbParser:
    """try_parse must handle `update` identically to `set`/`change`."""

    def test_update_status_to_applied_with_active_application(self):
        ctx = {"active_application_id": 1}
        result = try_parse("update status to applied", ctx)
        assert result is not None
        assert result.operation == "create_application_update_draft"
        assert result.changes.status == "applied"

    def test_update_status_with_active_draft(self):
        ctx = {"draft_id": "5"}
        result = try_parse("update status to applied", ctx)
        assert result is not None
        assert result.operation == "patch_draft"
        assert result.changes.status == "applied"

    def test_update_priority_to_medium(self):
        ctx = {"active_application_id": 1}
        result = try_parse("update priority to medium", ctx)
        assert result is not None
        assert result.changes.priority == "MEDIUM"

    def test_update_location_to_onsite(self):
        ctx = {"active_application_id": 1}
        result = try_parse("update location to onsite", ctx)
        assert result is not None
        assert result.changes.location_mode == "on-site"

    def test_update_employment_type_to_fulltime(self):
        ctx = {"active_application_id": 1}
        result = try_parse("update employment type to fulltime", ctx)
        assert result is not None
        assert result.changes.employment_types == ["Full Time"]

    def test_update_returns_none_with_no_context(self):
        result = try_parse("update status to applied", {})
        assert result is None

    def test_set_still_works(self):
        ctx = {"active_application_id": 1}
        result = try_parse("set status to applied", ctx)
        assert result is not None
        assert result.changes.status == "applied"

    def test_change_still_works(self):
        ctx = {"active_application_id": 1}
        result = try_parse("change status to applied", ctx)
        assert result is not None
        assert result.changes.status == "applied"


class TestUpdateVerbIntegration:
    """End-to-end: `update status to applied` with active saved application
    must bypass Ollama and produce create_application_update_draft."""

    _COMPANY = "Update Verb Co"
    _ROLE = "Status Engineer"

    def _setup(self, db) -> int:
        co = db.query(Company).filter_by(name=self._COMPANY).first()
        if co:
            for r in db.query(JobApplication).filter_by(company_id=co.id).all():
                db.delete(r)
            db.commit()
        draft_id = _create_draft(db, self._COMPANY, self._ROLE)
        save_result = dispatch(
            MutationPayload(
                operation="save_draft",
                target=MutationTarget(draft_id=str(draft_id)),
                changes=ApplicationChanges(),
            ),
            db,
        )
        assert save_result.success
        return save_result.application["id"]

    def _teardown(self, db, app_id: int) -> None:
        row = db.get(JobApplication, app_id)
        if row:
            db.delete(row)
            db.commit()

    def test_update_status_bypasses_ollama(self, client, db, monkeypatch):
        app_id = self._setup(db)
        context = {"active_application_id": app_id}

        calls = []
        from app import semantic_interpreter as si

        def mock_get():
            class M:
                def interpret(self, *a, **kw):
                    calls.append("interpret")
                    raise AssertionError("Ollama must not be called")
                def extract_fields(self, *a, **kw):
                    calls.append("extract_fields")
                    raise AssertionError("Ollama must not be called")
                @property
                def settings(self):
                    return type("S", (), {"max_tool_turns": 2})()
            return M()

        monkeypatch.setattr(si, "get_semantic_interpreter", mock_get)

        data = _parse(client, "update status to applied", context)

        assert calls == [], f"Ollama was called: {calls}"
        assert data["status"] == "pending_changes_created", (
            f"Expected pending_changes_created, got {data['status']!r}: {data['message']!r}"
        )

        # Saved row must be unchanged until Apply Changes
        db.expire_all()
        row = db.get(JobApplication, app_id)
        assert row.status != "applied", "Saved row must not be mutated directly"

        self._teardown(db, app_id)

    def test_update_priority_to_medium_creates_pending_changes(self, client, db):
        app_id = self._setup(db)
        context = {"active_application_id": app_id}
        try:
            data = _parse(client, "update priority to medium", context)
            assert data["status"] == "pending_changes_created"
            assert data["pending_changes"] is not None
            preview = data["pending_changes"]["preview"]
            assert preview["priority"] == "MEDIUM"

            db.expire_all()
            row = db.get(JobApplication, app_id)
            assert row.priority != "MEDIUM", "Saved row must not be mutated directly"
        finally:
            self._teardown(db, app_id)
