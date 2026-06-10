import logging
from datetime import datetime, timezone
from typing import List

from sqlalchemy.orm import Session

from .company_resolution import get_or_create_company
from .constants import (
    ALLOWED_CURRENT_STAGES,
    ALLOWED_EMPLOYMENT_TYPES,
    ALLOWED_LOCATIONS,
    ALLOWED_PRIORITIES,
    STATUS_OPTIONS,
    normalize_status_value,
)
from .models import ApplicationEvent, ApplicationNote, JobApplication
from .role_resolution import find_application_by_company_role, normalize_role_name
from .mutation_schemas import (
    ALLOWED_OPERATIONS,
    ApplicationChanges,
    MutationPayload,
    MutationResult,
    MutationTarget,
)

logger = logging.getLogger(__name__)

_OPERATION_TO_TRANSCRIPT_OP: dict[str, str] = {
    "create_draft": "create",
    "patch_draft": "create",
    "save_draft": "create",
    "patch_application": "update",
    "discard_draft": "none",
    "ask_clarification": "none",
    "append_note": "none",
    "archive_application": "none",
    "restore_application": "none",
    "delete_application_permanently": "none",
}

assert set(_OPERATION_TO_TRANSCRIPT_OP.keys()) == ALLOWED_OPERATIONS, (
    "OPERATION_TO_TRANSCRIPT_OP keys must exactly match ALLOWED_OPERATIONS. "
    f"Missing: {ALLOWED_OPERATIONS - set(_OPERATION_TO_TRANSCRIPT_OP.keys())} "
    f"Extra: {set(_OPERATION_TO_TRANSCRIPT_OP.keys()) - ALLOWED_OPERATIONS}"
)


def _error(operation: str, message: str) -> MutationResult:
    return MutationResult(success=False, operation=operation, message=message)


def _validate_enum_fields(changes: ApplicationChanges, operation: str) -> MutationResult | None:
    if changes.priority is not None and changes.priority not in ALLOWED_PRIORITIES:
        return _error(operation, f"Invalid priority '{changes.priority}'. Allowed: {ALLOWED_PRIORITIES}")
    if changes.location_mode is not None and changes.location_mode not in ALLOWED_LOCATIONS:
        return _error(operation, f"Invalid location_mode '{changes.location_mode}'. Allowed: {ALLOWED_LOCATIONS}")
    if changes.status is not None and changes.status != "":
        canonical = normalize_status_value(changes.status)
        if canonical is None:
            return _error(operation, f"Invalid status '{changes.status}'. Allowed: {STATUS_OPTIONS}")
        changes.status = canonical
    if changes.employment_types is not None:
        invalid = [v for v in changes.employment_types if v not in ALLOWED_EMPLOYMENT_TYPES]
        if invalid:
            return _error(operation, f"Invalid employment_type(s) {invalid}. Allowed: {ALLOWED_EMPLOYMENT_TYPES}")
    if changes.current_stages is not None:
        invalid = [v for v in changes.current_stages if v not in ALLOWED_CURRENT_STAGES]
        if invalid:
            return _error(operation, f"Invalid current_stage(s) {invalid}. Allowed: {ALLOWED_CURRENT_STAGES}")
    return None


def _application_to_dict(app: JobApplication) -> dict:
    return {
        "id": app.id,
        "company": app.company,
        "company_id": app.company_id,
        "role": app.role,
        "employment_types_json": app.employment_types_json,
        "job_link": app.job_link,
        "location": app.location,
        "status": app.status,
        "current_stages_json": app.current_stages_json,
        "priority": app.priority,
        "engaged_days": app.engaged_days,
        "next_action": app.next_action,
        "comments": app.comments,
        "is_draft": app.is_draft,
        "draft_created_at": app.draft_created_at.isoformat() if app.draft_created_at else None,
        "archived_at": app.archived_at.isoformat() if app.archived_at else None,
        "created_at": app.created_at.isoformat() if app.created_at else None,
        "updated_at": app.updated_at.isoformat() if app.updated_at else None,
    }


def _note_to_dict(note: ApplicationNote) -> dict:
    return {
        "id": note.id,
        "text": note.text,
        "created_at": note.created_at.isoformat() if note.created_at else None,
    }


def _apply_changes_to_application(app: JobApplication, changes: ApplicationChanges, db: Session) -> None:
    if changes.company is not None:
        company_obj = get_or_create_company(db, changes.company)
        app.company_id = company_obj.id
    if changes.role is not None:
        app.role = changes.role
        app.normalized_role = normalize_role_name(changes.role)
    if changes.status is not None:
        app.status = changes.status
    if changes.priority is not None:
        app.priority = changes.priority
    if changes.location_mode is not None:
        app.location = changes.location_mode
    if changes.job_link is not None:
        app.job_link = changes.job_link
    if changes.employment_types is not None:
        app.employment_types_json = list(changes.employment_types)
    if changes.current_stages is not None:
        app.current_stages_json = list(changes.current_stages)


def _append_notes(application_id: int, notes: List[str], db: Session) -> List[ApplicationNote]:
    created = []
    for text in notes:
        note = ApplicationNote(
            application_id=application_id,
            text=text,
            created_at=datetime.now(timezone.utc),
        )
        db.add(note)
        created.append(note)
    return created


def _append_event(application_id: int, event_type: str, payload: dict, db: Session) -> ApplicationEvent:
    event = ApplicationEvent(
        application_id=application_id,
        event_type=event_type,
        payload=payload,
        created_at=datetime.now(timezone.utc),
    )
    db.add(event)
    return event


def handle_create_draft(payload: MutationPayload, db: Session) -> MutationResult:
    if not payload.changes.company:
        return _error("create_draft", "company is required to create a draft")
    enum_error = _validate_enum_fields(payload.changes, "create_draft")
    if enum_error:
        return enum_error

    company_obj = get_or_create_company(db, payload.changes.company)
    incoming_role = payload.changes.role or ""
    incoming_status = payload.changes.status or ""

    # --- Uniqueness check: look for any existing row for this company + role ---
    existing = find_application_by_company_role(db, company_id=company_obj.id, role=incoming_role)
    if existing is not None:
        return _handle_reapply(existing, incoming_status, "create_draft", db)

    # No existing row — create a fresh draft.
    role_normalized = normalize_role_name(incoming_role)
    app = JobApplication(
        company_id=company_obj.id,
        role=incoming_role,
        normalized_role=role_normalized,
        employment_types_json=list(payload.changes.employment_types) if payload.changes.employment_types else [],
        job_link=payload.changes.job_link or "",
        location=payload.changes.location_mode or "",
        status=incoming_status,
        current_stages_json=list(payload.changes.current_stages) if payload.changes.current_stages else [],
        priority=payload.changes.priority or "",
        engaged_days=0,
        next_action="",
        comments="",
        is_draft=True,
        draft_created_at=datetime.now(timezone.utc),
    )
    db.add(app)
    db.commit()
    db.refresh(app)

    return MutationResult(
        success=True,
        operation="create_draft",
        message="Draft created.",
        draft=_application_to_dict(app),
    )


def _handle_reapply(existing: JobApplication, requested_status: str, operation: str, db: Session) -> MutationResult:
    """Apply reapply semantics when an existing company+role row is found.

    Reapply matrix:
      - existing is a draft          → return existing draft, no duplicate
      - already applied              → no-op, return truthful message
      - accepted                     → clarification required (don't downgrade silently)
      - rejected / in_touch / empty  → set status=applied, restore if archived
    """
    if existing.is_draft:
        db.refresh(existing)
        return MutationResult(
            success=True,
            operation="draft_updated",
            message="An existing draft for this company and role already exists.",
            draft=_application_to_dict(existing),
        )

    # Saved row (may be archived).
    current_status = existing.status

    if current_status == "accepted":
        return MutationResult(
            success=True,
            operation="ask_clarification",
            message="Clarification required.",
            clarification_question=(
                f"This application is currently marked as accepted. "
                f"Do you want to change it to applied?"
            ),
        )

    if current_status == "applied" and existing.archived_at is None:
        return MutationResult(
            success=True,
            operation="no_change",
            message="Application already exists and is marked as applied.",
            application=_application_to_dict(existing),
        )

    # For all other states (rejected, in_touch, empty) — or archived — reapply.
    was_archived = existing.archived_at is not None
    old_status = existing.status
    existing.status = "applied"
    existing.archived_at = None
    existing.updated_at = datetime.now(timezone.utc)

    if was_archived:
        _append_event(existing.id, "application_restored", {}, db)
        _append_event(existing.id, "status_changed", {"field": "status", "from": old_status, "to": "applied"}, db)
        message = "Existing archived application restored and marked as applied."
    elif old_status != "applied":
        _append_event(existing.id, "status_changed", {"field": "status", "from": old_status, "to": "applied"}, db)
        message = f"Existing application reused and status updated to applied (was {old_status!r})."
    else:
        message = "Application already exists and is marked as applied."

    db.commit()
    db.refresh(existing)
    return MutationResult(
        success=True,
        operation="updated",
        message=message,
        application=_application_to_dict(existing),
    )


def handle_patch_draft(payload: MutationPayload, db: Session) -> MutationResult:
    enum_error = _validate_enum_fields(payload.changes, "patch_draft")
    if enum_error:
        return enum_error

    if payload.notes_to_append:
        logger.warning("patch_draft received notes_to_append — ignoring; notes cannot be attached to drafts")

    if payload.target.draft_id is not None:
        try:
            draft_id = int(payload.target.draft_id)
        except (ValueError, TypeError):
            return _error("patch_draft", f"Invalid draft_id '{payload.target.draft_id}'")
        app = db.get(JobApplication, draft_id)
        if app is None or not app.is_draft:
            return _error("patch_draft", f"Draft with id {draft_id} not found")

        # Pre-flight uniqueness check when company or role is changing.
        company_changing = payload.changes.company is not None
        role_changing = payload.changes.role is not None
        if company_changing or role_changing:
            if company_changing:
                new_company_obj = get_or_create_company(db, payload.changes.company)
                new_company_id = new_company_obj.id
                new_company_name = new_company_obj.name
            else:
                new_company_id = app.company_id
                new_company_name = app.company
            new_normalized_role = (
                normalize_role_name(payload.changes.role) if role_changing else app.normalized_role
            )
            new_role_display = payload.changes.role if role_changing else app.role
            collision = (
                db.query(JobApplication)
                .filter(
                    JobApplication.company_id == new_company_id,
                    JobApplication.normalized_role == new_normalized_role,
                    JobApplication.id != app.id,
                )
                .first()
            )
            if collision is not None:
                return MutationResult(
                    success=False,
                    conflict=True,
                    operation="patch_draft",
                    message=f"An application for {new_company_name} — {new_role_display} already exists.",
                )

        _apply_changes_to_application(app, payload.changes, db)
        db.commit()
        db.refresh(app)
        return MutationResult(
            success=True,
            operation="patch_draft",
            message="Draft patched.",
            draft=_application_to_dict(app),
        )

    draft = _changes_to_dict(payload.changes)
    return MutationResult(
        success=True,
        operation="patch_draft",
        message="Draft patched (preview).",
        draft=draft,
    )


def handle_save_draft(payload: MutationPayload, db: Session) -> MutationResult:
    if payload.target.draft_id is not None:
        try:
            draft_id = int(payload.target.draft_id)
        except (ValueError, TypeError):
            return _error("save_draft", f"Invalid draft_id '{payload.target.draft_id}'")
        app = db.get(JobApplication, draft_id)
        if app is None or not app.is_draft:
            return _error("save_draft", f"Draft with id {draft_id} not found")
        if not app.company:
            return _error("save_draft", "Draft must have company to be saved.")

        # Enforce uniqueness: check for a saved row with same company + normalized_role.
        collision = (
            db.query(JobApplication)
            .filter(
                JobApplication.company_id == app.company_id,
                JobApplication.normalized_role == app.normalized_role,
                JobApplication.is_draft == False,  # noqa: E712
                JobApplication.id != app.id,
            )
            .first()
        )
        if collision is not None:
            # Safe merge: discard this draft, return the existing canonical row.
            db.delete(app)
            db.commit()
            db.refresh(collision)
            return MutationResult(
                success=True,
                operation="save_draft",
                message=(
                    f"An application for {collision.company} — {collision.role} already exists. "
                    f"Draft discarded; existing application returned."
                ),
                application=_application_to_dict(collision),
            )

        app.is_draft = False
        app.draft_created_at = None
        _append_event(app.id, "application_saved", {}, db)
        if payload.notes_to_append:
            created_notes = _append_notes(app.id, payload.notes_to_append, db)
            for note in created_notes:
                _append_event(app.id, "note_added", {"text": note.text}, db)
        db.commit()
        db.refresh(app)
        notes_list = [_note_to_dict(n) for n in app.notes_rel] if payload.notes_to_append else None
        return MutationResult(
            success=True,
            operation="save_draft",
            message="Draft saved as application.",
            application=_application_to_dict(app),
            notes=notes_list,
        )

    return _error("save_draft", "No active draft to save. Provide a draft_id.")


def handle_discard_draft(payload: MutationPayload, db: Session) -> MutationResult:
    if payload.target.draft_id is None:
        return MutationResult(
            success=True,
            operation="discard_draft",
            message="No active draft to discard.",
        )

    try:
        draft_id = int(payload.target.draft_id)
    except (ValueError, TypeError):
        return _error("discard_draft", f"Invalid draft_id '{payload.target.draft_id}'")
    app = db.get(JobApplication, draft_id)
    if app is None or not app.is_draft:
        return _error("discard_draft", f"Draft with id {draft_id} not found")
    db.delete(app)
    db.commit()
    return MutationResult(
        success=True,
        operation="discard_draft",
        message="Draft discarded.",
    )


def handle_patch_application(payload: MutationPayload, db: Session) -> MutationResult:
    if payload.target.application_id is None:
        return _error("patch_application", "application_id is required in target for patch_application")
    enum_error = _validate_enum_fields(payload.changes, "patch_application")
    if enum_error:
        return enum_error

    app = db.get(JobApplication, payload.target.application_id)
    if app is None:
        return _error("patch_application", f"Application {payload.target.application_id} not found")
    if app.is_draft:
        return _error("patch_application", "Cannot patch a draft application via patch_application. Use patch_draft.")
    if app.archived_at is not None:
        return _error("patch_application", f"Application {payload.target.application_id} is archived and cannot be patched.")

    old_status = app.status
    old_priority = app.priority
    old_location = app.location
    old_job_link = app.job_link
    old_company = app.company
    old_role = app.role
    old_employment_types = list(app.employment_types_json)
    old_current_stages = list(app.current_stages_json)

    # Pre-compute what the new company_id and normalized_role will be after changes.
    new_company_id = app.company_id
    if payload.changes.company is not None:
        new_company_obj = get_or_create_company(db, payload.changes.company)
        new_company_id = new_company_obj.id
    new_normalized_role = normalize_role_name(payload.changes.role) if payload.changes.role is not None else app.normalized_role

    # Check for collision with a different existing row only when company or role changes.
    company_or_role_changing = payload.changes.company is not None or payload.changes.role is not None
    if company_or_role_changing:
        collision = (
            db.query(JobApplication)
            .filter(
                JobApplication.company_id == new_company_id,
                JobApplication.normalized_role == new_normalized_role,
                JobApplication.id != app.id,
            )
            .first()
        )
        if collision is not None:
            collision_company = collision.company_rel.name if collision.company_rel else str(new_company_id)
            return _error(
                "patch_application",
                f"An application for {collision_company} — {collision.role} already exists.",
            )

    _apply_changes_to_application(app, payload.changes, db)

    if payload.changes.status is not None and app.status != old_status:
        _append_event(app.id, "status_changed", {"field": "status", "from": old_status, "to": app.status}, db)
    if payload.changes.priority is not None and app.priority != old_priority:
        _append_event(app.id, "field_changed", {"field": "priority", "from": old_priority, "to": app.priority}, db)
    if payload.changes.location_mode is not None and app.location != old_location:
        _append_event(app.id, "field_changed", {"field": "location", "from": old_location, "to": app.location}, db)
    if payload.changes.job_link is not None and app.job_link != old_job_link:
        _append_event(app.id, "field_changed", {"field": "job_link", "from": old_job_link, "to": app.job_link}, db)
    if payload.changes.company is not None and app.company != old_company:
        _append_event(app.id, "field_changed", {"field": "company", "from": old_company, "to": app.company}, db)
    if payload.changes.role is not None and app.role != old_role:
        _append_event(app.id, "field_changed", {"field": "role", "from": old_role, "to": app.role}, db)
    if payload.changes.employment_types is not None and app.employment_types_json != old_employment_types:
        _append_event(app.id, "field_changed", {"field": "employment_types", "from": old_employment_types, "to": app.employment_types_json}, db)
    if payload.changes.current_stages is not None and app.current_stages_json != old_current_stages:
        _append_event(app.id, "field_changed", {"field": "current_stages", "from": old_current_stages, "to": app.current_stages_json}, db)

    if payload.notes_to_append:
        created_notes = _append_notes(app.id, payload.notes_to_append, db)
        for note in created_notes:
            _append_event(app.id, "note_added", {"text": note.text}, db)

    db.commit()
    db.refresh(app)
    notes_list = [_note_to_dict(n) for n in app.notes_rel] if payload.notes_to_append else None
    return MutationResult(
        success=True,
        operation="patch_application",
        message="Application patched.",
        application=_application_to_dict(app),
        notes=notes_list,
    )


def handle_ask_clarification(payload: MutationPayload, db: Session) -> MutationResult:
    question = payload.notes_to_append[0] if payload.notes_to_append else "Please clarify."
    return MutationResult(
        success=True,
        operation="ask_clarification",
        message="Clarification required.",
        clarification_question=question,
    )


def handle_append_note(payload: MutationPayload, db: Session) -> MutationResult:
    if not payload.notes_to_append:
        return _error("append_note", "notes_to_append must not be empty")
    if payload.target.application_id is None:
        return _error("append_note", "application_id is required in target for append_note")

    app = db.get(JobApplication, payload.target.application_id)
    if app is None:
        return _error("append_note", f"Application {payload.target.application_id} not found")
    if app.is_draft:
        return _error("append_note", "Cannot append notes to a draft application")

    created_notes = _append_notes(app.id, payload.notes_to_append, db)
    for note in created_notes:
        _append_event(app.id, "note_added", {"text": note.text}, db)
    db.commit()
    for note in created_notes:
        db.refresh(note)

    return MutationResult(
        success=True,
        operation="append_note",
        message="Notes appended.",
        notes=[_note_to_dict(n) for n in created_notes],
    )


def handle_archive_application(payload: MutationPayload, db: Session) -> MutationResult:
    if payload.target.application_id is None:
        return _error("archive_application", "application_id is required in target for archive_application")

    app = db.get(JobApplication, payload.target.application_id)
    if app is None:
        return _error("archive_application", f"Application {payload.target.application_id} not found")
    if app.is_draft:
        return _error("archive_application", "Cannot archive a draft application")
    if app.archived_at is not None:
        return _error("archive_application", "Application is already archived")

    app.archived_at = datetime.now(timezone.utc)
    _append_event(app.id, "application_archived", {}, db)
    db.commit()
    db.refresh(app)
    return MutationResult(
        success=True,
        operation="archive_application",
        message="Application archived.",
        application=_application_to_dict(app),
    )


def handle_restore_application(payload: MutationPayload, db: Session) -> MutationResult:
    if payload.target.application_id is None:
        return _error("restore_application", "application_id is required in target for restore_application")

    app = db.get(JobApplication, payload.target.application_id)
    if app is None:
        return _error("restore_application", f"Application {payload.target.application_id} not found")
    if app.archived_at is None:
        return _error("restore_application", "Application is not archived")

    app.archived_at = None
    _append_event(app.id, "application_restored", {}, db)
    db.commit()
    db.refresh(app)
    return MutationResult(
        success=True,
        operation="restore_application",
        message="Application restored.",
        application=_application_to_dict(app),
    )


def handle_delete_application_permanently(payload: MutationPayload, db: Session) -> MutationResult:
    if payload.target.application_id is None:
        return _error("delete_application_permanently", "application_id is required in target for delete_application_permanently")

    app = db.get(JobApplication, payload.target.application_id)
    if app is None:
        return _error("delete_application_permanently", f"Application {payload.target.application_id} not found")
    if app.is_draft:
        return _error(
            "delete_application_permanently",
            "Drafts must be discarded through the draft workflow.",
        )
    if app.archived_at is None:
        return _error(
            "delete_application_permanently",
            "Only archived applications can be permanently deleted.",
        )

    # Notes and events are removed via cascade="all, delete-orphan" on the relationship.
    db.delete(app)
    db.commit()
    return MutationResult(
        success=True,
        operation="delete_application_permanently",
        message="Application permanently deleted.",
    )


def _changes_to_dict(changes: ApplicationChanges) -> dict:
    return {k: v for k, v in changes.model_dump().items() if v is not None}


def dispatch(payload: MutationPayload, db: Session) -> MutationResult:
    if payload.operation not in ALLOWED_OPERATIONS:
        return _error(payload.operation, f"Unknown operation '{payload.operation}'")

    handlers = {
        "create_draft": handle_create_draft,
        "patch_draft": handle_patch_draft,
        "save_draft": handle_save_draft,
        "discard_draft": handle_discard_draft,
        "patch_application": handle_patch_application,
        "ask_clarification": handle_ask_clarification,
        "append_note": handle_append_note,
        "archive_application": handle_archive_application,
        "restore_application": handle_restore_application,
        "delete_application_permanently": handle_delete_application_permanently,
    }
    return handlers[payload.operation](payload, db)
