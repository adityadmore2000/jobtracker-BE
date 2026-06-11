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
from .models import ApplicationChangeDraft, ApplicationEvent, ApplicationNote, JobApplication
from .role_resolution import find_application_by_company_role, normalize_role_name
from .mutation_schemas import (
    ALLOWED_OPERATIONS,
    ApplicationChanges,
    CollisionInfo,
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
    "append_note": "note",
    "archive_application": "update",
    "restore_application": "update",
    "delete_application_permanently": "none",
    "create_application_update_draft": "pending_changes",
    "patch_application_update_draft": "pending_changes",
    "apply_application_update_draft": "update",
    "discard_application_update_draft": "none",
    "set_active_application": "context",
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
    if changes.next_action is not None:
        app.next_action = changes.next_action
    if changes.comments is not None:
        app.comments = changes.comments
    if changes.engaged_days is not None:
        app.engaged_days = changes.engaged_days


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
        company_name = existing.company_rel.name if existing.company_rel else existing.company
        return MutationResult(
            success=True,
            operation="draft_updated",
            message=f"A draft already exists for {company_name} · {existing.role}.",
            draft=_application_to_dict(existing),
            collision=CollisionInfo(
                kind="draft",
                draft_id=existing.id,
                company=company_name,
                role=existing.role,
                archived=False,
            ),
        )

    # Saved row (may be archived).
    current_status = existing.status
    company_name = existing.company_rel.name if existing.company_rel else existing.company
    is_archived = existing.archived_at is not None
    collision = CollisionInfo(
        kind="archived_application" if is_archived else "active_application",
        application_id=existing.id,
        company=company_name,
        role=existing.role,
        archived=is_archived,
    )

    if current_status == "accepted":
        return MutationResult(
            success=True,
            operation="ask_clarification",
            message="Clarification required.",
            clarification_question=(
                f"This application is currently marked as accepted. "
                f"Do you want to change it to applied?"
            ),
            collision=collision,
        )

    if current_status == "applied" and existing.archived_at is None:
        return MutationResult(
            success=True,
            operation="no_change",
            message=f"An application already exists for {company_name} · {existing.role}.",
            application=_application_to_dict(existing),
            collision=collision,
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
        collision=collision,
    )


def _compute_effective_draft_changes(app: JobApplication, changes: ApplicationChanges) -> dict:
    """Return only the changes whose proposed value differs from the current draft value."""
    effective: dict = {}
    if changes.company is not None and changes.company != app.company:
        effective["company"] = changes.company
    if changes.role is not None and changes.role != app.role:
        effective["role"] = changes.role
    if changes.status is not None and changes.status != app.status:
        effective["status"] = changes.status
    if changes.priority is not None and changes.priority != app.priority:
        effective["priority"] = changes.priority
    if changes.location_mode is not None and changes.location_mode != app.location:
        effective["location_mode"] = changes.location_mode
    if changes.job_link is not None and changes.job_link != app.job_link:
        effective["job_link"] = changes.job_link
    if changes.employment_types is not None and list(changes.employment_types) != list(app.employment_types_json):
        effective["employment_types"] = changes.employment_types
    if changes.current_stages is not None and list(changes.current_stages) != list(app.current_stages_json):
        effective["current_stages"] = changes.current_stages
    if changes.next_action is not None and changes.next_action != app.next_action:
        effective["next_action"] = changes.next_action
    if changes.comments is not None and changes.comments != app.comments:
        effective["comments"] = changes.comments
    if changes.engaged_days is not None and changes.engaged_days != app.engaged_days:
        effective["engaged_days"] = changes.engaged_days
    return effective


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

        # No-op detection: skip commit when no field actually changes.
        effective_changes = _compute_effective_draft_changes(app, payload.changes)
        if not effective_changes:
            return MutationResult(
                success=True,
                operation="no_change",
                message="Draft already has those values.",
                draft=_application_to_dict(app),
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

    # Resolve target: prefer application_id, fall back to draft_id.
    # Draft notes are allowed and cascade-deleted when the draft is discarded.
    if payload.target.application_id is not None:
        app = db.get(JobApplication, payload.target.application_id)
        if app is None:
            return _error("append_note", f"Application {payload.target.application_id} not found")
    elif payload.target.draft_id is not None:
        try:
            draft_id = int(payload.target.draft_id)
        except (ValueError, TypeError):
            return _error("append_note", f"Invalid draft_id '{payload.target.draft_id}'")
        app = db.get(JobApplication, draft_id)
        if app is None or not app.is_draft:
            return _error("append_note", f"Draft {payload.target.draft_id} not found")
    else:
        return _error("append_note", "application_id or draft_id is required in target for append_note")

    created_notes = _append_notes(app.id, payload.notes_to_append, db)
    # Only log note_added events for saved (non-draft) applications.
    if not app.is_draft:
        for note in created_notes:
            _append_event(app.id, "note_added", {"text": note.text}, db)
    db.commit()
    for note in created_notes:
        db.refresh(note)

    return MutationResult(
        success=True,
        operation="append_note",
        message="Note added.",
        draft=_application_to_dict(app) if app.is_draft else None,
        application=_application_to_dict(app) if not app.is_draft else None,
        notes=[_note_to_dict(n) for n in created_notes],
    )


def handle_set_active_application(payload: MutationPayload, db: Session) -> MutationResult:
    """Context-selection sentinel. Returns the application so the frontend can set context.

    No DB mutation is performed. The operation tells the frontend to update
    its selected-application state so subsequent field-setter commands route
    to the correct row.
    """
    if payload.target.application_id is None:
        return _error("set_active_application", "application_id is required")
    app = db.get(JobApplication, payload.target.application_id)
    if app is None:
        return _error("set_active_application", f"Application {payload.target.application_id} not found")
    return MutationResult(
        success=True,
        operation="set_active_application",
        message=f"Now updating {app.company}{' — ' + app.role if app.role else ''}. Use set commands to stage changes.",
        application=_application_to_dict(app),
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


# ---------------------------------------------------------------------------
# Pending-changes helpers
# ---------------------------------------------------------------------------

_CHANGE_DRAFT_FIELD_MAP = {
    "company": "company",
    "role": "role",
    "status": "status",
    "priority": "priority",
    "location_mode": "location",
    "job_link": "job_link",
    "employment_types": "employment_types",
    "current_stages": "current_stages",
    "next_action": "next_action",
    "comments": "comments",
    "engaged_days": "engaged_days",
}


def _changes_to_changes_json(changes: ApplicationChanges) -> dict:
    """Convert ApplicationChanges to the storage dict (only non-None fields)."""
    result = {}
    raw = changes.model_dump()
    for schema_key, json_key in _CHANGE_DRAFT_FIELD_MAP.items():
        val = raw.get(schema_key)
        if val is not None:
            result[json_key] = val
    return result


def _merge_changes_json(existing: dict, incoming: dict) -> dict:
    """Merge new changes onto existing ones (last-write wins per field)."""
    merged = dict(existing)
    merged.update(incoming)
    return merged


def _build_preview_application(original: JobApplication, changes_json: dict) -> dict:
    """Build a preview dict representing what the application would look like after applying changes."""
    preview = _application_to_dict(original)
    for json_key, value in changes_json.items():
        if json_key == "location":
            preview["location"] = value
        elif json_key == "employment_types":
            preview["employment_types_json"] = list(value) if isinstance(value, list) else value
        elif json_key == "current_stages":
            preview["current_stages_json"] = list(value) if isinstance(value, list) else value
        elif json_key in {"next_action", "comments", "engaged_days", "company", "role", "status", "priority", "job_link"}:
            preview[json_key] = value
        else:
            preview[json_key] = value
    return preview


_JSON_KEY_TO_APP_DICT_KEY = {
    "employment_types": "employment_types_json",
    "current_stages": "current_stages_json",
    "location": "location",
}


def _compute_changed_fields(original: JobApplication, changes_json: dict) -> list[str]:
    orig = _application_to_dict(original)
    changed = []
    for json_key in changes_json:
        orig_key = _JSON_KEY_TO_APP_DICT_KEY.get(json_key, json_key)
        orig_val = orig.get(orig_key)
        new_val = changes_json[json_key]
        if isinstance(orig_val, list) and isinstance(new_val, list):
            if orig_val != new_val:
                changed.append(json_key)
        elif orig_val != new_val:
            changed.append(json_key)
    return changed


def _change_draft_to_dict(cd: ApplicationChangeDraft, original: JobApplication) -> dict:
    preview = _build_preview_application(original, cd.changes_json)
    changed_fields = _compute_changed_fields(original, cd.changes_json)
    return {
        "id": cd.id,
        "kind": cd.kind,
        "target_application_id": cd.target_application_id,
        "original": _application_to_dict(original),
        "preview": preview,
        "changed_fields": changed_fields,
        "changes_json": cd.changes_json,
        "created_at": cd.created_at.isoformat() if cd.created_at else None,
        "updated_at": cd.updated_at.isoformat() if cd.updated_at else None,
    }


def handle_create_application_update_draft(payload: MutationPayload, db: Session) -> MutationResult:
    if payload.target.application_id is None:
        return _error("create_application_update_draft", "application_id is required")

    enum_error = _validate_enum_fields(payload.changes, "create_application_update_draft")
    if enum_error:
        return enum_error

    app = db.get(JobApplication, payload.target.application_id)
    if app is None:
        return _error("create_application_update_draft", f"Application {payload.target.application_id} not found")
    if app.is_draft:
        return _error("create_application_update_draft", "Cannot create pending changes for a draft application")
    if app.archived_at is not None:
        return _error("create_application_update_draft", "Cannot create pending changes for an archived application")

    new_changes = _changes_to_changes_json(payload.changes)
    if not new_changes:
        return _error("create_application_update_draft", "No changes provided")

    existing_cd = (
        db.query(ApplicationChangeDraft)
        .filter(ApplicationChangeDraft.target_application_id == app.id)
        .first()
    )
    if existing_cd is not None:
        # Merge new changes into existing draft
        existing_cd.changes_json = _merge_changes_json(existing_cd.changes_json, new_changes)
        from datetime import datetime, timezone
        existing_cd.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(existing_cd)
        cd_dict = _change_draft_to_dict(existing_cd, app)
        return MutationResult(
            success=True,
            operation="patch_application_update_draft",
            message="Pending changes updated.",
            change_draft=cd_dict,
            application=_application_to_dict(app),
        )

    cd = ApplicationChangeDraft(
        kind="update",
        target_application_id=app.id,
        changes_json=new_changes,
    )
    db.add(cd)
    db.commit()
    db.refresh(cd)
    cd_dict = _change_draft_to_dict(cd, app)
    return MutationResult(
        success=True,
        operation="create_application_update_draft",
        message="Pending changes created. Review and apply when ready.",
        change_draft=cd_dict,
        application=_application_to_dict(app),
    )


def handle_patch_application_update_draft(payload: MutationPayload, db: Session) -> MutationResult:
    if payload.target.change_draft_id is None:
        return _error("patch_application_update_draft", "change_draft_id is required")

    enum_error = _validate_enum_fields(payload.changes, "patch_application_update_draft")
    if enum_error:
        return enum_error

    cd = db.get(ApplicationChangeDraft, payload.target.change_draft_id)
    if cd is None:
        return _error("patch_application_update_draft", f"Change draft {payload.target.change_draft_id} not found")

    app = db.get(JobApplication, cd.target_application_id)
    if app is None:
        return _error("patch_application_update_draft", "Target application not found")

    new_changes = _changes_to_changes_json(payload.changes)
    cd.changes_json = _merge_changes_json(cd.changes_json, new_changes)
    from datetime import datetime, timezone
    cd.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(cd)
    cd_dict = _change_draft_to_dict(cd, app)
    return MutationResult(
        success=True,
        operation="patch_application_update_draft",
        message="Pending changes updated.",
        change_draft=cd_dict,
        application=_application_to_dict(app),
    )


def handle_apply_application_update_draft(payload: MutationPayload, db: Session) -> MutationResult:
    if payload.target.change_draft_id is None:
        return _error("apply_application_update_draft", "change_draft_id is required")

    cd = db.get(ApplicationChangeDraft, payload.target.change_draft_id)
    if cd is None:
        return _error("apply_application_update_draft", f"Change draft {payload.target.change_draft_id} not found")

    app = db.get(JobApplication, cd.target_application_id)
    if app is None:
        db.delete(cd)
        db.commit()
        return _error("apply_application_update_draft", "Target application no longer exists")
    if app.archived_at is not None:
        return MutationResult(
            success=False,
            conflict=True,
            operation="apply_application_update_draft",
            message="Cannot apply changes: the application is archived. Restore it first or discard the pending changes.",
        )

    changes = cd.changes_json

    # Build ApplicationChanges from stored JSON for validation
    app_changes = ApplicationChanges(
        company=changes.get("company"),
        role=changes.get("role"),
        status=changes.get("status"),
        priority=changes.get("priority"),
        location_mode=changes.get("location"),
        job_link=changes.get("job_link"),
        employment_types=changes.get("employment_types"),
        current_stages=changes.get("current_stages"),
        next_action=changes.get("next_action"),
        comments=changes.get("comments"),
        engaged_days=changes.get("engaged_days"),
    )

    enum_error = _validate_enum_fields(app_changes, "apply_application_update_draft")
    if enum_error:
        return enum_error

    # Enforce uniqueness if company or role is changing
    new_company_id = app.company_id
    if app_changes.company is not None:
        new_company_obj = get_or_create_company(db, app_changes.company)
        new_company_id = new_company_obj.id
    new_normalized_role = normalize_role_name(app_changes.role) if app_changes.role is not None else app.normalized_role

    if app_changes.company is not None or app_changes.role is not None:
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
            return MutationResult(
                success=False,
                conflict=True,
                operation="apply_application_update_draft",
                message=f"Cannot apply: an application for {collision_company} — {collision.role} already exists.",
            )

    # Capture old values for event logging
    old_status = app.status
    old_priority = app.priority
    old_location = app.location
    old_job_link = app.job_link
    old_company = app.company
    old_role = app.role
    old_employment_types = list(app.employment_types_json)
    old_current_stages = list(app.current_stages_json)

    _apply_changes_to_application(app, app_changes, db)

    if app_changes.status is not None and app.status != old_status:
        _append_event(app.id, "status_changed", {"field": "status", "from": old_status, "to": app.status}, db)
    if app_changes.priority is not None and app.priority != old_priority:
        _append_event(app.id, "field_changed", {"field": "priority", "from": old_priority, "to": app.priority}, db)
    if app_changes.location_mode is not None and app.location != old_location:
        _append_event(app.id, "field_changed", {"field": "location", "from": old_location, "to": app.location}, db)
    if app_changes.job_link is not None and app.job_link != old_job_link:
        _append_event(app.id, "field_changed", {"field": "job_link", "from": old_job_link, "to": app.job_link}, db)
    if app_changes.company is not None and app.company != old_company:
        _append_event(app.id, "field_changed", {"field": "company", "from": old_company, "to": app.company}, db)
    if app_changes.role is not None and app.role != old_role:
        _append_event(app.id, "field_changed", {"field": "role", "from": old_role, "to": app.role}, db)
    if app_changes.employment_types is not None and app.employment_types_json != old_employment_types:
        _append_event(app.id, "field_changed", {"field": "employment_types", "from": old_employment_types, "to": app.employment_types_json}, db)
    if app_changes.current_stages is not None and app.current_stages_json != old_current_stages:
        _append_event(app.id, "field_changed", {"field": "current_stages", "from": old_current_stages, "to": app.current_stages_json}, db)

    db.delete(cd)
    db.commit()
    db.refresh(app)
    return MutationResult(
        success=True,
        operation="apply_application_update_draft",
        message="Changes applied.",
        application=_application_to_dict(app),
    )


def handle_discard_application_update_draft(payload: MutationPayload, db: Session) -> MutationResult:
    if payload.target.change_draft_id is None:
        return _error("discard_application_update_draft", "change_draft_id is required")

    cd = db.get(ApplicationChangeDraft, payload.target.change_draft_id)
    if cd is None:
        return _error("discard_application_update_draft", f"Change draft {payload.target.change_draft_id} not found")

    db.delete(cd)
    db.commit()
    return MutationResult(
        success=True,
        operation="discard_application_update_draft",
        message="Pending changes discarded.",
    )


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
        "create_application_update_draft": handle_create_application_update_draft,
        "patch_application_update_draft": handle_patch_application_update_draft,
        "apply_application_update_draft": handle_apply_application_update_draft,
        "discard_application_update_draft": handle_discard_application_update_draft,
        "set_active_application": handle_set_active_application,
    }
    return handlers[payload.operation](payload, db)
