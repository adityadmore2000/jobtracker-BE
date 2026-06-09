from datetime import datetime, timezone

from sqlalchemy.orm import Session

from .constants import (
    ALLOWED_CURRENT_STAGES,
    ALLOWED_EMPLOYMENT_TYPES,
    ALLOWED_LOCATIONS,
    ALLOWED_PRIORITIES,
    STATUS_OPTIONS,
)
from .models import JobApplication
from .mutation_schemas import (
    ALLOWED_OPERATIONS,
    ApplicationChanges,
    MutationPayload,
    MutationResult,
    MutationTarget,
)


def _error(operation: str, message: str) -> MutationResult:
    return MutationResult(success=False, operation=operation, message=message)


def _validate_enum_fields(changes: ApplicationChanges, operation: str) -> MutationResult | None:
    if changes.priority is not None and changes.priority not in ALLOWED_PRIORITIES:
        return _error(operation, f"Invalid priority '{changes.priority}'. Allowed: {ALLOWED_PRIORITIES}")
    if changes.location_mode is not None and changes.location_mode not in ALLOWED_LOCATIONS:
        return _error(operation, f"Invalid location_mode '{changes.location_mode}'. Allowed: {ALLOWED_LOCATIONS}")
    if changes.status is not None and changes.status not in STATUS_OPTIONS and changes.status != "":
        return _error(operation, f"Invalid status '{changes.status}'. Allowed: {STATUS_OPTIONS}")
    if changes.employment_type is not None and changes.employment_type not in ALLOWED_EMPLOYMENT_TYPES:
        return _error(operation, f"Invalid employment_type '{changes.employment_type}'. Allowed: {ALLOWED_EMPLOYMENT_TYPES}")
    if changes.current_stage is not None and changes.current_stage not in ALLOWED_CURRENT_STAGES:
        return _error(operation, f"Invalid current_stage '{changes.current_stage}'. Allowed: {ALLOWED_CURRENT_STAGES}")
    return None


def _application_to_dict(app: JobApplication) -> dict:
    return {
        "id": app.id,
        "company": app.company,
        "roles_json": app.roles_json,
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
        "created_at": app.created_at.isoformat() if app.created_at else None,
        "updated_at": app.updated_at.isoformat() if app.updated_at else None,
    }


def _apply_changes_to_application(app: JobApplication, changes: ApplicationChanges) -> None:
    if changes.company is not None:
        app.company = changes.company
    if changes.role is not None:
        app.roles_json = [changes.role]
    if changes.status is not None:
        app.status = changes.status
    if changes.priority is not None:
        app.priority = changes.priority
    if changes.location_mode is not None:
        app.location = changes.location_mode
    if changes.job_link is not None:
        app.job_link = changes.job_link
    if changes.employment_type is not None:
        app.employment_types_json = [changes.employment_type]
    if changes.current_stage is not None:
        app.current_stages_json = [changes.current_stage]


def handle_create_draft(payload: MutationPayload, db: Session) -> MutationResult:
    if not payload.changes.company:
        return _error("create_draft", "company is required to create a draft")
    enum_error = _validate_enum_fields(payload.changes, "create_draft")
    if enum_error:
        return enum_error

    app = JobApplication(
        company=payload.changes.company,
        roles_json=[payload.changes.role] if payload.changes.role else [],
        employment_types_json=[payload.changes.employment_type] if payload.changes.employment_type else [],
        job_link=payload.changes.job_link or "",
        location=payload.changes.location_mode or "",
        status=payload.changes.status or "",
        current_stages_json=[payload.changes.current_stage] if payload.changes.current_stage else [],
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


def handle_patch_draft(payload: MutationPayload, db: Session) -> MutationResult:
    enum_error = _validate_enum_fields(payload.changes, "patch_draft")
    if enum_error:
        return enum_error

    if payload.target.draft_id is not None:
        try:
            draft_id = int(payload.target.draft_id)
        except (ValueError, TypeError):
            return _error("patch_draft", f"Invalid draft_id '{payload.target.draft_id}'")
        app = db.get(JobApplication, draft_id)
        if app is None or not app.is_draft:
            return _error("patch_draft", f"Draft with id {draft_id} not found")
        _apply_changes_to_application(app, payload.changes)
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
        app.is_draft = False
        app.draft_created_at = None
        db.commit()
        db.refresh(app)
        return MutationResult(
            success=True,
            operation="save_draft",
            message="Draft saved as application.",
            application=_application_to_dict(app),
        )

    if not payload.changes.company:
        return _error("save_draft", "Draft must have company to be saved.")
    if not payload.changes.role:
        return _error("save_draft", "Draft must have role to be saved.")
    draft = _changes_to_dict(payload.changes)
    return MutationResult(
        success=True,
        operation="save_draft",
        message="Draft is ready to save.",
        draft=draft,
        requires_confirmation=True,
        confirmation_kind="save",
    )


def handle_discard_draft(payload: MutationPayload, db: Session) -> MutationResult:
    if payload.target.draft_id is not None:
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

    _apply_changes_to_application(app, payload.changes)
    db.commit()
    db.refresh(app)
    return MutationResult(
        success=True,
        operation="patch_application",
        message="Application patched.",
        application=_application_to_dict(app),
    )


def handle_ask_clarification(payload: MutationPayload, db: Session) -> MutationResult:
    question = payload.notes_to_append[0] if payload.notes_to_append else "Please clarify."
    return MutationResult(
        success=True,
        operation="ask_clarification",
        message="Clarification required.",
        clarification_question=question,
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
    }
    return handlers[payload.operation](payload, db)
