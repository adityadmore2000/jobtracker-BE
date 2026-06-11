from __future__ import annotations

from datetime import datetime
from typing import Any

from .models import JobApplication
from .mutation_schemas import MutationResult
from .public_schemas import (
    PublicApplicationChangeDraftDTO,
    PublicApplicationDTO,
    PublicCollisionDTO,
    PublicNoteDTO,
    PublicTranscriptResponse,
)
from .schemas import JobApplicationBase, SemanticTranscriptResponse


def to_public_application(source: JobApplication | dict) -> PublicApplicationDTO:
    """Convert a DB row or _application_to_dict snapshot to PublicApplicationDTO."""
    if isinstance(source, dict):
        return PublicApplicationDTO(
            id=source["id"],
            company=source["company"],
            role=source.get("role") or "",
            employment_types=source.get("employment_types_json") or [],
            job_link=source.get("job_link") or "",
            location=source.get("location") or "",
            status=source.get("status") or "",
            current_stages=source.get("current_stages_json") or [],
            priority=source.get("priority") or "",
            engaged_days=source.get("engaged_days") or 0,
            next_action=source.get("next_action") or "",
            comments=source.get("comments") or "",
            is_draft=source.get("is_draft", False),
            draft_created_at=source.get("draft_created_at"),
            archived_at=source.get("archived_at"),
            created_at=source["created_at"],
            updated_at=source["updated_at"],
        )
    return PublicApplicationDTO(
        id=source.id,
        company=source.company,
        role=source.role or "",
        employment_types=list(source.employment_types_json) if source.employment_types_json else [],
        job_link=source.job_link or "",
        location=source.location or "",
        status=source.status or "",
        current_stages=list(source.current_stages_json) if source.current_stages_json else [],
        priority=source.priority or "",
        engaged_days=source.engaged_days or 0,
        next_action=source.next_action or "",
        comments=source.comments or "",
        is_draft=source.is_draft,
        draft_created_at=source.draft_created_at,
        archived_at=source.archived_at,
        created_at=source.created_at,
        updated_at=source.updated_at,
    )


def _app_base_to_public_draft(base: JobApplicationBase, draft_id: str | None, draft_dict: dict | None) -> PublicApplicationDTO:
    """Build a public draft DTO from a JobApplicationBase (no DB id) or dict snapshot."""
    if draft_dict and "id" in draft_dict:
        return PublicApplicationDTO(
            id=draft_dict["id"],
            company=draft_dict.get("company") or base.company,
            role=draft_dict.get("role") or base.role,
            employment_types=draft_dict.get("employment_types_json") or list(base.employment_types_json),
            job_link=draft_dict.get("job_link") or base.job_link,
            location=draft_dict.get("location") or base.location,
            status=draft_dict.get("status") or base.status,
            current_stages=draft_dict.get("current_stages_json") or list(base.current_stages_json),
            priority=draft_dict.get("priority") or base.priority,
            engaged_days=draft_dict.get("engaged_days") or base.engaged_days,
            next_action=draft_dict.get("next_action") or base.next_action,
            comments=draft_dict.get("comments") or base.comments,
            is_draft=True,
            draft_created_at=draft_dict.get("draft_created_at"),
            archived_at=None,
            created_at=draft_dict.get("created_at") or draft_dict.get("draft_created_at"),
            updated_at=draft_dict.get("updated_at") or draft_dict.get("draft_created_at"),
        )
    return PublicApplicationDTO(
        id=0,
        company=base.company,
        role=base.role,
        employment_types=list(base.employment_types_json),
        job_link=base.job_link,
        location=base.location,
        status=base.status,
        current_stages=list(base.current_stages_json),
        priority=base.priority,
        engaged_days=base.engaged_days,
        next_action=base.next_action,
        comments=base.comments,
        is_draft=True,
        draft_created_at=None,
        archived_at=None,
        created_at=None,  # type: ignore[arg-type]
        updated_at=None,  # type: ignore[arg-type]
    )


def to_public_change_draft(cd_dict: dict) -> PublicApplicationChangeDraftDTO | None:
    """Convert a change_draft dict (from _change_draft_to_dict) to the public DTO."""
    if not cd_dict:
        return None
    original_raw = cd_dict.get("original")
    preview_raw = cd_dict.get("preview")
    if not original_raw or not preview_raw:
        return None

    def _dict_to_app_dto(d: dict) -> PublicApplicationDTO:
        created_raw = d.get("created_at")
        updated_raw = d.get("updated_at")
        created_dt = datetime.fromisoformat(created_raw) if isinstance(created_raw, str) else created_raw
        updated_dt = datetime.fromisoformat(updated_raw) if isinstance(updated_raw, str) else updated_raw
        return PublicApplicationDTO(
            id=d.get("id", 0),
            company=d.get("company", ""),
            role=d.get("role", ""),
            employment_types=d.get("employment_types_json") or [],
            job_link=d.get("job_link", ""),
            location=d.get("location", ""),
            status=d.get("status", ""),
            current_stages=d.get("current_stages_json") or [],
            priority=d.get("priority", ""),
            engaged_days=d.get("engaged_days", 0),
            next_action=d.get("next_action", ""),
            comments=d.get("comments", ""),
            is_draft=d.get("is_draft", False),
            draft_created_at=d.get("draft_created_at"),
            archived_at=d.get("archived_at"),
            created_at=created_dt,
            updated_at=updated_dt,
        )

    created_at_raw = cd_dict.get("created_at")
    updated_at_raw = cd_dict.get("updated_at")
    created_at = datetime.fromisoformat(created_at_raw) if isinstance(created_at_raw, str) else created_at_raw
    updated_at = datetime.fromisoformat(updated_at_raw) if isinstance(updated_at_raw, str) else updated_at_raw

    return PublicApplicationChangeDraftDTO(
        id=cd_dict["id"],
        kind=cd_dict.get("kind", "update"),
        target_application_id=cd_dict["target_application_id"],
        original=_dict_to_app_dto(original_raw),
        preview=_dict_to_app_dto(preview_raw),
        changed_fields=cd_dict.get("changed_fields", []),
        created_at=created_at,
        updated_at=updated_at,
    )


# Internal status → public status mapping
_INTERNAL_TO_PUBLIC: dict[str, str] = {
    "preview": "__resolve_by_operation__",
    "clarification_required": "clarification",
    "unsupported": "no_change",
    "unavailable": "error",
}

_OPERATION_TO_PUBLIC_STATUS: dict[str, str] = {
    "create_draft": "draft_created",
    "patch_draft": "draft_updated",
    "save_draft": "saved",
    "discard_draft": "discarded",
    "patch_application": "updated",
    "ask_clarification": "clarification",
    "append_note": "updated",
    "archive_application": "updated",
    "restore_application": "updated",
    "create_application_update_draft": "pending_changes_created",
    "patch_application_update_draft": "pending_changes_updated",
    "apply_application_update_draft": "changes_applied",
    "discard_application_update_draft": "changes_discarded",
}

_DEFAULT_MESSAGES: dict[str, str] = {
    "draft_created": "Draft created. Review it and save when ready.",
    "draft_updated": "Draft updated.",
    "saved": "Application saved.",
    "discarded": "Draft discarded.",
    "updated": "Application updated.",
    "clarification": "Please clarify.",
    "no_change": "No change was made.",
    "error": "An error occurred.",
    "pending_changes_created": "Pending changes created. Review and apply when ready.",
    "pending_changes_updated": "Pending changes updated.",
    "changes_applied": "Changes applied.",
    "changes_discarded": "Pending changes discarded.",
}


def to_public_transcript_response(internal: SemanticTranscriptResponse) -> PublicTranscriptResponse:
    """Map SemanticTranscriptResponse → PublicTranscriptResponse."""
    internal_status = internal.status
    operation = internal.operation

    if internal_status == "preview":
        if operation == "create":
            if internal.draft_id is not None:
                tool = getattr(internal.proposal, "tool_name", None) or ""
                if tool == "patch_active_draft":
                    if internal.confirmation_kind == "context":
                        public_status = "draft_updated"
                    else:
                        public_status = "draft_created"
                else:
                    public_status = "draft_created"
            elif internal.application_id is not None:
                public_status = "saved"
            else:
                public_status = "draft_created"
        elif operation == "update":
            public_status = "updated"
        elif operation == "pending_changes":
            # Distinguish create vs patch from internal operation
            internal_op = internal.operation if hasattr(internal, "operation") else ""
            if internal_op == "patch_application_update_draft":
                public_status = "pending_changes_updated"
            else:
                public_status = "pending_changes_created"
        else:
            public_status = "no_change"
    elif internal_status == "clarification_required":
        public_status = "clarification"
    elif internal_status == "unsupported":
        public_status = "no_change"
    elif internal_status == "unavailable":
        public_status = "error"
    else:
        public_status = "no_change"

    if public_status == "clarification":
        message = internal.clarification_question or _DEFAULT_MESSAGES["clarification"]
    elif public_status == "no_change":
        message = (internal.warnings[0] if internal.warnings else _DEFAULT_MESSAGES["no_change"])
    elif public_status == "error":
        message = (internal.warnings[0] if internal.warnings else _DEFAULT_MESSAGES["error"])
    else:
        message = _DEFAULT_MESSAGES.get(public_status, "Done.")

    public_draft: PublicApplicationDTO | None = None
    if internal.draft_dict is not None:
        # Prefer the persisted row dict — has the real id, created_at, updated_at.
        public_draft = to_public_application(internal.draft_dict)
    elif internal.draft is not None:
        public_draft = PublicApplicationDTO(
            id=0,
            company=internal.draft.company,
            role=internal.draft.role,
            employment_types=list(internal.draft.employment_types_json),
            job_link=internal.draft.job_link or "",
            location=internal.draft.location or "",
            status=internal.draft.status or "",
            current_stages=list(internal.draft.current_stages_json),
            priority=internal.draft.priority or "",
            engaged_days=internal.draft.engaged_days or 0,
            next_action=internal.draft.next_action or "",
            comments=internal.draft.comments or "",
            is_draft=True,
            draft_created_at=None,
            archived_at=None,
            created_at=None,  # type: ignore[arg-type]
            updated_at=None,  # type: ignore[arg-type]
        )

    public_change_draft: PublicApplicationChangeDraftDTO | None = None
    if internal.change_draft is not None:
        public_change_draft = to_public_change_draft(internal.change_draft)

    return PublicTranscriptResponse(
        status=public_status,  # type: ignore[arg-type]
        message=message,
        application_id=internal.application_id,
        draft_id=internal.draft_id,
        draft=public_draft,
        application=None,
        pending_changes=public_change_draft,
        warnings=list(internal.warnings),
        clarification_question=internal.clarification_question if public_status == "clarification" else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Controlled command layer: build response directly from MutationResult
# ─────────────────────────────────────────────────────────────────────────────

_MUTATION_OP_TO_PUBLIC_STATUS: dict[str, str] = {
    "create_draft": "draft_created",
    "draft_updated": "draft_updated",
    "patch_draft": "draft_updated",
    "save_draft": "saved",
    "discard_draft": "discarded",
    "patch_application": "updated",
    "updated": "updated",
    "no_change": "no_change",
    "ask_clarification": "clarification",
    "append_note": "note_added",
    "archive_application": "application_archived",
    "restore_application": "application_restored",
    "create_application_update_draft": "pending_changes_created",
    "patch_application_update_draft": "pending_changes_updated",
    "apply_application_update_draft": "changes_applied",
    "discard_application_update_draft": "changes_discarded",
    "set_active_application": "context_updated",
}

_DEFAULT_OP_MESSAGES: dict[str, str] = {
    "draft_created": "Draft created. Review it and save when ready.",
    "draft_updated": "Draft updated.",
    "saved": "Application saved.",
    "discarded": "Draft discarded.",
    "updated": "Application updated.",
    "clarification": "Please clarify.",
    "no_change": "No change was made.",
    "error": "An error occurred.",
    "pending_changes_created": "Pending changes created. Review and apply when ready.",
    "pending_changes_updated": "Pending changes updated.",
    "changes_applied": "Changes applied.",
    "changes_discarded": "Pending changes discarded.",
    "note_added": "Note added.",
    "application_archived": "Application archived.",
    "application_restored": "Application restored.",
    "context_updated": "Application selected.",
    "unsupported": "I could not identify a tracker command. Please use a supported command such as: set priority as medium.",
}


def mutation_result_to_public_response(
    result: MutationResult,
    *,
    pending_command: dict[str, Any] | None = None,
) -> PublicTranscriptResponse:
    """Convert a MutationResult from the controlled command layer to the public response."""
    public_status = _MUTATION_OP_TO_PUBLIC_STATUS.get(result.operation, "no_change")

    if result.clarification_question:
        message = result.clarification_question
    elif not result.success:
        message = result.message
    else:
        message = result.message or _DEFAULT_OP_MESSAGES.get(public_status, "Done.")

    # Build draft DTO
    public_draft: PublicApplicationDTO | None = None
    draft_id: str | None = None
    if result.draft:
        public_draft = to_public_application(result.draft)
        if result.draft.get("is_draft") and result.draft.get("id"):
            draft_id = str(result.draft["id"])

    # Build application DTO
    public_app: PublicApplicationDTO | None = None
    application_id: int | None = None
    if result.application:
        public_app = to_public_application(result.application)
        application_id = result.application.get("id")

    # Build note DTO (first note from the result)
    public_note: PublicNoteDTO | None = None
    if result.notes:
        first = result.notes[0]
        from datetime import datetime as _dt
        created_raw = first.get("created_at")
        created_dt = _dt.fromisoformat(created_raw) if isinstance(created_raw, str) else created_raw
        public_note = PublicNoteDTO(
            id=first["id"],
            text=first["text"],
            created_at=created_dt,
        )

    # Build change-draft DTO
    public_change_draft: PublicApplicationChangeDraftDTO | None = None
    if result.change_draft:
        public_change_draft = to_public_change_draft(result.change_draft)

    public_collision: PublicCollisionDTO | None = None
    if result.collision is not None:
        public_collision = PublicCollisionDTO(
            kind=result.collision.kind,  # type: ignore[arg-type]
            draft_id=result.collision.draft_id,
            application_id=result.collision.application_id,
            company=result.collision.company,
            role=result.collision.role,
            archived=result.collision.archived,
        )

    return PublicTranscriptResponse(
        status=public_status,  # type: ignore[arg-type]
        message=message,
        application_id=application_id,
        draft_id=draft_id,
        draft=public_draft,
        application=public_app,
        pending_changes=public_change_draft,
        warnings=[],
        clarification_question=result.clarification_question,
        note=public_note,
        pending_command=pending_command,
        collision=public_collision,
    )


def unsupported_command_response() -> PublicTranscriptResponse:
    """Safe response returned when no supported command anchor is recognised."""
    return PublicTranscriptResponse(
        status="unsupported",  # type: ignore[arg-type]
        message=_DEFAULT_OP_MESSAGES["unsupported"],
        warnings=[],
    )


def clarification_needed_response(
    question: str,
    pending_command: dict[str, Any] | None = None,
) -> PublicTranscriptResponse:
    """Response returned when the parser found a supported command but needs more info."""
    return PublicTranscriptResponse(
        status="clarification",
        message=question,
        clarification_question=question,
        pending_command=pending_command,
        warnings=[],
    )


def suggestion_only_response(
    message: str,
    *,
    clarification_question: str | None = None,
    suggested_phrasings: list[str] | None = None,
) -> PublicTranscriptResponse:
    """Safe no-mutation response that offers clickable rephrasings.

    Uses the ``unsupported`` status so the frontend never treats it as a mutation.
    """
    return PublicTranscriptResponse(
        status="unsupported",
        message=message,
        clarification_question=clarification_question,
        suggested_phrasings=suggested_phrasings or [],
        warnings=[],
    )


def mixed_intent_response(message: str) -> PublicTranscriptResponse:
    """Safe no-mutation response for a transcript that mixes a field update and a note."""
    return PublicTranscriptResponse(
        status="unsupported",
        message=message,
        warnings=[],
    )
