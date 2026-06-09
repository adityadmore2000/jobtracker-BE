from __future__ import annotations

from .models import JobApplication
from .public_schemas import PublicApplicationDTO, PublicTranscriptResponse
from .schemas import JobApplicationBase, SemanticTranscriptResponse


def to_public_application(source: JobApplication | dict) -> PublicApplicationDTO:
    """Convert a DB row or _application_to_dict snapshot to PublicApplicationDTO."""
    if isinstance(source, dict):
        return PublicApplicationDTO(
            id=source["id"],
            company=source["company"],
            roles=source.get("roles_json") or [],
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
        roles=list(source.roles_json) if source.roles_json else [],
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
            roles=draft_dict.get("roles_json") or list(base.roles_json),
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
    # Preview draft with no persisted row yet: use a sentinel id=0
    return PublicApplicationDTO(
        id=0,
        company=base.company,
        roles=list(base.roles_json),
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


# Internal status → public status mapping
_INTERNAL_TO_PUBLIC: dict[str, str] = {
    # SemanticTranscriptResponse.status → PublicTranscriptResponse.status
    "preview": "__resolve_by_operation__",   # resolved by MutationResult.operation
    "clarification_required": "clarification",
    "unsupported": "no_change",
    "unavailable": "error",
}

# Internal MutationResult.operation → public status (when internal status == "preview")
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
}


def to_public_transcript_response(internal: SemanticTranscriptResponse) -> PublicTranscriptResponse:
    """Map SemanticTranscriptResponse → PublicTranscriptResponse."""
    internal_status = internal.status
    operation = internal.operation  # "create" | "update" | "none"

    # Determine public status
    if internal_status == "preview":
        # Derive from the MutationResult operation embedded in the proposal
        # The proposal.tool_name tells us which handler ran; but the internal
        # response was built from MutationResult.operation. We recover it by
        # looking at the operation field ("create" / "update" / "none") and the
        # draft/application shape.
        if operation == "create":
            if internal.draft_id is not None:
                # draft_id present means create_draft or patch_draft ran
                # Distinguish: if the draft field contains an id, we check whether
                # draft was previously present (the semantic layer does not distinguish
                # create vs patch in the operation field, but uses "create" for both).
                # We use proposal.tool_name as the reliable discriminator.
                tool = getattr(internal.proposal, "tool_name", None) or ""
                if tool == "patch_active_draft":
                    # patch_active_draft maps to create_draft (no existing draft) OR
                    # patch_draft (existing draft). The mutation_result operation is in
                    # MutationResult but not surfaced in SemanticTranscriptResponse.
                    # We look at needs_confirmation / confirmation_kind as a proxy.
                    # The most reliable signal: if draft_id was already in context
                    # (from the incoming payload) and we returned the same id, it's
                    # a patch. But we can't recover context here.
                    # Safe approach: trust the draft DTO. If draft has no id (id==0
                    # or None), it's a pure preview without mutation; if it has an id,
                    # the mutation ran. The public status distinguishes create vs update
                    # only if the mutation was a patch_draft vs create_draft.
                    # Since the SemanticTranscriptResponse does NOT carry the internal
                    # MutationResult operation, we use confirmation_kind as a proxy:
                    # "context" → patch, "none" → create (new draft).
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
        else:
            # operation == "none" under "preview" should not occur in practice;
            # treat as no_change.
            public_status = "no_change"
    elif internal_status == "clarification_required":
        public_status = "clarification"
    elif internal_status == "unsupported":
        public_status = "no_change"
    elif internal_status == "unavailable":
        public_status = "error"
    else:
        public_status = "no_change"

    # Message: prefer warnings/clarification, then fall back to canonical default
    if public_status == "clarification":
        message = internal.clarification_question or _DEFAULT_MESSAGES["clarification"]
    elif public_status == "no_change":
        message = (internal.warnings[0] if internal.warnings else _DEFAULT_MESSAGES["no_change"])
    elif public_status == "error":
        message = (internal.warnings[0] if internal.warnings else _DEFAULT_MESSAGES["error"])
    else:
        message = _DEFAULT_MESSAGES.get(public_status, "Done.")

    # Build public draft DTO
    public_draft: PublicApplicationDTO | None = None
    if internal.draft is not None:
        public_draft = PublicApplicationDTO(
            id=0,
            company=internal.draft.company,
            roles=list(internal.draft.roles_json),
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

    return PublicTranscriptResponse(
        status=public_status,  # type: ignore[arg-type]
        message=message,
        application_id=internal.application_id,
        draft_id=internal.draft_id,
        draft=public_draft,
        application=None,
        warnings=list(internal.warnings),
        clarification_question=internal.clarification_question if public_status == "clarification" else None,
    )
