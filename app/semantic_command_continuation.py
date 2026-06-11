"""
Clarification continuation.

When the previous turn returned a clarification with a ``pending_command``, the
frontend echoes that command back together with the user's short reply (the
transcript). This module consumes the pending command *before* any normal
parsing and fills ONLY the declared missing field from the reply.

Strict safety rules:
  * The reply may fill only the single declared missing field (company or role).
  * The reply cannot change the operation, the changes, or the note.
  * The reply never reaches the LLM.
  * Cancel words clear the pending command.
  * If another field is still missing, an updated pending_command is returned.

All outcomes reuse the deterministic ``semantic_command_pipeline`` resolution so
the routing matches the rest of the system exactly.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy.orm import Session

from .mutation_schemas import ApplicationChanges, MutationPayload, MutationTarget
from .semantic_command_pipeline import (
    ClarificationOutcome,
    DispatchOutcome,
    PipelineOutcome,
    SuggestionOutcome,
    _resolve_archive_target,
    _resolve_note_target,
    _resolve_update_target,
)
from .semantic_command_schemas import SemanticChanges, SemanticCommand, SemanticTarget

logger = logging.getLogger(__name__)

_CANCEL_WORDS = {"cancel", "never mind", "nevermind", "stop", "forget it", "no"}

# Pending operations the continuation can resume, grouped by routing family.
# This covers both the fast-path-emitted operations (set_field, set_current_stages,
# update_application, remove_application) and the semantic-pipeline-emitted ones.
_UPDATE_OPERATIONS = {"update_application", "set_field", "set_current_stages"}
_NOTE_OPERATIONS = {"append_note"}
_ARCHIVE_OPERATIONS = {"archive_application", "remove_application"}
_CONTINUATION_OPERATIONS = _UPDATE_OPERATIONS | _NOTE_OPERATIONS | _ARCHIVE_OPERATIONS


class _PendingTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    company: Optional[str] = None
    role: Optional[str] = None
    application_id: Optional[int] = None


class _PendingCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: str
    target: _PendingTarget = _PendingTarget()
    changes: dict[str, Any] = {}
    note: Optional[str] = None
    missing_field: Optional[str] = None


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = " ".join(value.split())
    return stripped or None


def resume_pending_command(
    pending_command: dict | None,
    transcript: str,
    context: dict,
    db: Session,
) -> Optional[PipelineOutcome]:
    """Attempt to continue a pending clarification.

    Returns:
        None — no usable pending command (caller proceeds with normal parsing).
        PipelineOutcome — the continuation was consumed (dispatch / clarification / cancel).
    """
    if not pending_command:
        return None

    try:
        pending = _PendingCommand.model_validate(pending_command)
    except ValidationError:
        logger.info("semantic_continuation_rejected reason=invalid_pending_command")
        return None

    if pending.operation not in _CONTINUATION_OPERATIONS:
        # Lifecycle / fast-path-only pending commands are handled elsewhere.
        return None
    if pending.missing_field not in {"company", "role"}:
        return None

    reply = _clean(transcript)
    if reply is None:
        return None
    if reply.casefold() in _CANCEL_WORDS:
        logger.info("semantic_continuation_cancelled")
        return SuggestionOutcome(message="Okay, cancelled.")

    # Fill ONLY the declared missing field from the reply.
    company = _clean(pending.target.company)
    role = _clean(pending.target.role)
    if pending.missing_field == "company":
        company = reply
    else:  # role
        role = reply

    if pending.operation in _UPDATE_OPERATIONS:
        family = "update_application"
    elif pending.operation in _NOTE_OPERATIONS:
        family = "append_note"
    else:
        family = "archive_application"

    cmd = SemanticCommand(
        intent=family,
        target=SemanticTarget(company=company, role=role),
        changes=SemanticChanges(),
        note=pending.note,
    )

    logger.info(
        "semantic_continuation_invoked operation=%s family=%s missing=%s filled=%r",
        pending.operation, family, pending.missing_field, reply,
    )

    if family == "update_application":
        safe = _safe_changes(pending.changes)
        if not safe:
            # The original command carried no field to change — cannot resume safely.
            return SuggestionOutcome(message="I lost the field to change. Please send the update again.")
        resolved = _resolve_update_target(cmd, context, db)
        if resolved.clarification:
            return ClarificationOutcome(
                question=resolved.clarification,
                pending_command=_carry_forward(pending, company, role, resolved.missing_field),
            )
        return DispatchOutcome(
            MutationPayload(operation=resolved.operation, target=resolved.target, changes=ApplicationChanges(**safe))
        )

    if family == "append_note":
        note = _clean(pending.note)
        if not note:
            return SuggestionOutcome(message="I lost the note text. Please add the note again.")
        resolved = _resolve_note_target(cmd, context, db)
        if resolved.clarification:
            return ClarificationOutcome(
                question=resolved.clarification,
                pending_command=_carry_forward(pending, company, role, resolved.missing_field),
            )
        return DispatchOutcome(
            MutationPayload(
                operation="append_note",
                target=resolved.target,
                changes=ApplicationChanges(),
                notes_to_append=[note],
            )
        )

    # archive_application
    resolved = _resolve_archive_target(cmd, context, db)
    if resolved.clarification:
        return ClarificationOutcome(
            question=resolved.clarification,
            pending_command=_carry_forward(pending, company, role, resolved.missing_field),
        )
    return DispatchOutcome(
        MutationPayload(operation="archive_application", target=resolved.target, changes=ApplicationChanges())
    )


# The clarification reply cannot replace changes — only the original pending
# changes (already-validated field names) survive.
_PENDING_CHANGE_KEYS = {
    "status", "priority", "location_mode", "employment_types",
    "current_stages", "job_link", "engaged_days", "next_action", "comments",
}


def _safe_changes(raw: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in raw.items() if k in _PENDING_CHANGE_KEYS and v is not None}


def _carry_forward(pending: _PendingCommand, company: str | None, role: str | None, missing: str | None) -> dict:
    return {
        "operation": pending.operation,
        "target": {"company": company, "role": role, "application_id": None},
        "changes": _safe_changes(pending.changes),
        "note": pending.note,
        "missing_field": missing or "role",
    }
