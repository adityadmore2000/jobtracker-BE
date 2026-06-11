from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel


class PublicApplicationDTO(BaseModel):
    id: int
    company: str
    role: str
    employment_types: list[str]
    job_link: str
    location: str
    status: str
    current_stages: list[str]
    priority: str
    engaged_days: int
    next_action: str
    comments: str
    is_draft: bool
    draft_created_at: datetime | None
    archived_at: datetime | None
    # None only for unpersisted preview drafts (no DB row yet)
    created_at: datetime | None
    updated_at: datetime | None


class PublicApplicationChangeDraftDTO(BaseModel):
    id: int
    kind: str
    target_application_id: int
    original: PublicApplicationDTO
    preview: PublicApplicationDTO
    changed_fields: list[str]
    created_at: datetime
    updated_at: datetime


class PublicNoteDTO(BaseModel):
    id: int
    text: str
    created_at: datetime | None


# pending_command shape echoed back to the frontend for clarification continuation.
class PendingCommandTarget(BaseModel, extra="forbid"):
    company: str | None = None
    role: str | None = None
    application_id: int | None = None


class PendingCommand(BaseModel, extra="forbid"):
    operation: Literal[
        "remove_application",
        "update_application",
        "append_note",
        "set_role",
        "set_priority",
        "set_location",
        "set_employment_type",
        "set_current_stages",
    ]
    target: PendingCommandTarget
    changes: dict[str, Any]
    note: str | None = None
    missing_field: Literal["company", "role"] | None = None


TranscriptStatus = Literal[
    "draft_created",
    "draft_updated",
    "saved",
    "discarded",
    "updated",
    "clarification",
    "no_change",
    "error",
    "pending_changes_created",
    "pending_changes_updated",
    "changes_applied",
    "changes_discarded",
    "note_added",
    "application_archived",
    "application_restored",
    "context_updated",
    "unsupported",
]


class PublicTranscriptResponse(BaseModel):
    status: TranscriptStatus
    message: str
    application_id: int | None = None
    draft_id: str | None = None
    draft: PublicApplicationDTO | None = None
    application: PublicApplicationDTO | None = None
    pending_changes: PublicApplicationChangeDraftDTO | None = None
    warnings: list[str] = []
    clarification_question: str | None = None
    # New fields for controlled command layer
    note: PublicNoteDTO | None = None
    pending_command: dict[str, Any] | None = None
