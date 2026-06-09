from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class PublicApplicationDTO(BaseModel):
    id: int
    company: str
    roles: list[str]
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


TranscriptStatus = Literal[
    "draft_created",
    "draft_updated",
    "saved",
    "discarded",
    "updated",
    "clarification",
    "no_change",
    "error",
]


class PublicTranscriptResponse(BaseModel):
    status: TranscriptStatus
    message: str
    application_id: int | None = None
    draft_id: str | None = None
    draft: PublicApplicationDTO | None = None
    application: PublicApplicationDTO | None = None
    warnings: list[str] = []
    clarification_question: str | None = None
