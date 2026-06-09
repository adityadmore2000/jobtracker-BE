from typing import List, Optional

from pydantic import BaseModel


ALLOWED_OPERATIONS = {
    "create_draft",
    "patch_draft",
    "save_draft",
    "discard_draft",
    "patch_application",
    "ask_clarification",
    "append_note",
    "archive_application",
    "restore_application",
}


class MutationTarget(BaseModel):
    draft_id: Optional[str] = None
    application_id: Optional[int] = None


class ApplicationChanges(BaseModel):
    company: Optional[str] = None
    roles: Optional[List[str]] = None
    status: Optional[str] = None
    priority: Optional[str] = None
    location_mode: Optional[str] = None
    job_link: Optional[str] = None
    employment_types: Optional[List[str]] = None
    current_stages: Optional[List[str]] = None


class MutationPayload(BaseModel):
    operation: str
    target: MutationTarget
    changes: ApplicationChanges
    notes_to_append: List[str] = []


class MutationResult(BaseModel):
    success: bool
    operation: str
    message: str
    draft: Optional[dict] = None
    application: Optional[dict] = None
    requires_confirmation: bool = False
    confirmation_kind: Optional[str] = None
    clarification_question: Optional[str] = None
    notes: Optional[List[dict]] = None
