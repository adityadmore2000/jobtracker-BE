from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return value
    stripped = value.strip()
    return stripped or None


class SemanticFieldPatch(BaseModel):
    company: str | None = None
    roles: list[str] | None = None
    employment_types: list[str] | None = None
    job_link: str | None = None
    location: str | None = None
    status: str | None = None
    current_stages: list[str] | None = None
    priority: str | None = None
    engaged_days: int | None = Field(default=None, ge=0)
    next_action: str | None = None
    comments: str | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("company", "job_link", "location", "status", "priority", "next_action", "comments")
    @classmethod
    def normalize_text_fields(cls, value: str | None) -> str | None:
        return normalize_optional_text(value)

    @field_validator("roles", "employment_types", "current_stages")
    @classmethod
    def normalize_string_lists(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        normalized = [item.strip() for item in value if item and item.strip()]
        return normalized or []


class PatchActiveDraftArguments(BaseModel):
    fields: SemanticFieldPatch = Field(default_factory=SemanticFieldPatch)
    replace_explicit_fields: bool = True
    context_notes: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @field_validator("context_notes")
    @classmethod
    def normalize_context_notes(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item and item.strip()]


class PreviewExistingApplicationTarget(BaseModel):
    application_id: int | None = None
    company: str | None = None
    role: str | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("company", "role")
    @classmethod
    def normalize_text_fields(cls, value: str | None) -> str | None:
        return normalize_optional_text(value)


class PreviewExistingApplicationUpdateArguments(BaseModel):
    target: PreviewExistingApplicationTarget = Field(default_factory=PreviewExistingApplicationTarget)
    fields: SemanticFieldPatch = Field(default_factory=SemanticFieldPatch)
    replace_explicit_fields: bool = True

    model_config = ConfigDict(extra="forbid")


class RequestDraftSaveArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AttachLatestBrowserContextArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AskClarificationArguments(BaseModel):
    question: str = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question is required")
        return stripped


SemanticToolName = Literal[
    "patch_active_draft",
    "preview_existing_application_update",
    "request_draft_save",
    "attach_latest_browser_context",
    "ask_clarification",
]


class SemanticToolCallProposal(BaseModel):
    tool_name: SemanticToolName | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class SemanticInterpreterMetrics(BaseModel):
    latency_ms: int
    total_duration_ns: int | None = None
    load_duration_ns: int | None = None
    prompt_eval_duration_ns: int | None = None
    eval_duration_ns: int | None = None

    model_config = ConfigDict(extra="forbid")
