from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, TypeAdapter, field_validator

from .constants import (
    ALLOWED_CURRENT_STAGES,
    ALLOWED_EMPLOYMENT_TYPES,
    ALLOWED_LOCATIONS,
    ALLOWED_PRIORITIES,
    ALLOWED_ROLES,
    STATUS_OPTIONS,
)
from .semantic_schemas import SemanticInterpreterMetrics, SemanticToolCallProposal

http_url_adapter = TypeAdapter(HttpUrl)


def validate_allowed_list(values: list[str], allowed: list[str], field_name: str) -> list[str]:
    invalid = [value for value in values if value not in allowed]
    if invalid:
        allowed_values = ", ".join(allowed)
        raise ValueError(f"{field_name} contains invalid value(s): {', '.join(invalid)}. Allowed values: {allowed_values}")
    return values


def validate_optional_allowed(value: str, allowed: list[str], field_name: str) -> str:
    if value and value not in allowed:
        allowed_values = ", ".join(allowed)
        raise ValueError(f"{field_name} must be empty or one of: {allowed_values}")
    return value


def normalize_string(value: Any) -> Any:
    return "" if value is None else value


class JobApplicationBase(BaseModel):
    company: str = Field(min_length=1)
    roles_json: list[str] = Field(default_factory=list)
    employment_types_json: list[str] = Field(default_factory=list)
    job_link: str = ""
    location: str = ""
    status: str = ""
    current_stages_json: list[str] = Field(default_factory=list)
    priority: str = ""
    engaged_days: int = Field(default=0, ge=0)
    next_action: str = ""
    comments: str = ""

    @field_validator("company")
    @classmethod
    def company_required(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("company is required")
        return stripped

    @field_validator("roles_json")
    @classmethod
    def roles_are_allowed(cls, values: list[str]) -> list[str]:
        return validate_allowed_list(values, ALLOWED_ROLES, "roles_json")

    @field_validator("employment_types_json")
    @classmethod
    def employment_types_are_allowed(cls, values: list[str]) -> list[str]:
        return validate_allowed_list(values, ALLOWED_EMPLOYMENT_TYPES, "employment_types_json")

    @field_validator("current_stages_json")
    @classmethod
    def current_stages_are_allowed(cls, values: list[str]) -> list[str]:
        return validate_allowed_list(values, ALLOWED_CURRENT_STAGES, "current_stages_json")

    @field_validator("location")
    @classmethod
    def location_is_allowed(cls, value: str | None) -> str:
        return validate_optional_allowed(normalize_string(value), ALLOWED_LOCATIONS, "location")

    @field_validator("priority")
    @classmethod
    def priority_is_allowed(cls, value: str | None) -> str:
        return validate_optional_allowed(normalize_string(value), ALLOWED_PRIORITIES, "priority")

    @field_validator("job_link")
    @classmethod
    def job_link_is_url_or_empty(cls, value: str | None) -> str:
        value = normalize_string(value).strip()
        if not value:
            return ""
        http_url_adapter.validate_python(value)
        return value


class JobApplicationCreate(JobApplicationBase):
    pass


class ApplicationCreateCandidateRequest(JobApplicationBase):
    raw_transcript: str | None = None
    original_extracted_company_name: str | None = None
    audio_reference: str | None = None

    @field_validator("raw_transcript", "original_extracted_company_name", "audio_reference")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return value
        stripped = value.strip()
        return stripped or None


class ApplicationCompanyConfirmationRequest(ApplicationCreateCandidateRequest):
    confirmed_company_name: str = Field(min_length=1)

    @field_validator("confirmed_company_name")
    @classmethod
    def confirmed_company_name_required(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("confirmed_company_name is required")
        return stripped


class JobApplicationUpdate(BaseModel):
    company: str | None = Field(default=None, min_length=1)
    roles_json: list[str] | None = None
    employment_types_json: list[str] | None = None
    job_link: str | None = None
    location: str | None = None
    status: str | None = None
    current_stages_json: list[str] | None = None
    priority: str | None = None
    engaged_days: int | None = Field(default=None, ge=0)
    next_action: str | None = None
    comments: str | None = None

    @field_validator("company")
    @classmethod
    def company_required_when_present(cls, value: str | None) -> str | None:
        if value is None:
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("company is required")
        return stripped

    @field_validator("roles_json")
    @classmethod
    def roles_are_allowed(cls, values: list[str] | None) -> list[str] | None:
        return None if values is None else validate_allowed_list(values, ALLOWED_ROLES, "roles_json")

    @field_validator("employment_types_json")
    @classmethod
    def employment_types_are_allowed(cls, values: list[str] | None) -> list[str] | None:
        return None if values is None else validate_allowed_list(values, ALLOWED_EMPLOYMENT_TYPES, "employment_types_json")

    @field_validator("current_stages_json")
    @classmethod
    def current_stages_are_allowed(cls, values: list[str] | None) -> list[str] | None:
        return None if values is None else validate_allowed_list(values, ALLOWED_CURRENT_STAGES, "current_stages_json")

    @field_validator("location")
    @classmethod
    def location_is_allowed(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return validate_optional_allowed(value, ALLOWED_LOCATIONS, "location")

    @field_validator("priority")
    @classmethod
    def priority_is_allowed(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return validate_optional_allowed(value, ALLOWED_PRIORITIES, "priority")

    @field_validator("job_link")
    @classmethod
    def job_link_is_url_or_empty(cls, value: str | None) -> str | None:
        if value is None:
            return value
        value = value.strip()
        if not value:
            return ""
        http_url_adapter.validate_python(value)
        return value


class JobApplicationRead(JobApplicationBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class BrowserContextCreate(BaseModel):
    url: str
    page_title: str = ""

    @field_validator("url")
    @classmethod
    def url_must_be_http_or_https(cls, value: str) -> str:
        value = value.strip()
        parsed_url = http_url_adapter.validate_python(value)
        if parsed_url.scheme not in {"http", "https"}:
            raise ValueError("url must use http or https")
        return value

    @field_validator("page_title")
    @classmethod
    def normalize_page_title(cls, value: str | None) -> str:
        return "" if value is None else value


class BrowserContextRead(BaseModel):
    id: int
    url: str
    page_title: str
    captured_at: datetime

    model_config = ConfigDict(from_attributes=True)


class BrowserContextResponse(BaseModel):
    context: BrowserContextRead | None


class LiveKitTokenRequest(BaseModel):
    room_name: str = "job-tracker-local"

    @field_validator("room_name", mode="before")
    @classmethod
    def normalize_room_name(cls, value: str | None) -> str:
        if value is None:
            return "job-tracker-local"
        stripped = value.strip()
        return stripped or "job-tracker-local"


class LiveKitTokenResponse(BaseModel):
    url: str
    room_name: str
    participant_identity: str
    access_token: str
    expires_at: datetime


class ApplicationCreateCandidateRequiresConfirmation(BaseModel):
    status: Literal["confirmation_required"]
    requires_confirmation: Literal[True] = True
    candidate: ApplicationCreateCandidateRequest


class ApplicationCreateCandidateCreated(BaseModel):
    status: Literal["created"]
    requires_confirmation: Literal[False] = False
    application: JobApplicationRead


ApplicationCreateCandidateResponse = ApplicationCreateCandidateRequiresConfirmation | ApplicationCreateCandidateCreated


class TranscriptParseRequest(BaseModel):
    transcript: str = Field(min_length=1)
    context: dict[str, Any] | None = None

    @field_validator("transcript")
    @classmethod
    def transcript_required(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("transcript is required")
        return stripped


class SemanticTranscriptResponse(BaseModel):
    status: Literal["preview", "clarification_required", "unsupported", "unavailable"]
    operation: Literal["create", "update", "none"]
    raw_transcript: str
    proposal: SemanticToolCallProposal
    application_id: int | None = None
    draft: JobApplicationBase | None = None
    drafts: list[JobApplicationCreate] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    needs_confirmation: bool = False
    confirmation_kind: Literal["none", "multi_application", "context"] = "none"
    clarification_question: str | None = None
    interpreter_metrics: SemanticInterpreterMetrics | None = None


class AsrHotwordsResponse(BaseModel):
    hotwords: list[str]
    limit: int
