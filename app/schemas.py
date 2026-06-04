from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, TypeAdapter, field_validator

from .constants import (
    ALLOWED_CURRENT_STAGES,
    ALLOWED_EMPLOYMENT_TYPES,
    ALLOWED_LOCATIONS,
    ALLOWED_PRIORITIES,
    ALLOWED_ROLES,
)

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
