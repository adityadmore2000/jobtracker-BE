"""
Strict structural schemas for the single-call semantic extractor.

These schemas describe ONE structured envelope returned by a single Ollama JSON
call (see semantic_command_extractor.py). They are deliberately isolated from the
legacy tool-calling schemas in semantic_schemas.py.

Design rules
────────────
- ``extra="forbid"`` everywhere: an LLM that hallucinates an unknown key fails
  validation and is rejected safely (no mutation).
- ``target`` carries identity only (company / role / application_id).
- ``changes`` carries mutable application fields only — company and role must
  never appear here.
- ``note`` is first-class free-form prose and must never be copied into any
  ``changes.*`` field.
- Field names mirror the canonical backend ``ApplicationChanges`` schema so the
  pipeline can map 1:1 without a second representation.  The one alias is
  ``location_mode`` → backend ``ApplicationChanges.location_mode`` (which maps to
  the DB ``location`` column), kept for prompt clarity.

The schema is intentionally permissive about *values* (enums are validated later
in the pipeline) but strict about *shape*.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator


# Allowed primary domain intents the extractor may emit.
SemanticIntent = Literal[
    "create_application",
    "update_application",
    "append_note",
    "archive_application",
    "unsupported",
]

ALLOWED_SEMANTIC_INTENTS = {
    "create_application",
    "update_application",
    "append_note",
    "archive_application",
    "unsupported",
}


class SemanticTarget(BaseModel):
    """Identity of the application a command refers to. Identity only."""

    model_config = ConfigDict(extra="forbid")

    company: Optional[str] = None
    role: Optional[str] = None
    application_id: Optional[int] = None


class SemanticChanges(BaseModel):
    """Mutable application fields only. No identity fields (company/role) allowed."""

    model_config = ConfigDict(extra="forbid")

    status: Optional[str] = None
    priority: Optional[str] = None
    location_mode: Optional[str] = None
    employment_types: Optional[List[str]] = None
    current_stages: Optional[List[str]] = None
    job_link: Optional[str] = None
    engaged_days: Optional[int] = None
    next_action: Optional[str] = None
    comments: Optional[str] = None

    def has_any_field(self) -> bool:
        return any(
            value is not None
            for value in (
                self.status,
                self.priority,
                self.location_mode,
                self.employment_types,
                self.current_stages,
                self.job_link,
                self.engaged_days,
                self.next_action,
                self.comments,
            )
        )


class SemanticCommand(BaseModel):
    """One structured envelope produced by the single-call extractor."""

    model_config = ConfigDict(extra="forbid")

    intent: SemanticIntent
    target: SemanticTarget = SemanticTarget()
    changes: SemanticChanges = SemanticChanges()
    note: Optional[str] = None
    clarification: Optional[str] = None
    # Optional rephrasing suggestions the model proposes; validated (dry-run
    # parsed) by the pipeline before ever being shown to the user.
    suggested_phrasings: Optional[List[str]] = None

    @field_validator("target", mode="before")
    @classmethod
    def _coerce_target(cls, value):
        # Models routinely emit "target": null to mean "no explicit target".
        return SemanticTarget() if value is None else value

    @field_validator("changes", mode="before")
    @classmethod
    def _coerce_changes(cls, value):
        # Models routinely emit "changes": null to mean "no field changes".
        return SemanticChanges() if value is None else value
