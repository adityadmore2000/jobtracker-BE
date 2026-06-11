"""
Controlled natural-language command parser.

Architecture
────────────
1. Strip conversational prefixes ("please", "do me a favor,", "can you", etc.)
2. Search for a known command anchor anywhere in the normalised transcript
3. Parse only the substring beginning at that anchor
4. Validate extracted values with strict alias tables
5. Resolve target from context
6. Return a MutationPayload or a ClarificationNeeded sentinel

No LLM involvement. No fuzzy free-form matching.
Unrecognised commands → ParseMiss (caller must return unsupported response).
Clarification required → ClarificationNeeded (caller surfaces the question).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .constants import (
    ALLOWED_CURRENT_STAGES,
    EMPLOYMENT_TYPE_ALIASES,
    LOCATION_ALIASES,
    PRIORITY_ALIASES,
    STATUS_ALIASES,
    STATUS_OPTIONS,
)
from .mutation_schemas import ApplicationChanges, MutationPayload, MutationTarget

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


# ──────────────────────────────────────────────────────────────────────────────
# Sentinel types returned by try_parse_v2
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ParseMiss:
    """No supported command anchor was found."""


@dataclass
class ClarificationNeeded:
    """A supported command was recognised but the target is ambiguous."""
    question: str
    pending_command: dict


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ──────────────────────────────────────────────────────────────────────────────

# Conversational prefixes that may precede any command.
_CONV_PREFIX_RE = re.compile(
    r"^(?:"
    r"please[,\s]*|"
    r"do\s+me\s+a\s+favor[,\s]*|"
    r"can\s+you[,\s]*|"
    r"could\s+you[,\s]*|"
    r"okay[,\s]*|"
    r"ok[,\s]*|"
    r"hey[,\s]*"
    r")+",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _normalize_lookup(value: str) -> str:
    """Collapse whitespace, lowercase, replace separators with spaces for alias lookup."""
    return " ".join(value.strip().replace("-", " ").replace("_", " ").casefold().split())


def _strip_prefix(text: str) -> str:
    """Remove leading conversational prefixes; return remainder stripped."""
    return _CONV_PREFIX_RE.sub("", text.strip()).strip()


# ──────────────────────────────────────────────────────────────────────────────
# Context resolution helpers
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_patch_target(context: dict) -> tuple[str | None, MutationTarget | None]:
    draft_id = context.get("draft_id")
    if draft_id is not None:
        return "patch_draft", MutationTarget(draft_id=str(draft_id))
    app_id = context.get("active_application_id")
    if app_id is not None:
        return "create_application_update_draft", MutationTarget(application_id=int(app_id))
    return None, None


def _get_active_applications(context: dict) -> list[dict]:
    applications = context.get("applications")
    if not isinstance(applications, list):
        return []
    return [a for a in applications if isinstance(a, dict) and not a.get("archived_at")]


def _get_all_applications(context: dict) -> list[dict]:
    applications = context.get("applications")
    if not isinstance(applications, list):
        return []
    return [a for a in applications if isinstance(a, dict)]


def _match_apps_by_company(company_name: str, apps: list[dict]) -> list[dict]:
    normalized_target = _normalize(company_name)
    return [a for a in apps if _normalize(a.get("company", "")) == normalized_target]


def _match_apps_by_company_and_role(company_name: str, role: str, apps: list[dict]) -> list[dict]:
    normalized_company = _normalize(company_name)
    normalized_role = _normalize(role)
    return [
        a for a in apps
        if _normalize(a.get("company", "")) == normalized_company
        and _normalize(a.get("role", "")) == normalized_role
    ]


def _resolve_application_id_by_company(company_name: str, context: dict) -> int | None:
    apps = _get_active_applications(context)
    matches = _match_apps_by_company(company_name, apps)
    if len(matches) == 1:
        return matches[0].get("id")
    return None


def _resolve_archived_application_id_by_company(company_name: str, context: dict) -> int | None:
    all_apps = _get_all_applications(context)
    archived = [a for a in all_apps if a.get("archived_at")]
    matches = _match_apps_by_company(company_name, archived)
    if len(matches) == 1:
        return matches[0].get("id")
    return None


def _resolve_explicit_setter_target(
    company: str,
    role: str | None,
    context: dict,
) -> tuple[int | None, str | None]:
    """Return (application_id, clarification_question) for explicit-target setter commands.

    Never creates an application. Never patches a saved row directly.
    Always routes to create_application_update_draft.
    """
    active_apps = _get_active_applications(context)
    if role:
        matches = _match_apps_by_company_and_role(company, role, active_apps)
        if len(matches) == 1:
            return matches[0]["id"], None
        if len(matches) == 0:
            return None, f"No active application found for {company} — {role}."
        return None, f"Multiple applications match {company} — {role}. Please clarify."

    matches = _match_apps_by_company(company, active_apps)
    if len(matches) == 1:
        return matches[0]["id"], None
    if len(matches) > 1:
        roles_listed = ", ".join(a.get("role", "(no role)") for a in matches)
        return None, f"Which role at {company} should I update? ({roles_listed})"
    return None, f"No active application found for {company}."


def _resolve_note_target(
    company: str | None,
    role: str | None,
    context: dict,
) -> tuple[int | None, int | None, str | None]:
    """Return (draft_id_int, application_id, clarification_question).

    Priority:
    1. explicit company + role → unambiguous saved app
    2. explicit company only → single saved app
    3. active_application_id from context
    4. active draft_id from context
    5. cannot resolve → clarification_question
    """
    draft_id_raw = context.get("draft_id")
    active_app_id = context.get("active_application_id")

    if company:
        active_apps = _get_active_applications(context)
        if role:
            matches = _match_apps_by_company_and_role(company, role, active_apps)
            if len(matches) == 1:
                return None, matches[0]["id"], None
            if len(matches) == 0:
                return None, None, f"No active application found for {company} — {role}."
        company_matches = _match_apps_by_company(company, active_apps)
        if len(company_matches) == 1:
            return None, company_matches[0]["id"], None
        if len(company_matches) > 1:
            roles_listed = ", ".join(a.get("role", "(no role)") for a in company_matches)
            return None, None, f"Which role at {company} should I add this note to? ({roles_listed})"
        return None, None, f"No active application found for {company}."

    if active_app_id is not None:
        return None, int(active_app_id), None

    if draft_id_raw is not None:
        return int(draft_id_raw), None, None

    return None, None, "Which application should I add this note to?"


# ──────────────────────────────────────────────────────────────────────────────
# Value normalisation
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_status(value: str) -> str | None:
    key = _normalize_lookup(value)
    if key in STATUS_ALIASES:
        return STATUS_ALIASES[key]
    if value.strip() in STATUS_OPTIONS:
        return value.strip()
    return None


def _normalize_priority(value: str) -> str | None:
    return PRIORITY_ALIASES.get(_normalize_lookup(value))


def _normalize_location(value: str) -> str | None:
    return LOCATION_ALIASES.get(_normalize_lookup(value))


def _normalize_employment_type(value: str) -> str | None:
    return EMPLOYMENT_TYPE_ALIASES.get(_normalize_lookup(value))


def _normalize_stage_value(value: str) -> str | None:
    normalized = _normalize_lookup(value)
    for stage in ALLOWED_CURRENT_STAGES:
        if _normalize_lookup(stage) == normalized:
            return stage
    return None


def _parse_stages(raw: str) -> list[str] | None:
    """Parse a comma/and-separated stage list. Returns None if any value is invalid.

    Supports:
    - comma-separated: "tailored, networked"
    - and-separated: "tailored and networked"
    - Oxford-comma form: "tailored, networked, and engaged"
    """
    # Normalise "comma + and" (Oxford comma) → just comma before splitting.
    normalised = re.sub(r",\s*and\s+", ", ", raw.strip(), flags=re.IGNORECASE)
    # Then split on commas and standalone "and"
    parts = re.split(r"\s*,\s*|\s+and\s+", normalised, flags=re.IGNORECASE)
    result = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        canonical = _normalize_stage_value(part)
        if canonical is None:
            return None
        if canonical not in result:
            result.append(canonical)
    return result if result else None


def _strip_trailing_role_word(role: str) -> str:
    """Strip trailing ' role' only from explicit add-application grammar."""
    return re.sub(r"\s+role\s*$", "", role, flags=re.IGNORECASE).strip()


# ──────────────────────────────────────────────────────────────────────────────
# Regex patterns for command anchors
# ──────────────────────────────────────────────────────────────────────────────

# 1. add application for {role} [role] at {company}
_ADD_APP_RE = re.compile(
    r"(?:add|create|track)\s+application\s+for\s+(.+?)\s+at\s+(.+)$",
    re.IGNORECASE,
)

# 2. remove application [for {company} [for {role} role]]
_REMOVE_APP_RE = re.compile(
    r"remove\s+application(?:\s+for\s+(.+?)(?:\s+for\s+(.+?)\s+role)?)?\s*$",
    re.IGNORECASE,
)

# 3. update application [for {company} [for {role} role]]
_UPDATE_APP_RE = re.compile(
    r"update\s+application(?:\s+for\s+(.+?)(?:\s+for\s+(.+?)\s+role)?)?\s*$",
    re.IGNORECASE,
)

# 4. set role as {role}
_SET_ROLE_RE = re.compile(r"set\s+role\s+as\s+(.+)$", re.IGNORECASE)

# 5. set priority as {priority}
_SET_PRIORITY_RE = re.compile(r"set\s+priority\s+as\s+(.+)$", re.IGNORECASE)

# 6. set location as {location}
_SET_LOCATION_RE = re.compile(r"set\s+location\s+as\s+(.+)$", re.IGNORECASE)

# 7a. add [a] note [for {company} [for {role} role]] saying {text}
# Both the company clause and the role sub-clause are optional.
_ADD_NOTE_RE = re.compile(
    r"add\s+a?\s*note"
    r"(?:\s+for\s+(.+?)(?:\s+for\s+(.+?)\s+role)?)?"
    r"\s+saying\s+(.+)$",
    re.IGNORECASE,
)

# 8. set employment type as {type}
_SET_EMP_TYPE_RE = re.compile(r"set\s+employment\s+type\s+as\s+(.+)$", re.IGNORECASE)

# 9. set current stage[s] as {stage_list}
_SET_STAGES_RE = re.compile(r"set\s+current\s+stages?\s+as\s+(.+)$", re.IGNORECASE)

# 10. Explicit-target setter: set {field} {of|for} {company} [for {role} role] to {value}
# Field tokens: status | role | priority | location | employment type | current stages? | current stage
# Captured groups: (1) field, (2) company, (3) role-or-None, (4) value
_SET_FIELD_EXPLICIT_RE = re.compile(
    r"set\s+"
    r"(status|role|priority|location|employment\s+type|current\s+stages?)"
    r"\s+(?:of|for)\s+"
    r"(.+?)"
    r"(?:\s+for\s+(.+?)\s+role)?"
    r"\s+to\s+"
    r"(.+)$",
    re.IGNORECASE,
)

# Legacy: save / discard / apply changes / discard changes
_SAVE_TRIGGERS = {"save it", "save", "save this", "save draft"}
_DISCARD_TRIGGERS = {"discard it", "discard", "discard this", "discard draft", "cancel it", "cancel draft"}
_APPLY_CHANGES_TRIGGERS = {"apply changes"}
_DISCARD_CHANGES_TRIGGERS = {"discard changes"}

# Legacy field-update patterns kept for backward-compat
# "set priority to X", "set location to X", "set employment type to X"
_LEGACY_FIELD_UPDATE_RE = re.compile(
    r"^(?:set|change|update)\s+(status|priority|location|employment\s+type)\s+to\s+(.+)$",
    re.IGNORECASE,
)

# ──────────────────────────────────────────────────────────────────────────────
# Internal parse result types
# ──────────────────────────────────────────────────────────────────────────────

ParseResult = MutationPayload | ClarificationNeeded | ParseMiss


# ──────────────────────────────────────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────────────────────────────────────

def try_parse(transcript: str, context: dict) -> MutationPayload | None:
    """Original entry point — returns MutationPayload or None.

    Callers that need ClarificationNeeded should use try_parse_v2.
    ClarificationNeeded is collapsed to None here for backward-compat.
    """
    result = try_parse_v2(transcript, context)
    if isinstance(result, MutationPayload):
        return result
    return None


def try_parse_v2(transcript: str, context: dict) -> ParseResult:
    """Full controlled command parser.

    Returns:
    - MutationPayload   → dispatch immediately
    - ClarificationNeeded → surface question to user
    - ParseMiss         → no supported anchor found; do NOT fall through to LLM
    """
    normalized = _normalize(transcript)
    stripped = _strip_prefix(transcript.strip())
    stripped_norm = _normalize(stripped)

    # ── 1. apply changes ────────────────────────────────────────────────────
    if stripped_norm in _APPLY_CHANGES_TRIGGERS:
        change_draft_id = context.get("change_draft_id") or context.get("active_change_draft_id")
        if change_draft_id is None:
            return ClarificationNeeded(
                question="There are no pending changes to apply.",
                pending_command={},
            )
        return MutationPayload(
            operation="apply_application_update_draft",
            target=MutationTarget(change_draft_id=int(change_draft_id)),
            changes=ApplicationChanges(),
        )

    # ── 2. discard changes ───────────────────────────────────────────────────
    if stripped_norm in _DISCARD_CHANGES_TRIGGERS:
        change_draft_id = context.get("change_draft_id") or context.get("active_change_draft_id")
        if change_draft_id is None:
            return ClarificationNeeded(
                question="There are no pending changes to discard.",
                pending_command={},
            )
        return MutationPayload(
            operation="discard_application_update_draft",
            target=MutationTarget(change_draft_id=int(change_draft_id)),
            changes=ApplicationChanges(),
        )

    # ── 3. save draft ─────────────────────────────────────────────────────────
    if stripped_norm in _SAVE_TRIGGERS:
        draft_id = context.get("draft_id")
        if draft_id is None:
            return ParseMiss()
        return MutationPayload(
            operation="save_draft",
            target=MutationTarget(draft_id=str(draft_id)),
            changes=ApplicationChanges(),
        )

    # ── 4. discard draft ──────────────────────────────────────────────────────
    if stripped_norm in _DISCARD_TRIGGERS:
        draft_id = context.get("draft_id")
        if draft_id is None:
            return ParseMiss()
        return MutationPayload(
            operation="discard_draft",
            target=MutationTarget(draft_id=str(draft_id)),
            changes=ApplicationChanges(),
        )

    # ── 5. add note ───────────────────────────────────────────────────────────
    m = _ADD_NOTE_RE.search(stripped)
    if m:
        company_raw = m.group(1)
        role_raw = m.group(2)
        note_text = re.sub(r"\s+", " ", m.group(3).strip())
        company = re.sub(r"\s+", " ", company_raw.strip()) if company_raw else None
        role = re.sub(r"\s+", " ", role_raw.strip()) if role_raw else None

        draft_id_int, app_id, question = _resolve_note_target(company, role, context)
        if question:
            return ClarificationNeeded(
                question=question,
                pending_command={
                    "operation": "append_note",
                    "target": {
                        "company": company,
                        "role": role,
                        "application_id": None,
                    },
                    "changes": {},
                    "note": note_text,
                    "missing_field": "role" if company and not role else "company",
                },
            )
        if draft_id_int is not None:
            return MutationPayload(
                operation="append_note",
                target=MutationTarget(draft_id=str(draft_id_int)),
                changes=ApplicationChanges(),
                notes_to_append=[note_text],
            )
        return MutationPayload(
            operation="append_note",
            target=MutationTarget(application_id=app_id),
            changes=ApplicationChanges(),
            notes_to_append=[note_text],
        )

    # ── 6. add application ────────────────────────────────────────────────────
    m = _ADD_APP_RE.search(stripped)
    if m:
        raw_role = re.sub(r"\s+", " ", m.group(1).strip())
        raw_company = re.sub(r"\s+", " ", m.group(2).strip())
        if raw_role and raw_company:
            role_clean = _strip_trailing_role_word(raw_role)
            return MutationPayload(
                operation="create_draft",
                target=MutationTarget(),
                changes=ApplicationChanges(company=raw_company, role=role_clean),
            )
        return ParseMiss()

    # ── 7. remove application ─────────────────────────────────────────────────
    m = _REMOVE_APP_RE.search(stripped)
    if m:
        company_raw = m.group(1)
        role_raw = m.group(2)
        company = re.sub(r"\s+", " ", company_raw.strip()) if company_raw else None
        role = re.sub(r"\s+", " ", role_raw.strip()) if role_raw else None

        # Resolve target
        active_app_id = context.get("active_application_id")
        draft_id_raw = context.get("draft_id")

        if company and role:
            active_apps = _get_active_applications(context)
            matches = _match_apps_by_company_and_role(company, role, active_apps)
            if len(matches) == 1:
                app_id = matches[0]["id"]
            elif len(matches) == 0:
                return ClarificationNeeded(
                    question=f"No active application found for {company} — {role}.",
                    pending_command={},
                )
            else:
                return ClarificationNeeded(
                    question=f"Multiple applications match {company} — {role}. Please clarify.",
                    pending_command={},
                )
        elif company:
            app_id = _resolve_application_id_by_company(company, context)
            if app_id is None:
                active_apps = _get_active_applications(context)
                matches = _match_apps_by_company(company, active_apps)
                if len(matches) > 1:
                    roles_listed = ", ".join(a.get("role", "(no role)") for a in matches)
                    return ClarificationNeeded(
                        question=f"Which role at {company} should I remove? ({roles_listed})",
                        pending_command={
                            "operation": "remove_application",
                            "target": {"company": company, "role": None, "application_id": None},
                            "changes": {},
                            "note": None,
                            "missing_field": "role",
                        },
                    )
                return ClarificationNeeded(
                    question=f"No active application found for {company}.",
                    pending_command={},
                )
        elif active_app_id is not None:
            app_id = int(active_app_id)
        elif draft_id_raw is not None:
            # Cannot archive a draft; only saved apps can be archived
            return ClarificationNeeded(
                question="The active item is a draft. Did you mean to discard the draft?",
                pending_command={},
            )
        else:
            return ClarificationNeeded(
                question="Which application should I remove?",
                pending_command={
                    "operation": "remove_application",
                    "target": {"company": None, "role": None, "application_id": None},
                    "changes": {},
                    "note": None,
                    "missing_field": "company",
                },
            )

        return MutationPayload(
            operation="archive_application",
            target=MutationTarget(application_id=app_id),
            changes=ApplicationChanges(),
        )

    # ── 8. update application ─────────────────────────────────────────────────
    m = _UPDATE_APP_RE.search(stripped)
    if m:
        company_raw = m.group(1)
        role_raw = m.group(2)
        company = re.sub(r"\s+", " ", company_raw.strip()) if company_raw else None
        role = re.sub(r"\s+", " ", role_raw.strip()) if role_raw else None

        if company and role:
            active_apps = _get_active_applications(context)
            matches = _match_apps_by_company_and_role(company, role, active_apps)
            if len(matches) == 1:
                app_id = matches[0]["id"]
                return MutationPayload(
                    operation="set_active_application",
                    target=MutationTarget(application_id=app_id),
                    changes=ApplicationChanges(),
                )
            elif len(matches) == 0:
                return ClarificationNeeded(
                    question=f"No active application found for {company} — {role}.",
                    pending_command={},
                )
        elif company:
            active_apps = _get_active_applications(context)
            matches = _match_apps_by_company(company, active_apps)
            if len(matches) == 1:
                app_id = matches[0]["id"]
                return MutationPayload(
                    operation="set_active_application",
                    target=MutationTarget(application_id=app_id),
                    changes=ApplicationChanges(),
                )
            elif len(matches) > 1:
                roles_listed = ", ".join(a.get("role", "(no role)") for a in matches)
                return ClarificationNeeded(
                    question=f"Which role at {company} should I update? ({roles_listed})",
                    pending_command={
                        "operation": "update_application",
                        "target": {"company": company, "role": None, "application_id": None},
                        "changes": {},
                        "note": None,
                        "missing_field": "role",
                    },
                )
            elif len(matches) == 0:
                return ClarificationNeeded(
                    question=f"No active application found for {company}.",
                    pending_command={},
                )
        else:
            return ClarificationNeeded(
                question="Which application should I update?",
                pending_command={
                    "operation": "update_application",
                    "target": {"company": None, "role": None, "application_id": None},
                    "changes": {},
                    "note": None,
                    "missing_field": "company",
                },
            )

    # ── 9. Explicit-target setter: set {field} {of|for} {company} [for {role} role] to {value}
    m = _SET_FIELD_EXPLICIT_RE.search(stripped)
    if m:
        field_raw = re.sub(r"\s+", " ", m.group(1).strip().lower())
        company_raw = re.sub(r"\s+", " ", m.group(2).strip())
        role_raw = re.sub(r"\s+", " ", m.group(3).strip()) if m.group(3) else None
        value_raw = re.sub(r"\s+", " ", m.group(4).strip())

        # Resolve target application — no creation, no direct patch
        app_id, question = _resolve_explicit_setter_target(company_raw, role_raw, context)
        if question:
            missing = "role" if _match_apps_by_company(company_raw, _get_active_applications(context)) else "company"
            return ClarificationNeeded(
                question=question,
                pending_command={
                    "operation": "set_field",
                    "target": {"company": company_raw, "role": role_raw, "application_id": None},
                    "changes": {},
                    "note": None,
                    "missing_field": missing,
                },
            )

        # Normalise value for the matched field
        if field_raw == "status":
            canonical = _normalize_status(value_raw)
            if canonical is None:
                return ParseMiss()
            changes = ApplicationChanges(status=canonical)

        elif field_raw == "role":
            if not value_raw:
                return ParseMiss()
            changes = ApplicationChanges(role=value_raw)

        elif field_raw == "priority":
            canonical = _normalize_priority(value_raw)
            if canonical is None:
                return ParseMiss()
            changes = ApplicationChanges(priority=canonical)

        elif field_raw == "location":
            canonical = _normalize_location(value_raw)
            if canonical is None:
                return ParseMiss()
            changes = ApplicationChanges(location_mode=canonical)

        elif field_raw == "employment type":
            canonical = _normalize_employment_type(value_raw)
            if canonical is None:
                return ParseMiss()
            changes = ApplicationChanges(employment_types=[canonical])

        else:  # current stage / current stages
            stages = _parse_stages(value_raw)
            if stages is None:
                return ParseMiss()
            changes = ApplicationChanges(current_stages=stages)

        return MutationPayload(
            operation="create_application_update_draft",
            target=MutationTarget(application_id=app_id),
            changes=changes,
        )

    # ── 10. set role (contextual) ────────────────────────────────────────────
    m = _SET_ROLE_RE.search(stripped)
    if m:
        raw_role = re.sub(r"\s+", " ", m.group(1).strip())
        if not raw_role:
            return ParseMiss()
        operation, target = _resolve_patch_target(context)
        if operation is None or target is None:
            return ClarificationNeeded(
                question="Which application should I set the role for?",
                pending_command={},
            )
        return MutationPayload(
            operation=operation,
            target=target,
            changes=ApplicationChanges(role=raw_role),
        )

    # ── 11. set priority (contextual) ────────────────────────────────────────
    m = _SET_PRIORITY_RE.search(stripped)
    if m:
        raw_value = re.sub(r"\s+", " ", m.group(1).strip())
        canonical = _normalize_priority(raw_value)
        if canonical is None:
            return ParseMiss()
        operation, target = _resolve_patch_target(context)
        if operation is None or target is None:
            return ClarificationNeeded(
                question="Which application should I set the priority for?",
                pending_command={},
            )
        return MutationPayload(
            operation=operation,
            target=target,
            changes=ApplicationChanges(priority=canonical),
        )

    # ── 12. set location (contextual) ────────────────────────────────────────
    m = _SET_LOCATION_RE.search(stripped)
    if m:
        raw_value = re.sub(r"\s+", " ", m.group(1).strip())
        canonical = _normalize_location(raw_value)
        if canonical is None:
            return ParseMiss()
        operation, target = _resolve_patch_target(context)
        if operation is None or target is None:
            return ClarificationNeeded(
                question="Which application should I set the location for?",
                pending_command={},
            )
        return MutationPayload(
            operation=operation,
            target=target,
            changes=ApplicationChanges(location_mode=canonical),
        )

    # ── 13. set employment type (contextual) ────────────────────────────────
    m = _SET_EMP_TYPE_RE.search(stripped)
    if m:
        raw_value = re.sub(r"\s+", " ", m.group(1).strip())
        canonical = _normalize_employment_type(raw_value)
        if canonical is None:
            return ParseMiss()
        operation, target = _resolve_patch_target(context)
        if operation is None or target is None:
            return ClarificationNeeded(
                question="Which application should I set the employment type for?",
                pending_command={},
            )
        return MutationPayload(
            operation=operation,
            target=target,
            changes=ApplicationChanges(employment_types=[canonical]),
        )

    # ── 14. set current stages (contextual) ─────────────────────────────────
    m = _SET_STAGES_RE.search(stripped)
    if m:
        raw_value = m.group(1).strip()
        stages = _parse_stages(raw_value)
        if stages is None:
            return ParseMiss()
        operation, target = _resolve_patch_target(context)
        if operation is None or target is None:
            return ClarificationNeeded(
                question="Which application should I update the stages for?",
                pending_command={
                    "operation": "set_current_stages",
                    "target": {"company": None, "role": None, "application_id": None},
                    "changes": {"current_stages": stages},
                    "note": None,
                    "missing_field": "company",
                },
            )
        return MutationPayload(
            operation=operation,
            target=target,
            changes=ApplicationChanges(current_stages=stages),
        )

    # ── Legacy: priority shorthand triggers ──────────────────────────────────
    priority_high_triggers = {"priority high", "high priority", "set priority high", "set high priority"}
    if stripped_norm in priority_high_triggers:
        operation, target = _resolve_patch_target(context)
        if operation is None or target is None:
            return ParseMiss()
        return MutationPayload(operation=operation, target=target, changes=ApplicationChanges(priority="HIGH"))

    priority_medium_triggers = {"priority medium", "medium priority", "set priority medium", "set medium priority"}
    if stripped_norm in priority_medium_triggers:
        operation, target = _resolve_patch_target(context)
        if operation is None or target is None:
            return ParseMiss()
        return MutationPayload(operation=operation, target=target, changes=ApplicationChanges(priority="MEDIUM"))

    priority_low_triggers = {"priority low", "low priority", "set priority low", "set low priority"}
    if stripped_norm in priority_low_triggers:
        operation, target = _resolve_patch_target(context)
        if operation is None or target is None:
            return ParseMiss()
        return MutationPayload(operation=operation, target=target, changes=ApplicationChanges(priority="LOW"))

    # ── Legacy: location single-word triggers ────────────────────────────────
    remote_triggers = {"remote", "it is remote", "location remote", "set remote", "working remotely"}
    if stripped_norm in remote_triggers:
        operation, target = _resolve_patch_target(context)
        if operation is None or target is None:
            return ParseMiss()
        return MutationPayload(operation=operation, target=target, changes=ApplicationChanges(location_mode="remote"))

    hybrid_triggers = {"hybrid", "it is hybrid", "location hybrid", "set hybrid"}
    if stripped_norm in hybrid_triggers:
        operation, target = _resolve_patch_target(context)
        if operation is None or target is None:
            return ParseMiss()
        return MutationPayload(operation=operation, target=target, changes=ApplicationChanges(location_mode="hybrid"))

    onsite_triggers = {"onsite", "on site", "it is onsite", "location onsite", "set onsite", "on-site"}
    if stripped_norm in onsite_triggers:
        operation, target = _resolve_patch_target(context)
        if operation is None or target is None:
            return ParseMiss()
        return MutationPayload(operation=operation, target=target, changes=ApplicationChanges(location_mode="on-site"))

    # ── Legacy: status triggers ──────────────────────────────────────────────
    applied_triggers = {"mark applied", "applied", "i applied", "i have applied", "already applied", "status applied"}
    if stripped_norm in applied_triggers:
        operation, target = _resolve_patch_target(context)
        if operation is None or target is None:
            return ParseMiss()
        return MutationPayload(operation=operation, target=target, changes=ApplicationChanges(status="applied"))

    rejected_triggers = {"mark rejected", "rejected", "got rejected", "they rejected", "status rejected"}
    if stripped_norm in rejected_triggers:
        operation, target = _resolve_patch_target(context)
        if operation is None or target is None:
            return ParseMiss()
        return MutationPayload(operation=operation, target=target, changes=ApplicationChanges(status="rejected"))

    # ── Legacy: archive/restore with company name prefix ─────────────────────
    archive_prefixes = ["archive ", "remove ", "hide "]
    for prefix in archive_prefixes:
        if stripped_norm.startswith(prefix):
            company_name = stripped_norm[len(prefix):].strip()
            if company_name and not company_name.startswith("application"):
                application_id = _resolve_application_id_by_company(company_name, context)
                if application_id is None:
                    return ParseMiss()
                return MutationPayload(
                    operation="archive_application",
                    target=MutationTarget(application_id=application_id),
                    changes=ApplicationChanges(),
                )

    restore_prefixes = ["restore ", "unarchive ", "bring back "]
    for prefix in restore_prefixes:
        if stripped_norm.startswith(prefix):
            company_name = stripped_norm[len(prefix):].strip()
            if company_name:
                application_id = _resolve_archived_application_id_by_company(company_name, context)
                if application_id is None:
                    return ParseMiss()
                return MutationPayload(
                    operation="restore_application",
                    target=MutationTarget(application_id=application_id),
                    changes=ApplicationChanges(),
                )

    # ── Legacy: "set/change {field} to {value}" ──────────────────────────────
    m = _LEGACY_FIELD_UPDATE_RE.match(stripped)
    if m:
        field_key = re.sub(r"\s+", " ", m.group(1).strip().lower())
        raw_value = re.sub(r"\s+", " ", m.group(2).strip())
        operation, target = _resolve_patch_target(context)
        if operation is None or target is None:
            return ParseMiss()

        if field_key == "status":
            canonical = _normalize_status(raw_value)
            if canonical is None:
                return ParseMiss()
            return MutationPayload(operation=operation, target=target, changes=ApplicationChanges(status=canonical))

        if field_key == "priority":
            canonical = _normalize_priority(raw_value)
            if canonical is None:
                return ParseMiss()
            return MutationPayload(operation=operation, target=target, changes=ApplicationChanges(priority=canonical))

        if field_key == "location":
            canonical = _normalize_location(raw_value)
            if canonical is None:
                return ParseMiss()
            return MutationPayload(operation=operation, target=target, changes=ApplicationChanges(location_mode=canonical))

        if field_key == "employment type":
            canonical = _normalize_employment_type(raw_value)
            if canonical is None:
                return ParseMiss()
            return MutationPayload(operation=operation, target=target, changes=ApplicationChanges(employment_types=[canonical]))

    # ── 15. No anchor found ───────────────────────────────────────────────────
    return ParseMiss()
