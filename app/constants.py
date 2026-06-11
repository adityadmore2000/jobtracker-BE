# Roles are free-form strings. ALLOWED_ROLES is a UI / ASR-hotword hint only and is
# NOT enforced for validation anywhere in the backend. Any non-blank role string is valid.
ALLOWED_ROLES = [
    "AI Engineer",
    "Generative AI Engineer",
    "GenAI Engineer",
    "LLM Engineer",
    "RAG Engineer",
    "AI Systems Engineer",
    "ML Engineer",
    "Computer Vision Engineer",
    "Agentic AI Engineer",
    "Data Science",
    "Prompt Engineer",
    "Platform Engineer",
    "GET",
    "AI Product Engineer",
]

ALLOWED_EMPLOYMENT_TYPES = [
    "Internship",
    "Full Time",
    "Part Time",
]

ALLOWED_LOCATIONS = [
    "remote",
    "hybrid",
    "on-site",
]

ALLOWED_CURRENT_STAGES = [
    "Tailored",
    "Applied",
    "Networked",
    "Engaged",
    "COLD_MAIL",
    "Followed up",
]

ALLOWED_PRIORITIES = [
    "LOW",
    "MEDIUM",
    "HIGH",
]

STATUS_OPTIONS = [
    "in_touch",
    "applied",
    "accepted",
    "rejected",
]

# Maps normalized alias text → canonical STATUS_OPTIONS value.
# Aliases must never become gatekeeping: only well-known variants are listed.
# Keys use space as the only separator (hyphens/underscores are normalised
# to spaces before lookup by normalize_status_value).
STATUS_ALIASES: dict[str, str] = {
    "in touch": "in_touch",  # covers "in-touch", "in_touch", "intouch" after sep normalisation
    "intouch": "in_touch",
    "applied": "applied",
    "submitted application": "applied",
    "application sent": "applied",
    "already applied": "applied",
    "i applied": "applied",
    "accepted": "accepted",
    "selected": "accepted",
    "offer accepted": "accepted",
    "rejected": "rejected",
    "declined": "rejected",
    "got rejected": "rejected",
}

def normalize_status_value(value: str) -> str | None:
    """Return canonical status for *value*, or None if unrecognised.

    Matching is case-insensitive, collapses internal whitespace, and
    treats hyphens, underscores, and spaces as equivalent separators.
    Returns None for any value not in STATUS_OPTIONS or STATUS_ALIASES.
    """
    key = " ".join(value.strip().replace("-", " ").replace("_", " ").casefold().split())
    if key in STATUS_ALIASES:
        return STATUS_ALIASES[key]
    # Accept already-canonical lower-snake values directly
    if value.strip() in STATUS_OPTIONS:
        return value.strip()
    return None


# --------------------------------------------------------------------------- #
# Import-time consistency guards: alias target values must be canonical.       #
# Roles are explicitly excluded — unknown roles are always valid.              #
# --------------------------------------------------------------------------- #

_STATUS_ALIAS_TARGETS = set(STATUS_ALIASES.values())
assert _STATUS_ALIAS_TARGETS.issubset(set(STATUS_OPTIONS)), (
    f"STATUS_ALIASES contains non-canonical targets: {_STATUS_ALIAS_TARGETS - set(STATUS_OPTIONS)}"
)

EMPLOYMENT_TYPE_ALIASES: dict[str, str] = {
    "internship": "Internship",
    "intern": "Internship",
    "full time": "Full Time",
    "fulltime": "Full Time",
    "part time": "Part Time",
    "parttime": "Part Time",
}
_ET_ALIAS_TARGETS = set(EMPLOYMENT_TYPE_ALIASES.values())
assert _ET_ALIAS_TARGETS.issubset(set(ALLOWED_EMPLOYMENT_TYPES)), (
    f"EMPLOYMENT_TYPE_ALIASES contains non-canonical targets: {_ET_ALIAS_TARGETS - set(ALLOWED_EMPLOYMENT_TYPES)}"
)

LOCATION_ALIASES: dict[str, str] = {
    "remote": "remote",
    "work from home": "remote",
    "wfh": "remote",
    "hybrid": "hybrid",
    "onsite": "on-site",
    "on site": "on-site",
    "on-site": "on-site",
}
_LOC_ALIAS_TARGETS = set(LOCATION_ALIASES.values())
assert _LOC_ALIAS_TARGETS.issubset(set(ALLOWED_LOCATIONS)), (
    f"LOCATION_ALIASES contains non-canonical targets: {_LOC_ALIAS_TARGETS - set(ALLOWED_LOCATIONS)}"
)

PRIORITY_ALIASES: dict[str, str] = {
    "low": "LOW",
    "medium": "MEDIUM",
    "high": "HIGH",
}
_PRIORITY_ALIAS_TARGETS = set(PRIORITY_ALIASES.values())
assert _PRIORITY_ALIAS_TARGETS.issubset(set(ALLOWED_PRIORITIES)), (
    f"PRIORITY_ALIASES contains non-canonical targets: {_PRIORITY_ALIAS_TARGETS - set(ALLOWED_PRIORITIES)}"
)


def _normalize_alias_key(value: str) -> str:
    """Collapse whitespace, casefold, and treat -/_ as spaces for alias lookup."""
    return " ".join(value.strip().replace("-", " ").replace("_", " ").casefold().split())


def normalize_priority_value(value: str) -> str | None:
    return PRIORITY_ALIASES.get(_normalize_alias_key(value))


def normalize_location_value(value: str) -> str | None:
    return LOCATION_ALIASES.get(_normalize_alias_key(value))


def normalize_employment_type_value(value: str) -> str | None:
    return EMPLOYMENT_TYPE_ALIASES.get(_normalize_alias_key(value))


def normalize_current_stage_value(value: str) -> str | None:
    normalized = _normalize_alias_key(value)
    for stage in ALLOWED_CURRENT_STAGES:
        if _normalize_alias_key(stage) == normalized:
            return stage
    return None


EVENT_TYPES = {
    "application_saved",
    "field_changed",
    "note_added",
    "status_changed",
    "application_archived",
    "application_restored",
}
