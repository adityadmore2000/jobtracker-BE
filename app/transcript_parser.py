import re

from .constants import (
    ALLOWED_CURRENT_STAGES,
    ALLOWED_EMPLOYMENT_TYPES,
    ALLOWED_LOCATIONS,
    ALLOWED_PRIORITIES,
    ALLOWED_ROLES,
    STATUS_OPTIONS,
)
from .schemas import JobDraftPatch, ParsedTranscriptCommand


ROLE_ALIASES = {
    "ai engineer": "AI Engineer",
    "generative ai engineer": "Generative AI Engineer",
    "gen ai engineer": "GenAI Engineer",
    "genai engineer": "GenAI Engineer",
    "llm engineer": "LLM Engineer",
    "rag engineer": "RAG Engineer",
    "ai systems engineer": "AI Systems Engineer",
    "ml engineer": "ML Engineer",
    "computer vision engineer": "Computer Vision Engineer",
    "cv engineer": "Computer Vision Engineer",
    "agentic ai engineer": "Agentic AI Engineer",
    "data science": "Data Science",
    "prompt engineer": "Prompt Engineer",
    "platform engineer": "Platform Engineer",
    "get": "GET",
    "ai product engineer": "AI Product Engineer",
}

TYPE_ALIASES = {
    "internship": "Internship",
    "intern": "Internship",
    "full time": "Full Time",
    "full-time": "Full Time",
    "part time": "Part Time",
    "part-time": "Part Time",
}

STAGE_ALIASES = {
    "tailored": "Tailored",
    "applied": "Applied",
    "networked": "Networked",
    "engaged": "Engaged",
    "cold_mail": "COLD_MAIL",
    "cold mail": "COLD_MAIL",
    "followed up": "Followed up",
    "follow up": "Followed up",
}

FILLER_WORDS = {
    "a",
    "an",
    "application",
    "for",
    "job",
    "new",
    "role",
    "save",
    "current",
    "page",
}


def parse_transcript(transcript: str, *, correction: bool = False) -> ParsedTranscriptCommand:
    raw_transcript = transcript.strip()
    normalized = normalize_text(raw_transcript)
    sentences = split_sentences(raw_transcript)
    patch = JobDraftPatch()
    warnings: list[str] = []

    patch.roles_add = find_matches(normalized, ROLE_ALIASES)
    patch.employment_types_add = find_matches(normalized, TYPE_ALIASES)
    patch.location = parse_location(normalized)
    patch.priority = parse_priority(normalized)
    patch.use_latest_browser_url = has_current_link_request(normalized)
    patch.status = parse_status(normalized)
    patch.engaged_days = parse_engaged_days(normalized)
    patch.next_action = parse_next_action(sentences)
    comments_replace, comments_append = parse_comments(sentences)
    patch.comments_replace = comments_replace
    patch.comments_append = comments_append

    added_stages, removed_stages = parse_stage_changes(sentences)
    patch.current_stages_add = added_stages
    patch.current_stages_remove = removed_stages

    role_removals = parse_removals(sentences, ROLE_ALIASES)
    type_removals = parse_removals(sentences, TYPE_ALIASES)
    patch.roles_remove = role_removals
    patch.employment_types_remove = type_removals
    patch.roles_add = remove_values(patch.roles_add, patch.roles_remove)
    patch.employment_types_add = remove_values(patch.employment_types_add, patch.employment_types_remove)

    if correction:
        intent = parse_correction_intent(normalized)
    else:
        patch.company = parse_company(sentences, patch.roles_add, patch.employment_types_add)
        intent = parse_intent(normalized)
        if intent == "UNKNOWN" and is_narrative_add_application(normalized):
            intent = "ADD_APPLICATION"
        elif intent == "UNKNOWN" and has_meaningful_patch(patch):
            intent = "ADD_APPLICATION"
        if intent == "ADD_APPLICATION" and not patch.company:
            warnings.append("Company could not be extracted.")

    if intent == "UNKNOWN" and not has_meaningful_patch(patch):
        warnings.append("No supported command was detected.")

    return ParsedTranscriptCommand(intent=intent, patch=patch, raw_transcript=raw_transcript, warnings=warnings)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("-", " ").replace("_", " ")).strip().lower()


def split_sentences(transcript: str) -> list[str]:
    return [part.strip() for part in re.split(r"[.\n]+", transcript) if part.strip()]


def find_matches(normalized: str, aliases: dict[str, str]) -> list[str]:
    matches: list[str] = []
    remaining = normalized
    for alias, value in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        pattern = rf"\b{re.escape(alias)}\b"
        if re.search(pattern, remaining) and value not in matches:
            matches.append(value)
            remaining = re.sub(pattern, " ", remaining)
    return matches


def remove_values(values: list[str], removals: list[str]) -> list[str]:
    return [value for value in values if value not in removals]


def parse_location(normalized: str) -> str | None:
    if re.search(r"\bon\s*site\b", normalized):
        return "onsite"
    for location in ALLOWED_LOCATIONS:
        if re.search(rf"\b{re.escape(location)}\b", normalized):
            return location
    return None


def parse_priority(normalized: str) -> str | None:
    if not re.search(r"\b(priority|prioritize)\b", normalized):
        return None
    for priority in ALLOWED_PRIORITIES:
        if re.search(rf"\b{priority.lower()}\b", normalized):
            return priority
    return None


def has_current_link_request(normalized: str) -> bool:
    phrases = [
        "use current link",
        "use the current link",
        "use current url",
        "use the current url",
        "use current tab",
        "use the current tab",
        "attach current page",
        "attach the current page",
        "use latest captured url",
        "use the latest captured url",
    ]
    return any(phrase in normalized for phrase in phrases)


def parse_status(normalized: str) -> str | None:
    if re.search(r"\b(applied for|i applied for|submitted my application for)\b", normalized):
        return "Applied"
    if not re.search(
        r"\b(set|mark|change|update)\s+status\s+(to|as)?\b|\bstatus\s+(to|as|is|:)?\b",
        normalized,
    ):
        return None
    for status in STATUS_OPTIONS:
        if re.search(rf"\b{re.escape(status.lower())}\b", normalized):
            return status
    return None


def parse_engaged_days(normalized: str) -> int | None:
    patterns = [
        r"\bengaged\s+days\s+(\d+)\b",
        r"\bengaged\s+(\d+)\s+days\b",
        r"\bengaged\s+number\s+of\s+days\s+(\d+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return int(match.group(1))
    return None


def parse_next_action(sentences: list[str]) -> str | None:
    for sentence in sentences:
        match = re.search(r"\b(?:next action|next step|action item)\s*(?:is|to|:)?\s*(.+)$", sentence, re.IGNORECASE)
        if match:
            return clean_dictated_text(match.group(1))
        future_action_match = re.search(r"\b(?:i should|i need to|my next step is)\s+(.+)$", sentence, re.IGNORECASE)
        if future_action_match:
            return clean_dictated_text(future_action_match.group(1))
    return None


def parse_comments(sentences: list[str]) -> tuple[str | None, str | None]:
    replace_value = None
    append_value = None
    for sentence in sentences:
        replace_match = re.search(r"\b(?:replace|change)\s+(?:comment|comments|note|notes)\s+(?:with|to)\s+(.+)$", sentence, re.IGNORECASE)
        if replace_match:
            replace_value = clean_dictated_text(replace_match.group(1))
            continue

        append_match = re.search(
            r"\b(?:append|add)\s+(?:a\s+)?(?:comment|comments|note|notes)\s*(?:saying|with|:)?\s*(.+)$",
            sentence,
            re.IGNORECASE,
        )
        if append_match:
            append_value = clean_dictated_text(append_match.group(1))
    return replace_value, append_value


def parse_stage_changes(sentences: list[str]) -> tuple[list[str], list[str]]:
    additions: list[str] = []
    removals: list[str] = []
    for sentence in sentences:
        normalized = normalize_text(sentence)
        has_stage_marker = re.search(r"\b(stage|stages|current stage|current stages)\b", normalized)
        has_narrative_stage_marker = re.search(r"\btailored\s+my\s+application\b|\bstarted\s+engaging\b|\balready\s+tailored\b", normalized)
        if not re.search(r"\b(add|remove|delete)\b", normalized) and not has_stage_marker and not has_narrative_stage_marker:
            continue
        if re.search(r"\btailored\s+my\s+application\b|\balready\s+tailored\b", normalized):
            append_unique(additions, "Tailored")
        if re.search(r"\bstarted\s+engaging\b", normalized):
            append_unique(additions, "Engaged")
        for alias, stage in STAGE_ALIASES.items():
            if not re.search(rf"\b{re.escape(alias)}\b", normalized):
                continue
            if re.search(r"\b(remove|delete)\b", normalized):
                append_unique(removals, stage)
            elif has_stage_marker or has_narrative_stage_marker or is_explicit_stage_sentence(normalized, alias):
                append_unique(additions, stage)
    return additions, removals


def is_explicit_stage_sentence(normalized_sentence: str, alias: str) -> bool:
    if re.search(r"\b(stage|stages|current stage)\b", normalized_sentence):
        return True
    return alias in {"tailored", "networked", "engaged", "cold_mail", "cold mail", "followed up", "follow up"}


def parse_removals(sentences: list[str], aliases: dict[str, str]) -> list[str]:
    removals: list[str] = []
    for sentence in sentences:
        normalized = normalize_text(sentence)
        if not re.search(r"\b(remove|delete)\b", normalized):
            continue
        remaining = normalized
        for alias, value in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
            pattern = rf"\b{re.escape(alias)}\b"
            if re.search(pattern, remaining):
                append_unique(removals, value)
                remaining = re.sub(pattern, " ", remaining)
    return removals


def append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def clean_dictated_text(value: str) -> str:
    cleaned = value.strip()
    cleaned = re.sub(r"^(saying|with|to|is|as)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" .")


def parse_company(sentences: list[str], roles: list[str], employment_types: list[str]) -> str | None:
    for sentence in sentences:
        explicit = re.search(r"\bcompany\s+([A-Za-z0-9][A-Za-z0-9&.,' -]*?)\s*$", sentence, re.IGNORECASE)
        if explicit:
            return clean_company(explicit.group(1), roles, employment_types)

        trailing = re.search(r"\b([A-Za-z0-9][A-Za-z0-9&.'-]*)\s+company\b", sentence, re.IGNORECASE)
        if trailing:
            return clean_company(trailing.group(1), roles, employment_types)

    for sentence in sentences:
        narrative = re.search(
            r"\b(?:applied for|i applied for|submitted my application for|interested in|considering)\s+(?:an?\s+|the\s+)?(?:[A-Za-z0-9&.' -]+?\s+)?(?:role|job|application|internship|intern|full time|full-time|part time|part-time)?\s+at\s+([A-Za-z0-9][A-Za-z0-9&.' -]*?)\s*$",
            sentence,
            re.IGNORECASE,
        )
        if narrative:
            return clean_company(narrative.group(1), roles, employment_types)

        implicit = re.search(
            r"\b(?:add|new|save)\s+(?:an?\s+)?(?:application\s+for\s+|job\s+for\s+)?([A-Za-z0-9][A-Za-z0-9&.' -]*?)\s+(?:role|job|application|internship|intern|full time|full-time|part time|part-time|generative ai engineer|agentic ai engineer|ai product engineer|ai systems engineer|computer vision engineer|genai engineer|gen ai engineer|prompt engineer|platform engineer|ai engineer|llm engineer|rag engineer|ml engineer|cv engineer|data science|get)\b",
            sentence,
            re.IGNORECASE,
        )
        if implicit:
            company = clean_company(implicit.group(1), roles, employment_types)
            if company:
                return company
    return None


def clean_company(value: str, roles: list[str], employment_types: list[str]) -> str | None:
    cleaned = value.strip(" .")
    for role in roles:
        cleaned = re.sub(rf"\b{re.escape(role)}\b", "", cleaned, flags=re.IGNORECASE)
    for employment_type in employment_types:
        cleaned = re.sub(rf"\b{re.escape(employment_type)}\b", "", cleaned, flags=re.IGNORECASE)
    words = [word for word in cleaned.split() if word.lower() not in FILLER_WORDS]
    cleaned = " ".join(words).strip(" .")
    return cleaned or None


def has_meaningful_patch(patch: JobDraftPatch) -> bool:
    return bool(
        patch.company
        or patch.roles_add
        or patch.roles_remove
        or patch.employment_types_add
        or patch.employment_types_remove
        or patch.job_link
        or patch.use_latest_browser_url
        or patch.location
        or patch.status
        or patch.current_stages_add
        or patch.current_stages_remove
        or patch.priority
        or patch.engaged_days is not None
        or patch.next_action
        or patch.comments_append
        or patch.comments_replace
    )


def is_narrative_add_application(normalized: str) -> bool:
    return bool(
        re.search(
            r"\b(applied for|i applied for|submitted my application for|interested in|considering)\b",
            normalized,
        )
    )


def parse_intent(normalized: str) -> str:
    if re.search(r"\b(cancel|discard|clear draft)\b", normalized):
        return "CANCEL_ACTIVE_DRAFT"
    if re.search(r"\b(final save|confirm|persist|save)\b", normalized):
        if re.search(r"\b(save current job|save current page)\b", normalized):
            return "ADD_APPLICATION"
        return "SAVE_ACTIVE_DRAFT"
    if re.search(r"\b(add application|add job|new application|new job)\b", normalized):
        return "ADD_APPLICATION"
    if is_narrative_add_application(normalized):
        return "ADD_APPLICATION"
    if re.search(r"\badd\b", normalized) and find_matches(normalized, ROLE_ALIASES | TYPE_ALIASES):
        return "ADD_APPLICATION"
    if re.search(r"\b(add|change|update|modify|remove|delete tag)\b", normalized):
        return "PATCH_ACTIVE_DRAFT"
    return "UNKNOWN"


def parse_correction_intent(normalized: str) -> str:
    if re.search(r"\b(cancel|discard|clear draft)\b", normalized):
        return "CANCEL_ACTIVE_DRAFT"
    if re.search(r"\b(final save|confirm|persist|save)\b", normalized):
        return "SAVE_ACTIVE_DRAFT"
    if re.search(r"\b(add|change|update|modify|remove|delete|set|use|append|replace)\b", normalized):
        return "PATCH_ACTIVE_DRAFT"
    return "PATCH_ACTIVE_DRAFT"
