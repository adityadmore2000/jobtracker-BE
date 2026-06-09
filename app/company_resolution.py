from sqlalchemy.orm import Session

from .company_matching import normalize_company_name
from .models import CanonicalCompany, CompanyAlias, JobApplication


def get_canonical_company_by_normalized_name(db: Session, normalized_name: str) -> CanonicalCompany | None:
    for canonical_company in db.query(CanonicalCompany).order_by(CanonicalCompany.id.asc()).all():
        if normalize_company_name(canonical_company.canonical_name) == normalized_name:
            return canonical_company
    return None


def get_company_alias_by_normalized_name(db: Session, normalized_name: str) -> CompanyAlias | None:
    for alias in db.query(CompanyAlias).order_by(CompanyAlias.id.asc()).all():
        if normalize_company_name(alias.alias_text) == normalized_name:
            return alias
    return None


def get_existing_application_company(db: Session, normalized_name: str) -> str | None:
    seen_names: set[str] = set()
    applications = db.query(JobApplication.company).filter(JobApplication.is_draft == False).order_by(JobApplication.id.asc()).all()  # noqa: E712
    for (company_name,) in applications:
        if company_name in seen_names:
            continue
        seen_names.add(company_name)
        if normalize_company_name(company_name) == normalized_name:
            return company_name
    return None


def ensure_canonical_company(db: Session, company_name: str) -> CanonicalCompany:
    normalized_name = normalize_company_name(company_name)
    existing_company = get_canonical_company_by_normalized_name(db, normalized_name)
    if existing_company is not None:
        return existing_company

    canonical_company = CanonicalCompany(canonical_name=company_name.strip())
    db.add(canonical_company)
    db.flush()
    return canonical_company


def resolve_company_name(db: Session, company_name: str) -> tuple[str | None, CanonicalCompany | None]:
    normalized_name = normalize_company_name(company_name)
    if not normalized_name:
        return None, None

    canonical_company = get_canonical_company_by_normalized_name(db, normalized_name)
    if canonical_company is not None:
        return canonical_company.canonical_name, canonical_company

    alias = get_company_alias_by_normalized_name(db, normalized_name)
    if alias is not None:
        return alias.canonical_company.canonical_name, alias.canonical_company

    application_company = get_existing_application_company(db, normalized_name)
    if application_company is not None:
        return application_company, ensure_canonical_company(db, application_company)

    return None, None


def get_application_matches_for_company(db: Session, company_name: str) -> list[JobApplication]:
    normalized_name = normalize_company_name(company_name)
    return [
        application
        for application in db.query(JobApplication).filter(JobApplication.is_draft == False).order_by(JobApplication.id.asc()).all()  # noqa: E712
        if normalize_company_name(application.company) == normalized_name
    ]


def detect_explicit_known_companies(db: Session, utterance: str) -> list[str]:
    normalized_utterance = normalize_company_name(utterance)
    if not normalized_utterance:
        return []

    utterance_tokens = normalized_utterance.split()
    known_phrases: dict[tuple[str, ...], set[str]] = {}

    for canonical_company in db.query(CanonicalCompany).order_by(CanonicalCompany.id.asc()).all():
        normalized_company = normalize_company_name(canonical_company.canonical_name)
        if normalized_company:
            known_phrases.setdefault(tuple(normalized_company.split()), set()).add(canonical_company.canonical_name)

    for alias in db.query(CompanyAlias).order_by(CompanyAlias.id.asc()).all():
        normalized_alias = normalize_company_name(alias.alias_text)
        if normalized_alias:
            known_phrases.setdefault(tuple(normalized_alias.split()), set()).add(alias.canonical_company.canonical_name)

    if not known_phrases:
        return []

    max_phrase_length = max(len(tokens) for tokens in known_phrases)
    occupied = [False] * len(utterance_tokens)
    matches: list[str] = []

    for phrase_length in range(max_phrase_length, 0, -1):
        for start_index in range(0, len(utterance_tokens) - phrase_length + 1):
            end_index = start_index + phrase_length
            if any(occupied[start_index:end_index]):
                continue
            phrase_tokens = tuple(utterance_tokens[start_index:end_index])
            canonical_names = known_phrases.get(phrase_tokens)
            if not canonical_names:
                continue
            for canonical_name in sorted(canonical_names):
                if canonical_name not in matches:
                    matches.append(canonical_name)
            for index in range(start_index, end_index):
                occupied[index] = True

    return matches
