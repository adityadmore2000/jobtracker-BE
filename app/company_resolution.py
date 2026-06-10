from sqlalchemy.orm import Session

from .company_matching import normalize_company_name
from .models import CanonicalCompany, Company, CompanyAlias, JobApplication


# ---------------------------------------------------------------------------
# New Company table helpers (used by mutation dispatcher and main.py)
# ---------------------------------------------------------------------------

def get_company_by_normalized_name(db: Session, normalized_name: str) -> Company | None:
    return db.query(Company).filter(Company.normalized_name == normalized_name).first()


def get_or_create_company(db: Session, name: str) -> Company:
    """Return the Company that matches the normalized form of *name*, creating one if absent.

    The display name of an existing company is not overwritten — only the first
    spelling that created the record is retained. This is intentional: the DB
    record was created from a confirmed or user-supplied name and should not be
    silently overwritten by a later variant.
    """
    if not name or not name.strip():
        raise ValueError("company name must not be blank")
    normalized = normalize_company_name(name)
    if not normalized:
        raise ValueError("company name must not be blank after normalization")
    existing = get_company_by_normalized_name(db, normalized)
    if existing is not None:
        return existing
    company = Company(name=name.strip(), normalized_name=normalized)
    db.add(company)
    db.flush()
    return company


# ---------------------------------------------------------------------------
# Legacy CanonicalCompany helpers (retained for the ASR correction-event flow)
# ---------------------------------------------------------------------------

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
    companies = db.query(Company).order_by(Company.id.asc()).all()
    for company in companies:
        if company.name in seen_names:
            continue
        seen_names.add(company.name)
        if normalize_company_name(company.name) == normalized_name:
            return company.name
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
    matches = []
    for application in (
        db.query(JobApplication)
        .filter(JobApplication.is_draft == False)  # noqa: E712
        .order_by(JobApplication.id.asc())
        .all()
    ):
        if normalize_company_name(application.company) == normalized_name:
            matches.append(application)
    return matches


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
