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
    applications = db.query(JobApplication.company).order_by(JobApplication.id.asc()).all()
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
        for application in db.query(JobApplication).order_by(JobApplication.id.asc()).all()
        if normalize_company_name(application.company) == normalized_name
    ]
