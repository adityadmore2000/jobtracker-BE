import re

from sqlalchemy.orm import Session

from .models import JobApplication


def normalize_role_name(role: str) -> str:
    """Conservative role normalization: trim, collapse whitespace, casefold."""
    return re.sub(r"\s+", " ", role.strip()).casefold()


def find_application_by_company_role(
    db: Session,
    *,
    company_id: int,
    role: str,
) -> "JobApplication | None":
    """Return the first JobApplication matching company_id + normalized_role, or None."""
    normalized = normalize_role_name(role)
    return (
        db.query(JobApplication)
        .filter(
            JobApplication.company_id == company_id,
            JobApplication.normalized_role == normalized,
        )
        .first()
    )
