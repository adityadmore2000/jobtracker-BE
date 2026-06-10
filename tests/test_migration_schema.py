"""Tests for migration schema correctness and round-trip."""

from __future__ import annotations

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from app.models import Company, JobApplication


def test_companies_table_exists(db_session: Session):
    inspector = inspect(db_session.bind)
    tables = inspector.get_table_names()
    assert "companies" in tables


def test_job_applications_has_company_id_not_company_text(db_session: Session):
    inspector = inspect(db_session.bind)
    columns = {col["name"] for col in inspector.get_columns("job_applications")}
    assert "company_id" in columns
    assert "company" not in columns


def test_job_applications_has_role_not_roles_json(db_session: Session):
    inspector = inspect(db_session.bind)
    columns = {col["name"] for col in inspector.get_columns("job_applications")}
    assert "role" in columns
    assert "roles_json" not in columns


def test_companies_has_normalized_name_unique(db_session: Session):
    inspector = inspect(db_session.bind)
    indexes = inspector.get_indexes("companies")
    unique_index_cols = {
        idx["column_names"][0]
        for idx in indexes
        if idx.get("unique") and len(idx["column_names"]) == 1
    }
    assert "normalized_name" in unique_index_cols


def test_company_fk_from_job_applications(db_session: Session):
    inspector = inspect(db_session.bind)
    fks = inspector.get_foreign_keys("job_applications")
    fk_targets = {(fk["referred_table"], tuple(fk["referred_columns"])) for fk in fks}
    assert ("companies", ("id",)) in fk_targets


def test_company_unique_constraint_prevents_duplicate_normalized_name(db_session: Session):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    db_session.add(Company(name="Rockwell", normalized_name="rockwell", created_at=now, updated_at=now))
    db_session.flush()
    db_session.add(Company(name="ROCKWELL", normalized_name="rockwell", created_at=now, updated_at=now))
    from sqlalchemy.exc import IntegrityError
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()
