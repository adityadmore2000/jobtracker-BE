"""Add companies table and company_id FK to job_applications.

Revision ID: 20260610_0008
Revises: 20260610_0007
Create Date: 2026-06-10 00:00:00

Strategy:
1. Create companies table with normalized_name unique index.
2. Add nullable company_id to job_applications.
3. Backfill: find-or-create a Company for each distinct company text.
4. Populate company_id from the backfill mapping.
5. Make company_id NOT NULL.
6. Drop legacy job_applications.company text column.
7. Add FK + index.

Downgrade:
- Re-add company text column, backfill from companies.name.
- Drop company_id FK and column.
- Drop companies table.
- Downgraded rows will have company text but no roles_json (role stays).
"""

from __future__ import annotations

from datetime import datetime, timezone

import sqlalchemy as sa
from alembic import op
from sqlalchemy import text


revision = "20260610_0008"
down_revision = "20260610_0007"
branch_labels = None
depends_on = None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_company_name(value: str) -> str:
    import re
    import string
    trimmed = value.strip().lower()
    collapsed = re.sub(r"\s+", " ", trimmed)
    punct_table = str.maketrans({char: " " for char in string.punctuation})
    separated = collapsed.translate(punct_table)
    no_punct = separated.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", no_punct).strip()


def upgrade() -> None:
    conn = op.get_bind()

    # 1. Create companies table
    op.create_table(
        "companies",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("normalized_name", sa.Text(), nullable=False),
        sa.Column("website", sa.Text(), nullable=True),
        sa.Column("career_page", sa.Text(), nullable=True),
        sa.Column("linkedin_url", sa.Text(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("normalized_name"),
    )
    op.execute(
        "CREATE SEQUENCE IF NOT EXISTS companies_id_seq START 1 INCREMENT 1"
    )
    op.execute(
        "ALTER TABLE companies ALTER COLUMN id SET DEFAULT nextval('companies_id_seq')"
    )
    op.execute(
        "ALTER SEQUENCE companies_id_seq OWNED BY companies.id"
    )
    op.create_index("ix_companies_id", "companies", ["id"], unique=False)
    op.create_index("ix_companies_normalized_name", "companies", ["normalized_name"], unique=True)

    # 2. Add nullable company_id to job_applications
    op.add_column(
        "job_applications",
        sa.Column("company_id", sa.BigInteger(), nullable=True),
    )

    # 3. Backfill: collect distinct company names from job_applications
    rows = conn.execute(
        text("SELECT DISTINCT company FROM job_applications WHERE company IS NOT NULL AND company != ''")
    ).fetchall()

    now = _utcnow()
    company_name_to_id: dict[str, int] = {}

    for (company_name,) in rows:
        normalized = _normalize_company_name(company_name)
        if normalized in {_normalize_company_name(k) for k in company_name_to_id}:
            # find existing
            existing_norm_key = next(
                k for k in company_name_to_id
                if _normalize_company_name(k) == normalized
            )
            company_name_to_id[company_name] = company_name_to_id[existing_norm_key]
            continue

        result = conn.execute(
            text(
                "INSERT INTO companies (name, normalized_name, notes, created_at, updated_at) "
                "VALUES (:name, :normalized_name, '', :created_at, :updated_at) "
                "RETURNING id"
            ),
            {"name": company_name, "normalized_name": normalized, "created_at": now, "updated_at": now},
        )
        company_id = result.scalar()
        company_name_to_id[company_name] = company_id

    # 4. Populate company_id on each row
    for company_name, company_id in company_name_to_id.items():
        conn.execute(
            text(
                "UPDATE job_applications SET company_id = :cid WHERE company = :name"
            ),
            {"cid": company_id, "name": company_name},
        )

    # 5. Make company_id NOT NULL
    op.alter_column("job_applications", "company_id", nullable=False)

    # 6. Drop legacy company text column and its index
    op.drop_index("ix_job_applications_company", table_name="job_applications")
    op.drop_column("job_applications", "company")

    # 7. Add FK and index on company_id
    op.create_foreign_key(
        "fk_job_applications_company_id",
        "job_applications",
        "companies",
        ["company_id"],
        ["id"],
    )
    op.create_index("ix_job_applications_company_id", "job_applications", ["company_id"], unique=False)


def downgrade() -> None:
    conn = op.get_bind()

    # Re-add company text column
    op.add_column(
        "job_applications",
        sa.Column("company", sa.String(), nullable=True),
    )

    # Backfill company text from companies.name via company_id
    conn.execute(
        text(
            "UPDATE job_applications ja "
            "SET company = c.name "
            "FROM companies c "
            "WHERE ja.company_id = c.id"
        )
    )

    # Make company NOT NULL with default
    op.alter_column("job_applications", "company", nullable=False, server_default="")
    op.create_index("ix_job_applications_company", "job_applications", ["company"], unique=False)

    # Drop company_id FK and column
    op.drop_constraint("fk_job_applications_company_id", "job_applications", type_="foreignkey")
    op.drop_index("ix_job_applications_company_id", table_name="job_applications")
    op.drop_column("job_applications", "company_id")

    # Drop companies table
    op.drop_index("ix_companies_normalized_name", table_name="companies")
    op.drop_index("ix_companies_id", table_name="companies")
    op.drop_table("companies")
