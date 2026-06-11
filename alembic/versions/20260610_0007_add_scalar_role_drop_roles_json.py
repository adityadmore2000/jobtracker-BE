"""Add scalar role column and drop roles_json from job_applications.

Revision ID: 20260610_0007
Revises: 20260609_0006
Create Date: 2026-06-10 00:00:00

NOTE: This migration was already applied to the developer DB before the file
was lost. Recreated as a faithful stub so alembic history is coherent.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260610_0007"
down_revision = "20260609_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "job_applications",
        sa.Column("role", sa.String(), nullable=False, server_default=""),
    )
    op.drop_column("job_applications", "roles_json")


def downgrade() -> None:
    op.add_column(
        "job_applications",
        sa.Column("roles_json", sa.JSON(), nullable=False, server_default="[]"),
    )
    op.drop_column("job_applications", "role")
