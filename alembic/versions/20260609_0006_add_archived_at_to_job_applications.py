"""Add archived_at column to job_applications.

Revision ID: 20260609_0006
Revises: 20260609_0005
Create Date: 2026-06-09 00:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260609_0006"
down_revision = "20260609_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "job_applications",
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True, server_default=None),
    )


def downgrade() -> None:
    op.drop_column("job_applications", "archived_at")
