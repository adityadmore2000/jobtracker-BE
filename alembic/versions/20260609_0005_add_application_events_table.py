"""Add application_events table.

Revision ID: 20260609_0005
Revises: 20260609_0004
Create Date: 2026-06-09 00:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260609_0005"
down_revision = "20260609_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "application_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("application_id", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["application_id"], ["job_applications.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_application_events_application_id", "application_events", ["application_id"])


def downgrade() -> None:
    op.drop_index("ix_application_events_application_id", table_name="application_events")
    op.drop_table("application_events")
