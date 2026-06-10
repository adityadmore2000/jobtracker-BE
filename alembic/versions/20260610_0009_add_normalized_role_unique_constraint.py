"""Add normalized_role and enforce company_id + normalized_role uniqueness.

Revision ID: 20260610_0009
Revises: 20260610_0008
Create Date: 2026-06-10 00:00:00

Strategy:
1. Add nullable normalized_role TEXT column.
2. Backfill normalized_role from role (trim + collapse whitespace + casefold).
3. Audit duplicate (company_id, normalized_role) groups.
4. For conflict-free groups: consolidate — keep best canonical row, re-parent
   notes and events from redundant rows, add consolidation event, delete
   redundant rows.
5. For conflicting groups: abort with detail (prefer safe path B).
6. Make normalized_role NOT NULL.
7. Add UNIQUE constraint on (company_id, normalized_role).
8. Add supporting lookup index.

Downgrade:
- Drop UNIQUE constraint and index.
- Drop normalized_role column.
- Does NOT recreate removed duplicate rows (that data is gone).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import sqlalchemy as sa
from alembic import op
from sqlalchemy import text


revision = "20260610_0009"
down_revision = "20260610_0008"
branch_labels = None
depends_on = None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_role(role: str) -> str:
    return re.sub(r"\s+", " ", role.strip()).casefold()


# ---------------------------------------------------------------------------
# Business-field keys that, if they differ between rows in a group, mean the
# group is "conflicting" and cannot be auto-consolidated.
# ---------------------------------------------------------------------------
_CONFLICT_FIELDS = ("status", "priority", "job_link", "location", "next_action", "comments")


def upgrade() -> None:
    conn = op.get_bind()

    # -----------------------------------------------------------------------
    # 1. Add nullable normalized_role
    # -----------------------------------------------------------------------
    op.add_column(
        "job_applications",
        sa.Column("normalized_role", sa.Text(), nullable=True),
    )

    # -----------------------------------------------------------------------
    # 2. Backfill normalized_role from role
    # -----------------------------------------------------------------------
    rows = conn.execute(
        text("SELECT id, role FROM job_applications")
    ).fetchall()

    for (app_id, role) in rows:
        normalized = _normalize_role(role or "")
        conn.execute(
            text("UPDATE job_applications SET normalized_role = :nr WHERE id = :id"),
            {"nr": normalized, "id": app_id},
        )

    # -----------------------------------------------------------------------
    # 3. Audit duplicate (company_id, normalized_role) groups
    # -----------------------------------------------------------------------
    dup_groups_rows = conn.execute(
        text(
            "SELECT company_id, normalized_role, array_agg(id ORDER BY "
            "  CASE WHEN is_draft = false AND archived_at IS NULL THEN 0 "
            "       WHEN is_draft = false THEN 1 "
            "       ELSE 2 END, "
            "  updated_at DESC, id DESC"
            ") AS ids "
            "FROM job_applications "
            "GROUP BY company_id, normalized_role "
            "HAVING count(*) > 1"
        )
    ).fetchall()

    if not dup_groups_rows:
        # No duplicates — proceed directly to constraint.
        pass
    else:
        # -----------------------------------------------------------------------
        # 4 & 5. Separate conflict-free from conflicting groups
        # -----------------------------------------------------------------------
        conflicting_groups: list[dict] = []
        safe_groups: list[dict] = []

        for group_row in dup_groups_rows:
            company_id = group_row[0]
            norm_role = group_row[1]
            ids = list(group_row[2])  # already ordered: canonical first

            app_rows = conn.execute(
                text(
                    "SELECT id, status, priority, job_link, location, "
                    "next_action, comments, is_draft, archived_at, updated_at, "
                    "(SELECT count(*) FROM application_notes WHERE application_id = ja.id) AS note_count, "
                    "(SELECT count(*) FROM application_events WHERE application_id = ja.id) AS event_count "
                    "FROM job_applications ja WHERE id = ANY(:ids)"
                ),
                {"ids": ids},
            ).fetchall()

            # Map id → row for convenience
            by_id = {r[0]: r for r in app_rows}

            # Check for conflicts on business fields
            # Empty-string values are treated as "no value set" for conflict purposes.
            reference_id = ids[0]
            reference = by_id[reference_id]
            conflict_detail: list[str] = []

            for other_id in ids[1:]:
                other = by_id[other_id]
                for field_idx, field_name in enumerate(_CONFLICT_FIELDS):
                    # Column positions in SELECT: id=0,status=1,priority=2,job_link=3,location=4,next_action=5,comments=6
                    col_offset = 1 + field_idx
                    ref_val = reference[col_offset] or ""
                    other_val = other[col_offset] or ""
                    if ref_val != other_val and ref_val != "" and other_val != "":
                        conflict_detail.append(
                            f"  company_id={company_id} normalized_role={norm_role!r}: "
                            f"id={reference_id}.{field_name}={ref_val!r} vs id={other_id}.{field_name}={other_val!r}"
                        )

            if conflict_detail:
                conflicting_groups.append({
                    "company_id": company_id,
                    "norm_role": norm_role,
                    "ids": ids,
                    "detail": conflict_detail,
                })
            else:
                safe_groups.append({
                    "company_id": company_id,
                    "norm_role": norm_role,
                    "ids": ids,
                    "by_id": by_id,
                })

        if conflicting_groups:
            lines = ["Migration aborted: conflicting duplicate (company_id, normalized_role) groups found."]
            lines.append("Manual remediation required before this migration can proceed.")
            lines.append("Conflicting groups:")
            for g in conflicting_groups:
                lines.append(f"  company_id={g['company_id']} normalized_role={g['norm_role']!r} ids={g['ids']}")
                lines.extend(g["detail"])
            raise RuntimeError("\n".join(lines))

        # -----------------------------------------------------------------------
        # Consolidate safe groups
        # -----------------------------------------------------------------------
        for group in safe_groups:
            ids = group["ids"]
            canonical_id = ids[0]
            redundant_ids = ids[1:]
            now = _utcnow()

            # Re-parent notes
            for rid in redundant_ids:
                conn.execute(
                    text(
                        "UPDATE application_notes SET application_id = :canonical "
                        "WHERE application_id = :redundant"
                    ),
                    {"canonical": canonical_id, "redundant": rid},
                )

            # Re-parent events
            for rid in redundant_ids:
                conn.execute(
                    text(
                        "UPDATE application_events SET application_id = :canonical "
                        "WHERE application_id = :redundant"
                    ),
                    {"canonical": canonical_id, "redundant": rid},
                )

            # Add consolidation event to canonical row
            import json as _json
            conn.execute(
                text(
                    "INSERT INTO application_events (application_id, event_type, payload, created_at) "
                    "VALUES (:app_id, 'application_duplicates_consolidated', "
                    "  CAST(:payload AS jsonb), :created_at)"
                ),
                {
                    "app_id": canonical_id,
                    "payload": _json.dumps({
                        "merged_application_ids": redundant_ids,
                        "kept_application_id": canonical_id,
                        "reason": "same_company_normalized_role",
                    }),
                    "created_at": now,
                },
            )

            # Delete redundant rows (notes/events already re-parented so cascade won't lose them)
            conn.execute(
                text("DELETE FROM job_applications WHERE id = ANY(:ids)"),
                {"ids": redundant_ids},
            )

    # -----------------------------------------------------------------------
    # 6. Make normalized_role NOT NULL
    # -----------------------------------------------------------------------
    op.alter_column("job_applications", "normalized_role", nullable=False)

    # -----------------------------------------------------------------------
    # 7. Add UNIQUE constraint on (company_id, normalized_role)
    # -----------------------------------------------------------------------
    op.create_unique_constraint(
        "uq_job_applications_company_role",
        "job_applications",
        ["company_id", "normalized_role"],
    )

    # -----------------------------------------------------------------------
    # 8. Add lookup index (the UNIQUE constraint already creates one in PG,
    #    but add an explicit non-unique index on normalized_role alone for
    #    role-only queries if needed in future).
    # -----------------------------------------------------------------------
    op.create_index(
        "ix_job_applications_normalized_role",
        "job_applications",
        ["normalized_role"],
        unique=False,
    )


def downgrade() -> None:
    # Drop the unique constraint and indexes
    op.drop_index("ix_job_applications_normalized_role", table_name="job_applications")
    op.drop_constraint("uq_job_applications_company_role", "job_applications", type_="unique")
    # Drop normalized_role column
    op.drop_column("job_applications", "normalized_role")
    # Note: duplicate rows that were removed during upgrade are NOT recreated.
