from __future__ import annotations

import argparse
from pathlib import Path
import sys

import psycopg
from psycopg import sql
from sqlalchemy.engine import make_url


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_DIR))

from app.database_config import derive_admin_database_url, get_required_database_url, require_database_name  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the job_tracker PostgreSQL databases if they do not already exist."
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="job_tracker primary database URL. Defaults to DATABASE_URL.",
    )
    parser.add_argument(
        "--test-database-url",
        default=None,
        help="job_tracker test database URL. Defaults to TEST_DATABASE_URL.",
    )
    return parser.parse_args()


def ensure_database(admin_database_url: str, database_name: str) -> bool:
    with psycopg.connect(admin_database_url, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database_name,))
            if cursor.fetchone():
                return False
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))
            return True


def to_psycopg_connection_url(database_url: str) -> str:
    parsed_url = make_url(database_url)
    drivername = parsed_url.drivername.replace("+psycopg", "")
    return parsed_url.set(drivername=drivername).render_as_string(hide_password=False)


def main() -> None:
    args = parse_args()
    database_url = args.database_url or get_required_database_url("DATABASE_URL")
    test_database_url = args.test_database_url or get_required_database_url("TEST_DATABASE_URL")
    require_database_name(database_url, "job_tracker", env_var="DATABASE_URL")
    require_database_name(test_database_url, "job_tracker_test", env_var="TEST_DATABASE_URL")
    admin_database_url = to_psycopg_connection_url(derive_admin_database_url(database_url, admin_database="postgres"))

    created_primary = ensure_database(admin_database_url, "job_tracker")
    created_test = ensure_database(admin_database_url, "job_tracker_test")

    print(
        {
            "database": "job_tracker",
            "created": created_primary,
            "database_url_env": "DATABASE_URL",
        }
    )
    print(
        {
            "database": "job_tracker_test",
            "created": created_test,
            "database_url_env": "TEST_DATABASE_URL",
        }
    )


if __name__ == "__main__":
    main()
