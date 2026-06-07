import os
import sys
from pathlib import Path

from alembic import command
from alembic.config import Config
import pytest
from sqlalchemy import create_engine, text


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

from app.database_config import get_required_database_url  # noqa: E402

TEST_DATABASE_URL = get_required_database_url("TEST_DATABASE_URL")
os.environ["DATABASE_URL"] = TEST_DATABASE_URL

from app.database import SessionLocal, engine  # noqa: E402


def alembic_config() -> Config:
    config = Config(str(PROJECT_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_DIR / "alembic"))
    return config


def reset_public_schema(database_url: str) -> None:
    admin_engine = create_engine(database_url, isolation_level="AUTOCOMMIT", pool_pre_ping=True)
    with admin_engine.connect() as connection:
        connection.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
        connection.execute(text("GRANT ALL ON SCHEMA public TO CURRENT_USER"))
        connection.execute(text("GRANT ALL ON SCHEMA public TO public"))
    admin_engine.dispose()


def truncate_all_tables() -> None:
    with engine.begin() as connection:
        table_names = connection.execute(
            text(
                """
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = 'public'
                ORDER BY tablename ASC
                """
            )
        ).scalars().all()
        if not table_names:
            return
        joined = ", ".join(f'"{table_name}"' for table_name in table_names)
        connection.execute(text(f"TRUNCATE TABLE {joined} RESTART IDENTITY CASCADE"))


@pytest.fixture(scope="session", autouse=True)
def prepare_test_database() -> None:
    reset_public_schema(TEST_DATABASE_URL)
    command.upgrade(alembic_config(), "head")
    yield
    truncate_all_tables()


@pytest.fixture(autouse=True)
def reset_database_state():
    truncate_all_tables()
    yield
    truncate_all_tables()


@pytest.fixture
def db_session():
    with SessionLocal() as session:
        yield session
