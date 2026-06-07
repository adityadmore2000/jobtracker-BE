import os
import sys
from pathlib import Path

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine, inspect, text

from app import database_config, migrations
from app.main import app
from scripts import bootstrap_postgres


PROJECT_DIR = Path(__file__).resolve().parents[1]
TEST_DATABASE_URL = os.environ["TEST_DATABASE_URL"]


def alembic_config() -> Config:
    config = Config(str(PROJECT_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_DIR / "alembic"))
    return config


@pytest.fixture(autouse=True)
def reset_config_state() -> None:
    database_config.reset_backend_environment_cache()
    migrations.reset_auto_migrate_state()
    yield
    database_config.reset_backend_environment_cache()
    migrations.reset_auto_migrate_state()


def test_backend_dotenv_is_loaded_when_os_values_are_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    expected_url = "postgresql+psycopg://dotenv:dotenv@localhost:5432/job_tracker"
    env_file.write_text(f"DATABASE_URL={expected_url}\n", encoding="utf-8")

    monkeypatch.setattr(database_config, "get_backend_env_path", lambda: env_file)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    assert database_config.get_required_database_url("DATABASE_URL") == expected_url


def test_os_environment_variables_override_backend_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DATABASE_URL=postgresql+psycopg://dotenv:dotenv@localhost:5432/job_tracker\n",
        encoding="utf-8",
    )
    expected_url = "postgresql+psycopg://env:env@localhost:5432/job_tracker"

    monkeypatch.setattr(database_config, "get_backend_env_path", lambda: env_file)
    monkeypatch.setenv("DATABASE_URL", expected_url)

    assert database_config.get_required_database_url("DATABASE_URL") == expected_url


def test_missing_database_url_fails_clearly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(database_config, "get_backend_env_path", lambda: tmp_path / ".env")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="DATABASE_URL is required"):
        database_config.get_required_database_url("DATABASE_URL")


def test_auto_migrate_false_does_not_trigger_alembic(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    monkeypatch.setenv("AUTO_MIGRATE", "false")
    monkeypatch.setattr(migrations, "run_alembic_upgrade_head", lambda: calls.append("upgrade"))

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert calls == []


def test_auto_migrate_true_triggers_alembic_once_before_serving(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    monkeypatch.setenv("AUTO_MIGRATE", "true")
    monkeypatch.setattr(migrations, "run_alembic_upgrade_head", lambda: calls.append("upgrade"))

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200

    migrations.run_startup_migrations_if_enabled()

    assert calls == ["upgrade"]


def test_auto_migrate_failure_prevents_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTO_MIGRATE", "true")

    def fail_upgrade() -> None:
        raise RuntimeError("alembic upgrade head failed")

    monkeypatch.setattr(migrations, "run_alembic_upgrade_head", fail_upgrade)

    with pytest.raises(RuntimeError, match="alembic upgrade head failed"):
        with TestClient(app):
            pass


def test_bootstrap_postgres_uses_backend_dotenv_without_manual_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "DATABASE_URL=postgresql+psycopg://dotenv:dotenv@localhost:5432/job_tracker",
                "TEST_DATABASE_URL=postgresql+psycopg://dotenv:dotenv@localhost:5432/job_tracker_test",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    created_names: list[str] = []

    monkeypatch.setattr(database_config, "get_backend_env_path", lambda: env_file)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    monkeypatch.setattr(bootstrap_postgres, "ensure_database", lambda _admin_url, name: created_names.append(name) or False)
    monkeypatch.setattr(sys, "argv", ["bootstrap_postgres.py"])

    bootstrap_postgres.main()

    captured = capsys.readouterr()
    assert created_names == ["job_tracker", "job_tracker_test"]
    assert "job_tracker" in captured.out
    assert "job_tracker_test" in captured.out


def test_alembic_upgrades_fresh_job_tracker_test_database_successfully() -> None:
    engine = create_engine(TEST_DATABASE_URL, isolation_level="AUTOCOMMIT", pool_pre_ping=True)
    with engine.connect() as connection:
        connection.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
        connection.execute(text("GRANT ALL ON SCHEMA public TO CURRENT_USER"))
        connection.execute(text("GRANT ALL ON SCHEMA public TO public"))
    engine.dispose()

    command.upgrade(alembic_config(), "head")

    inspection_engine = create_engine(TEST_DATABASE_URL, pool_pre_ping=True)
    inspector = inspect(inspection_engine)
    table_names = set(inspector.get_table_names())
    inspection_engine.dispose()

    assert table_names == {
        "alembic_version",
        "asr_company_correction_events",
        "browser_context",
        "canonical_companies",
        "company_aliases",
        "job_applications",
    }
