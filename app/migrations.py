from __future__ import annotations

from threading import Lock

from alembic import command
from alembic.config import Config

from .database_config import get_backend_root, get_bool_env


_AUTO_MIGRATE_LOCK = Lock()
_AUTO_MIGRATE_COMPLETED = False


def build_alembic_config() -> Config:
    backend_root = get_backend_root()
    config = Config(str(backend_root / "alembic.ini"))
    config.set_main_option("script_location", str(backend_root / "alembic"))
    return config


def run_alembic_upgrade_head() -> None:
    command.upgrade(build_alembic_config(), "head")


def run_startup_migrations_if_enabled() -> bool:
    global _AUTO_MIGRATE_COMPLETED
    if not get_bool_env("AUTO_MIGRATE", default=False):
        return False

    with _AUTO_MIGRATE_LOCK:
        if _AUTO_MIGRATE_COMPLETED:
            return False
        run_alembic_upgrade_head()
        _AUTO_MIGRATE_COMPLETED = True
        return True


def reset_auto_migrate_state() -> None:
    global _AUTO_MIGRATE_COMPLETED
    _AUTO_MIGRATE_COMPLETED = False
