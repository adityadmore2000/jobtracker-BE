import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.engine import URL, make_url


def get_backend_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_backend_env_path() -> Path:
    return get_backend_root() / ".env"


def _load_backend_environment_cached(env_path_str: str) -> str | None:
    env_path = Path(env_path_str)
    if not env_path.exists():
        return None
    load_dotenv(dotenv_path=env_path, override=False)
    return str(env_path)


def load_backend_environment() -> Path | None:
    loaded_path = _load_backend_environment_cached(str(get_backend_env_path()))
    return Path(loaded_path) if loaded_path else None


def reset_backend_environment_cache() -> None:
    _load_backend_environment_cached.cache_clear()


_load_backend_environment_cached = lru_cache(maxsize=None)(_load_backend_environment_cached)


def validate_postgresql_database_url(raw_url: str, *, env_var: str = "DATABASE_URL") -> str:
    try:
        parsed_url = make_url(raw_url)
    except Exception as exc:  # pragma: no cover - exact parser errors vary.
        raise RuntimeError(f"{env_var} is invalid. Provide a PostgreSQL SQLAlchemy URL.") from exc

    if parsed_url.get_backend_name() != "postgresql":
        raise RuntimeError(f"{env_var} must use PostgreSQL. SQLite is no longer supported.")

    if not parsed_url.database:
        raise RuntimeError(f"{env_var} must include a database name.")

    return raw_url


def get_optional_env_value(env_var: str) -> str:
    load_backend_environment()
    return os.getenv(env_var, "").strip()


def get_required_database_url(env_var: str = "DATABASE_URL") -> str:
    raw_url = get_optional_env_value(env_var)
    if not raw_url:
        raise RuntimeError(
            f"{env_var} is required. Set it in the OS environment or jobtracker-BE/.env. SQLite is no longer supported."
        )
    return validate_postgresql_database_url(raw_url, env_var=env_var)


def get_bool_env(env_var: str, *, default: bool = False) -> bool:
    raw_value = get_optional_env_value(env_var)
    if not raw_value:
        return default
    return raw_value.strip().casefold() in {"1", "true", "yes", "on"}


def get_database_name(database_url: str) -> str:
    return make_url(database_url).database or ""


def require_database_name(database_url: str, expected_name: str, *, env_var: str) -> None:
    actual_name = get_database_name(database_url)
    if actual_name != expected_name:
        raise RuntimeError(f"{env_var} must target the PostgreSQL database named '{expected_name}'.")


def derive_admin_database_url(database_url: str, *, admin_database: str = "postgres") -> str:
    parsed_url = make_url(database_url)
    admin_url: URL = parsed_url.set(database=admin_database)
    return admin_url.render_as_string(hide_password=False)
