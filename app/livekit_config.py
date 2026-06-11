from dataclasses import dataclass

from .database_config import get_optional_env_value


class LiveKitConfigurationError(RuntimeError):
    """Raised when required LiveKit settings are missing."""


@dataclass(frozen=True)
class LiveKitSettings:
    url: str
    api_key: str
    api_secret: str


def get_livekit_settings() -> LiveKitSettings:
    url = get_optional_env_value("LIVEKIT_URL")
    api_key = get_optional_env_value("LIVEKIT_API_KEY")
    api_secret = get_optional_env_value("LIVEKIT_API_SECRET")

    missing = [name for name, value in [("LIVEKIT_URL", url), ("LIVEKIT_API_KEY", api_key), ("LIVEKIT_API_SECRET", api_secret)] if not value]
    if missing:
        joined = ", ".join(missing)
        raise LiveKitConfigurationError(
            f"LiveKit server configuration is incomplete. Set {joined} in the OS environment or jobtracker-BE/.env."
        )

    return LiveKitSettings(url=url, api_key=api_key, api_secret=api_secret)
