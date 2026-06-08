from datetime import datetime, timezone
import json

import pytest
from httpx import ASGITransport, AsyncClient
from livekit.api import TokenVerifier

from app.main import DEFAULT_LIVEKIT_ROOM_NAME, app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client


@pytest.fixture
def livekit_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    values = {
        "LIVEKIT_URL": "ws://127.0.0.1:7880",
        "LIVEKIT_API_KEY": "devkey",
        "LIVEKIT_API_SECRET": "secret",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return values


def parse_utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None
    return parsed.astimezone(timezone.utc)


@pytest.mark.anyio
async def test_livekit_token_creation_with_explicit_room(client, livekit_env):
    response = await client.post("/livekit/token", json={"room_name": "custom-room"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["url"] == livekit_env["LIVEKIT_URL"]
    assert payload["room_name"] == "custom-room"
    assert payload["participant_identity"].startswith("browser-")
    assert payload["access_token"]

    expires_at = parse_utc_datetime(payload["expires_at"])
    assert expires_at > datetime.now(timezone.utc)

    claims = TokenVerifier(livekit_env["LIVEKIT_API_KEY"], livekit_env["LIVEKIT_API_SECRET"]).verify(payload["access_token"])
    assert claims.identity == payload["participant_identity"]
    assert claims.video is not None
    assert claims.video.room == "custom-room"
    assert claims.video.room_join is True
    assert claims.video.can_publish is True
    assert claims.video.can_publish_data is True
    assert claims.video.can_subscribe is True
    assert claims.video.can_publish_sources == ["microphone"]


@pytest.mark.anyio
async def test_livekit_token_defaults_room_name(client, livekit_env):
    response = await client.post("/livekit/token", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["room_name"] == DEFAULT_LIVEKIT_ROOM_NAME

    claims = TokenVerifier(livekit_env["LIVEKIT_API_KEY"], livekit_env["LIVEKIT_API_SECRET"]).verify(payload["access_token"])
    assert claims.video is not None
    assert claims.video.room == DEFAULT_LIVEKIT_ROOM_NAME


@pytest.mark.anyio
async def test_livekit_token_participant_identities_are_unique(client, livekit_env):
    first_response = await client.post("/livekit/token", json={})
    second_response = await client.post("/livekit/token", json={})

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    first_identity = first_response.json()["participant_identity"]
    second_identity = second_response.json()["participant_identity"]
    assert first_identity != second_identity
    assert first_identity.startswith("browser-")
    assert second_identity.startswith("browser-")


@pytest.mark.anyio
@pytest.mark.parametrize("missing_env_var", ["LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"])
async def test_livekit_token_missing_configuration_fails_safely(client, livekit_env, monkeypatch: pytest.MonkeyPatch, missing_env_var: str):
    monkeypatch.setenv(missing_env_var, "")

    response = await client.post("/livekit/token", json={})

    assert response.status_code == 503
    payload = response.json()
    assert missing_env_var in payload["detail"]
    assert "secret" not in payload["detail"].replace("LIVEKIT_API_SECRET", "")
    assert livekit_env["LIVEKIT_API_SECRET"] not in payload["detail"]


@pytest.mark.anyio
async def test_livekit_token_response_does_not_expose_secret_fields(client, livekit_env):
    response = await client.post("/livekit/token", json={"room_name": "custom-room"})

    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {"url", "room_name", "participant_identity", "access_token", "expires_at"}

    serialized_payload = json.dumps(payload).lower()
    assert "livekit_api_secret" not in serialized_payload
    assert "\"api_secret\"" not in serialized_payload
    assert "\"secret\"" not in serialized_payload
