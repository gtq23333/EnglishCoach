from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import AppConfig, load_config
from app.serve import create_app
from app.service import CoachService


def test_settings_redacts_secrets_and_writes_toml(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[auth]
api_key = "secret-duplex"
resource_id = "volc.speech.dialog"

[llm]
api_key = "secret-llm"
model = "demo"

[session]
voice = "en_female_dacey_uranus_bigtts"
speed = 0
loudness = 0
""".strip()
        + "\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    cfg.db_path = str(tmp_path / "coach.sqlite")
    cfg.session_dir = str(tmp_path / "sessions")
    service = CoachService(cfg)
    app = create_app(service)
    with TestClient(app) as client:
        got = client.get("/v1/settings")
        assert got.status_code == 200
        settings = got.json()["settings"]
        assert "api_key" not in (settings.get("auth") or {})
        assert settings["auth"]["has_api_key"] is True
        assert "api_key" not in (settings.get("llm") or {})
        patched = client.patch(
            "/v1/settings",
            json={"session": {"voice": "en_male_tim_uranus_bigtts", "speed": 12}},
        )
        assert patched.status_code == 200
        assert patched.json()["settings"]["session"]["voice"] == "en_male_tim_uranus_bigtts"
        assert service.cfg.speaker == "en_male_tim_uranus_bigtts"
        assert service.cfg.tts_speed == 12
        disk = path.read_text(encoding="utf-8")
        assert "en_male_tim_uranus_bigtts" in disk
        assert "secret-duplex" in disk
