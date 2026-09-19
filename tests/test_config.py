from __future__ import annotations

from pathlib import Path

from app.config import load_config
from app.realtime.client import RealtimeClient


def test_load_config_prefers_duplex_native_voice_speed_keys(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[auth]
api_key = "k"

[session]
model = "1.2.6.1"
voice = "en_female_stokie_uranus_bigtts"
speaker = "en_female_anna_mars_bigtts"
speed = 15
speech_rate = 99
loudness = -5
loudness_rate = 8
asr_format = "pcm"
tts_format = "pcm_s16le"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.speaker == "en_female_stokie_uranus_bigtts"
    assert cfg.tts_speed == 15
    assert cfg.tts_loudness == -5
    session, _ = RealtimeClient(cfg, "sid").build_session_config()
    output = session["audio"]["output"]
    assert output["voice"] == "en_female_stokie_uranus_bigtts"
    assert output["speed"] == 15
    assert output["loudness"] == -5


def test_load_config_accepts_legacy_speaker_aliases(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[auth]
api_key = "k"

[session]
speaker = "en_male_tim_uranus_bigtts"
speech_rate = 20
loudness_rate = 10
""".strip()
        + "\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.speaker == "en_male_tim_uranus_bigtts"
    assert cfg.tts_speed == 20
    assert cfg.tts_loudness == 10


def test_load_config_rvc_section(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[auth]
api_key = "k"

[voice.rvc]
enabled = true
mode = "utterance"
f0_up_key = -2
index_rate = 0.5
protect = 0.33
pth = "model_cache/KFK_V2_500.pth/KFK_V2_500.pth"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.rvc_enabled is True
    assert cfg.rvc_f0_up_key == -2
    assert cfg.rvc_index_rate == 0.5
    assert cfg.rvc_mode == "utterance"
    assert cfg.rvc_protect == 0.33
    assert cfg.rvc_pth.endswith("KFK_V2_500.pth")


def test_load_config_avatar_section(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[auth]
api_key = "k"

[avatar]
enabled = true
show_preview = true
character = "data/avatar/kafka_512.png"
weights_dir = "model_cache/tha3/separable_float"
display_scale = 1
fps = 12
""".strip()
        + "\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.avatar_enabled is True
    assert cfg.avatar_show_preview is True
    assert cfg.avatar_display_scale == 1
    assert cfg.avatar_fps == 12
    assert cfg.avatar_weights_dir.endswith("separable_float")


def test_load_config_items_dir_parallel_to_sessions(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[auth]
api_key = "k"

[logging]
session_dir = "data/sessions"
items_dir = "data/items"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.session_dir == "data/sessions"
    assert cfg.items_dir == "data/items"


def test_load_config_store_section(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[auth]
api_key = "k"

[store]
db_path = "data/coach.sqlite"
owner_id = "local"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.db_path == "data/coach.sqlite"
    assert cfg.owner_id == "local"


def test_load_config_max_speech_seconds(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[auth]
api_key = "k"

[voice.uplink_gate]
max_speech_seconds = 25
""".strip()
        + "\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.max_speech_seconds == 25
