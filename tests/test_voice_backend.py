from __future__ import annotations

import base64

import pytest

from app.config import AppConfig
from app.voice.duplex import DuplexVoiceBackend, normalize_duplex_event
from app.voice.events import VoiceEventType
from app.voice.factory import create_voice_backend


def test_normalize_transcript_and_text_done():
    done = normalize_duplex_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "I want a coffee",
        }
    )
    assert done is not None
    assert done.type == VoiceEventType.USER_TRANSCRIPT_DONE
    assert done.text == "I want a coffee"

    chat = normalize_duplex_event(
        {"type": "response.output_text.done", "text": "What size?"}
    )
    assert chat is not None
    assert chat.type == VoiceEventType.ASSISTANT_TEXT_DONE
    assert chat.text == "What size?"


def test_normalize_audio_delta_decodes_bytes():
    payload = base64.b64encode(b"\x01\x02").decode("ascii")
    event = normalize_duplex_event(
        {"type": "response.output_audio.delta", "delta": payload}
    )
    assert event is not None
    assert event.type == VoiceEventType.ASSISTANT_AUDIO_DELTA
    assert event.audio == b"\x01\x02"


def test_normalize_function_call_and_skips_noise():
    event = normalize_duplex_event(
        {
            "type": "response.function_call_arguments.done",
            "items": [
                {
                    "call_id": "c1",
                    "name": "set_scene",
                    "arguments": '{"scene":"cafe"}',
                }
            ],
        }
    )
    assert event is not None
    assert event.type == VoiceEventType.FUNCTION_CALL
    assert event.function_calls[0].name == "set_scene"
    assert normalize_duplex_event({"type": "response.done"}) is None
    assert normalize_duplex_event({"type": "response.output_audio.delta"}) is None


def test_factory_builds_duplex_backend():
    backend = create_voice_backend(AppConfig(), "sid")
    assert isinstance(backend, DuplexVoiceBackend)


def test_factory_rejects_unknown_and_unimplemented_pipeline():
    with pytest.raises(ValueError, match="unknown voice.backend"):
        create_voice_backend(AppConfig(voice_backend="nope"), "sid")
    with pytest.raises(NotImplementedError, match="pipeline"):
        create_voice_backend(AppConfig(voice_backend="pipeline"), "sid")


def test_session_config_includes_voice_speed_loudness():
    from app.realtime.client import RealtimeClient

    cfg = AppConfig(speaker="en_female_dacey_uranus_bigtts", tts_speed=20, tts_loudness=-10)
    client = RealtimeClient(cfg, "sid")
    session, _extension = client.build_session_config()
    output = session["audio"]["output"]
    assert output["voice"] == "en_female_dacey_uranus_bigtts"
    assert output["speed"] == 20
    assert output["loudness"] == -10

    hot = RealtimeClient(AppConfig(tts_speed=999, tts_loudness=-80), "sid")
    output = hot.build_session_config()[0]["audio"]["output"]
    assert output["speed"] == 100
    assert output["loudness"] == -50
