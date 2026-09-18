from __future__ import annotations

from app.config import AppConfig
from app.realtime.client import RealtimeClient
from app.voice.backend import VoiceBackend
from app.voice.duplex import DuplexVoiceBackend


def create_voice_backend(cfg: AppConfig, session_id: str) -> VoiceBackend:
    name = (cfg.voice_backend or "duplex").strip().lower()
    if name == "duplex":
        return DuplexVoiceBackend(RealtimeClient(cfg, session_id=session_id))
    if name == "pipeline":
        raise NotImplementedError(
            'voice.backend="pipeline" (separate ASR + LLM + TTS) is not implemented yet. '
            'Use backend = "duplex".'
        )
    raise ValueError(f"unknown voice.backend: {name!r}")
