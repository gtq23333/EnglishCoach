"""Offline / import-stable RVC helpers.

Realtime playback uses ``app.voice.rvc_engine.PlaybackRvc``. This module keeps
the older names so tests and one-off scripts still import ``RvcConverter``.
"""

from app.voice.rvc_engine import (  # noqa: F401
    DEFAULT_INDEX,
    DEFAULT_PTH,
    DEFAULT_RVC_ROOT,
    PlaybackRvc as RvcConverter,
    RvcNotReadyError,
    RvcPaths,
)
