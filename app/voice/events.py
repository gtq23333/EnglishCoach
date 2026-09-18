from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class VoiceEventType(str, Enum):
    SESSION_READY = "session_ready"
    SESSION_CLOSED = "session_closed"
    USER_SPEECH_STARTED = "user_speech_started"
    USER_TRANSCRIPT_DELTA = "user_transcript_delta"
    USER_TRANSCRIPT_DONE = "user_transcript_done"
    USER_TRANSCRIPT_FAILED = "user_transcript_failed"
    ASSISTANT_TEXT_DELTA = "assistant_text_delta"
    ASSISTANT_TEXT_DONE = "assistant_text_done"
    ASSISTANT_AUDIO_STARTED = "assistant_audio_started"
    ASSISTANT_AUDIO_DELTA = "assistant_audio_delta"
    ASSISTANT_AUDIO_DONE = "assistant_audio_done"
    FUNCTION_CALL = "function_call"
    ERROR = "error"


@dataclass
class FunctionCallRequest:
    call_id: str
    name: str
    arguments: str


@dataclass
class ToolResult:
    call_id: str
    output: str


@dataclass
class VoiceEvent:
    type: VoiceEventType
    text: str = ""
    audio: bytes = b""
    function_calls: List[FunctionCallRequest] = field(default_factory=list)
    error: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
