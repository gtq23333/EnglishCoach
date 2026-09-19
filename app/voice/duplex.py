from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

from app.realtime.audio_io import extract_audio_b64
from app.realtime.client import (
    TYPE_CONVERSATION_ITEM_ADDED,
    TYPE_CONVERSATION_ITEM_DELETED,
    TYPE_CONVERSATION_ITEM_RETRIEVED,
    TYPE_CONVERSATION_ITEM_UPDATED,
    TYPE_ERROR,
    TYPE_INPUT_AUDIO_BUFFER_COMMITTED,
    TYPE_RESPONSE_CANCELED,
    TYPE_RESPONSE_DONE,
    TYPE_RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE,
    TYPE_RESPONSE_OUTPUT_AUDIO_DELTA,
    TYPE_RESPONSE_OUTPUT_AUDIO_DONE,
    TYPE_RESPONSE_OUTPUT_AUDIO_STARTED,
    TYPE_RESPONSE_OUTPUT_TEXT_DELTA,
    TYPE_RESPONSE_OUTPUT_TEXT_DONE,
    TYPE_SESSION_CLOSED,
    TYPE_SESSION_CREATED,
    TYPE_SESSION_UPDATED,
    TYPE_TRANSCRIPTION_COMPLETED,
    TYPE_TRANSCRIPTION_DELTA,
    TYPE_TRANSCRIPTION_FAILED,
    TYPE_TRANSCRIPTION_STARTED,
    RealtimeClient,
)
from app.voice.backend import VoiceBackend
from app.voice.events import FunctionCallRequest, ToolResult, VoiceEvent, VoiceEventType


def normalize_duplex_event(event: Dict[str, Any]) -> Optional[VoiceEvent]:
    """Map a Volcengine duplex JSON event to a VoiceEvent, or None to skip."""
    event_type = event.get("type")

    if event_type in (TYPE_SESSION_CREATED, TYPE_SESSION_UPDATED):
        session = event.get("session") or {}
        return VoiceEvent(
            type=VoiceEventType.SESSION_READY,
            extra={"dialog_id": session.get("id") or "", "raw_type": event_type},
        )

    if event_type == TYPE_SESSION_CLOSED:
        return VoiceEvent(type=VoiceEventType.SESSION_CLOSED)

    if event_type == TYPE_TRANSCRIPTION_STARTED:
        return VoiceEvent(type=VoiceEventType.USER_SPEECH_STARTED)

    if event_type == TYPE_TRANSCRIPTION_DELTA:
        return VoiceEvent(
            type=VoiceEventType.USER_TRANSCRIPT_DELTA,
            text=str(event.get("delta") or ""),
        )

    if event_type == TYPE_TRANSCRIPTION_COMPLETED:
        text = event.get("transcript") or event.get("text") or ""
        return VoiceEvent(type=VoiceEventType.USER_TRANSCRIPT_DONE, text=str(text))

    if event_type == TYPE_TRANSCRIPTION_FAILED:
        return VoiceEvent(
            type=VoiceEventType.USER_TRANSCRIPT_FAILED,
            error=str(event.get("error") or event),
        )

    if event_type == TYPE_RESPONSE_OUTPUT_TEXT_DELTA:
        return VoiceEvent(
            type=VoiceEventType.ASSISTANT_TEXT_DELTA,
            text=str(event.get("delta") or ""),
        )

    if event_type == TYPE_RESPONSE_OUTPUT_TEXT_DONE:
        return VoiceEvent(
            type=VoiceEventType.ASSISTANT_TEXT_DONE,
            text=str(event.get("text") or ""),
        )

    if event_type == TYPE_RESPONSE_OUTPUT_AUDIO_STARTED:
        return VoiceEvent(
            type=VoiceEventType.ASSISTANT_AUDIO_STARTED,
            extra={"tts_type": event.get("tts_type")},
        )

    if event_type == TYPE_RESPONSE_OUTPUT_AUDIO_DELTA:
        payload = extract_audio_b64(event)
        if not payload:
            return None
        try:
            audio = base64.b64decode(payload)
        except Exception:
            return VoiceEvent(
                type=VoiceEventType.ERROR,
                error="assistant audio base64 decode failed",
            )
        if not audio:
            return None
        return VoiceEvent(type=VoiceEventType.ASSISTANT_AUDIO_DELTA, audio=audio)

    if event_type == TYPE_RESPONSE_OUTPUT_AUDIO_DONE:
        return VoiceEvent(
            type=VoiceEventType.ASSISTANT_AUDIO_DONE,
            extra={"status_code": event.get("status_code")},
        )

    if event_type == TYPE_RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE:
        calls = [
            FunctionCallRequest(
                call_id=str(item.get("call_id", "")),
                name=str(item.get("name", "")),
                arguments=str(item.get("arguments", "")),
            )
            for item in (event.get("items") or [])
        ]
        if not calls:
            return None
        return VoiceEvent(type=VoiceEventType.FUNCTION_CALL, function_calls=calls)

    if event_type == TYPE_ERROR:
        return VoiceEvent(type=VoiceEventType.ERROR, error=str(event.get("error") or event))

    if event_type in (
        TYPE_INPUT_AUDIO_BUFFER_COMMITTED,
        TYPE_CONVERSATION_ITEM_ADDED,
        TYPE_CONVERSATION_ITEM_RETRIEVED,
        TYPE_CONVERSATION_ITEM_UPDATED,
        TYPE_CONVERSATION_ITEM_DELETED,
        TYPE_RESPONSE_CANCELED,
        TYPE_RESPONSE_DONE,
    ):
        return None

    print(f"[duplex] unhandled event type={event_type}")
    return None


class DuplexVoiceBackend(VoiceBackend):
    """Volcengine end-to-end realtime dialogue, behind VoiceBackend."""

    def __init__(self, client: RealtimeClient):
        self._client = client

    async def start(self, instructions: str, tools: List[Dict[str, Any]]) -> None:
        self._client.apply_bundle(instructions, tools)
        await self._client.connect()
        await self._client.session_create()

    async def update(self, instructions: str, tools: List[Dict[str, Any]]) -> None:
        await self._client.apply_and_update(instructions, tools)

    async def send_audio(self, pcm: bytes) -> None:
        await self._client.input_audio_append(pcm)

    async def commit_audio(self) -> None:
        await self._client.input_audio_commit()

    async def speak_text(self, text: str) -> None:
        await self._client.speak_text(text)

    async def inject_assistant_text(self, text: str) -> None:
        await self._client.inject_assistant_text(text)

    async def submit_tool_results(self, results: List[ToolResult]) -> None:
        items = [
            {
                "type": "message",
                "role": "tool",
                "call_id": result.call_id,
                "content": [{"type": "input_text", "text": result.output}],
            }
            for result in results
        ]
        await self._client.conversation_item_create(items)

    async def recv_event(self) -> VoiceEvent:
        while True:
            raw = await self._client.recv_event()
            mapped = normalize_duplex_event(raw)
            if mapped is not None:
                return mapped

    async def reconnect(self, instructions: str, tools: List[Dict[str, Any]]) -> None:
        await self._client.hard_reset()
        await self.start(instructions, tools)

    async def close(self) -> None:
        await self._client.close()
