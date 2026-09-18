from __future__ import annotations

import asyncio
import base64
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

import websockets

from app.config import AppConfig


TYPE_SESSION_CREATE = "session.create"
TYPE_SESSION_UPDATE = "session.update"
TYPE_SESSION_CLOSE = "session.close"
TYPE_INPUT_AUDIO_BUFFER_APPEND = "input_audio_buffer.append"
TYPE_INPUT_AUDIO_BUFFER_COMMIT = "input_audio_buffer.commit"
TYPE_SPEECH_TEXT_BUFFER_APPEND = "speech_text_buffer.append"
TYPE_SPEECH_TEXT_BUFFER_COMMIT = "speech_text_buffer.commit"
TYPE_SPEECH_TEXT_BUFFER_REPLACEMENT_APPEND = "speech_text_buffer.replacement.append"
TYPE_SPEECH_TEXT_BUFFER_REPLACEMENT_COMMIT = "speech_text_buffer.replacement.commit"
TYPE_CONVERSATION_ITEM_CREATE = "conversation.item.create"
TYPE_CONVERSATION_ITEM_UPDATE = "conversation.item.update"
TYPE_CONVERSATION_ITEM_RETRIEVE = "conversation.item.retrieve"
TYPE_CONVERSATION_ITEM_DELETE = "conversation.item.delete"

TYPE_SESSION_CREATED = "session.created"
TYPE_SESSION_UPDATED = "session.updated"
TYPE_SESSION_CLOSED = "session.closed"
TYPE_INPUT_AUDIO_BUFFER_COMMITTED = "input_audio_buffer.committed"
TYPE_TRANSCRIPTION_STARTED = "conversation.item.input_audio_transcription.started"
TYPE_TRANSCRIPTION_DELTA = "conversation.item.input_audio_transcription.delta"
TYPE_TRANSCRIPTION_COMPLETED = "conversation.item.input_audio_transcription.completed"
TYPE_TRANSCRIPTION_FAILED = "conversation.item.input_audio_transcription.failed"
TYPE_RESPONSE_OUTPUT_TEXT_DELTA = "response.output_text.delta"
TYPE_RESPONSE_OUTPUT_TEXT_DONE = "response.output_text.done"
TYPE_RESPONSE_OUTPUT_AUDIO_STARTED = "response.output_audio.started"
TYPE_RESPONSE_OUTPUT_AUDIO_DELTA = "response.output_audio.delta"
TYPE_RESPONSE_OUTPUT_AUDIO_DONE = "response.output_audio.done"
TYPE_CONVERSATION_ITEM_ADDED = "conversation.item.added"
TYPE_CONVERSATION_ITEM_RETRIEVED = "conversation.item.retrieved"
TYPE_CONVERSATION_ITEM_UPDATED = "conversation.item.updated"
TYPE_CONVERSATION_ITEM_DELETED = "conversation.item.deleted"
TYPE_RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE = "response.function_call_arguments.done"
TYPE_RESPONSE_CANCELED = "response.canceled"
TYPE_RESPONSE_DONE = "response.done"
TYPE_ERROR = "error"


class RealtimeClient:
    def __init__(self, cfg: AppConfig, session_id: Optional[str] = None):
        self.cfg = cfg
        self.ws: Any = None
        self.session_id = session_id or str(uuid.uuid4())
        self.dialog_id = ""
        self.instructions = ""
        self.tools: List[Dict[str, Any]] = []
        self._event_id = 0
        self._write_lock = asyncio.Lock()

    def apply_bundle(self, instructions: str, tools: Optional[List[Dict[str, Any]]] = None) -> None:
        self.instructions = instructions
        self.tools = tools or []

    def new_event_id(self) -> str:
        self._event_id += 1
        return f"event_{self._event_id}"

    async def connect(self) -> None:
        headers = {
            "X-Api-Key": self.cfg.api_key,
            "X-Api-Resource-Id": self.cfg.resource_id or "volc.speech.dialog",
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }
        if self.cfg.app_id:
            headers["X-Api-App-Id"] = self.cfg.app_id
        print(f"connect url={self.cfg.endpoint_url}")
        connect_kwargs: Dict[str, Any] = {"ping_interval": None}
        try:
            try:
                self.ws = await websockets.connect(
                    self.cfg.endpoint_url,
                    additional_headers=headers,
                    **connect_kwargs,
                )
            except TypeError:
                self.ws = await websockets.connect(
                    self.cfg.endpoint_url,
                    extra_headers=headers,
                    **connect_kwargs,
                )
        except Exception as exc:
            raise RuntimeError(self._handshake_error(exc)) from exc
        logid = None
        response = getattr(self.ws, "response", None)
        if response is not None:
            logid = response.headers.get("X-Tt-Logid")
        else:
            response_headers = getattr(self.ws, "response_headers", None)
            if response_headers is not None:
                logid = response_headers.get("X-Tt-Logid")
        if logid:
            print(f"dialog server response logid: {logid}")

    @staticmethod
    def _handshake_error(exc: Exception) -> str:
        body = ""
        response = getattr(exc, "response", None)
        if response is not None:
            raw = getattr(response, "body", b"")
            if isinstance(raw, (bytes, bytearray)):
                body = raw.decode("utf-8", errors="replace")
            elif raw:
                body = str(raw)
        detail = body or str(exc)
        if "45000030" in detail or "resource not granted" in detail:
            return (
                "实时语音握手失败：当前 API Key 未授权资源 volc.speech.dialog。"
                "请到火山引擎控制台为该 Key 开通「端到端实时语音」。"
                f" 服务端返回：{detail}"
            )
        return f"实时语音握手失败：{detail}"

    async def close(self) -> None:
        if self.ws is None:
            return
        try:
            await self.session_close()
            await self._wait_session_closed()
        except Exception as e:
            print(f"session.close error: {e}")
        await self.ws.close()

    async def _wait_session_closed(self, timeout: float = 3.0) -> None:
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                print("[session.closed] not received before timeout")
                return
            try:
                event = await asyncio.wait_for(self.recv_event(), timeout=remaining)
            except asyncio.TimeoutError:
                print("[session.closed] not received before timeout")
                return
            except Exception:
                return
            if event.get("type") == TYPE_SESSION_CLOSED:
                print("[session.closed]")
                return

    async def send_event(self, event: Dict[str, Any]) -> None:
        if self.ws is None:
            raise RuntimeError("websocket is not connected")
        payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        if event.get("type") != TYPE_INPUT_AUDIO_BUFFER_APPEND:
            print(f"[send event] {payload[:500]}")
        async with self._write_lock:
            await self.ws.send(payload)

    async def recv_event(self) -> Dict[str, Any]:
        if self.ws is None:
            raise RuntimeError("websocket is not connected")
        frame = await self.ws.recv()
        if isinstance(frame, bytes):
            frame = frame.decode("utf-8")
        event = json.loads(frame)
        event_type = event.get("type")
        if event_type != TYPE_RESPONSE_OUTPUT_AUDIO_DELTA:
            print(f"[recv event] type={event_type} frame={frame[:500]}")
        return event

    def build_session_config(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        session = {
            "id": self.session_id,
            "model": self.cfg.duplex_model,
            "instructions": self.instructions,
            "audio": {
                "input": {
                    "format": {"type": self.cfg.asr_format, "rate": 16000},
                },
                "output": {
                    "format": {"type": self.cfg.tts_format, "rate": 24000},
                    "voice": self.cfg.speaker,
                    "speed": _clamp_tts_rate(self.cfg.tts_speed),
                    "loudness": _clamp_tts_rate(self.cfg.tts_loudness),
                },
            },
            "tools": self.tools,
        }
        extension = {
            "asr": {"extra": {}},
            "tts": {"extra": {}},
            "dialog": {
                "extra": {
                    "audit_response": (
                        "Sorry, I cannot continue that topic. Let's go back to our scene."
                    ),
                    "enable_loudness_norm": True,
                    "enable_music": False,
                },
            },
        }
        return session, extension

    async def session_create(self) -> None:
        session, extension = self.build_session_config()
        await self.send_event(
            {
                "type": TYPE_SESSION_CREATE,
                "event_id": self.new_event_id(),
                "session": session,
                "extension": extension,
            }
        )
        while True:
            event = await self.recv_event()
            event_type = event.get("type")
            if event_type == TYPE_SESSION_CREATED:
                self.dialog_id = event.get("session", {}).get("id", "")
                print(f"Session created, dialog_id={self.dialog_id}")
                return
            if event_type == TYPE_ERROR:
                raise RuntimeError(f"session.create error: {event}")
            print(f"Ignore event before session.created: {event_type}")

    async def session_update(
        self, session: Dict[str, Any], extension: Optional[Dict[str, Any]] = None
    ) -> None:
        if "id" not in session:
            session["id"] = self.session_id
        await self.send_event(
            {
                "type": TYPE_SESSION_UPDATE,
                "event_id": self.new_event_id(),
                "session": session,
                "extension": extension,
            }
        )

    async def apply_and_update(self, instructions: str, tools: List[Dict[str, Any]]) -> None:
        self.apply_bundle(instructions, tools)
        session, extension = self.build_session_config()
        await self.session_update(session, extension)

    async def session_close(self) -> None:
        await self.send_event({"type": TYPE_SESSION_CLOSE, "event_id": self.new_event_id()})

    async def input_audio_append(self, data: bytes) -> None:
        await self.send_event(
            {
                "type": TYPE_INPUT_AUDIO_BUFFER_APPEND,
                "audio": base64.b64encode(data).decode("ascii"),
            }
        )

    async def input_audio_commit(self) -> None:
        await self.send_event(
            {"type": TYPE_INPUT_AUDIO_BUFFER_COMMIT, "event_id": self.new_event_id()}
        )

    async def speech_text_append(self, speech_id: str, text: str) -> None:
        await self.send_event(
            {
                "type": TYPE_SPEECH_TEXT_BUFFER_APPEND,
                "event_id": self.new_event_id(),
                "speech_id": speech_id,
                "text": text,
            }
        )

    async def speech_text_commit(self, speech_id: str, text: str) -> None:
        await self.send_event(
            {
                "type": TYPE_SPEECH_TEXT_BUFFER_COMMIT,
                "event_id": self.new_event_id(),
                "speech_id": speech_id,
                "text": text,
            }
        )

    async def speak_text(self, text: str) -> str:
        speech_id = str(uuid.uuid4())
        await self.speech_text_commit(speech_id, text)
        return speech_id

    async def conversation_item_create(self, items: List[Dict[str, Any]]) -> None:
        await self.send_event(
            {
                "type": TYPE_CONVERSATION_ITEM_CREATE,
                "event_id": self.new_event_id(),
                "items": items,
            }
        )

    async def inject_assistant_text(self, text: str) -> None:
        await self.conversation_item_create(
            [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "input_text", "text": text}],
                }
            ]
        )


def _clamp_tts_rate(value: int) -> int:
    return max(-50, min(100, int(value)))
