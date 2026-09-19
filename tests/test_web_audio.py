from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from app.config import AppConfig
from app.dialogue.mic_session import MicSession
from app.prompts.base import PromptBundle
from app.realtime.transport import WebQueueTransport
from app.serve import create_app
from app.service import CoachService
from app.voice.events import VoiceEvent, VoiceEventType


class IdleVoice:
    def __init__(self) -> None:
        self.sent: list[bytes] = []
        self._q: asyncio.Queue = asyncio.Queue()
        self.reconnects = 0

    async def start(self, instructions, tools=None) -> None:
        return None

    async def update(self, instructions, tools=None) -> None:
        return None

    async def send_audio(self, pcm: bytes) -> None:
        self.sent.append(pcm)

    async def speak_text(self, text: str) -> None:
        return None

    async def inject_assistant_text(self, text: str) -> None:
        return None

    async def submit_tool_results(self, results) -> None:
        return None

    async def recv_event(self) -> VoiceEvent:
        return await self._q.get()

    async def close(self) -> None:
        await self._q.put(VoiceEvent(type=VoiceEventType.SESSION_CLOSED))

    async def reconnect(self, instructions, tools=None) -> None:
        self.reconnects += 1


def _service(tmp_path: Path) -> tuple[CoachService, IdleVoice]:
    voice = IdleVoice()
    cfg = AppConfig(
        session_dir=str(tmp_path / "sessions"),
        items_dir=str(tmp_path / "items"),
        db_path=str(tmp_path / "coach.sqlite"),
        api_key="k",
        avatar_enabled=False,
        rvc_enabled=False,
    )
    service = CoachService(cfg, voice_backend_factory=lambda _cfg, _sid: voice)
    return service, voice


def test_web_mode_audio_ws_roundtrip(tmp_path: Path):
    service, voice = _service(tmp_path)
    app = create_app(service)
    with TestClient(app) as client:
        created = client.post("/v1/sessions", json={"mode": "web"})
        assert created.status_code == 200, created.text
        session_id = created.json()["session_id"]
        transport = service.web_audio_transport(session_id)
        assert isinstance(transport, WebQueueTransport)

        with client.websocket_connect(f"/v1/sessions/{session_id}/audio") as ws:
            hello = ws.receive_json()
            assert hello["uplink_sr"] == 16000
            assert hello["downlink_sr"] == 24000
            ws.send_bytes(b"\x01\x00" * 320)
            for _ in range(40):
                if voice.sent:
                    break
                import time

                time.sleep(0.05)
            assert voice.sent, "uplink pcm should reach the voice backend"
            transport.write_output(b"\x02\x00" * 10)
            down = ws.receive_bytes()
            assert down.startswith(b"\x02\x00")

        client.post(f"/v1/sessions/{session_id}/stop")


def test_second_web_session_replaces_the_first(tmp_path: Path):
    service, _voice = _service(tmp_path)
    app = create_app(service)
    with TestClient(app) as client:
        first = client.post("/v1/sessions", json={"mode": "web"})
        assert first.status_code == 200, first.text
        first_id = first.json()["session_id"]
        second = client.post("/v1/sessions", json={"mode": "web"})
        assert second.status_code == 200, second.text
        second_id = second.json()["session_id"]
        assert second_id != first_id
        assert service.get_session(second_id)["status"] == "running"
        client.post(f"/v1/sessions/{second_id}/stop")


class _FakeOrch:
    session_id = "opening-rvc"
    should_exit = False
    bundle = PromptBundle(
        instructions="bootstrap",
        tools=[],
        injected_user_prompt="Hello! Please describe the scene.",
    )

    @property
    def tools(self):
        return self.bundle.tools

    def on_injected_prompt(self, text: str) -> None:
        return None

    def on_opening(self, text: str) -> None:
        return None

    def on_user_text(self, text: str) -> None:
        return None

    def close(self) -> None:
        return None


class _FakeRvc:
    def attach_thread(self) -> None:
        return None

    def begin(self) -> None:
        return None

    def push_pcm24(self, pcm: bytes) -> list[bytes]:
        return [pcm] if pcm else []

    def flush(self) -> list[bytes]:
        return []


async def test_opening_does_not_wait_for_slow_rvc(tmp_path, monkeypatch):
    order: list[str] = []

    async def fake_load(self: MicSession) -> None:
        await asyncio.sleep(0.3)
        self._rvc = _FakeRvc()
        order.append("rvc")

    async def fake_inject(self: MicSession) -> None:
        order.append("opening")

    monkeypatch.setattr(MicSession, "_load_rvc", fake_load)
    monkeypatch.setattr(MicSession, "_inject_opening", fake_inject)

    cfg = AppConfig(
        session_dir=str(tmp_path / "sessions"),
        items_dir=str(tmp_path / "items"),
        db_path=str(tmp_path / "coach.sqlite"),
        api_key="k",
        avatar_enabled=False,
        rvc_enabled=True,
    )
    voice = IdleVoice()
    session = MicSession(voice, _FakeOrch(), cfg, transport=WebQueueTransport())
    task = asyncio.create_task(session.start())
    try:
        for _ in range(50):
            if "opening" in order or task.done():
                break
            await asyncio.sleep(0.02)
        if task.done() and task.exception() is not None:
            raise task.exception()
        assert order[0] == "opening"
        assert "rvc" not in order
    finally:
        session.running = False
        await voice._q.put(VoiceEvent(type=VoiceEventType.SESSION_CLOSED))
        await asyncio.wait_for(task, timeout=3)


async def test_cancel_while_rvc_loading_does_not_hang(tmp_path, monkeypatch):
    async def slow_load(self: MicSession) -> None:
        await asyncio.sleep(30)
        self._rvc = _FakeRvc()

    monkeypatch.setattr(MicSession, "_load_rvc", slow_load)
    cfg = AppConfig(
        session_dir=str(tmp_path / "sessions"),
        items_dir=str(tmp_path / "items"),
        db_path=str(tmp_path / "coach.sqlite"),
        api_key="k",
        avatar_enabled=False,
        rvc_enabled=True,
    )
    voice = IdleVoice()
    session = MicSession(voice, _FakeOrch(), cfg, transport=WebQueueTransport())
    task = asyncio.create_task(session.start())
    for _ in range(20):
        if task.done():
            break
        await asyncio.sleep(0.02)
    session.running = False
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)


async def test_sami_stream_reset_reconnects_instead_of_ending(tmp_path):
    events: list[str] = []
    voice = IdleVoice()
    cfg = AppConfig(
        session_dir=str(tmp_path / "sessions"),
        items_dir=str(tmp_path / "items"),
        db_path=str(tmp_path / "coach.sqlite"),
        api_key="k",
        avatar_enabled=False,
        rvc_enabled=False,
    )
    session = MicSession(
        voice,
        _FakeOrch(),
        cfg,
        emit=lambda event: events.append(event.type),
        transport=WebQueueTransport(),
    )
    task = asyncio.create_task(session.start())
    try:
        for _ in range(50):
            if "voice.ready" in events or task.done():
                break
            await asyncio.sleep(0.02)
        await voice._q.put(
            VoiceEvent(
                type=VoiceEventType.USER_TRANSCRIPT_DELTA,
                text="a long explanation about react",
            )
        )
        await voice._q.put(
            VoiceEvent(
                type=VoiceEventType.ERROR,
                error=(
                    "{'type': 'Internal Server Error', 'code': '55000000', "
                    "'message': 'sami error: codes=50700000, RST_STREAM'}"
                ),
            )
        )
        for _ in range(50):
            if voice.reconnects >= 1:
                break
            await asyncio.sleep(0.02)
        assert voice.reconnects == 1
        assert "voice.connecting" in events
        assert "asr.completed" in events
        assert "session.ended" not in events
    finally:
        session.running = False
        await voice._q.put(VoiceEvent(type=VoiceEventType.SESSION_CLOSED))
        await asyncio.wait_for(task, timeout=3)


def test_flush_transcript_persists_open_turns(tmp_path: Path):
    from app.dialogue.orchestrator import DialogueOrchestrator
    from app.dialogue.session_log import SessionLog, load_records
    from app.prompts.scene_practice import ScenePracticeAssembler

    log = SessionLog(tmp_path, "flush-1", mode="web")
    orch = DialogueOrchestrator(ScenePracticeAssembler(), log, session_id="flush-1")
    cfg = AppConfig(
        session_dir=str(tmp_path / "sessions"),
        items_dir=str(tmp_path / "items"),
        db_path=str(tmp_path / "coach.sqlite"),
        api_key="k",
        avatar_enabled=False,
        rvc_enabled=False,
    )
    session = MicSession(IdleVoice(), orch, cfg, transport=WebQueueTransport())
    session._last_asr_text = "I want a coffee please"
    session._assistant_text = "What size would you like?"
    session.flush_transcript()
    records = load_records(log.path)
    texts = [row.get("text") for row in records if row.get("type") == "utterance"]
    assert "I want a coffee please" in texts
    assert "What size would you like?" in texts
    before = len(records)
    session.flush_transcript()
    assert len(load_records(log.path)) == before


def test_jpeg_sink_encodes_rgb():
    import numpy as np

    from app.avatar.jpeg_sink import JpegQueueSink

    sink = JpegQueueSink()
    sink.push_frame(np.zeros((16, 16, 3), dtype=np.uint8))
    frame = sink.pull(timeout=1)
    assert frame and frame[:2] == b"\xff\xd8"
