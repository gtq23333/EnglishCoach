from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from app.config import AppConfig, OGG_OPUS
from app.dialogue.mic_session import MicSession
from app.dialogue.orchestrator import DialogueOrchestrator
from app.dialogue.session_log import SessionLog, load_records
from app.dialogue.text_session import TextSession
from app.events import CoachEvent, EventBus
from app.items.generator import generate_for_session, load_items, session_jsonl_path
from app.llm import LLMClient, OpenAICompatClient
from app.prompts.base import SceneSpec
from app.prompts.registry import get_assembler
from app.voice import create_voice_backend
from app.voice.backend import VoiceBackend


class SessionBusyError(RuntimeError):
    pass


class SessionNotFoundError(KeyError):
    pass


class SessionModeError(RuntimeError):
    pass


VoiceFactory = Callable[[AppConfig, str], VoiceBackend]


@dataclass
class ActiveSession:
    session_id: str
    mode: str
    jsonl_path: Path
    orchestrator: DialogueOrchestrator
    task: asyncio.Task
    text_session: Optional[TextSession] = None
    mic_session: Optional[MicSession] = None


class CoachService:
    def __init__(
        self,
        cfg: AppConfig,
        *,
        llm: Optional[LLMClient] = None,
        voice_backend_factory: Optional[VoiceFactory] = None,
    ):
        self.cfg = cfg
        self.bus = EventBus()
        self._llm = llm
        self._voice_factory = voice_backend_factory or create_voice_backend
        self._active: Optional[ActiveSession] = None
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._avatar = None

    def emit(self, event: CoachEvent) -> None:
        self.bus.emit(event)

    def _emit(self, session_id: str, event_type: str, payload: Optional[dict] = None) -> None:
        self.bus.emit(
            CoachEvent(type=event_type, session_id=session_id, payload=payload or {})
        )

    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = OpenAICompatClient(
                self.cfg.llm_base_url, self.cfg.llm_api_key, self.cfg.llm_model
            )
        return self._llm

    def is_busy(self) -> bool:
        return self._active is not None and not self._active.task.done()

    def _require_active(self, session_id: str) -> ActiveSession:
        if self._active is None or self._active.session_id != session_id:
            raise SessionNotFoundError(session_id)
        return self._active

    async def create_session(
        self, mode: str, scene: Optional[str] = None
    ) -> Dict[str, Any]:
        if mode not in {"mic", "text"}:
            raise SessionModeError(f"unsupported mode: {mode}")
        if self.is_busy():
            raise SessionBusyError("an active practice session is already running")
        session_id = str(uuid.uuid4())
        assembler = get_assembler(self.cfg.assembler)
        initial_scene = SceneSpec.from_cli(scene) if scene else None
        session_log = SessionLog(self.cfg.session_dir, session_id, mode=mode)
        orchestrator = DialogueOrchestrator(
            assembler,
            session_log,
            initial_scene=initial_scene,
            emit=self.emit,
            session_id=session_id,
        )
        jsonl_path = session_log.path
        if mode == "text":
            text_session = TextSession(orchestrator, self.llm(), emit=self.emit)
            task = asyncio.create_task(text_session.start(), name=f"text-{session_id}")
            self._active = ActiveSession(
                session_id=session_id,
                mode=mode,
                jsonl_path=jsonl_path,
                orchestrator=orchestrator,
                task=task,
                text_session=text_session,
            )
        else:
            if not self.cfg.api_key:
                raise RuntimeError("auth.api_key is required for mic mode")
            if self.cfg.tts_format == OGG_OPUS:
                raise RuntimeError(
                    'Python client cannot play ogg_opus. Use tts_format = "pcm_s16le".'
                )
            backend = self._voice_factory(self.cfg, session_id)
            mic_session = MicSession(
                backend, orchestrator, self.cfg, emit=self.emit
            )
            task = asyncio.create_task(mic_session.start(), name=f"mic-{session_id}")
            self._active = ActiveSession(
                session_id=session_id,
                mode=mode,
                jsonl_path=jsonl_path,
                orchestrator=orchestrator,
                task=task,
                mic_session=mic_session,
            )
        self._emit(
            session_id,
            "session.started",
            {
                "mode": mode,
                "jsonl_path": str(jsonl_path),
                "phase": orchestrator.phase,
            },
        )
        await self._start_avatar(session_id)
        return {
            "session_id": session_id,
            "jsonl_path": str(jsonl_path),
            "mode": mode,
            "phase": orchestrator.phase,
        }

    async def stop_session(self, session_id: str) -> Dict[str, Any]:
        if self._active is None or self._active.session_id != session_id:
            return {"ok": True, "already_stopped": True}
        active = self._active
        active.orchestrator.request_exit()
        if active.text_session is not None:
            await active.text_session.stop()
        if active.mic_session is not None:
            active.mic_session.running = False
        try:
            await asyncio.wait_for(active.task, timeout=1.5)
        except asyncio.TimeoutError:
            active.task.cancel()
            try:
                await active.task
            except asyncio.CancelledError:
                pass
        except asyncio.CancelledError:
            pass
        await self._stop_avatar()
        return {"ok": True}

    async def _start_avatar(self, session_id: str) -> None:
        if not self.cfg.avatar_enabled or not self.cfg.avatar_show_preview:
            return
        from app.avatar.session import AvatarSession

        runtime = AvatarSession(self.cfg, self.bus)
        runtime.start(session_id, asyncio.get_running_loop())
        self._avatar = runtime
        self._emit(
            session_id,
            "avatar.started",
            {
                "display_scale": self.cfg.avatar_display_scale,
                "show_preview": True,
            },
        )

    async def _stop_avatar(self) -> None:
        runtime = self._avatar
        self._avatar = None
        if runtime is None:
            return
        await runtime.stop()

    async def wait_until_stopped(self, session_id: Optional[str] = None) -> None:
        if self._active is None:
            return
        if session_id and self._active.session_id != session_id:
            return
        try:
            await self._active.task
        except asyncio.CancelledError:
            pass

    async def send_text(self, session_id: str, text: str) -> Dict[str, Any]:
        active = self._require_active(session_id)
        if active.mode != "text" or active.text_session is None:
            raise SessionModeError("send_text is only valid in text mode")
        if active.task.done():
            raise SessionModeError("session has ended")
        await active.text_session.submit(text)
        await active.text_session.wait_idle()
        return {"ok": True}

    def get_session(self, session_id: str) -> Dict[str, Any]:
        if self._active is not None and self._active.session_id == session_id:
            active = self._active
            muted = False
            reason = None
            if active.mic_session is not None:
                muted = not active.mic_session.gate.uplink_enabled
                reason = active.mic_session.gate.mute_reason
            status = "running" if not active.task.done() else "ended"
            return {
                "session_id": session_id,
                "status": status,
                "mode": active.mode,
                "phase": active.orchestrator.phase,
                "scene": active.orchestrator.scene_dict(),
                "uplink_muted": muted,
                "uplink_mute_reason": reason,
                "jsonl_path": str(active.jsonl_path),
                "transcript": [
                    {
                        "role": entry.role,
                        "text": entry.text,
                        "source": entry.source,
                        "phase": entry.phase,
                        "timestamp": entry.timestamp,
                    }
                    for entry in active.orchestrator.transcript.entries
                ],
            }
        path = session_jsonl_path(self.cfg.session_dir, session_id)
        if not path.exists():
            raise SessionNotFoundError(session_id)
        records = load_records(path)
        scene = None
        transcript = []
        mode = ""
        for record in records:
            if record.get("type") == "session_start":
                mode = str(record.get("mode") or "")
            elif record.get("type") == "scene":
                scene = record.get("scene")
            elif record.get("type") == "utterance":
                transcript.append(
                    {
                        "role": record.get("role"),
                        "text": record.get("text"),
                        "source": record.get("source"),
                        "phase": record.get("phase"),
                        "timestamp": record.get("timestamp"),
                    }
                )
        return {
            "session_id": session_id,
            "status": "ended",
            "mode": mode,
            "phase": "practice" if scene else "bootstrap",
            "scene": scene,
            "uplink_muted": False,
            "uplink_mute_reason": None,
            "jsonl_path": str(path),
            "transcript": transcript,
        }

    def list_sessions(self) -> List[Dict[str, Any]]:
        directory = Path(self.cfg.session_dir)
        if not directory.exists():
            return []
        items = []
        for path in sorted(directory.glob("*.jsonl")):
            if path.name.endswith(".errors.jsonl") or path.name.endswith(".items.jsonl"):
                continue
            session_id = path.stem
            created = ""
            mode = ""
            try:
                records = load_records(path)
            except Exception:
                continue
            start = next((row for row in records if row.get("type") == "session_start"), None)
            if start:
                created = str(start.get("created_at") or "")
                mode = str(start.get("mode") or "")
            items.append(
                {
                    "session_id": session_id,
                    "created_at": created,
                    "mode": mode,
                    "jsonl_path": str(path),
                }
            )
        return items

    async def generate_items(
        self,
        session_id: str,
        max_user_turns_per_slice: int = 2,
        char_budget: int = 800,
        parse_retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "job_id": job_id,
            "session_id": session_id,
            "status": "running",
        }

        def progress(event_type: str, payload: dict) -> None:
            self._emit(session_id, event_type, {"job_id": job_id, **payload})

        async def _run() -> None:
            try:
                result = await generate_for_session(
                    session_id=session_id,
                    session_dir=self.cfg.session_dir,
                    items_dir=self.cfg.items_dir,
                    llm=self.llm(),
                    max_user_turns_per_slice=max_user_turns_per_slice,
                    char_budget=char_budget,
                    parse_retries=(
                        self.cfg.itemgen_parse_retries
                        if parse_retries is None
                        else parse_retries
                    ),
                    progress=progress,
                )
                self._jobs[job_id]["status"] = "done"
                self._jobs[job_id]["cases"] = len(result.get("cases") or [])
            except Exception as exc:
                self._jobs[job_id]["status"] = "error"
                self._jobs[job_id]["error"] = str(exc)
                self._emit(
                    session_id,
                    "itemgen.error",
                    {"job_id": job_id, "message": str(exc)},
                )

        asyncio.create_task(_run(), name=f"itemgen-{job_id}")
        return {"job_id": job_id, "status": "running", "session_id": session_id}

    def get_items(self, session_id: str) -> Dict[str, Any]:
        data = load_items(self.cfg.items_dir, session_id)
        if data is None:
            raise SessionNotFoundError(session_id)
        return data

    def get_job(self, job_id: str) -> Dict[str, Any]:
        job = self._jobs.get(job_id)
        if not job:
            raise SessionNotFoundError(job_id)
        return job
