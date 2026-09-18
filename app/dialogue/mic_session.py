from __future__ import annotations

import asyncio
import queue as thread_queue
import threading
from typing import Callable, Optional

from app.config import AppConfig
from app.dialogue.orchestrator import DialogueOrchestrator
from app.dialogue.uplink_gate import UplinkGate
from app.events import CoachEvent
from app.realtime.audio_io import (
    MIC_CHUNK,
    AudioDevices,
    clear_queue,
)
from app.voice.backend import VoiceBackend
from app.voice.events import ToolResult, VoiceEvent, VoiceEventType
from app.voice.rvc_engine import PlaybackRvc, load_playback_rvc

SILENCE_CHUNK = b"\x00" * (MIC_CHUNK * 2)
EmitFn = Callable[[CoachEvent], None]


class MicSession:
    def __init__(
        self,
        backend: VoiceBackend,
        orchestrator: DialogueOrchestrator,
        cfg: AppConfig,
        emit: Optional[EmitFn] = None,
        gate: Optional[UplinkGate] = None,
    ):
        self.backend = backend
        self.orchestrator = orchestrator
        self.cfg = cfg
        self._emit_fn = emit
        self.running = True
        self.is_injecting_speech = False
        self.audio_queue: "thread_queue.Queue" = thread_queue.Queue()
        self._tts_chunks = 0
        self._tts_bytes = 0
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.devices = AudioDevices(cfg.asr_format, cfg.tts_format, cfg.output_device)
        self.mic_thread: Optional[threading.Thread] = None
        self.play_thread: Optional[threading.Thread] = None
        self.gate = gate or UplinkGate(
            delta_idle_seconds=cfg.delta_idle_seconds,
            commit_timeout_seconds=cfg.commit_timeout_seconds,
            mute_during_tts=cfg.mute_during_tts,
            unmute_holdoff_seconds=cfg.unmute_holdoff_seconds,
        )
        self._gate_task: Optional[asyncio.Task] = None
        self._play_lock = threading.Lock()
        self._server_tts_done = False
        self._play_writing = False
        self._playback_finished_sent = False
        self._tts_gen = 0
        self._rvc: Optional[PlaybackRvc] = None
        self._rvc_flushed = True
        self._playback_started_sent = False

    def emit(self, event_type: str, payload: Optional[dict] = None) -> None:
        if self._emit_fn is None:
            return
        self._emit_fn(
            CoachEvent(
                type=event_type,
                session_id=self.orchestrator.session_id,
                payload=payload or {},
            )
        )

    def _emit_from_thread(self, event_type: str, payload: Optional[dict] = None) -> None:
        loop = self.loop
        if loop is None:
            self.emit(event_type, payload)
            return

        def _go() -> None:
            self.emit(event_type, payload)

        loop.call_soon_threadsafe(_go)

    async def start(self) -> None:
        self.devices.ensure_supported()
        await self._load_rvc()
        await self.backend.start(
            self.orchestrator.bundle.instructions, self.orchestrator.tools
        )
        try:
            await self._inject_opening()
            self.loop = asyncio.get_event_loop()
            self.devices.open()
            self._start_audio_threads()
            self._gate_task = asyncio.create_task(self._gate_loop())
            await self._receive_loop()
        finally:
            self.running = False
            if self._gate_task is not None:
                self._gate_task.cancel()
                try:
                    await self._gate_task
                except asyncio.CancelledError:
                    pass
            self.audio_queue.put(None)
            self._join_audio_threads()
            self.devices.cleanup()
            await self.backend.close()
            self.orchestrator.close()
            self.emit("session.ended")

    async def _inject_opening(self) -> None:
        bundle = self.orchestrator.bundle
        if bundle.injected_user_prompt:
            text = bundle.injected_user_prompt
            self.is_injecting_speech = True
            await self.backend.inject_assistant_text(text)
            await self.backend.speak_text(text)
            self.orchestrator.on_injected_prompt(text)
            return
        if bundle.opening_text:
            text = bundle.opening_text
            self.is_injecting_speech = True
            await self.backend.speak_text(text)
            self.orchestrator.on_opening(text)

    async def _load_rvc(self) -> None:
        if not self.cfg.rvc_enabled:
            return
        self.emit("rvc.loading")
        try:
            self._rvc = await asyncio.to_thread(
                load_playback_rvc,
                enabled=True,
                pth=self.cfg.rvc_pth,
                index=self.cfg.rvc_index,
                hubert=self.cfg.rvc_hubert,
                rmvpe=self.cfg.rvc_rmvpe,
                f0_up_key=self.cfg.rvc_f0_up_key,
                index_rate=self.cfg.rvc_index_rate,
                hop_seconds=self.cfg.rvc_hop_seconds,
                extra_seconds=self.cfg.rvc_extra_seconds,
                mode=self.cfg.rvc_mode,
                protect=self.cfg.rvc_protect,
                pad_seconds=self.cfg.rvc_pad_seconds,
                rms_mix_rate=self.cfg.rvc_rms_mix_rate,
            )
        except Exception as exc:
            self._rvc = None
            self.emit(
                "rvc.failed",
                {"message": f"RVC unavailable, playing original TTS: {exc}"},
            )
            return
        self.emit(
            "rvc.ready",
            {
                "mode": self.cfg.rvc_mode,
                "pad_seconds": self.cfg.rvc_pad_seconds,
            },
        )

    def _rvc_begin(self, *, expect_audio: bool) -> None:
        if self._rvc is not None:
            self._rvc.begin()
        with self._play_lock:
            self._rvc_flushed = (self._rvc is None) or (not expect_audio)

    def _play_converted(self, pcm: bytes) -> list[bytes]:
        if self._rvc is None:
            return [pcm] if pcm else []
        try:
            return self._rvc.push_pcm24(pcm)
        except Exception as exc:
            self._rvc = None
            self.emit("rvc.failed", {"message": f"RVC convert failed, fallback TTS: {exc}"})
            return [pcm] if pcm else []

    def _flush_rvc_if_ready(self) -> None:
        if self._rvc is None:
            with self._play_lock:
                self._rvc_flushed = True
            return
        with self._play_lock:
            if self._rvc_flushed or not self._server_tts_done or not self.audio_queue.empty():
                return
            self._rvc_flushed = True
        try:
            pieces = self._rvc.flush()
        except Exception as exc:
            self._rvc = None
            self.emit("rvc.failed", {"message": f"RVC flush failed: {exc}"})
            return
        if not pieces:
            return
        self._mark_playback_started(pieces)
        with self._play_lock:
            self._play_writing = True
        try:
            for chunk in pieces:
                self.devices.write_output(chunk)
        except Exception:
            pass
        with self._play_lock:
            self._play_writing = False

    def _mark_playback_started(self, pieces: list[bytes]) -> None:
        with self._play_lock:
            if self._playback_started_sent:
                return
            self._playback_started_sent = True
        nbytes = sum(len(p) for p in pieces)
        sample_rate = 24000
        duration_s = nbytes / (sample_rate * 2) if nbytes else 0.0
        self._emit_from_thread(
            "playback.started",
            {
                "duration_s": duration_s,
                "bytes": nbytes,
                "sample_rate": sample_rate,
            },
        )

    def _start_audio_threads(self) -> None:
        self.mic_thread = threading.Thread(target=self._mic_worker, daemon=True)
        self.play_thread = threading.Thread(target=self._play_worker, daemon=True)
        self.mic_thread.start()
        self.play_thread.start()

    def _join_audio_threads(self) -> None:
        for thread in (self.mic_thread, self.play_thread):
            if thread is not None:
                thread.join(timeout=0.5)

    def _mic_worker(self) -> None:
        if self.devices.input_stream is None or self.loop is None:
            return
        while self.running:
            try:
                captured = self.devices.read_mic()
            except Exception as exc:
                self.emit("session.error", {"message": f"mic read failed: {exc}"})
                break
            if not self.running:
                break
            data = captured if self.gate.uplink_enabled else SILENCE_CHUNK
            if not data:
                continue
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self.backend.send_audio(data), self.loop
                )
                fut.result(timeout=2)
            except Exception:
                break

    def _play_worker(self) -> None:
        if self.devices.output_stream is None:
            return
        if self._rvc is not None:
            try:
                self._rvc.attach_thread()
            except Exception as exc:
                self.emit("rvc.failed", {"message": f"RVC GPU attach failed: {exc}"})
                self._rvc = None
                with self._play_lock:
                    self._rvc_flushed = True
        while self.running:
            try:
                data = self.audio_queue.get(timeout=0.1)
            except thread_queue.Empty:
                self._flush_rvc_if_ready()
                self._notify_playback_finished_if_ready()
                continue
            if data is None:
                break
            pieces = self._play_converted(data)
            if pieces:
                self._mark_playback_started(pieces)
            with self._play_lock:
                self._play_writing = True
            try:
                for chunk in pieces:
                    self.devices.write_output(chunk)
            except Exception:
                pass
            with self._play_lock:
                self._play_writing = False
            self._flush_rvc_if_ready()
            self._notify_playback_finished_if_ready()

    def _notify_playback_finished_if_ready(self) -> None:
        with self._play_lock:
            if self._playback_finished_sent:
                return
            if (
                not self._server_tts_done
                or self._play_writing
                or not self.audio_queue.empty()
                or not self._rvc_flushed
            ):
                return
            self._playback_finished_sent = True
            gen = self._tts_gen
        if self.loop is not None:
            asyncio.run_coroutine_threadsafe(
                self._on_playback_finished(gen), self.loop
            )

    async def _on_playback_finished(self, gen: int) -> None:
        if gen != self._tts_gen:
            return
        self.emit("playback.finished", {"gen": gen})
        await self._apply_gate(self.gate.on_playback_idle())
        holdoff = self.cfg.unmute_holdoff_seconds
        if holdoff > 0:
            await asyncio.sleep(holdoff)
        if gen != self._tts_gen:
            return
        await self._apply_gate(self.gate.tick())

    async def _gate_loop(self) -> None:
        while self.running:
            await asyncio.sleep(0.2)
            await self._apply_gate(self.gate.tick())

    async def _apply_gate(self, actions: list[str]) -> None:
        for action in actions:
            if action == "mute":
                self.emit(
                    "uplink.muted",
                    {"reason": self.gate.mute_reason or "delta_idle"},
                )
            elif action == "unmute":
                self.emit("uplink.unmuted", {"reason": "ready"})
            elif action == "commit":
                try:
                    await self.backend.commit_audio()
                except Exception as exc:
                    self.emit("session.error", {"message": f"audio commit failed: {exc}"})

    async def _receive_loop(self) -> None:
        while self.running and not self.orchestrator.should_exit:
            event = await self.backend.recv_event()
            done = await self._handle_event(event)
            if done:
                return

    async def _handle_event(self, event: VoiceEvent) -> bool:
        event_type = event.type
        if event_type == VoiceEventType.SESSION_READY:
            self.emit(
                "session.ready",
                {"dialog_id": event.extra.get("dialog_id") or ""},
            )

        elif event_type == VoiceEventType.SESSION_CLOSED:
            return True

        elif event_type == VoiceEventType.USER_SPEECH_STARTED:
            clear_queue(self.audio_queue)
            self._rvc_begin(expect_audio=False)
            self.emit("asr.started")
            await self._apply_gate(self.gate.on_speech_started())

        elif event_type == VoiceEventType.USER_TRANSCRIPT_DELTA:
            self.emit("asr.delta", {"text": event.text})
            await self._apply_gate(self.gate.on_delta(event.text))

        elif event_type == VoiceEventType.USER_TRANSCRIPT_DONE:
            self.emit("asr.completed", {"text": event.text})
            if event.text:
                self.orchestrator.on_user_text(event.text)
            await self._apply_gate(self.gate.on_completed())

        elif event_type == VoiceEventType.USER_TRANSCRIPT_FAILED:
            self.emit("asr.failed", {"error": event.error})

        elif event_type == VoiceEventType.ASSISTANT_TEXT_DELTA:
            self.emit("assistant.text_delta", {"text": event.text})

        elif event_type == VoiceEventType.ASSISTANT_TEXT_DONE:
            self.emit("assistant.text_done", {"text": event.text})
            if event.text:
                self.orchestrator.on_assistant_text(event.text)

        elif event_type == VoiceEventType.ASSISTANT_AUDIO_STARTED:
            self._tts_chunks = 0
            self._tts_bytes = 0
            with self._play_lock:
                self._tts_gen += 1
                self._server_tts_done = False
                self._playback_finished_sent = False
                self._playback_started_sent = False
            self._rvc_begin(expect_audio=True)
            self.emit("tts.started", {"tts_type": event.extra.get("tts_type")})
            if self.is_injecting_speech:
                clear_queue(self.audio_queue)
                self.is_injecting_speech = False
            await self._apply_gate(self.gate.on_tts_started())

        elif event_type == VoiceEventType.ASSISTANT_AUDIO_DELTA:
            self._tts_chunks += 1
            self._tts_bytes += len(event.audio)
            self.audio_queue.put(event.audio)

        elif event_type == VoiceEventType.ASSISTANT_AUDIO_DONE:
            self.is_injecting_speech = False
            self.emit(
                "tts.done",
                {
                    "status_code": event.extra.get("status_code"),
                    "chunks": self._tts_chunks,
                    "bytes": self._tts_bytes,
                },
            )
            if self._rvc is not None:
                self.emit("rvc.converting")
            with self._play_lock:
                self._server_tts_done = True
            await self._apply_gate(self.gate.on_tts_done())
            self._notify_playback_finished_if_ready()

        elif event_type == VoiceEventType.FUNCTION_CALL:
            asyncio.create_task(self._handle_function_call(event))

        elif event_type == VoiceEventType.ERROR:
            self.emit("session.error", {"message": event.error})
            return True

        return self.orchestrator.should_exit

    async def _handle_function_call(self, event: VoiceEvent) -> None:
        if not event.function_calls:
            return
        last_user = ""
        for entry in reversed(self.orchestrator.transcript.entries):
            if entry.role == "user":
                last_user = entry.text
                break
        results = []
        new_bundle = None
        for call in event.function_calls:
            output, bundle = self.orchestrator.handle_tool(
                call.name, call.arguments, raw_fallback=last_user
            )
            if bundle is not None:
                new_bundle = bundle
            results.append(ToolResult(call_id=call.call_id, output=output))
        await self.backend.submit_tool_results(results)
        if new_bundle is not None:
            await self.backend.update(new_bundle.instructions, new_bundle.tools)
            if new_bundle.opening_text:
                self.is_injecting_speech = True
                await self.backend.speak_text(new_bundle.opening_text)
                self.orchestrator.on_opening(new_bundle.opening_text)
        if self.orchestrator.should_exit:
            self.running = False
