from __future__ import annotations

import asyncio
import queue as thread_queue
import threading
import time
from typing import Callable, Optional

from app.config import AppConfig
from app.diag import classify_exception, classify_voice_error
from app.dialogue.orchestrator import DialogueOrchestrator
from app.dialogue.uplink_gate import UplinkGate
from app.events import CoachEvent
from app.realtime.audio_io import MIC_CHUNK, clear_queue
from app.realtime.transport import AudioTransport, SoundDeviceTransport
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
        transport: Optional[AudioTransport] = None,
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
        self.transport: AudioTransport = transport or SoundDeviceTransport(
            cfg.asr_format, cfg.tts_format, cfg.output_device
        )
        self.mic_thread: Optional[threading.Thread] = None
        self.play_thread: Optional[threading.Thread] = None
        self.gate = gate or UplinkGate(
            delta_idle_seconds=cfg.delta_idle_seconds,
            commit_timeout_seconds=cfg.commit_timeout_seconds,
            mute_during_tts=cfg.mute_during_tts,
            unmute_holdoff_seconds=cfg.unmute_holdoff_seconds,
            max_speech_seconds=cfg.max_speech_seconds,
        )
        self._gate_task: Optional[asyncio.Task] = None
        self._play_lock = threading.Lock()
        self._server_tts_done = False
        self._play_writing = False
        self._playback_finished_sent = False
        self._tts_gen = 0
        self._rvc: Optional[PlaybackRvc] = None
        self._rvc_attached = False
        self._rvc_flushed = True
        self._rvc_flushing = False
        self._playback_started_sent = False
        self._playback_until = 0.0
        self._rvc_task: Optional[asyncio.Task] = None
        self._voice_reconnects = 0
        self._last_asr_text = ""
        self._assistant_text = ""
        self._recovering_voice = False

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
        await asyncio.sleep(0)
        self.transport.ensure_supported()
        self._rvc_task = asyncio.create_task(self._load_rvc(), name="rvc-load")
        self.emit(
            "voice.connecting",
            {
                "component": "voice",
                "endpoint": self.cfg.endpoint_url,
                "detail": "正在连接火山引擎语音对话…",
            },
        )
        try:
            await asyncio.wait_for(
                self.backend.start(
                    self.orchestrator.bundle.instructions, self.orchestrator.tools
                ),
                timeout=12,
            )
        except asyncio.TimeoutError as exc:
            await self._cancel_rvc_task()
            payload = classify_exception(exc, component="voice")
            payload["detail"] = f"语音对话 API 12 秒未连通：{self.cfg.endpoint_url}"
            self.emit("voice.failed", payload)
            self.emit("session.error", {"message": payload["detail"], **payload})
            return
        except Exception as exc:
            await self._cancel_rvc_task()
            payload = classify_exception(exc, component="voice")
            payload["detail"] = payload.get("hint") or str(exc)
            self.emit("voice.failed", payload)
            self.emit("session.error", {"message": payload["detail"], **payload})
            return
        self.emit(
            "voice.ready",
            {
                "component": "voice",
                "detail": "语音对话已接通，可以说话",
                "endpoint": self.cfg.endpoint_url,
            },
        )
        try:
            if not self.running:
                return
            self.loop = asyncio.get_event_loop()
            self.transport.open()
            self._start_audio_threads()
            await self._inject_opening()
            self._gate_task = asyncio.create_task(self._gate_loop())
            await self._receive_loop()
        finally:
            self.running = False
            await self._cancel_rvc_task()
            if self._gate_task is not None:
                self._gate_task.cancel()
                try:
                    await self._gate_task
                except asyncio.CancelledError:
                    pass
            self.audio_queue.put(None)
            self._join_audio_threads()
            self.transport.cleanup()
            try:
                self.flush_transcript()
            except Exception:
                pass
            try:
                await asyncio.wait_for(self.backend.close(), timeout=1.0)
            except Exception:
                pass
            try:
                self.orchestrator.close()
            except Exception:
                pass
            self.emit("session.ended")

    async def _cancel_rvc_task(self) -> None:
        task = self._rvc_task
        self._rvc_task = None
        if task is None:
            return
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=0.1)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

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
            self.emit(
                "rvc.skipped",
                {"component": "rvc", "detail": "配置未开启变声"},
            )
            return
        self.emit(
            "rvc.loading",
            {
                "component": "rvc",
                "detail": "正在从本地缓存加载变声模型（不会访问 HuggingFace）…",
            },
        )
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
            payload = classify_exception(exc, component="rvc")
            payload["detail"] = payload["hint"]
            payload["message"] = f"{payload['hint']} {payload['error']}"
            self.emit("rvc.failed", payload)
            return
        if not self.running:
            return
        self.emit(
            "rvc.ready",
            {
                "component": "rvc",
                "detail": "变声已就绪（本地权重）",
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
        self._ensure_rvc_attached()
        if self._rvc is None:
            return [pcm] if pcm else []
        with self._play_lock:
            use_rvc = not self._rvc_flushed
        if not use_rvc:
            return [pcm] if pcm else []
        try:
            return self._rvc.push_pcm24(pcm)
        except Exception as exc:
            self._rvc = None
            self.emit("rvc.failed", {"message": f"RVC convert failed, fallback TTS: {exc}"})
            return [pcm] if pcm else []

    def _output_duration(self, pieces: list[bytes]) -> float:
        nbytes = sum(len(p) for p in pieces)
        rate = int(getattr(self.transport, "playback_rate", 24000) or 24000)
        return nbytes / (rate * 2) if nbytes and rate > 0 else 0.0

    def _note_output_written(self, pieces: list[bytes]) -> None:
        duration_s = self._output_duration(pieces)
        now = time.perf_counter()
        with self._play_lock:
            if not self._playback_started_sent:
                self._playback_started_sent = True
                self._playback_until = now + max(duration_s, 0.05)
                first = True
            else:
                self._playback_until = max(self._playback_until, now) + duration_s
                first = False
        if not first:
            return
        self._emit_from_thread(
            "playback.started",
            {
                "duration_s": max(duration_s, 0.05),
                "bytes": sum(len(p) for p in pieces),
                "sample_rate": int(getattr(self.transport, "playback_rate", 24000) or 24000),
            },
        )

    def _flush_rvc_if_ready(self) -> None:
        if self._rvc is None:
            with self._play_lock:
                self._rvc_flushed = True
            return
        with self._play_lock:
            if (
                self._rvc_flushed
                or self._rvc_flushing
                or not self._server_tts_done
                or not self.audio_queue.empty()
            ):
                return
            self._rvc_flushing = True
        try:
            pieces = self._rvc.flush()
        except Exception as exc:
            self._rvc = None
            self.emit("rvc.failed", {"message": f"RVC flush failed: {exc}"})
            pieces = []
        finally:
            with self._play_lock:
                self._rvc_flushing = False
                self._rvc_flushed = True
        if not pieces:
            return
        self._note_output_written(pieces)
        with self._play_lock:
            self._play_writing = True
        try:
            for chunk in pieces:
                self.transport.write_output(chunk)
        except Exception:
            pass
        with self._play_lock:
            self._play_writing = False

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
        if self.loop is None:
            return
        while self.running:
            try:
                captured = self.transport.read_mic()
            except Exception as exc:
                self.emit("session.error", {"message": f"mic read failed: {exc}"})
                break
            if not self.running:
                break
            if self._recovering_voice:
                continue
            # 火山端到端靠 20ms 音频包保活，没有独立心跳事件。
            # 浏览器包可能空一拍，空窗补静音帧，不能跳过发包。
            if not self.gate.uplink_enabled or not captured:
                data = SILENCE_CHUNK
            else:
                data = captured
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self.backend.send_audio(data), self.loop
                )
                fut.result(timeout=2)
            except Exception:
                if not self.running:
                    break

    def _ensure_rvc_attached(self) -> None:
        if self._rvc is None or self._rvc_attached:
            return
        try:
            self._rvc.attach_thread()
            self._rvc_attached = True
        except Exception as exc:
            self._emit_from_thread(
                "rvc.failed",
                {"message": f"RVC GPU attach failed: {exc}"},
            )
            self._rvc = None
            with self._play_lock:
                self._rvc_flushed = True

    def _play_worker(self) -> None:
        while self.running:
            self._ensure_rvc_attached()
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
                self._note_output_written(pieces)
            with self._play_lock:
                self._play_writing = True
            try:
                for chunk in pieces:
                    self.transport.write_output(chunk)
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
                or self._rvc_flushing
                or time.perf_counter() < self._playback_until
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
                reason = self.gate.mute_reason or "delta_idle"
                payload = {"reason": reason}
                if reason == "max_speech":
                    payload["detail"] = (
                        f"这一句已超过 {int(self.gate.max_speech_seconds)} 秒，"
                        "先按已识别内容让助手接话，避免语音流被服务端掐断。"
                    )
                self.emit("uplink.muted", payload)
            elif action == "unmute":
                self.emit("uplink.unmuted", {"reason": "ready"})
            elif action == "commit":
                try:
                    await self.backend.commit_audio()
                except Exception as exc:
                    self.emit(
                        "session.error",
                        {"message": f"audio commit failed: {exc}", "recoverable": True},
                    )

    async def _receive_loop(self) -> None:
        while self.running and not self.orchestrator.should_exit:
            try:
                event = await self.backend.recv_event()
            except Exception as exc:
                if await self._recover_voice(str(exc)):
                    continue
                payload = classify_exception(exc, component="voice")
                self.emit(
                    "session.error",
                    {"message": payload["hint"], "detail": payload["hint"], **payload},
                )
                return
            done = await self._handle_event(event)
            if done:
                return

    def _salvage_open_asr(self) -> None:
        text = (self._last_asr_text or "").strip()
        self._last_asr_text = ""
        if not text:
            return
        self.emit("asr.completed", {"text": text})
        self.orchestrator.on_user_text(text)

    def _flush_assistant(self) -> None:
        text = (self._assistant_text or "").strip()
        self._assistant_text = ""
        if not text:
            return
        self.emit("assistant.text_done", {"text": text})
        self.orchestrator.on_assistant_text(text)

    def flush_transcript(self) -> None:
        """Persist in-flight user/assistant text before hangup or crash-stop."""
        self._salvage_open_asr()
        self._flush_assistant()

    async def _recover_voice(self, error_text: str) -> bool:
        payload = classify_voice_error(error_text)
        if not payload.get("recoverable") or self._voice_reconnects >= 2:
            return False
        self._voice_reconnects += 1
        self._salvage_open_asr()
        self._recovering_voice = True
        await self._apply_gate(self.gate.on_backend_reset())
        detail = (
            f"{payload['hint']} 正在重连语音通道（{self._voice_reconnects}/2）…"
        )
        self.emit(
            "voice.connecting",
            {"component": "voice", "cause": payload["cause"], "detail": detail},
        )
        try:
            await asyncio.wait_for(
                self.backend.reconnect(
                    self.orchestrator.bundle.instructions,
                    self.orchestrator.tools,
                ),
                timeout=12,
            )
        except Exception as exc:
            failed = classify_exception(exc, component="voice")
            failed["detail"] = failed.get("hint") or str(exc)
            self.emit("voice.failed", failed)
            self._recovering_voice = False
            return False
        self._recovering_voice = False
        self.emit(
            "voice.ready",
            {
                "component": "voice",
                "detail": "语音通道已重连，可以继续说",
            },
        )
        return True

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
            self._flush_assistant()
            clear_queue(self.audio_queue)
            self._rvc_begin(expect_audio=False)
            self.emit("asr.started")
            await self._apply_gate(self.gate.on_speech_started())

        elif event_type == VoiceEventType.USER_TRANSCRIPT_DELTA:
            if event.text:
                self._last_asr_text = event.text
            self.emit("asr.delta", {"text": event.text})
            await self._apply_gate(self.gate.on_delta(event.text))

        elif event_type == VoiceEventType.USER_TRANSCRIPT_DONE:
            text = (event.text or self._last_asr_text or "").strip()
            self._last_asr_text = ""
            self.emit("asr.completed", {"text": text})
            if text:
                self.orchestrator.on_user_text(text)
            await self._apply_gate(self.gate.on_completed())

        elif event_type == VoiceEventType.USER_TRANSCRIPT_FAILED:
            self.emit("asr.failed", {"error": event.error})

        elif event_type == VoiceEventType.ASSISTANT_TEXT_DELTA:
            if event.text:
                self._assistant_text += event.text
            self.emit("assistant.text_delta", {"text": self._assistant_text or event.text})

        elif event_type == VoiceEventType.ASSISTANT_TEXT_DONE:
            text = (event.text or self._assistant_text or "").strip()
            self._assistant_text = ""
            self.emit("assistant.text_done", {"text": text})
            if text:
                self.orchestrator.on_assistant_text(text)

        elif event_type == VoiceEventType.ASSISTANT_AUDIO_STARTED:
            self._tts_chunks = 0
            self._tts_bytes = 0
            with self._play_lock:
                self._tts_gen += 1
                self._server_tts_done = False
                self._playback_finished_sent = False
                self._playback_started_sent = False
                self._rvc_flushing = False
                self._playback_until = 0.0
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
            if self._assistant_text.strip():
                self._flush_assistant()
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
            if await self._recover_voice(event.error):
                return False
            payload = classify_voice_error(event.error)
            self.emit(
                "session.error",
                {
                    "message": payload["hint"],
                    "detail": payload["hint"],
                    **payload,
                },
            )
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
