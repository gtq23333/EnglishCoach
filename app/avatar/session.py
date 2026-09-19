"""Attach an AvatarDriver to a coach session.

Loads the GPU engine on a worker thread (same idea as RVC's play thread) so
the asyncio loop stays free. A Tk sink is optional until the real UI exists.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import Optional

from app.avatar.driver import AvatarDriver, FrameSink
from app.avatar.engine import character_preview_rgb, get_or_create_avatar_engine
from app.config import AppConfig
from app.events import CoachEvent, EventBus

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


def _resolve(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((ROOT / path).resolve())


class AvatarSession:
    def __init__(self, cfg: AppConfig, bus: EventBus, sink: Optional[FrameSink] = None):
        self.cfg = cfg
        self.bus = bus
        self._external_sink = sink
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pump: Optional[asyncio.Task] = None
        self._driver: Optional[AvatarDriver] = None
        self._queue: asyncio.Queue | None = None
        self._inbox: list = []
        self._inbox_lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._session_id = ""

    def start(self, session_id: str, loop: asyncio.AbstractEventLoop) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._loop = loop
        self._session_id = session_id
        self._thread = threading.Thread(
            target=self._run,
            name="avatar-preview",
            daemon=True,
        )
        self._thread.start()
        self._pump = loop.create_task(self._pump_events(session_id), name="avatar-events")

    async def stop(self) -> None:
        self._stop.set()
        if self._queue is not None:
            self.bus.unsubscribe(self._queue)
            self._queue = None
        if self._pump is not None:
            self._pump.cancel()
            try:
                await self._pump
            except asyncio.CancelledError:
                pass
            self._pump = None
        if self._thread is not None:
            self._thread.join(timeout=0.4)
            self._thread = None
        self._driver = None
        self._loop = None

    def _emit(self, event_type: str, payload: Optional[dict] = None) -> None:
        event = CoachEvent(
            type=event_type,
            session_id=self._session_id,
            payload=payload or {},
        )
        loop = self._loop
        if loop is None:
            self.bus.emit(event)
            return
        loop.call_soon_threadsafe(self.bus.emit, event)

    async def _pump_events(self, session_id: str) -> None:
        self._queue = self.bus.subscribe(session_id)
        try:
            while not self._stop.is_set():
                event = await self._queue.get()
                with self._inbox_lock:
                    driver = self._driver
                    if driver is None:
                        self._inbox.append(event)
                        continue
                driver.on_event(event)
        except asyncio.CancelledError:
            return
        finally:
            if self._queue is not None:
                self.bus.unsubscribe(self._queue)
                self._queue = None

    def _run(self) -> None:
        sink: FrameSink
        if self._external_sink is not None:
            sink = self._external_sink
        elif self.cfg.avatar_show_preview:
            from app.avatar.preview import TkFrameSink

            sink = TkFrameSink()
        else:
            sink = _NullSink()
        try:
            try:
                sink.push_frame(character_preview_rgb(_resolve(self.cfg.avatar_character)))
            except Exception:
                logger.exception("avatar placeholder failed")
            self._emit(
                "avatar.loading",
                {
                    "component": "avatar",
                    "detail": "正在加载本地 THA3 权重（不访问 HuggingFace）…",
                },
            )
            engine = get_or_create_avatar_engine(
                character=_resolve(self.cfg.avatar_character),
                weights_dir=_resolve(self.cfg.avatar_weights_dir),
            )
            try:
                import torch

                if engine.device.type == "cuda":
                    torch.cuda.set_device(engine.device)
                    torch.zeros(1, device=engine.device)
                    torch.cuda.synchronize()
            except Exception:
                logger.exception("avatar cuda attach failed")
            driver = AvatarDriver(engine, fps=self.cfg.avatar_fps)
            with self._inbox_lock:
                self._driver = driver
                pending = list(self._inbox)
                self._inbox.clear()
            for event in pending:
                driver.on_event(event)
            logger.info(
                "avatar ready native=%s weights=%s",
                engine.native_size,
                self.cfg.avatar_weights_dir,
            )
            self._emit(
                "avatar.ready",
                {
                    "component": "avatar",
                    "detail": f"数字人已就绪 device={engine.device}",
                    "native_size": list(engine.native_size),
                    "device": str(engine.device),
                },
            )
            from app.avatar.driver import attach_sink_loop

            attach_sink_loop(driver, sink, self._stop.is_set)
        except Exception as exc:
            logger.exception("avatar thread failed")
            from app.diag import classify_exception

            payload = classify_exception(exc, component="avatar")
            payload["message"] = f"{payload['hint']} {payload['error']}"
            payload["detail"] = payload["hint"]
            self._emit("avatar.failed", payload)
            self._stop.set()


class _NullSink:
    def push_frame(self, rgb) -> None:
        return

    def close(self) -> None:
        return
