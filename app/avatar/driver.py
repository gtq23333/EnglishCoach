"""Drive AvatarEngine from coach events. UI-agnostic.

A FrameSink receives 1x RGB frames. Tk preview is one sink; the future
bottom-right widget should be another that calls the same ``tick()``.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Callable, Optional, Protocol

import numpy as np

from app.avatar.engine import AvatarEngine
from app.avatar.viseme import Timeline, plan_visemes
from app.events import CoachEvent


class FrameSink(Protocol):
    def push_frame(self, rgb: np.ndarray) -> None: ...

    def close(self) -> None: ...


class AvatarDriver:
    def __init__(self, engine: AvatarEngine, fps: float = 15.0):
        self.engine = engine
        self.fps = fps
        self._timeline: Timeline = [(0.0, {}), (0.2, {})]
        self._t0: Optional[float] = None
        self._duration = 0.2
        self._pending_text = ""
        self._speaking = False
        self.latest: Optional[np.ndarray] = None
        self._lock = threading.RLock()

    def on_event(self, event: CoachEvent) -> None:
        kind = event.type
        payload = event.payload or {}
        if kind == "assistant.text_delta":
            piece = str(payload.get("text") or "")
            if piece:
                with self._lock:
                    self._pending_text += piece
                if self._speaking:
                    self._refresh_timeline()
        elif kind in {"assistant.text_done", "utterance"}:
            if kind == "utterance" and payload.get("role") != "assistant":
                return
            text = str(payload.get("text") or "")
            if text:
                with self._lock:
                    self._pending_text = text
                if self._speaking:
                    self._refresh_timeline()
        elif kind == "playback.started":
            duration = float(payload.get("duration_s") or 0.0)
            with self._lock:
                text = self._pending_text
            self.speak(text, duration)
        elif kind in {"playback.finished", "asr.started"}:
            self.rest()

    def speak(self, text: str, duration_s: float) -> None:
        text_est = max(0.45, 0.055 * max(len(text), 8))
        duration = max(float(duration_s or 0.0), text_est)
        timeline = plan_visemes(text, duration)
        with self._lock:
            self._timeline = timeline
            self._duration = duration
            if self._t0 is None:
                self._t0 = time.perf_counter()
            self._speaking = True

    def _refresh_timeline(self) -> None:
        with self._lock:
            if not self._speaking:
                return
            text = self._pending_text
            elapsed = (time.perf_counter() - self._t0) if self._t0 else 0.0
            duration = max(self._duration, elapsed + 0.4)
            self._duration = duration
            self._timeline = plan_visemes(text, duration)

    def rest(self) -> None:
        with self._lock:
            self._speaking = False
            self._t0 = None
            self._pending_text = ""
            self._timeline = [(0.0, {}), (0.2, {})]

    def _with_idle(self, pose: np.ndarray) -> np.ndarray:
        names = getattr(self.engine, "name_to_index", None) or {}
        index = names.get("breathing")
        if index is None:
            return pose
        out = np.array(pose, dtype=np.float32, copy=True)
        out[index] = 0.42 + 0.38 * math.sin(time.perf_counter() * 1.35)
        return out

    def tick(self) -> np.ndarray:
        with self._lock:
            speaking = self._speaking
            t0 = self._t0
            duration = self._duration
            timeline = self._timeline
        if not speaking or t0 is None:
            pose = self.engine.rest_pose()
        else:
            elapsed = time.perf_counter() - t0
            wrap = duration if duration > 0 else 0.45
            pose = self.engine.pose_from_timeline(elapsed % wrap, timeline)
        pose = self._with_idle(pose)
        frame = self.engine.render_rgb(pose)
        self.latest = frame
        return frame


def attach_sink_loop(
    driver: AvatarDriver,
    sink: FrameSink,
    should_stop: Callable[[], bool],
    idle_rest: bool = True,
) -> None:
    """Blocking render loop for a preview thread. UI can replace this."""
    interval = 1.0 / max(driver.fps, 1.0)
    if idle_rest:
        sink.push_frame(driver.tick())
    while not should_stop():
        t0 = time.perf_counter()
        sink.push_frame(driver.tick())
        delay = interval - (time.perf_counter() - t0)
        if delay > 0:
            time.sleep(delay)
    sink.close()
