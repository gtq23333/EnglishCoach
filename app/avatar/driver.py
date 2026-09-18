"""Drive AvatarEngine from coach events. UI-agnostic.

A FrameSink receives 1x RGB frames. Tk preview is one sink; the future
bottom-right widget should be another that calls the same ``tick()``.
"""

from __future__ import annotations

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
        self._lock = threading.Lock()

    def on_event(self, event: CoachEvent) -> None:
        kind = event.type
        payload = event.payload or {}
        if kind == "assistant.text_done":
            with self._lock:
                self._pending_text = str(payload.get("text") or "")
        elif kind == "utterance" and payload.get("role") == "assistant":
            with self._lock:
                self._pending_text = str(payload.get("text") or self._pending_text)
        elif kind == "playback.started":
            duration = float(payload.get("duration_s") or 0.0)
            with self._lock:
                text = self._pending_text
            if duration <= 0:
                duration = max(0.4, 0.06 * max(len(text), 4))
            self.speak(text, duration)
        elif kind in {"playback.finished", "asr.started"}:
            self.rest()

    def speak(self, text: str, duration_s: float) -> None:
        timeline = plan_visemes(text, duration_s)
        with self._lock:
            self._timeline = timeline
            self._duration = max(duration_s, 0.12)
            self._t0 = time.perf_counter()
            self._speaking = True

    def rest(self) -> None:
        with self._lock:
            self._speaking = False
            self._t0 = None
            self._timeline = [(0.0, {}), (0.2, {})]

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
            if elapsed >= duration:
                self.rest()
                pose = self.engine.rest_pose()
            else:
                pose = self.engine.pose_from_timeline(elapsed, timeline)
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
