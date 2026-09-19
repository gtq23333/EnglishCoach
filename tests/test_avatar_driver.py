from __future__ import annotations

import numpy as np

from app.avatar.driver import AvatarDriver
from app.events import CoachEvent


class FakeEngine:
    name_to_index = {}

    def rest_pose(self):
        return np.zeros(4, dtype=np.float32)

    def pose_from_timeline(self, t, timeline):
        self.last_t = t
        self.last_timeline = timeline
        active = any(keys for _, keys in timeline)
        return np.ones(4, dtype=np.float32) if active else np.zeros(4, dtype=np.float32)

    def render_rgb(self, pose):
        self.last_pose = np.asarray(pose)
        return np.zeros((2, 2, 3), dtype=np.uint8)


def _event(kind: str, **payload) -> CoachEvent:
    return CoachEvent(type=kind, session_id="s", payload=payload)


def test_playback_before_transcript_still_opens_mouth():
    driver = AvatarDriver(FakeEngine(), fps=10)
    driver.on_event(_event("playback.started", duration_s=0.08))
    driver.tick()
    assert driver._speaking is True
    mouths = [keys for _, keys in driver._timeline if keys]
    assert mouths, "empty transcript should still get a fallback viseme cycle"


def test_tick_keeps_speaking_past_estimated_duration():
    driver = AvatarDriver(FakeEngine(), fps=10)
    driver.on_event(_event("assistant.text_done", text="hello there"))
    driver.on_event(_event("playback.started", duration_s=0.05))
    driver._duration = 0.01
    driver._t0 = driver._t0 - 1.0 if driver._t0 else 0.0
    driver.tick()
    assert driver._speaking is True
    assert np.any(driver.engine.last_pose)


def test_tts_started_does_not_open_mouth():
    driver = AvatarDriver(FakeEngine(), fps=10)
    driver.on_event(_event("assistant.text_done", text="hello there"))
    driver.on_event(_event("tts.started"))
    driver.tick()
    assert driver._speaking is False
    assert np.allclose(driver.engine.last_pose, 0)


def test_playback_finished_returns_to_rest():
    driver = AvatarDriver(FakeEngine(), fps=10)
    driver.on_event(_event("playback.started", duration_s=1.0))
    driver.on_event(_event("playback.finished"))
    driver.tick()
    assert driver._speaking is False
    assert np.allclose(driver.engine.last_pose, 0)


def test_text_deltas_accumulate_before_playback():
    driver = AvatarDriver(FakeEngine(), fps=10)
    driver.on_event(_event("assistant.text_delta", text="hel"))
    driver.on_event(_event("assistant.text_delta", text="lo"))
    driver.on_event(_event("playback.started", duration_s=1.0))
    assert driver._pending_text == "hello"
