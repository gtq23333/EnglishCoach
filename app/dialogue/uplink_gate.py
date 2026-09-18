from __future__ import annotations

import time
from typing import Callable, List, Optional


Clock = Callable[[], float]


class UplinkGate:
    """Decide when to send mic PCM vs silence frames.

    Hardware capture stays open. The gate only flips `uplink_enabled`.
    """

    def __init__(
        self,
        delta_idle_seconds: float = 6.0,
        commit_timeout_seconds: float = 3.0,
        mute_during_tts: bool = True,
        complete_unmute_seconds: float = 2.0,
        unmute_holdoff_seconds: float = 0.08,
        clock: Optional[Clock] = None,
    ):
        self.delta_idle_seconds = float(delta_idle_seconds)
        self.commit_timeout_seconds = float(commit_timeout_seconds)
        self.mute_during_tts = bool(mute_during_tts)
        self.complete_unmute_seconds = float(complete_unmute_seconds)
        self.unmute_holdoff_seconds = float(unmute_holdoff_seconds)
        self._clock = clock or time.monotonic
        self.uplink_enabled = True
        self.mute_reason: Optional[str] = None
        self._speech_open = False
        self._last_nonempty_at: Optional[float] = None
        self._silenced_at: Optional[float] = None
        self._completed_at: Optional[float] = None
        self._commit_sent = False
        self._tts_playing = False
        self._tts_done = False
        self._playback_idle = True
        self._playback_idle_at: Optional[float] = None

    def on_speech_started(self) -> List[str]:
        self._speech_open = True
        self._last_nonempty_at = self._clock()
        self._completed_at = None
        self._commit_sent = False
        return self.tick()

    def on_delta(self, text: str) -> List[str]:
        if (text or "").strip():
            self._last_nonempty_at = self._clock()
        return self.tick()

    def on_completed(self) -> List[str]:
        self._speech_open = False
        self._last_nonempty_at = None
        self._completed_at = self._clock()
        return self.tick()

    def on_tts_started(self) -> List[str]:
        self._tts_playing = True
        self._tts_done = False
        self._playback_idle = False
        self._playback_idle_at = None
        actions: List[str] = []
        if self.mute_during_tts:
            actions.extend(self._mute("tts_playback"))
        actions.extend(self.tick())
        return _dedupe(actions)

    def on_tts_done(self) -> List[str]:
        self._tts_playing = False
        self._tts_done = True
        # Mid-stream empty-queue idle does not count. Need a fresh
        # playback-finished signal after the last chunk has actually played.
        self._playback_idle = False
        self._playback_idle_at = None
        return self.tick()

    def on_playback_idle(self) -> List[str]:
        self._playback_idle = True
        self._playback_idle_at = self._clock()
        return self.tick()

    def tick(self) -> List[str]:
        now = self._clock()
        actions: List[str] = []
        if (
            self.uplink_enabled
            and self._speech_open
            and self._last_nonempty_at is not None
            and now - self._last_nonempty_at >= self.delta_idle_seconds
        ):
            actions.extend(self._mute("delta_idle"))
        if (
            not self.uplink_enabled
            and self._speech_open
            and self._silenced_at is not None
            and not self._commit_sent
            and now - self._silenced_at >= self.commit_timeout_seconds
        ):
            self._commit_sent = True
            actions.append("commit")
        if self._should_unmute(now):
            actions.extend(self._unmute())
        return _dedupe(actions)

    def _should_unmute(self, now: float) -> bool:
        if self.uplink_enabled or self._tts_playing or not self._playback_idle:
            return False
        if self._tts_done:
            idle_at = self._playback_idle_at
            if idle_at is None:
                return False
            return now - idle_at >= self.unmute_holdoff_seconds
        if (
            self._completed_at is not None
            and now - self._completed_at >= self.complete_unmute_seconds
        ):
            return True
        return False

    def _mute(self, reason: str) -> List[str]:
        if not self.uplink_enabled:
            self.mute_reason = reason
            return []
        self.uplink_enabled = False
        self.mute_reason = reason
        self._silenced_at = self._clock()
        self._commit_sent = False
        return ["mute"]

    def _unmute(self) -> List[str]:
        if self.uplink_enabled:
            return []
        self.uplink_enabled = True
        self.mute_reason = None
        self._silenced_at = None
        self._commit_sent = False
        self._tts_done = False
        self._completed_at = None
        self._playback_idle_at = None
        return ["unmute"]


def _dedupe(actions: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in actions:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
