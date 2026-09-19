from app.dialogue.uplink_gate import UplinkGate


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def test_empty_delta_does_not_reset_idle_timer():
    clock = FakeClock()
    gate = UplinkGate(delta_idle_seconds=6, commit_timeout_seconds=3, clock=clock)
    assert gate.on_speech_started() == []
    clock.advance(2)
    assert gate.on_delta("hello") == []
    clock.advance(5)
    assert gate.on_delta("") == []
    clock.advance(1.1)
    actions = gate.tick()
    assert "mute" in actions
    assert gate.uplink_enabled is False
    assert gate.mute_reason == "delta_idle"


def test_silence_then_commit_then_unmute_after_tts():
    clock = FakeClock()
    gate = UplinkGate(delta_idle_seconds=6, commit_timeout_seconds=3, clock=clock)
    gate.on_speech_started()
    clock.advance(6.1)
    assert "mute" in gate.tick()
    clock.advance(3.1)
    assert "commit" in gate.tick()
    gate.on_completed()
    assert "unmute" not in gate.tick()
    gate.on_tts_started()
    assert gate.uplink_enabled is False
    clock.advance(0.2)
    gate.on_tts_done()
    assert gate.uplink_enabled is False
    assert "unmute" not in gate.on_playback_idle()
    clock.advance(0.08)
    actions = gate.tick()
    assert "unmute" in actions
    assert gate.uplink_enabled is True


def test_mute_during_tts_without_idle():
    clock = FakeClock()
    gate = UplinkGate(mute_during_tts=True, clock=clock)
    actions = gate.on_tts_started()
    assert "mute" in actions
    assert gate.mute_reason == "tts_playback"
    gate.on_tts_done()
    assert gate.uplink_enabled is False
    assert "unmute" not in gate.tick()
    assert "unmute" not in gate.on_playback_idle()
    clock.advance(0.08)
    actions = gate.tick()
    assert "unmute" in actions


def test_midstream_idle_does_not_unmute_on_tts_done():
    clock = FakeClock()
    gate = UplinkGate(mute_during_tts=True, unmute_holdoff_seconds=0.08, clock=clock)
    gate.on_tts_started()
    gate.on_playback_idle()
    clock.advance(0.5)
    gate.on_tts_done()
    assert gate.uplink_enabled is False
    clock.advance(1.0)
    assert "unmute" not in gate.tick()
    assert "unmute" not in gate.on_playback_idle()
    clock.advance(0.07)
    assert "unmute" not in gate.tick()
    clock.advance(0.01)
    assert "unmute" in gate.tick()
    assert gate.uplink_enabled is True


def test_holdoff_blocks_immediate_unmute():
    clock = FakeClock()
    gate = UplinkGate(unmute_holdoff_seconds=0.08, clock=clock)
    gate.on_tts_started()
    gate.on_tts_done()
    assert "unmute" not in gate.on_playback_idle()
    clock.advance(0.07)
    assert "unmute" not in gate.tick()
    clock.advance(0.01)
    assert "unmute" in gate.tick()


def test_long_speech_is_not_cut_by_default():
    clock = FakeClock()
    gate = UplinkGate(delta_idle_seconds=60, clock=clock)
    assert gate.max_speech_seconds == 0
    gate.on_speech_started()
    clock.advance(120)
    assert gate.on_delta("still talking after two minutes") == []
    assert gate.uplink_enabled is True


def test_max_speech_mutes_and_commits_while_deltas_continue():
    clock = FakeClock()
    gate = UplinkGate(max_speech_seconds=20, delta_idle_seconds=60, clock=clock)
    assert gate.on_speech_started() == []
    clock.advance(10)
    assert gate.on_delta("still talking") == []
    clock.advance(10.1)
    actions = gate.on_delta("still talking more")
    assert "mute" in actions
    assert "commit" in actions
    assert gate.uplink_enabled is False
    assert gate.mute_reason == "max_speech"
