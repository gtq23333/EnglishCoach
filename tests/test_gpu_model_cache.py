from app.avatar.engine import get_or_create_avatar_engine, reset_avatar_engine_cache
from app.voice.rvc_engine import load_playback_rvc, reset_playback_rvc_cache


def test_playback_rvc_cache_reuses_instance(monkeypatch):
    created: list[object] = []

    class FakePlayback:
        def __init__(self, *args, **kwargs):
            created.append(self)

        def load(self) -> None:
            return None

    monkeypatch.setattr("app.voice.rvc_engine.PlaybackRvc", FakePlayback)
    reset_playback_rvc_cache()
    first = load_playback_rvc(enabled=True, pth="a.pth")
    second = load_playback_rvc(enabled=True, pth="a.pth")
    assert first is second
    assert len(created) == 1
    reset_playback_rvc_cache()


def test_avatar_engine_cache_reuses_instance(monkeypatch):
    created: list[object] = []

    class FakeEngine:
        def __init__(self, *args, **kwargs):
            created.append(self)
            self.native_size = (1, 1)
            self.device = "cpu"

    monkeypatch.setattr("app.avatar.engine.AvatarEngine", FakeEngine)
    reset_avatar_engine_cache()
    first = get_or_create_avatar_engine("char.png", "weights")
    second = get_or_create_avatar_engine("char.png", "weights")
    assert first is second
    assert len(created) == 1
    reset_avatar_engine_cache()
