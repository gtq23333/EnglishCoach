from pathlib import Path

from app.voice.rvc_sdk import RvcConverter, RvcPaths


def test_rvc_paths_know_default_speaker_pack():
    paths = RvcPaths()
    assert paths.model_pth.name == "KFK_V2_500.pth"
    assert paths.index.name.endswith(".index")
    missing = set(paths.missing())
    if (Path("third_party/rvc/infer/modules/vc/modules.py")).exists():
        assert "rvc_root" not in missing
    if paths.model_pth.exists():
        assert "model_pth" not in missing
    if paths.index.exists():
        assert "index" not in missing


def test_playback_defaults_to_utterance_mode():
    paths = RvcPaths()
    if paths.missing():
        return
    engine = RvcConverter(paths)
    assert engine.mode == "utterance"
    assert engine.protect == 0.33
    assert engine.hop_16 % 160 == 0
