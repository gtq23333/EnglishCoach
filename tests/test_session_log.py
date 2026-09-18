from __future__ import annotations

from pathlib import Path

from app.dialogue.session_log import SessionLog, load_records
from app.dialogue.transcript import Transcript
from app.prompts.base import SceneSpec


def test_session_log_flushes_each_record(tmp_path: Path):
    log = SessionLog(tmp_path, "sess-1", mode="text")
    path = log.path
    assert path.exists()
    first = load_records(path)
    assert first[0]["type"] == "session_start"
    assert first[0]["mode"] == "text"
    assert first[0]["session_id"] == "sess-1"

    transcript = Transcript()
    entry = transcript.add("user", "hello", source="user", phase="practice")
    log.append_utterance(entry)
    after_utterance = load_records(path)
    assert after_utterance[-1]["type"] == "utterance"
    assert after_utterance[-1]["text"] == "hello"
    assert after_utterance[-1]["role"] == "user"

    log.append_scene(SceneSpec(raw="cafe", scene="cafe", user_role="customer"))
    after_scene = load_records(path)
    assert after_scene[-1]["type"] == "scene"
    assert after_scene[-1]["scene"]["scene"] == "cafe"

    log.close()
    log.close()
    records = load_records(path)
    assert records[-1]["type"] == "session_end"
    assert sum(1 for item in records if item["type"] == "session_end") == 1
