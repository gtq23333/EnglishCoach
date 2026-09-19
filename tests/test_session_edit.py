from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import AppConfig
from app.dialogue.session_log import SessionLog
from app.dialogue.transcript import Transcript
from app.prompts.base import SceneSpec
from app.serve import create_app
from app.service import CoachService


def test_session_records_patch_and_delete(tmp_path: Path):
    cfg = AppConfig(
        session_dir=str(tmp_path / "sessions"),
        items_dir=str(tmp_path / "items"),
        db_path=str(tmp_path / "coach.sqlite"),
        llm_api_key="test",
        avatar_enabled=False,
    )
    service = CoachService(cfg)
    log = SessionLog(cfg.session_dir, "sess-edit", mode="text")
    transcript = Transcript()
    log.append_utterance(transcript.add("user", "I go store", source="user", phase="practice"))
    log.append_scene(SceneSpec(raw="cafe", scene="cafe", user_role="customer"))
    log.close()
    items = Path(cfg.items_dir)
    items.mkdir(parents=True)
    (items / "sess-edit.json").write_text("{}", encoding="utf-8")

    app = create_app(service)
    with TestClient(app) as client:
        listed = client.get("/v1/sessions")
        assert any(row["session_id"] == "sess-edit" for row in listed.json()["sessions"])
        recs = client.get("/v1/sessions/sess-edit/records")
        assert recs.status_code == 200
        utterance = next(row for row in recs.json()["records"] if row["type"] == "utterance")
        patched = client.patch(
            f"/v1/sessions/sess-edit/records/{utterance['seq']}",
            json={"text": "I went to the store"},
        )
        assert patched.status_code == 200
        assert patched.json()["text"] == "I went to the store"
        scene = next(row for row in recs.json()["records"] if row["type"] == "scene")
        scene_patch = client.patch(
            f"/v1/sessions/sess-edit/records/{scene['seq']}",
            json={"scene": {"scene": "bookstore"}},
        )
        assert scene_patch.json()["scene"]["scene"] == "bookstore"
        deleted = client.delete("/v1/sessions/sess-edit")
        assert deleted.status_code == 200
        assert client.get("/v1/sessions/sess-edit").status_code == 404
        assert not (items / "sess-edit.json").exists()


def test_list_sessions_newest_first(tmp_path: Path):
    cfg = AppConfig(
        session_dir=str(tmp_path / "sessions"),
        items_dir=str(tmp_path / "items"),
        db_path=str(tmp_path / "coach.sqlite"),
        llm_api_key="test",
        avatar_enabled=False,
    )
    older = SessionLog(cfg.session_dir, "old-sess", mode="text")
    older.close()
    newer = SessionLog(cfg.session_dir, "new-sess", mode="web")
    newer.close()
    service = CoachService(cfg)
    listed = service.list_sessions()
    ids = [row["session_id"] for row in listed]
    assert ids[0] == "new-sess"
    assert ids[-1] == "old-sess"
