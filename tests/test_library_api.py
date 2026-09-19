from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import AppConfig
from app.serve import create_app
from app.service import CoachService
from app.store.db import Store
from tests.test_library_ingest import SAMPLE_ITEMS


def _client(tmp_path: Path) -> tuple[TestClient, Store]:
    cfg = AppConfig(
        session_dir=str(tmp_path / "sessions"),
        items_dir=str(tmp_path / "items"),
        db_path=str(tmp_path / "coach.sqlite"),
        llm_api_key="test",
        owner_id="local",
    )
    store = Store.from_config(cfg)
    service = CoachService(cfg, store=store)
    app = create_app(service, store=store)
    return TestClient(app), store


def test_library_quiz_review_http(tmp_path: Path):
    items_dir = tmp_path / "items"
    items_dir.mkdir()
    (items_dir / "sess-1.json").write_text(
        json.dumps(SAMPLE_ITEMS, ensure_ascii=False), encoding="utf-8"
    )
    client, _store = _client(tmp_path)
    with client:
        missing = client.post("/v1/library/ingest/no-such")
        assert missing.status_code == 404

        ingested = client.post("/v1/library/ingest/sess-1")
        assert ingested.status_code == 200
        body = ingested.json()
        assert body["cases"] == 1
        assert body["mcqs"] == 2

        listed = client.get("/v1/library/cases", params={"source_session_id": "sess-1"})
        assert listed.status_code == 200
        assert listed.json()["cases"][0]["id"] == "case_sess-1_2"

        got = client.get("/v1/library/cases/case_sess-1_2")
        assert got.status_code == 200
        assert len(got.json()["mcqs"]) == 2
        assert len(got.json()["sentence_pairs"]) == 1

        patched = client.patch(
            "/v1/library/mcqs/case_sess-1_2_q1",
            json={"stem": "User: I will <blank> today."},
        )
        assert patched.status_code == 200
        assert patched.json()["stem"].endswith("today.")

        created = client.post("/v1/library/cases", json={"scenario": "manual", "raw_user": "I go"})
        assert created.status_code == 200
        case_id = created.json()["id"]
        mcq = client.post(
            f"/v1/library/cases/{case_id}/mcqs",
            json={
                "stem": "User: I <blank>.",
                "correct_answer": "went",
                "original_distractor": "go",
                "additional_distractors": ["goed", "gone"],
            },
        )
        assert mcq.status_code == 200
        pair = client.post(
            f"/v1/library/cases/{case_id}/pairs",
            json={
                "original_sentence": "I go",
                "polished_sentence": "I went",
                "analysis": "时态",
            },
        )
        assert pair.status_code == 200
        pair_id = pair.json()["id"]

        deleted = client.delete(f"/v1/library/mcqs/{mcq.json()['id']}")
        assert deleted.status_code == 200
        assert client.get(f"/v1/library/mcqs/{mcq.json()['id']}").status_code == 404

        quiz = client.post("/v1/quizzes", json={"source": "session", "session_id": "sess-1", "count": 2})
        assert quiz.status_code == 200
        quiz_body = quiz.json()
        assert quiz_body["total"] == 2
        item = quiz_body["items"][0]
        fetched = client.get(f"/v1/quizzes/{quiz_body['id']}")
        assert fetched.json()["items"][0]["options"] == item["options"]

        detail = client.get(f"/v1/library/mcqs/{item['mcq_id']}")
        correct = detail.json()["correct_answer"]
        wrong = next(opt for opt in item["options"] if opt != correct)
        answered = client.post(
            f"/v1/quizzes/{quiz_body['id']}/answers",
            json={"mcq_id": item["mcq_id"], "selected": wrong, "elapsed_ms": 40},
        )
        assert answered.status_code == 200
        assert answered.json()["last"]["correct"] is False

        due = client.get("/v1/reviews/due")
        ids = {card["id"] for card in due.json()["cards"]}
        assert "case_sess-1_2_s1" in ids
        assert pair_id in ids
        graded = client.post(f"/v1/reviews/{pair_id}/grade", json={"grade": "good"})
        assert graded.status_code == 200
        assert graded.json()["review"]["interval_days"] == 1.0
        due_after = client.get("/v1/reviews/due")
        assert pair_id not in {card["id"] for card in due_after.json()["cards"]}
