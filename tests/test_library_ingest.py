from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.config import AppConfig
from app.library.ingest import ingest_payload, ingest_session_json
from app.library.repo import LibraryRepo
from app.service import CoachService
from app.store.db import Store
from tests.test_itemgen import GoodLLM, _write_session


SAMPLE_ITEMS = {
    "session_id": "sess-1",
    "cases": [
        {
            "case_id": "case_sess-1_2",
            "scenario": "interview",
            "source_seq": 2,
            "raw_interaction": {
                "speaker_other": "How do you handle delays?",
                "speaker_user": "I will catch up the project progress.",
            },
            "context_adaptation_note": "未改写",
            "questions": [
                {
                    "question_id": "case_sess-1_2_q1",
                    "category": "Idiomatic Collocation",
                    "stem": "User: I will <blank>.",
                    "correct_answer": "get the project back on track",
                    "original_distractor": "catch up the project progress",
                    "additional_distractors": [
                        "chase the delivery schedule",
                        "speed up the timeline gap",
                    ],
                    "analysis": "中式搭配",
                },
                {
                    "question_id": "case_sess-1_2_q2",
                    "category": "Grammar",
                    "stem": "User: I <blank> yesterday.",
                    "correct_answer": "caught up",
                    "original_distractor": "catch up",
                    "additional_distractors": ["catched up", "catching up"],
                    "analysis": "时态",
                },
            ],
            "sentence_pairs": [
                {
                    "pair_id": "case_sess-1_2_s1",
                    "original_sentence": "I will catch up the project progress.",
                    "polished_sentence": "I will get the project back on track.",
                    "analysis": "用地道块替换",
                }
            ],
        }
    ],
}


async def make_store(tmp_path: Path) -> Store:
    store = Store(tmp_path / "coach.sqlite", owner_id="local")
    await store.create_tables()
    return store


async def test_ingest_splits_cases_mcqs_and_pairs(tmp_path: Path):
    store = await make_store(tmp_path)
    try:
        stats = await ingest_payload(store, SAMPLE_ITEMS)
        assert stats["cases"] == 1
        assert stats["mcqs"] == 2
        assert stats["pairs"] == 1
        assert stats["skipped_cases"] == 0
        repo = LibraryRepo(store.owner_id)
        async with store.session() as session:
            row, mcqs, pairs = await repo.get_case(session, "case_sess-1_2")
            assert row.source_session_id == "sess-1"
            assert row.raw_user == "I will catch up the project progress."
            assert {item.id for item in mcqs} == {"case_sess-1_2_q1", "case_sess-1_2_q2"}
            assert mcqs[0].distractor_1 == "chase the delivery schedule"
            assert pairs[0].polished_sentence == "I will get the project back on track."
    finally:
        await store.dispose()


async def test_second_ingest_does_not_overwrite(tmp_path: Path):
    store = await make_store(tmp_path)
    try:
        await ingest_payload(store, SAMPLE_ITEMS)
        mutated = {
            "session_id": "sess-1",
            "cases": [
                {
                    "case_id": "case_sess-1_2",
                    "scenario": "SHOULD_NOT_WRITE",
                    "source_seq": 99,
                    "raw_interaction": {"speaker_other": "x", "speaker_user": "y"},
                    "questions": [
                        {
                            "question_id": "case_sess-1_2_q1",
                            "stem": "CHANGED STEM",
                            "correct_answer": "CHANGED",
                            "original_distractor": "CHANGED",
                            "additional_distractors": ["a", "b"],
                            "analysis": "CHANGED",
                        }
                    ],
                    "sentence_pairs": [
                        {
                            "pair_id": "case_sess-1_2_s1",
                            "original_sentence": "CHANGED",
                            "polished_sentence": "CHANGED",
                            "analysis": "CHANGED",
                        }
                    ],
                }
            ],
        }
        stats = await ingest_payload(store, mutated)
        assert stats["skipped_cases"] == 1
        assert stats["skipped_mcqs"] == 1
        assert stats["skipped_pairs"] == 1
        assert stats["cases"] == 0
        repo = LibraryRepo(store.owner_id)
        async with store.session() as session:
            row, mcqs, pairs = await repo.get_case(session, "case_sess-1_2")
            assert row.scenario == "interview"
            assert row.source_seq == 2
            q1 = next(item for item in mcqs if item.id == "case_sess-1_2_q1")
            assert q1.stem == "User: I will <blank>."
            assert q1.correct_answer == "get the project back on track"
            assert pairs[0].original_sentence == "I will catch up the project progress."
    finally:
        await store.dispose()


async def test_ingest_session_json_and_soft_delete(tmp_path: Path):
    items_dir = tmp_path / "items"
    items_dir.mkdir()
    (items_dir / "sess-1.json").write_text(
        json.dumps(SAMPLE_ITEMS, ensure_ascii=False),
        encoding="utf-8",
    )
    store = await make_store(tmp_path)
    try:
        stats = await ingest_session_json(store, "sess-1", items_dir)
        assert stats["session_id"] == "sess-1"
        assert stats["mcqs"] == 2
        repo = LibraryRepo(store.owner_id)
        async with store.session() as session:
            await repo.patch_mcq(session, "case_sess-1_2_q1", {"stem": "User: I will <blank> now."})
            await repo.soft_delete_mcq(session, "case_sess-1_2_q2")
            await session.commit()
            _, mcqs, _ = await repo.get_case(session, "case_sess-1_2")
            assert [item.id for item in mcqs] == ["case_sess-1_2_q1"]
            assert mcqs[0].stem.endswith("now.")
    finally:
        await store.dispose()


async def test_generate_items_auto_ingests(tmp_path: Path):
    _write_session(tmp_path, "s-good", "I will catch up the project progress.")
    store = await make_store(tmp_path)
    try:
        cfg = AppConfig(
            session_dir=str(tmp_path),
            items_dir=str(tmp_path / "items"),
            db_path=str(tmp_path / "coach.sqlite"),
            llm_api_key="test",
        )
        service = CoachService(cfg, llm=GoodLLM(), store=store)
        job = await service.generate_items("s-good")
        info = None
        for _ in range(80):
            info = service.get_job(job["job_id"])
            if info.get("status") != "running":
                break
            await asyncio.sleep(0.05)
        assert info is not None
        assert info["status"] == "done"
        assert info["ingested"]["cases"] == 1
        assert info["ingested"]["mcqs"] == 1
        repo = LibraryRepo(store.owner_id)
        async with store.session() as session:
            row, mcqs, _pairs = await repo.get_case(session, "case_s_2")
            assert row.source_session_id == "s-good"
            assert mcqs[0].original_distractor == "catch up the project progress"
    finally:
        await store.dispose()


class BoomLLM:
    async def chat(self, messages, tools=None, response_format=None):
        raise RuntimeError("connection timeout")


async def test_generate_items_llm_failure_is_error(tmp_path: Path):
    _write_session(tmp_path, "s-boom", "I will catch up the project progress.")
    store = await make_store(tmp_path)
    try:
        cfg = AppConfig(
            session_dir=str(tmp_path),
            items_dir=str(tmp_path / "items"),
            db_path=str(tmp_path / "coach.sqlite"),
            llm_api_key="test",
        )
        service = CoachService(cfg, llm=BoomLLM(), store=store)
        job = await service.generate_items("s-boom")
        info = None
        for _ in range(80):
            info = service.get_job(job["job_id"])
            if info.get("status") != "running":
                break
            await asyncio.sleep(0.05)
        assert info is not None
        assert info["status"] == "error"
        assert "timeout" in (info.get("error") or "")
    finally:
        await store.dispose()
