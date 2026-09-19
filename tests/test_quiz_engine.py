from __future__ import annotations

import random
from pathlib import Path

from app.library.ingest import ingest_payload
from app.library.repo import LibraryRepo
from app.quiz.engine import QuizEngine, QuizError
from app.store.db import Store
from tests.test_library_ingest import SAMPLE_ITEMS, make_store


async def test_quiz_shuffles_options_and_stores_them(tmp_path: Path):
    store = await make_store(tmp_path)
    try:
        await ingest_payload(store, SAMPLE_ITEMS)
        engine = QuizEngine(store, rng=random.Random(0))
        quiz = await engine.start(source="all", count=1)
        assert quiz["total"] == 1
        item = quiz["items"][0]
        expected = {
            "get the project back on track",
            "catch up the project progress",
            "chase the delivery schedule",
            "speed up the timeline gap",
        }
        if item["mcq_id"] == "case_sess-1_2_q2":
            expected = {"caught up", "catch up", "catched up", "catching up"}
        assert set(item["options"]) == expected
        assert "correct_answer" not in item
        again = await engine.get(quiz["id"])
        assert again["items"][0]["options"] == item["options"]
        repo = LibraryRepo(store.owner_id)
        async with store.session() as session:
            mcq = await repo.get_mcq(session, item["mcq_id"])
            assert mcq.correct_answer in expected
    finally:
        await store.dispose()


async def test_quiz_grades_and_wrong_pool(tmp_path: Path):
    store = await make_store(tmp_path)
    try:
        await ingest_payload(store, SAMPLE_ITEMS)
        engine = QuizEngine(store, rng=random.Random(1))
        quiz = await engine.start(source="all", count=2)
        first = quiz["items"][0]
        repo = LibraryRepo(store.owner_id)
        async with store.session() as session:
            mcq = await repo.get_mcq(session, first["mcq_id"])
            correct = mcq.correct_answer
        wrong = next(opt for opt in first["options"] if opt != correct)
        result = await engine.answer(quiz["id"], mcq_id=first["mcq_id"], selected=wrong, elapsed_ms=120)
        assert result["last"]["correct"] is False
        assert result["items"][0]["answered"] is True
        async with store.session() as session:
            mcq = await repo.get_mcq(session, first["mcq_id"])
            assert mcq.correct_answer == correct

        wrong_quiz = await engine.start(source="wrong", count=10)
        assert [item["mcq_id"] for item in wrong_quiz["items"]] == [first["mcq_id"]]

        right = await engine.answer(
            wrong_quiz["id"],
            mcq_id=first["mcq_id"],
            selected=correct,
            elapsed_ms=10,
        )
        assert right["last"]["correct"] is True
        empty = await engine.start(source="wrong", count=10)
        assert empty["items"] == []

        try:
            await engine.answer(quiz["id"], mcq_id=first["mcq_id"], selected=wrong)
            assert False, "expected already answered"
        except QuizError:
            pass
    finally:
        await store.dispose()
