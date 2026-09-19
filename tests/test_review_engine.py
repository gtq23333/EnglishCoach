from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from app.library.ingest import ingest_payload
from app.review.engine import ReviewEngine
from tests.test_library_ingest import SAMPLE_ITEMS, make_store


async def test_review_again_stays_due_good_advances(tmp_path: Path):
    store = await make_store(tmp_path)
    now = datetime(2026, 9, 19, 12, 0, 0)
    try:
        await ingest_payload(store, SAMPLE_ITEMS)
        engine = ReviewEngine(store, now=lambda: now)
        due = await engine.due()
        assert [card["id"] for card in due] == ["case_sess-1_2_s1"]
        assert due[0]["review"]["interval_days"] == 0.0

        again = await engine.grade("case_sess-1_2_s1", "again")
        assert again["review"]["last_grade"] == "again"
        assert again["review"]["interval_days"] == 0.0
        still_due = await engine.due()
        assert [card["id"] for card in still_due] == ["case_sess-1_2_s1"]

        good = await engine.grade("case_sess-1_2_s1", "good")
        assert good["review"]["last_grade"] == "good"
        assert good["review"]["interval_days"] == 1.0
        due_at = datetime.fromisoformat(good["review"]["due_at"].replace("Z", "+00:00")).replace(tzinfo=None)
        assert due_at == now + timedelta(days=1)
        assert await engine.due() == []

        later = ReviewEngine(store, now=lambda: now + timedelta(days=1, seconds=1))
        due_later = await later.due()
        assert [card["id"] for card in due_later] == ["case_sess-1_2_s1"]
    finally:
        await store.dispose()
