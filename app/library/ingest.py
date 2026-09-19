from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.items.generator import load_items
from app.store.db import Store
from app.store.models import ItemCase, Mcq, ReviewState, SentencePair, utcnow


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _distractors(question: dict[str, Any]) -> tuple[str, str]:
    extra = question.get("additional_distractors") or []
    if not isinstance(extra, list):
        extra = []
    first = _text(extra[0]) if len(extra) > 0 else _text(question.get("distractor_1"))
    second = _text(extra[1]) if len(extra) > 1 else _text(question.get("distractor_2"))
    return first, second


async def ingest_payload(store: Store, payload: dict[str, Any]) -> dict[str, int]:
    session_id = _text(payload.get("session_id") or (payload.get("meta") or {}).get("session_id")) or None
    cases = payload.get("cases") or []
    inserted = {
        "cases": 0,
        "mcqs": 0,
        "pairs": 0,
        "skipped_cases": 0,
        "skipped_mcqs": 0,
        "skipped_pairs": 0,
    }

    async with store.session() as session:
        for case in cases:
            if not isinstance(case, dict):
                continue
            case_id = _text(case.get("case_id") or case.get("id"))
            if not case_id:
                continue
            existing_case = (
                await session.execute(
                    select(ItemCase.id).where(
                        ItemCase.owner_id == store.owner_id, ItemCase.id == case_id
                    )
                )
            ).scalar_one_or_none()
            if existing_case is None:
                raw = case.get("raw_interaction") if isinstance(case.get("raw_interaction"), dict) else {}
                session.add(
                    ItemCase(
                        id=case_id,
                        owner_id=store.owner_id,
                        source_session_id=session_id,
                        scenario=_text(case.get("scenario")),
                        source_seq=_int(case.get("source_seq")),
                        raw_other=_text(raw.get("speaker_other")),
                        raw_user=_text(raw.get("speaker_user")),
                        note=_text(case.get("context_adaptation_note") or case.get("note")),
                    )
                )
                inserted["cases"] += 1
            else:
                inserted["skipped_cases"] += 1

            for question in case.get("questions") or []:
                if not isinstance(question, dict):
                    continue
                question_id = _text(question.get("question_id") or question.get("id"))
                if not question_id:
                    continue
                existing_mcq = (
                    await session.execute(
                        select(Mcq.id).where(Mcq.owner_id == store.owner_id, Mcq.id == question_id)
                    )
                ).scalar_one_or_none()
                if existing_mcq is not None:
                    inserted["skipped_mcqs"] += 1
                    continue
                d1, d2 = _distractors(question)
                session.add(
                    Mcq(
                        id=question_id,
                        owner_id=store.owner_id,
                        case_id=case_id,
                        category=_text(question.get("category")),
                        stem=_text(question.get("stem")),
                        correct_answer=_text(question.get("correct_answer")),
                        original_distractor=_text(question.get("original_distractor")),
                        distractor_1=d1,
                        distractor_2=d2,
                        analysis=_text(question.get("analysis")),
                    )
                )
                inserted["mcqs"] += 1

            for pair in case.get("sentence_pairs") or []:
                if not isinstance(pair, dict):
                    continue
                pair_id = _text(pair.get("pair_id") or pair.get("id"))
                if not pair_id:
                    continue
                existing_pair = (
                    await session.execute(
                        select(SentencePair.id).where(
                            SentencePair.owner_id == store.owner_id, SentencePair.id == pair_id
                        )
                    )
                ).scalar_one_or_none()
                if existing_pair is not None:
                    inserted["skipped_pairs"] += 1
                    continue
                session.add(
                    SentencePair(
                        id=pair_id,
                        owner_id=store.owner_id,
                        case_id=case_id,
                        original_sentence=_text(pair.get("original_sentence")),
                        polished_sentence=_text(pair.get("polished_sentence")),
                        analysis=_text(pair.get("analysis")),
                    )
                )
                existing_review = (
                    await session.execute(
                        select(ReviewState.pk).where(
                            ReviewState.owner_id == store.owner_id,
                            ReviewState.pair_id == pair_id,
                        )
                    )
                ).scalar_one_or_none()
                if existing_review is None:
                    session.add(
                        ReviewState(
                            owner_id=store.owner_id,
                            pair_id=pair_id,
                            due_at=utcnow(),
                            interval_days=0.0,
                            ease=2.5,
                            reps=0,
                        )
                    )
                inserted["pairs"] += 1

        await session.commit()
    return inserted


async def ingest_session_json(
    store: Store,
    session_id: str,
    items_dir: str | Path,
) -> dict[str, Any]:
    payload = load_items(items_dir, session_id)
    if payload is None:
        raise FileNotFoundError(session_id)
    stats = await ingest_payload(store, payload)
    stats["session_id"] = session_id
    return stats
