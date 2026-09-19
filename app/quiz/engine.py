from __future__ import annotations

import json
import random
import uuid
from typing import Any, Optional

from sqlalchemy import func, select

from app.library.repo import LibraryNotFoundError, iso
from app.store.db import Store
from app.store.models import Attempt, ItemCase, Mcq, QuizItem, QuizRun

VALID_SOURCES = {"all", "session", "wrong"}


class QuizError(RuntimeError):
    pass


class QuizEngine:
    def __init__(self, store: Store, rng: Optional[random.Random] = None):
        self.store = store
        self.rng = rng or random.Random()

    def _shuffle_options(self, mcq: Mcq) -> list[str]:
        options = [
            mcq.correct_answer,
            mcq.original_distractor,
            mcq.distractor_1,
            mcq.distractor_2,
        ]
        self.rng.shuffle(options)
        return options

    async def _pool(self, session, source: str, source_session_id: Optional[str]) -> list[Mcq]:
        base = select(Mcq).where(Mcq.owner_id == self.store.owner_id, Mcq.deleted_at.is_(None))
        if source == "all":
            stmt = base
        elif source == "session":
            if not source_session_id:
                raise QuizError("session_id is required when source=session")
            stmt = base.join(
                ItemCase,
                (ItemCase.owner_id == Mcq.owner_id) & (ItemCase.id == Mcq.case_id),
            ).where(
                ItemCase.source_session_id == source_session_id,
                ItemCase.deleted_at.is_(None),
            )
        elif source == "wrong":
            latest = (
                select(Attempt.mcq_id, func.max(Attempt.pk).label("max_pk"))
                .where(Attempt.owner_id == self.store.owner_id)
                .group_by(Attempt.mcq_id)
            ).subquery()
            stmt = (
                base.join(Attempt, Attempt.mcq_id == Mcq.id)
                .join(latest, latest.c.max_pk == Attempt.pk)
                .where(Attempt.owner_id == self.store.owner_id, Attempt.is_correct == 0)
            )
        else:
            raise QuizError(f"unknown source: {source}")
        return list((await session.execute(stmt.order_by(Mcq.id))).scalars().all())

    async def start(
        self,
        *,
        source: str = "all",
        count: int = 10,
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        source = (source or "all").strip().lower()
        if source not in VALID_SOURCES:
            raise QuizError(f"unknown source: {source}")
        count = max(1, int(count))
        run_id = str(uuid.uuid4())
        async with self.store.session() as session:
            pool = await self._pool(session, source, session_id)
            self.rng.shuffle(pool)
            chosen = pool[:count]
            session.add(
                QuizRun(
                    id=run_id,
                    owner_id=self.store.owner_id,
                    source=source,
                    source_session_id=session_id,
                )
            )
            await session.flush()
            for position, mcq in enumerate(chosen):
                session.add(
                    QuizItem(
                        quiz_run_id=run_id,
                        mcq_id=mcq.id,
                        position=position,
                        options_json=json.dumps(self._shuffle_options(mcq), ensure_ascii=False),
                    )
                )
            await session.commit()
        return await self.get(run_id)

    async def get(self, run_id: str) -> dict[str, Any]:
        async with self.store.session() as session:
            run = await session.get(QuizRun, run_id)
            if run is None or run.owner_id != self.store.owner_id:
                raise LibraryNotFoundError(run_id)
            items = list(
                (
                    await session.execute(
                        select(QuizItem)
                        .where(QuizItem.quiz_run_id == run_id)
                        .order_by(QuizItem.position)
                    )
                )
                .scalars()
                .all()
            )
            mcq_ids = [item.mcq_id for item in items]
            mcqs = {}
            if mcq_ids:
                rows = (
                    await session.execute(
                        select(Mcq).where(Mcq.owner_id == self.store.owner_id, Mcq.id.in_(mcq_ids))
                    )
                ).scalars()
                mcqs = {row.id: row for row in rows}
            payload_items = []
            correct_count = 0
            answered_count = 0
            for item in items:
                mcq = mcqs.get(item.mcq_id)
                options = json.loads(item.options_json or "[]")
                answered = bool(item.answered)
                entry: dict[str, Any] = {
                    "mcq_id": item.mcq_id,
                    "position": item.position,
                    "options": options,
                    "answered": answered,
                    "selected": item.selected,
                    "correct": bool(item.correct) if item.correct is not None else None,
                }
                if mcq is not None:
                    entry["stem"] = mcq.stem
                    entry["category"] = mcq.category
                    if answered:
                        entry["correct_answer"] = mcq.correct_answer
                        entry["analysis"] = mcq.analysis
                if answered:
                    answered_count += 1
                    if item.correct:
                        correct_count += 1
                payload_items.append(entry)
            return {
                "id": run.id,
                "source": run.source,
                "source_session_id": run.source_session_id,
                "created_at": iso(run.created_at),
                "items": payload_items,
                "answered": answered_count,
                "correct": correct_count,
                "total": len(payload_items),
            }

    async def answer(
        self,
        run_id: str,
        *,
        mcq_id: str,
        selected: str,
        elapsed_ms: int = 0,
    ) -> dict[str, Any]:
        async with self.store.session() as session:
            run = await session.get(QuizRun, run_id)
            if run is None or run.owner_id != self.store.owner_id:
                raise LibraryNotFoundError(run_id)
            item = (
                await session.execute(
                    select(QuizItem).where(
                        QuizItem.quiz_run_id == run_id, QuizItem.mcq_id == mcq_id
                    )
                )
            ).scalar_one_or_none()
            if item is None:
                raise LibraryNotFoundError(mcq_id)
            if item.answered:
                raise QuizError("already answered")
            options = json.loads(item.options_json or "[]")
            if selected not in options:
                raise QuizError("selected option is not in this quiz item")
            mcq = (
                await session.execute(
                    select(Mcq).where(Mcq.owner_id == self.store.owner_id, Mcq.id == mcq_id)
                )
            ).scalar_one_or_none()
            if mcq is None:
                raise LibraryNotFoundError(mcq_id)
            is_correct = 1 if selected == mcq.correct_answer else 0
            item.answered = 1
            item.selected = selected
            item.correct = is_correct
            session.add(
                Attempt(
                    owner_id=self.store.owner_id,
                    mcq_id=mcq_id,
                    quiz_run_id=run_id,
                    selected=selected,
                    is_correct=is_correct,
                    elapsed_ms=max(0, int(elapsed_ms)),
                )
            )
            await session.commit()
        result = await self.get(run_id)
        result["last"] = {
            "mcq_id": mcq_id,
            "selected": selected,
            "correct": bool(is_correct),
            "correct_answer": mcq.correct_answer,
            "analysis": mcq.analysis,
        }
        return result
