from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from sqlalchemy import select

from app.library.repo import LibraryNotFoundError, pair_dict, review_dict
from app.store.db import Store
from app.store.models import ReviewState, SentencePair, utcnow

VALID_GRADES = {"again", "good", "easy"}


class ReviewError(RuntimeError):
    pass


class ReviewEngine:
    def __init__(self, store: Store, now: Optional[Callable[[], datetime]] = None):
        self.store = store
        self._now_fn = now or utcnow

    def _now(self) -> datetime:
        value = self._now_fn()
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value

    def _apply_grade(self, state: ReviewState, grade: str, now: datetime) -> None:
        if grade == "again":
            state.interval_days = 0.0
            state.ease = max(1.3, (state.ease or 2.5) - 0.2)
            state.due_at = now
        elif grade == "good":
            if state.reps <= 0 or state.interval_days <= 0:
                state.interval_days = 1.0
            else:
                state.interval_days = max(1.0, state.interval_days * (state.ease or 2.5))
            state.due_at = now + timedelta(days=state.interval_days)
            state.reps = (state.reps or 0) + 1
        else:
            state.ease = min(3.0, (state.ease or 2.5) + 0.15)
            if state.reps <= 0 or state.interval_days <= 0:
                state.interval_days = 3.0
            else:
                state.interval_days = max(1.0, state.interval_days * state.ease * 1.3)
            state.due_at = now + timedelta(days=state.interval_days)
            state.reps = (state.reps or 0) + 1
        state.last_grade = grade

    async def due(self, *, limit: int = 50) -> list[dict[str, Any]]:
        now = self._now()
        async with self.store.session() as session:
            rows = list(
                (
                    await session.execute(
                        select(ReviewState, SentencePair)
                        .join(
                            SentencePair,
                            (SentencePair.owner_id == ReviewState.owner_id)
                            & (SentencePair.id == ReviewState.pair_id),
                        )
                        .where(
                            ReviewState.owner_id == self.store.owner_id,
                            SentencePair.deleted_at.is_(None),
                            ReviewState.due_at <= now,
                        )
                        .order_by(ReviewState.due_at, ReviewState.pair_id)
                        .limit(max(1, int(limit)))
                    )
                ).all()
            )
            cards = []
            for state, pair in rows:
                card = pair_dict(pair)
                card["review"] = review_dict(state)
                cards.append(card)
            return cards

    async def grade(self, pair_id: str, grade: str) -> dict[str, Any]:
        grade = (grade or "").strip().lower()
        if grade not in VALID_GRADES:
            raise ReviewError(f"unknown grade: {grade}")
        now = self._now()
        async with self.store.session() as session:
            pair = (
                await session.execute(
                    select(SentencePair).where(
                        SentencePair.owner_id == self.store.owner_id,
                        SentencePair.id == pair_id,
                        SentencePair.deleted_at.is_(None),
                    )
                )
            ).scalar_one_or_none()
            if pair is None:
                raise LibraryNotFoundError(pair_id)
            state = (
                await session.execute(
                    select(ReviewState).where(
                        ReviewState.owner_id == self.store.owner_id,
                        ReviewState.pair_id == pair_id,
                    )
                )
            ).scalar_one_or_none()
            if state is None:
                state = ReviewState(
                    owner_id=self.store.owner_id,
                    pair_id=pair_id,
                    due_at=now,
                    interval_days=0.0,
                    ease=2.5,
                    reps=0,
                )
                session.add(state)
                await session.flush()
            self._apply_grade(state, grade, now)
            await session.commit()
            await session.refresh(state)
            payload = pair_dict(pair)
            payload["review"] = review_dict(state)
            return payload
