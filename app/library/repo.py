from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.store.models import ItemCase, Mcq, ReviewState, SentencePair, utcnow


class LibraryNotFoundError(KeyError):
    pass


def iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def case_dict(row: ItemCase, *, mcqs: list[Mcq] | None = None, pairs: list[SentencePair] | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": row.id,
        "owner_id": row.owner_id,
        "source_session_id": row.source_session_id,
        "scenario": row.scenario,
        "source_seq": row.source_seq,
        "raw_other": row.raw_other,
        "raw_user": row.raw_user,
        "note": row.note,
        "created_at": iso(row.created_at),
        "deleted_at": iso(row.deleted_at),
    }
    if mcqs is not None:
        data["mcqs"] = [mcq_dict(item) for item in mcqs]
    if pairs is not None:
        data["sentence_pairs"] = [pair_dict(item) for item in pairs]
    return data


def mcq_dict(row: Mcq) -> dict[str, Any]:
    return {
        "id": row.id,
        "case_id": row.case_id,
        "category": row.category,
        "stem": row.stem,
        "correct_answer": row.correct_answer,
        "original_distractor": row.original_distractor,
        "distractor_1": row.distractor_1,
        "distractor_2": row.distractor_2,
        "analysis": row.analysis,
        "created_at": iso(row.created_at),
        "deleted_at": iso(row.deleted_at),
    }


def pair_dict(row: SentencePair) -> dict[str, Any]:
    return {
        "id": row.id,
        "case_id": row.case_id,
        "original_sentence": row.original_sentence,
        "polished_sentence": row.polished_sentence,
        "analysis": row.analysis,
        "created_at": iso(row.created_at),
        "deleted_at": iso(row.deleted_at),
    }


def review_dict(row: ReviewState) -> dict[str, Any]:
    return {
        "pair_id": row.pair_id,
        "due_at": iso(row.due_at),
        "interval_days": row.interval_days,
        "ease": row.ease,
        "reps": row.reps,
        "last_grade": row.last_grade,
    }


class LibraryRepo:
    def __init__(self, owner_id: str):
        self.owner_id = owner_id

    async def _get_case(self, session: AsyncSession, case_id: str, *, include_deleted: bool = False) -> ItemCase:
        stmt = select(ItemCase).where(ItemCase.owner_id == self.owner_id, ItemCase.id == case_id)
        if not include_deleted:
            stmt = stmt.where(ItemCase.deleted_at.is_(None))
        row = (await session.execute(stmt)).scalar_one_or_none()
        if row is None:
            raise LibraryNotFoundError(case_id)
        return row

    async def _get_mcq(self, session: AsyncSession, mcq_id: str, *, include_deleted: bool = False) -> Mcq:
        stmt = select(Mcq).where(Mcq.owner_id == self.owner_id, Mcq.id == mcq_id)
        if not include_deleted:
            stmt = stmt.where(Mcq.deleted_at.is_(None))
        row = (await session.execute(stmt)).scalar_one_or_none()
        if row is None:
            raise LibraryNotFoundError(mcq_id)
        return row

    async def _get_pair(self, session: AsyncSession, pair_id: str, *, include_deleted: bool = False) -> SentencePair:
        stmt = select(SentencePair).where(
            SentencePair.owner_id == self.owner_id, SentencePair.id == pair_id
        )
        if not include_deleted:
            stmt = stmt.where(SentencePair.deleted_at.is_(None))
        row = (await session.execute(stmt)).scalar_one_or_none()
        if row is None:
            raise LibraryNotFoundError(pair_id)
        return row

    async def list_cases(
        self,
        session: AsyncSession,
        *,
        source_session_id: Optional[str] = None,
    ) -> list[ItemCase]:
        stmt = select(ItemCase).where(
            ItemCase.owner_id == self.owner_id, ItemCase.deleted_at.is_(None)
        )
        if source_session_id:
            stmt = stmt.where(ItemCase.source_session_id == source_session_id)
        stmt = stmt.order_by(ItemCase.source_seq, ItemCase.id)
        return list((await session.execute(stmt)).scalars().all())

    async def get_case(self, session: AsyncSession, case_id: str) -> tuple[ItemCase, list[Mcq], list[SentencePair]]:
        row = await self._get_case(session, case_id)
        mcqs = list(
            (
                await session.execute(
                    select(Mcq)
                    .where(
                        Mcq.owner_id == self.owner_id,
                        Mcq.case_id == case_id,
                        Mcq.deleted_at.is_(None),
                    )
                    .order_by(Mcq.id)
                )
            )
            .scalars()
            .all()
        )
        pairs = list(
            (
                await session.execute(
                    select(SentencePair)
                    .where(
                        SentencePair.owner_id == self.owner_id,
                        SentencePair.case_id == case_id,
                        SentencePair.deleted_at.is_(None),
                    )
                    .order_by(SentencePair.id)
                )
            )
            .scalars()
            .all()
        )
        return row, mcqs, pairs

    async def create_case(
        self,
        session: AsyncSession,
        *,
        case_id: str,
        source_session_id: Optional[str] = None,
        scenario: str = "",
        source_seq: int = 0,
        raw_other: str = "",
        raw_user: str = "",
        note: str = "",
    ) -> ItemCase:
        row = ItemCase(
            id=case_id,
            owner_id=self.owner_id,
            source_session_id=source_session_id,
            scenario=scenario,
            source_seq=source_seq,
            raw_other=raw_other,
            raw_user=raw_user,
            note=note,
        )
        session.add(row)
        await session.flush()
        return row

    async def patch_case(self, session: AsyncSession, case_id: str, fields: dict[str, Any]) -> ItemCase:
        row = await self._get_case(session, case_id)
        for key, value in fields.items():
            setattr(row, key, value)
        await session.flush()
        return row

    async def soft_delete_case(self, session: AsyncSession, case_id: str) -> ItemCase:
        row = await self._get_case(session, case_id)
        now = utcnow()
        row.deleted_at = now
        mcqs = (
            await session.execute(
                select(Mcq).where(
                    Mcq.owner_id == self.owner_id,
                    Mcq.case_id == case_id,
                    Mcq.deleted_at.is_(None),
                )
            )
        ).scalars()
        for mcq in mcqs:
            mcq.deleted_at = now
        pairs = (
            await session.execute(
                select(SentencePair).where(
                    SentencePair.owner_id == self.owner_id,
                    SentencePair.case_id == case_id,
                    SentencePair.deleted_at.is_(None),
                )
            )
        ).scalars()
        for pair in pairs:
            pair.deleted_at = now
        await session.flush()
        return row

    async def create_mcq(
        self,
        session: AsyncSession,
        *,
        mcq_id: str,
        case_id: str,
        category: str = "",
        stem: str = "",
        correct_answer: str = "",
        original_distractor: str = "",
        distractor_1: str = "",
        distractor_2: str = "",
        analysis: str = "",
    ) -> Mcq:
        await self._get_case(session, case_id)
        row = Mcq(
            id=mcq_id,
            owner_id=self.owner_id,
            case_id=case_id,
            category=category,
            stem=stem,
            correct_answer=correct_answer,
            original_distractor=original_distractor,
            distractor_1=distractor_1,
            distractor_2=distractor_2,
            analysis=analysis,
        )
        session.add(row)
        await session.flush()
        return row

    async def get_mcq(self, session: AsyncSession, mcq_id: str) -> Mcq:
        return await self._get_mcq(session, mcq_id)

    async def patch_mcq(self, session: AsyncSession, mcq_id: str, fields: dict[str, Any]) -> Mcq:
        row = await self._get_mcq(session, mcq_id)
        for key, value in fields.items():
            setattr(row, key, value)
        await session.flush()
        return row

    async def soft_delete_mcq(self, session: AsyncSession, mcq_id: str) -> Mcq:
        row = await self._get_mcq(session, mcq_id)
        row.deleted_at = utcnow()
        await session.flush()
        return row

    async def create_pair(
        self,
        session: AsyncSession,
        *,
        pair_id: str,
        case_id: str,
        original_sentence: str = "",
        polished_sentence: str = "",
        analysis: str = "",
    ) -> SentencePair:
        await self._get_case(session, case_id)
        row = SentencePair(
            id=pair_id,
            owner_id=self.owner_id,
            case_id=case_id,
            original_sentence=original_sentence,
            polished_sentence=polished_sentence,
            analysis=analysis,
        )
        session.add(row)
        session.add(
            ReviewState(
                owner_id=self.owner_id,
                pair_id=pair_id,
                due_at=utcnow(),
                interval_days=0.0,
                ease=2.5,
                reps=0,
            )
        )
        await session.flush()
        return row

    async def get_pair(self, session: AsyncSession, pair_id: str) -> SentencePair:
        return await self._get_pair(session, pair_id)

    async def patch_pair(self, session: AsyncSession, pair_id: str, fields: dict[str, Any]) -> SentencePair:
        row = await self._get_pair(session, pair_id)
        for key, value in fields.items():
            setattr(row, key, value)
        await session.flush()
        return row

    async def soft_delete_pair(self, session: AsyncSession, pair_id: str) -> SentencePair:
        row = await self._get_pair(session, pair_id)
        row.deleted_at = utcnow()
        await session.flush()
        return row
