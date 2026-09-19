from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class ItemCase(Base):
    __tablename__ = "item_cases"
    __table_args__ = (UniqueConstraint("owner_id", "id", name="uq_case_owner_id"),)

    pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(128), index=True)
    owner_id: Mapped[str] = mapped_column(String(64), index=True)
    source_session_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    scenario: Mapped[str] = mapped_column(Text, default="")
    source_seq: Mapped[int] = mapped_column(Integer, default=0)
    raw_other: Mapped[str] = mapped_column(Text, default="")
    raw_user: Mapped[str] = mapped_column(Text, default="")
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class Mcq(Base):
    __tablename__ = "mcqs"
    __table_args__ = (UniqueConstraint("owner_id", "id", name="uq_mcq_owner_id"),)

    pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(128), index=True)
    owner_id: Mapped[str] = mapped_column(String(64), index=True)
    case_id: Mapped[str] = mapped_column(String(128), index=True)
    category: Mapped[str] = mapped_column(String(128), default="")
    stem: Mapped[str] = mapped_column(Text, default="")
    correct_answer: Mapped[str] = mapped_column(Text, default="")
    original_distractor: Mapped[str] = mapped_column(Text, default="")
    distractor_1: Mapped[str] = mapped_column(Text, default="")
    distractor_2: Mapped[str] = mapped_column(Text, default="")
    analysis: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class SentencePair(Base):
    __tablename__ = "sentence_pairs"
    __table_args__ = (UniqueConstraint("owner_id", "id", name="uq_pair_owner_id"),)

    pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(128), index=True)
    owner_id: Mapped[str] = mapped_column(String(64), index=True)
    case_id: Mapped[str] = mapped_column(String(128), index=True)
    original_sentence: Mapped[str] = mapped_column(Text, default="")
    polished_sentence: Mapped[str] = mapped_column(Text, default="")
    analysis: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class QuizRun(Base):
    __tablename__ = "quiz_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(64), index=True)
    source: Mapped[str] = mapped_column(String(32), default="all")
    source_session_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class QuizItem(Base):
    __tablename__ = "quiz_items"

    pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    quiz_run_id: Mapped[str] = mapped_column(ForeignKey("quiz_runs.id"), index=True)
    mcq_id: Mapped[str] = mapped_column(String(128), index=True)
    position: Mapped[int] = mapped_column(Integer, default=0)
    options_json: Mapped[str] = mapped_column(Text, default="[]")
    answered: Mapped[int] = mapped_column(Integer, default=0)
    selected: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    correct: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class Attempt(Base):
    __tablename__ = "attempts"

    pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_id: Mapped[str] = mapped_column(String(64), index=True)
    mcq_id: Mapped[str] = mapped_column(String(128), index=True)
    quiz_run_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    selected: Mapped[str] = mapped_column(Text, default="")
    is_correct: Mapped[int] = mapped_column(Integer, default=0)
    elapsed_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ReviewState(Base):
    __tablename__ = "review_states"
    __table_args__ = (UniqueConstraint("owner_id", "pair_id", name="uq_review_owner_pair"),)

    pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_id: Mapped[str] = mapped_column(String(64), index=True)
    pair_id: Mapped[str] = mapped_column(String(128), index=True)
    due_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    interval_days: Mapped[float] = mapped_column(Float, default=0.0)
    ease: Mapped[float] = mapped_column(Float, default=2.5)
    reps: Mapped[int] = mapped_column(Integer, default=0)
    last_grade: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
