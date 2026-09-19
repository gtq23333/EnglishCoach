from __future__ import annotations

import argparse
import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError

from app.config import load_config
from app.library.ingest import ingest_session_json
from app.library.repo import (
    LibraryNotFoundError,
    LibraryRepo,
    case_dict,
    mcq_dict,
    pair_dict,
)
from app.quiz.engine import QuizEngine, QuizError
from app.review.engine import ReviewEngine, ReviewError
from app.service import (
    CoachService,
    SessionBusyError,
    SessionModeError,
    SessionNotFoundError,
)
from app.store.db import Store


class CreateSessionBody(BaseModel):
    mode: str
    scene: Optional[str] = None


class TextBody(BaseModel):
    text: str


class GenerateItemsBody(BaseModel):
    max_user_turns_per_slice: int = 2
    char_budget: int = 800
    parse_retries: Optional[int] = None


class CaseCreateBody(BaseModel):
    id: Optional[str] = None
    source_session_id: Optional[str] = None
    scenario: str = ""
    source_seq: int = 0
    raw_other: str = ""
    raw_user: str = ""
    note: str = ""


class CasePatchBody(BaseModel):
    scenario: Optional[str] = None
    source_seq: Optional[int] = None
    raw_other: Optional[str] = None
    raw_user: Optional[str] = None
    note: Optional[str] = None
    source_session_id: Optional[str] = None


class McqCreateBody(BaseModel):
    id: Optional[str] = None
    category: str = ""
    stem: str = ""
    correct_answer: str = ""
    original_distractor: str = ""
    distractor_1: str = ""
    distractor_2: str = ""
    additional_distractors: Optional[list[str]] = None
    analysis: str = ""


class McqPatchBody(BaseModel):
    category: Optional[str] = None
    stem: Optional[str] = None
    correct_answer: Optional[str] = None
    original_distractor: Optional[str] = None
    distractor_1: Optional[str] = None
    distractor_2: Optional[str] = None
    analysis: Optional[str] = None


class PairCreateBody(BaseModel):
    id: Optional[str] = None
    original_sentence: str = ""
    polished_sentence: str = ""
    analysis: str = ""


class PairPatchBody(BaseModel):
    original_sentence: Optional[str] = None
    polished_sentence: Optional[str] = None
    analysis: Optional[str] = None


class CreateQuizBody(BaseModel):
    source: str = "all"
    count: int = 10
    session_id: Optional[str] = None


class QuizAnswerBody(BaseModel):
    mcq_id: str
    selected: str
    elapsed_ms: int = 0


class ReviewGradeBody(BaseModel):
    grade: str


class RecordPatchBody(BaseModel):
    text: Optional[str] = None
    scene: Optional[dict] = None


WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"


def _http_library(exc: Exception) -> HTTPException:
    if isinstance(exc, LibraryNotFoundError):
        return HTTPException(status_code=404, detail=str(exc) or "not found")
    if isinstance(exc, (QuizError, ReviewError)):
        message = str(exc)
        code = 409 if "already answered" in message else 400
        return HTTPException(status_code=code, detail=message)
    if isinstance(exc, IntegrityError):
        return HTTPException(status_code=409, detail="duplicate id")
    raise exc


def _mcq_distractors(body: McqCreateBody) -> tuple[str, str]:
    extra = body.additional_distractors or []
    first = extra[0] if len(extra) > 0 else body.distractor_1
    second = extra[1] if len(extra) > 1 else body.distractor_2
    return first or "", second or ""


def create_app(service: CoachService, store: Store | None = None) -> FastAPI:
    store = store or service.store or Store.from_config(service.cfg)
    if service.store is None:
        service.store = store
    repo = LibraryRepo(store.owner_id)
    quiz = QuizEngine(store)
    review = ReviewEngine(store)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await store.create_tables()
        yield
        await store.dispose()

    app = FastAPI(title="English Coach", version="1.0", lifespan=lifespan)
    app.state.store = store
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/v1/sessions")
    async def create_session(body: CreateSessionBody):
        try:
            return await service.create_session(body.mode, scene=body.scene)
        except SessionBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except SessionModeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/sessions")
    async def list_sessions():
        return {"sessions": service.list_sessions()}

    @app.get("/v1/sessions/{session_id}")
    async def get_session(session_id: str):
        try:
            return service.get_session(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/sessions/{session_id}/stop")
    async def stop_session(session_id: str):
        return await service.stop_session(session_id)

    @app.post("/v1/sessions/{session_id}/text")
    async def send_text(session_id: str, body: TextBody):
        try:
            return await service.send_text(session_id, body.text)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SessionModeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/sessions/{session_id}/items/generate")
    async def generate_items(
        session_id: str, body: Optional[GenerateItemsBody] = None
    ):
        payload = body or GenerateItemsBody()
        try:
            return await service.generate_items(
                session_id,
                max_user_turns_per_slice=payload.max_user_turns_per_slice,
                char_budget=payload.char_budget,
                parse_retries=payload.parse_retries,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/jobs")
    async def list_jobs():
        return {"jobs": service.list_jobs()}

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str):
        try:
            return service.get_job(job_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/sessions/{session_id}/items")
    async def get_items(session_id: str):
        try:
            return service.get_items(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/library/cases")
    async def list_cases(source_session_id: Optional[str] = None):
        async with store.session() as session:
            rows = await repo.list_cases(session, source_session_id=source_session_id)
            return {"cases": [case_dict(row) for row in rows]}

    @app.post("/v1/library/cases")
    async def create_case(body: CaseCreateBody):
        case_id = body.id or str(uuid.uuid4())
        try:
            async with store.session() as session:
                row = await repo.create_case(
                    session,
                    case_id=case_id,
                    source_session_id=body.source_session_id,
                    scenario=body.scenario,
                    source_seq=body.source_seq,
                    raw_other=body.raw_other,
                    raw_user=body.raw_user,
                    note=body.note,
                )
                await session.commit()
                return case_dict(row)
        except Exception as exc:
            raise _http_library(exc) from exc

    @app.get("/v1/library/cases/{case_id}")
    async def get_case(case_id: str):
        try:
            async with store.session() as session:
                row, mcqs, pairs = await repo.get_case(session, case_id)
                return case_dict(row, mcqs=mcqs, pairs=pairs)
        except Exception as exc:
            raise _http_library(exc) from exc

    @app.patch("/v1/library/cases/{case_id}")
    async def patch_case(case_id: str, body: CasePatchBody):
        fields = body.model_dump(exclude_unset=True)
        try:
            async with store.session() as session:
                row = await repo.patch_case(session, case_id, fields)
                await session.commit()
                return case_dict(row)
        except Exception as exc:
            raise _http_library(exc) from exc

    @app.delete("/v1/library/cases/{case_id}")
    async def delete_case(case_id: str):
        try:
            async with store.session() as session:
                await repo.soft_delete_case(session, case_id)
                await session.commit()
                return {"ok": True, "id": case_id}
        except Exception as exc:
            raise _http_library(exc) from exc

    @app.post("/v1/library/cases/{case_id}/mcqs")
    async def create_mcq(case_id: str, body: McqCreateBody):
        mcq_id = body.id or str(uuid.uuid4())
        d1, d2 = _mcq_distractors(body)
        try:
            async with store.session() as session:
                row = await repo.create_mcq(
                    session,
                    mcq_id=mcq_id,
                    case_id=case_id,
                    category=body.category,
                    stem=body.stem,
                    correct_answer=body.correct_answer,
                    original_distractor=body.original_distractor,
                    distractor_1=d1,
                    distractor_2=d2,
                    analysis=body.analysis,
                )
                await session.commit()
                return mcq_dict(row)
        except Exception as exc:
            raise _http_library(exc) from exc

    @app.get("/v1/library/mcqs/{mcq_id}")
    async def get_mcq(mcq_id: str):
        try:
            async with store.session() as session:
                return mcq_dict(await repo.get_mcq(session, mcq_id))
        except Exception as exc:
            raise _http_library(exc) from exc

    @app.patch("/v1/library/mcqs/{mcq_id}")
    async def patch_mcq(mcq_id: str, body: McqPatchBody):
        fields = body.model_dump(exclude_unset=True)
        try:
            async with store.session() as session:
                row = await repo.patch_mcq(session, mcq_id, fields)
                await session.commit()
                return mcq_dict(row)
        except Exception as exc:
            raise _http_library(exc) from exc

    @app.delete("/v1/library/mcqs/{mcq_id}")
    async def delete_mcq(mcq_id: str):
        try:
            async with store.session() as session:
                await repo.soft_delete_mcq(session, mcq_id)
                await session.commit()
                return {"ok": True, "id": mcq_id}
        except Exception as exc:
            raise _http_library(exc) from exc

    @app.post("/v1/library/cases/{case_id}/pairs")
    async def create_pair(case_id: str, body: PairCreateBody):
        pair_id = body.id or str(uuid.uuid4())
        try:
            async with store.session() as session:
                row = await repo.create_pair(
                    session,
                    pair_id=pair_id,
                    case_id=case_id,
                    original_sentence=body.original_sentence,
                    polished_sentence=body.polished_sentence,
                    analysis=body.analysis,
                )
                await session.commit()
                return pair_dict(row)
        except Exception as exc:
            raise _http_library(exc) from exc

    @app.get("/v1/library/pairs/{pair_id}")
    async def get_pair(pair_id: str):
        try:
            async with store.session() as session:
                return pair_dict(await repo.get_pair(session, pair_id))
        except Exception as exc:
            raise _http_library(exc) from exc

    @app.patch("/v1/library/pairs/{pair_id}")
    async def patch_pair(pair_id: str, body: PairPatchBody):
        fields = body.model_dump(exclude_unset=True)
        try:
            async with store.session() as session:
                row = await repo.patch_pair(session, pair_id, fields)
                await session.commit()
                return pair_dict(row)
        except Exception as exc:
            raise _http_library(exc) from exc

    @app.delete("/v1/library/pairs/{pair_id}")
    async def delete_pair(pair_id: str):
        try:
            async with store.session() as session:
                await repo.soft_delete_pair(session, pair_id)
                await session.commit()
                return {"ok": True, "id": pair_id}
        except Exception as exc:
            raise _http_library(exc) from exc

    @app.post("/v1/library/ingest/{session_id}")
    async def ingest_session(session_id: str):
        try:
            return await ingest_session_json(store, session_id, service.cfg.items_dir)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/quizzes")
    async def create_quiz(body: CreateQuizBody):
        try:
            return await quiz.start(
                source=body.source, count=body.count, session_id=body.session_id
            )
        except Exception as exc:
            raise _http_library(exc) from exc

    @app.get("/v1/quizzes/{quiz_id}")
    async def get_quiz(quiz_id: str):
        try:
            return await quiz.get(quiz_id)
        except Exception as exc:
            raise _http_library(exc) from exc

    @app.post("/v1/quizzes/{quiz_id}/answers")
    async def answer_quiz(quiz_id: str, body: QuizAnswerBody):
        try:
            return await quiz.answer(
                quiz_id,
                mcq_id=body.mcq_id,
                selected=body.selected,
                elapsed_ms=body.elapsed_ms,
            )
        except Exception as exc:
            raise _http_library(exc) from exc

    @app.get("/v1/reviews/due")
    async def reviews_due(limit: int = 50):
        return {"cards": await review.due(limit=limit)}

    @app.post("/v1/reviews/{pair_id}/grade")
    async def grade_review(pair_id: str, body: ReviewGradeBody):
        try:
            return await review.grade(pair_id, body.grade)
        except Exception as exc:
            raise _http_library(exc) from exc

    @app.get("/v1/sessions/{session_id}/records")
    async def get_records(session_id: str):
        try:
            return {"records": service.get_records(session_id)}
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.patch("/v1/sessions/{session_id}/records/{seq}")
    async def patch_record(session_id: str, seq: int, body: RecordPatchBody):
        fields: dict[str, Any] = body.model_dump(exclude_unset=True)
        try:
            return service.patch_record(session_id, seq, fields)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SessionBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except SessionModeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/v1/sessions/{session_id}")
    async def delete_session(session_id: str):
        try:
            return service.delete_saved_session(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SessionBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/v1/settings")
    async def get_settings():
        return {"settings": service.get_settings()}

    @app.patch("/v1/settings")
    async def patch_settings_route(body: dict[str, Any]):
        try:
            return service.update_settings(body)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.websocket("/v1/sessions/{session_id}/audio")
    async def session_audio(websocket: WebSocket, session_id: str):
        try:
            transport = service.web_audio_transport(session_id)
        except SessionNotFoundError:
            await websocket.close(code=4404)
            return
        except SessionModeError:
            await websocket.close(code=4400)
            return
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "hello",
                "uplink_sr": 16000,
                "downlink_sr": 24000,
                "channels": 1,
            }
        )

        async def push_downlink() -> None:
            while True:
                chunk = await asyncio.to_thread(transport.pull_downlink, 0.2)
                if chunk is None:
                    if getattr(transport, "_closed", False):
                        return
                    continue
                await websocket.send_bytes(chunk)

        task = asyncio.create_task(push_downlink())
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                data = message.get("bytes")
                if data:
                    transport.push_uplink(data)
        except WebSocketDisconnect:
            pass
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @app.websocket("/v1/sessions/{session_id}/avatar")
    async def session_avatar(websocket: WebSocket, session_id: str):
        try:
            sink = service.avatar_sink(session_id)
        except SessionNotFoundError:
            await websocket.close(code=4404)
            return
        await websocket.accept()
        if sink is None:
            await websocket.send_json({"type": "disabled"})
            try:
                while True:
                    await websocket.receive()
            except WebSocketDisconnect:
                return
        try:
            while True:
                frame = await asyncio.to_thread(sink.pull, 0.3)
                if frame is None:
                    if getattr(sink, "_closed", False):
                        return
                    continue
                await websocket.send_bytes(frame)
        except WebSocketDisconnect:
            pass

    @app.websocket("/v1/sessions/{session_id}/events")
    async def session_events(websocket: WebSocket, session_id: str, last_n: int = 50):
        await websocket.accept()
        for event in service.bus.history(session_id, last_n=last_n):
            await websocket.send_json(event.to_dict())
        queue = service.bus.subscribe(session_id)
        try:
            while True:
                event = await queue.get()
                await websocket.send_json(event.to_dict())
        except WebSocketDisconnect:
            pass
        finally:
            service.bus.unsubscribe(queue)

    if WEB_DIST.is_dir():
        assets = WEB_DIST / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="web-assets")

        @app.get("/{full_path:path}")
        async def spa_fallback(full_path: str):
            if full_path.startswith("v1/") or full_path == "v1":
                raise HTTPException(status_code=404, detail="not found")
            candidate = WEB_DIST / full_path
            if candidate.is_file():
                return FileResponse(candidate)
            index = WEB_DIST / "index.html"
            if index.is_file():
                return FileResponse(index)
            raise HTTPException(status_code=404, detail="frontend not built")

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="English coach local HTTP API")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    store = Store.from_config(cfg)
    service = CoachService(cfg, store=store)
    app = create_app(service, store=store)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
