from __future__ import annotations

import argparse
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import load_config
from app.service import (
    CoachService,
    SessionBusyError,
    SessionModeError,
    SessionNotFoundError,
)


class CreateSessionBody(BaseModel):
    mode: str
    scene: Optional[str] = None


class TextBody(BaseModel):
    text: str


class GenerateItemsBody(BaseModel):
    max_user_turns_per_slice: int = 2
    char_budget: int = 800
    parse_retries: Optional[int] = None


def create_app(service: CoachService) -> FastAPI:
    app = FastAPI(title="English Coach", version="1.0")
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

    @app.get("/v1/sessions/{session_id}/items")
    async def get_items(session_id: str):
        try:
            return service.get_items(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

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
    service = CoachService(cfg)
    app = create_app(service)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
