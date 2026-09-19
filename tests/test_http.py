from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import AppConfig
from app.llm import ChatResult, ToolCall
from app.serve import create_app
from app.service import CoachService


class ScriptedMainLLM:
    async def chat(self, messages, tools=None, response_format=None) -> ChatResult:
        last = messages[-1]
        content = last.get("content") or ""
        if last.get("role") == "user" and "咖啡" in content:
            return ChatResult(
                tool_calls=[
                    ToolCall(
                        id="call_scene",
                        name="set_scene",
                        arguments=json.dumps(
                            {
                                "scene": "ordering coffee",
                                "setting": "a coffee shop",
                                "user_role": "customer",
                                "assistant_role": "barista",
                                "goals": "order a drink",
                                "raw": content,
                            },
                            ensure_ascii=False,
                        ),
                    )
                ]
            )
        if last.get("role") == "user":
            return ChatResult(content="Sure, what size would you like?")
        return ChatResult(content="Okay.")


def _service(tmp_path: Path) -> CoachService:
    cfg = AppConfig(
        session_dir=str(tmp_path),
        llm_api_key="test",
        db_path=str(tmp_path / "coach.sqlite"),
    )
    return CoachService(cfg, llm=ScriptedMainLLM())


def test_http_create_and_text_turn(tmp_path: Path):
    service = _service(tmp_path)
    app = create_app(service)
    with TestClient(app) as client:
        first = client.post("/v1/sessions", json={"mode": "text"})
        assert first.status_code == 200
        session_id = first.json()["session_id"]

        got = client.get(f"/v1/sessions/{session_id}")
        assert got.status_code == 200
        assert got.json()["mode"] == "text"

        text = client.post(
            f"/v1/sessions/{session_id}/text",
            json={"text": "我想练咖啡店点单，你是店员我是顾客"},
        )
        assert text.status_code == 200
        scene = None
        for _ in range(40):
            snapshot = client.get(f"/v1/sessions/{session_id}")
            scene = snapshot.json().get("scene")
            if scene:
                break
            time.sleep(0.05)
        assert scene is not None
        assert scene["assistant_role"] == "barista"

        with client.websocket_connect(f"/v1/sessions/{session_id}/events?last_n=50") as ws:
            types = set()
            for _ in range(12):
                event = ws.receive_json()
                types.add(event["type"])
                if "utterance" in types and "session.started" in types:
                    break
            assert "session.started" in types or "utterance" in types

        client.post(f"/v1/sessions/{session_id}/stop")
