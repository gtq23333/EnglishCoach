from __future__ import annotations

import json
from pathlib import Path

from app.dialogue.orchestrator import DialogueOrchestrator
from app.dialogue.session_log import SessionLog, load_records
from app.dialogue.text_session import run_text_session
from app.llm import ChatResult, ToolCall
from app.prompts.scene_practice import BOOTSTRAP_INJECTED_PROMPT, ScenePracticeAssembler


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


async def test_text_session_writes_utterances_incrementally(tmp_path: Path):
    session_id = "text-e2e"
    log = SessionLog(tmp_path, session_id, mode="text")
    orch = DialogueOrchestrator(ScenePracticeAssembler(), log)

    lines = [
        "我想练咖啡店点单，你是店员我是顾客",
        "I am want a coffee",
        "Thanks, I go now",
        "exit",
    ]
    iterator = iter(lines)

    async def input_fn(_prompt: str) -> str:
        return next(iterator)

    outputs: list[str] = []
    await run_text_session(
        orch,
        ScriptedMainLLM(),
        input_fn=input_fn,
        output_fn=outputs.append,
    )

    assert any(BOOTSTRAP_INJECTED_PROMPT in item for item in outputs)
    records = load_records(log.path)
    types = [item["type"] for item in records]
    assert types[0] == "session_start"
    assert types[-1] == "session_end"
    utterances = [item for item in records if item["type"] == "utterance"]
    assert utterances[0]["role"] == "assistant"
    assert utterances[0]["source"] == "injected"
    assert utterances[1]["role"] == "user"
    assert "咖啡店" in utterances[1]["text"]
    scenes = [item for item in records if item["type"] == "scene"]
    assert len(scenes) == 1
    assert scenes[0]["scene"]["assistant_role"] == "barista"
    user_practice = [
        item for item in utterances if item["role"] == "user" and item["phase"] == "practice"
    ]
    assert user_practice[0]["text"] == "I am want a coffee"
    assert user_practice[1]["text"] == "Thanks, I go now"
    assistant_practice = [
        item
        for item in utterances
        if item["role"] == "assistant" and item["phase"] == "practice"
    ]
    assert any("size" in item["text"] for item in assistant_practice)
    assert orch.scene is not None
    assert orch.scene.assistant_role == "barista"
    seqs = [item["seq"] for item in records]
    assert seqs == list(range(1, len(records) + 1))
