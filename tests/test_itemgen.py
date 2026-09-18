from __future__ import annotations

import json
from pathlib import Path

from app.items.generator import ItemGenError, generate_for_session, parse_model_json
from app.llm import ChatResult


class RefuseLLM:
    async def chat(self, messages, tools=None, response_format=None) -> ChatResult:
        return ChatResult(
            content=json.dumps(
                {
                    "meta": {
                        "session_id": "",
                        "scenario": "",
                        "user_turns_reviewed": 1,
                        "issues_found": 0,
                    },
                    "alignment": {
                        "task_restatement": "只从 User 应答里找用得不好的表达",
                        "extraction_scope_confirmed": "user_utterances_only",
                        "skipped_summary": "疑似上游ASR/回声噪声，不予出题：整段复读面试官问句",
                    },
                    "turn_reviews": [
                        {
                            "source_seq": 2,
                            "user_text": "hello",
                            "preceding_assistant_text": "hi",
                            "verdict": "skip",
                            "skip_reason": "noise",
                            "clean_reason": None,
                            "issue_spots": [],
                        }
                    ],
                    "cases": [],
                    "self_check": {"violations": []},
                },
                ensure_ascii=False,
            )
        )


class GoodLLM:
    async def chat(self, messages, tools=None, response_format=None) -> ChatResult:
        return ChatResult(
            content=json.dumps(
                {
                    "meta": {
                        "session_id": "s",
                        "scenario": "interview",
                        "user_turns_reviewed": 1,
                        "issues_found": 1,
                    },
                    "alignment": {
                        "task_restatement": "只从 User 应答里找用得不好的表达",
                        "extraction_scope_confirmed": "user_utterances_only",
                        "skipped_summary": "无跳过",
                    },
                    "turn_reviews": [],
                    "cases": [
                        {
                            "case_id": "case_s_2",
                            "scenario": "interview",
                            "source_seq": 2,
                            "raw_interaction": {
                                "speaker_other": "How do you handle delays?",
                                "speaker_user": "I will catch up the project progress.",
                            },
                            "context_adaptation_note": "未改写",
                            "questions": [
                                {
                                    "question_id": "case_s_2_q1",
                                    "category": "Idiomatic Collocation",
                                    "stem": "User: I will <blank>.",
                                    "correct_answer": "get the project back on track",
                                    "original_distractor": "catch up the project progress",
                                    "additional_distractors": [
                                        "chase the delivery schedule",
                                        "speed up the timeline gap",
                                    ],
                                    "analysis": "中式搭配",
                                }
                            ],
                            "sentence_pairs": [
                                {
                                    "pair_id": "case_s_2_s1",
                                    "original_sentence": "I will catch up the project progress.",
                                    "polished_sentence": "I will get the project back on track.",
                                    "analysis": "用地道块替换",
                                }
                            ],
                        }
                    ],
                    "self_check": {"violations": []},
                },
                ensure_ascii=False,
            )
        )


def _write_session(tmp_path: Path, session_id: str, user_text: str) -> None:
    path = tmp_path / f"{session_id}.jsonl"
    lines = [
        {
            "v": 1,
            "session_id": session_id,
            "seq": 1,
            "type": "session_start",
            "mode": "text",
        },
        {
            "v": 1,
            "session_id": session_id,
            "seq": 2,
            "type": "utterance",
            "role": "assistant",
            "text": "How do you handle delays?",
            "source": "model",
            "phase": "practice",
        },
        {
            "v": 1,
            "session_id": session_id,
            "seq": 3,
            "type": "utterance",
            "role": "user",
            "text": user_text,
            "source": "user",
            "phase": "practice",
        },
    ]
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in lines) + "\n",
        encoding="utf-8",
    )


async def test_itemgen_model_refuse(tmp_path: Path):
    _write_session(tmp_path, "s-refuse", "I will catch up the project progress with the team tomorrow.")
    result = await generate_for_session(
        session_id="s-refuse",
        session_dir=tmp_path,
        items_dir=tmp_path / "items",
        llm=RefuseLLM(),
    )
    assert result["cases"] == []
    assert any(item.get("reason") == "model_refused" for item in result["skipped"])


async def test_itemgen_gate_accepts_valid_case(tmp_path: Path):
    _write_session(tmp_path, "s-good", "I will catch up the project progress.")
    result = await generate_for_session(
        session_id="s-good",
        session_dir=tmp_path,
        items_dir=tmp_path / "items",
        llm=GoodLLM(),
    )
    assert len(result["cases"]) == 1
    assert result["cases"][0]["questions"][0]["original_distractor"] == (
        "catch up the project progress"
    )
    assert (tmp_path / "items" / "s-good.json").exists()
    assert not (tmp_path / "s-good.items.json").exists()


def test_parse_model_json_extracts_fenced_and_trailing_text():
    payload = {"cases": [], "alignment": {"skipped_summary": "无跳过"}}
    fenced = "```json\n" + json.dumps(payload) + "\n```"
    assert parse_model_json(fenced)["cases"] == []
    wrapped = "here you go\n" + json.dumps(payload) + "\nthanks"
    assert parse_model_json(wrapped)["cases"] == []
    try:
        parse_model_json("not json at all")
        assert False, "expected ItemGenError"
    except ItemGenError:
        pass


class FlakyThenGoodLLM:
    def __init__(self, good: str) -> None:
        self.good = good
        self.calls = 0

    async def chat(self, messages, tools=None, response_format=None) -> ChatResult:
        self.calls += 1
        if self.calls == 1:
            return ChatResult(content="sorry, I cannot emit JSON right now")
        return ChatResult(content=self.good)


class AlwaysBadJSONLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, messages, tools=None, response_format=None) -> ChatResult:
        self.calls += 1
        return ChatResult(content="still not json {")


async def test_itemgen_retries_bad_json_then_succeeds(tmp_path: Path):
    _write_session(tmp_path, "s-retry", "I will catch up the project progress.")
    good = GoodLLM()
    first = await good.chat([{"role": "user", "content": "x"}])
    llm = FlakyThenGoodLLM(first.content or "")
    result = await generate_for_session(
        session_id="s-retry",
        session_dir=tmp_path,
        items_dir=tmp_path / "items",
        llm=llm,
        parse_retries=2,
    )
    assert llm.calls == 2
    assert len(result["cases"]) == 1


async def test_itemgen_gives_up_after_parse_retries(tmp_path: Path):
    _write_session(tmp_path, "s-fail", "I will catch up the project progress.")
    llm = AlwaysBadJSONLLM()
    result = await generate_for_session(
        session_id="s-fail",
        session_dir=tmp_path,
        items_dir=tmp_path / "items",
        llm=llm,
        parse_retries=1,
    )
    assert llm.calls == 2
    assert result["cases"] == []
    assert any(str(item.get("reason", "")).startswith("llm_failed:") for item in result["skipped"])
