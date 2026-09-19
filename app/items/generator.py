from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from app.dialogue.session_log import load_records
from app.items.filter import filter_session_records
from app.items.gate import admit_case
from app.items.prompt import ITEMGEN_SYSTEM_PROMPT, SKIP_PREFIXES
from app.items.slicer import slice_keep_turns
from app.llm import LLMClient


logger = logging.getLogger(__name__)


FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I | re.M)
DEFAULT_PARSE_RETRIES = 2
REPAIR_USER_TEMPLATE = (
    "The previous output was not valid JSON for this task.\n"
    "Parse error: {error}\n"
    "Return exactly one JSON object matching the schema. "
    "No markdown, no code fences, no preface.\n"
    "Previous output:\n{raw}"
)


class ItemGenError(RuntimeError):
    pass


def items_path(items_dir: str | Path, session_id: str) -> Path:
    return Path(items_dir) / f"{session_id}.json"


def session_jsonl_path(session_dir: str | Path, session_id: str) -> Path:
    return Path(session_dir) / f"{session_id}.jsonl"


def _extract_json_text(raw: str) -> str:
    text = FENCE_RE.sub("", (raw or "").strip()).strip()
    if not text:
        raise ItemGenError("empty model output")
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ItemGenError("no json object found")
    return text[start : end + 1]


def parse_model_json(raw: str) -> Dict[str, Any]:
    text = _extract_json_text(raw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ItemGenError(f"invalid json: {exc}") from exc
    if not isinstance(data, dict):
        raise ItemGenError("model output is not an object")
    if "cases" not in data:
        raise ItemGenError("missing cases array")
    if not isinstance(data.get("cases"), list):
        raise ItemGenError("cases is not an array")
    alignment = data.get("alignment")
    if alignment is not None and not isinstance(alignment, dict):
        raise ItemGenError("alignment is not an object")
    return data


async def _chat_json_with_retries(
    llm: LLMClient,
    payload: dict,
    *,
    parse_retries: int = DEFAULT_PARSE_RETRIES,
    progress: Optional[Callable[[str, dict], None]] = None,
    slice_index: int = 0,
) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": ITEMGEN_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    attempts = 1 + max(0, int(parse_retries))
    last_error: Optional[Exception] = None
    raw = ""
    for attempt in range(1, attempts + 1):
        try:
            if progress:
                progress(
                    "itemgen.progress",
                    {
                        "stage": "llm",
                        "slice": slice_index,
                        "attempt": attempt,
                    },
                )
            logger.info(
                "itemgen llm slice=%s attempt=%s/%s", slice_index, attempt, attempts
            )
            result = await llm.chat(
                messages=messages,
                response_format={"type": "json_object"},
            )
            raw = result.content or ""
            return parse_model_json(raw)
        except ItemGenError as exc:
            last_error = exc
            if attempt >= attempts:
                break
            if progress:
                progress(
                    "itemgen.progress",
                    {
                        "stage": "parse_retry",
                        "slice": slice_index,
                        "attempt": attempt,
                        "error": str(exc),
                    },
                )
            if raw:
                messages = messages + [
                    {"role": "assistant", "content": raw[:4000]},
                    {
                        "role": "user",
                        "content": REPAIR_USER_TEMPLATE.format(
                            error=exc, raw=raw[:2000]
                        ),
                    },
                ]
            raw = ""
        except Exception as exc:
            last_error = exc
            logger.exception(
                "itemgen llm call failed slice=%s attempt=%s", slice_index, attempt
            )
            break
    if isinstance(last_error, ItemGenError):
        raise ItemGenError(
            f"JSON parse failed after {attempts} attempts: {last_error}"
        ) from last_error
    raise ItemGenError(f"LLM call failed: {last_error}") from last_error


def _scene_from_records(records: List[dict]) -> dict:
    for record in reversed(records):
        if record.get("type") == "scene" and isinstance(record.get("scene"), dict):
            return record["scene"]
    start = next((item for item in records if item.get("type") == "session_start"), None)
    return {}


async def generate_for_session(
    *,
    session_id: str,
    session_dir: str | Path,
    items_dir: str | Path,
    llm: LLMClient,
    max_user_turns_per_slice: int = 2,
    char_budget: int = 800,
    parse_retries: int = DEFAULT_PARSE_RETRIES,
    progress: Optional[Callable[[str, dict], None]] = None,
) -> Dict[str, Any]:
    path = session_jsonl_path(session_dir, session_id)
    if not path.exists():
        raise FileNotFoundError(f"session jsonl not found: {path}")
    records = load_records(path)
    scene = _scene_from_records(records)
    scenario = str(scene.get("scene") or "")
    decisions = filter_session_records(records)
    skipped = [
        {
            "source_seq": item.source_seq,
            "action": item.action,
            "reason": item.reason,
            "user_text": item.user_text if item.action == "drop" else "",
        }
        for item in decisions
        if item.action == "drop"
    ]
    slices = slice_keep_turns(
        decisions,
        max_user_turns_per_slice=max_user_turns_per_slice,
        char_budget=char_budget,
        session_id=session_id,
        scenario=scenario,
    )
    if progress:
        progress("itemgen.progress", {"stage": "filtered", "slices": len(slices), "skipped": len(skipped)})

    cases: List[dict] = []
    reviews: List[dict] = []
    alignments: List[dict] = []
    slice_skips: List[dict] = []

    for index, slc in enumerate(slices, start=1):
        if progress:
            progress(
                "itemgen.progress",
                {"stage": "slice", "index": index, "total": len(slices)},
            )
        try:
            payload = slc.to_prompt_input(session_id=session_id, scene=scene)
            parsed = await _chat_json_with_retries(
                llm,
                payload,
                parse_retries=parse_retries,
                progress=progress,
                slice_index=index,
            )
        except Exception as exc:
            slice_skips.append({"slice": index, "reason": f"llm_failed:{exc}"})
            continue
        alignments.append(parsed.get("alignment") or {})
        reviews.extend(parsed.get("turn_reviews") or [])
        summary = str((parsed.get("alignment") or {}).get("skipped_summary") or "")
        if any(summary.startswith(prefix) for prefix in SKIP_PREFIXES):
            slice_skips.append(
                {"slice": index, "reason": "model_refused", "detail": summary}
            )
        user_texts = [pair.user_text for pair in slc.pairs]
        assistant_texts = [pair.preceding_assistant for pair in slc.pairs]
        for case in parsed.get("cases") or []:
            if not isinstance(case, dict):
                continue
            violations = admit_case(case, user_texts, assistant_texts)
            top_check = parsed.get("self_check") if isinstance(parsed.get("self_check"), dict) else {}
            if top_check.get("violations"):
                violations.append("top_self_check_not_empty")
            if violations:
                slice_skips.append(
                    {
                        "slice": index,
                        "reason": "gate_rejected",
                        "case_id": case.get("case_id"),
                        "violations": violations,
                    }
                )
                continue
            cases.append(case)

    output = {
        "v": 1,
        "session_id": session_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario,
        "slices": len(slices),
        "skipped": skipped + slice_skips,
        "cases": cases,
        "turn_reviews": reviews,
        "alignments": alignments,
    }
    out_path = items_path(items_dir, session_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    if progress:
        progress(
            "itemgen.done",
            {"path": str(out_path), "cases": len(cases), "skipped": len(output["skipped"])},
        )
    return output


def load_items(items_dir: str | Path, session_id: str) -> Optional[Dict[str, Any]]:
    path = items_path(items_dir, session_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
