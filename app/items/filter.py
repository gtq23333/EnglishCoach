from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional


META_PHRASES = {
    "keep going",
    "i can hear you",
    "can you hear me",
    "hello",
    "hi",
    "ok",
    "okay",
    "yes",
    "yeah",
    "let's do this tomorrow",
    "lets do this tomorrow",
    "we talk enough today",
    "i think we talk enough today",
}

TTS_ARTIFACTS = (
    "the voice you selected does not support this language",
    "the voice use a",
    "the voice you selected",
)

TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+", re.I)


@dataclass
class FilterDecision:
    action: str
    reason: str
    user_text: str
    source_seq: int
    preceding_assistant: str = ""
    phase: str = ""


def tokenize(text: str) -> List[str]:
    return [item.lower() for item in TOKEN_RE.findall(text or "")]


def tokenize_spans(text: str) -> List[tuple[str, int, int]]:
    return [(m.group().lower(), m.start(), m.end()) for m in TOKEN_RE.finditer(text or "")]


def word_similar(left: str, right: str) -> bool:
    if left == right:
        return True
    if min(len(left), len(right)) >= 3 and (left in right or right in left):
        return True
    return SequenceMatcher(None, left, right).ratio() >= 0.75


def jaccard(left: List[str], right: List[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def strip_echo_prefix(assistant: str, user: str) -> tuple[str, int]:
    a_words = tokenize(assistant)
    spans = tokenize_spans(user)
    u_words = [item[0] for item in spans]
    if not a_words or not u_words:
        return user, 0
    best_j = 0
    limit = min(len(u_words), len(a_words) + 8)
    for end in range(1, limit + 1):
        window = u_words[:end]
        asst_window = a_words[: min(len(a_words), end + 4)]
        ratio = SequenceMatcher(
            None, " ".join(window), " ".join(asst_window)
        ).ratio()
        overlap = jaccard(window, asst_window)
        if ratio >= 0.52 or overlap >= 0.45:
            best_j = end
    consumed_all = best_j >= len(u_words)
    if consumed_all and best_j >= 3:
        return "", best_j
    if best_j < 6 and not (best_j >= 4 and best_j >= int(len(u_words) * 0.4)):
        return user, 0
    remainder = user[spans[best_j][1] :].lstrip(" .,:;!?")
    return remainder, best_j


def _is_meta(text: str) -> bool:
    compact = " ".join(tokenize(text))
    if compact in META_PHRASES:
        return True
    if compact in {"hello", "hi hello"}:
        return True
    return False


def _is_tts_artifact(text: str) -> bool:
    compact = " ".join(tokenize(text))
    return any(marker in compact for marker in TTS_ARTIFACTS)


def _is_wrapup(text: str) -> bool:
    compact = " ".join(tokenize(text))
    wrap_markers = (
        "talk enough",
        "do this tomorrow",
        "that's all for today",
        "thats all for today",
        "see you tomorrow",
    )
    return any(marker in compact for marker in wrap_markers)


def _too_short(text: str) -> bool:
    words = tokenize(text)
    if len(words) >= 6:
        return False
    if len((text or "").strip()) >= 32 and len(words) >= 4:
        return False
    return True


def filter_user_turn(
    *,
    source_seq: int,
    user_text: str,
    phase: str,
    preceding_assistant: str = "",
) -> FilterDecision:
    original = user_text or ""
    if phase and phase != "practice":
        return FilterDecision("drop", "not_practice", original, source_seq, preceding_assistant, phase)
    if _is_tts_artifact(original):
        return FilterDecision("drop", "tts_artifact", original, source_seq, preceding_assistant, phase)

    stripped, matched = strip_echo_prefix(preceding_assistant, original)
    working = stripped if matched else original
    stripped_used = bool(matched) and stripped != original

    if stripped_used and not working.strip():
        return FilterDecision(
            "drop", "echo_only", original, source_seq, preceding_assistant, phase
        )

    if not stripped_used:
        user_words = tokenize(original)
        asst_words = tokenize(preceding_assistant)
        if user_words and asst_words and jaccard(user_words, asst_words) >= 0.6:
            return FilterDecision(
                "drop", "echo_overlap", original, source_seq, preceding_assistant, phase
            )

    if _is_meta(working):
        return FilterDecision("drop", "meta", original, source_seq, preceding_assistant, phase)
    if _is_wrapup(working):
        return FilterDecision("drop", "wrapup", original, source_seq, preceding_assistant, phase)
    if _too_short(working):
        return FilterDecision("drop", "too_short", original, source_seq, preceding_assistant, phase)

    if stripped_used:
        return FilterDecision(
            "strip_echo_prefix",
            "echo_prefix",
            working,
            source_seq,
            preceding_assistant,
            phase,
        )
    return FilterDecision("keep", "ok", working, source_seq, preceding_assistant, phase)


def filter_session_records(records: List[dict]) -> List[FilterDecision]:
    last_assistant = ""
    decisions: List[FilterDecision] = []
    for record in records:
        if record.get("type") == "utterance" and record.get("role") == "assistant":
            last_assistant = str(record.get("text") or "")
            continue
        if record.get("type") != "utterance" or record.get("role") != "user":
            continue
        decisions.append(
            filter_user_turn(
                source_seq=int(record.get("seq") or 0),
                user_text=str(record.get("text") or ""),
                phase=str(record.get("phase") or ""),
                preceding_assistant=last_assistant,
            )
        )
    return decisions
