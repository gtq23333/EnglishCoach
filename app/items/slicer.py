from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.items.filter import FilterDecision


@dataclass
class DialogueSlice:
    pairs: List[FilterDecision]
    session_id: str = ""
    scenario: str = ""

    @property
    def user_turns(self) -> int:
        return len(self.pairs)

    @property
    def char_count(self) -> int:
        return sum(len(item.user_text) + len(item.preceding_assistant) for item in self.pairs)

    def to_prompt_input(self, session_id: str = "", scene: dict | None = None) -> dict:
        turns = []
        seq = 1
        for pair in self.pairs:
            if pair.preceding_assistant:
                turns.append(
                    {
                        "seq": seq,
                        "role": "assistant",
                        "phase": "practice",
                        "text": pair.preceding_assistant,
                    }
                )
                seq += 1
            turns.append(
                {
                    "seq": seq,
                    "role": "user",
                    "phase": "practice",
                    "text": pair.user_text,
                    "source_seq": pair.source_seq,
                }
            )
            seq += 1
        return {
            "session_id": session_id or self.session_id,
            "scene": scene or {},
            "turns": turns,
        }


def slice_keep_turns(
    decisions: List[FilterDecision],
    *,
    max_user_turns_per_slice: int = 2,
    char_budget: int = 800,
    session_id: str = "",
    scenario: str = "",
) -> List[DialogueSlice]:
    kept = [item for item in decisions if item.action in {"keep", "strip_echo_prefix"}]
    slices: List[DialogueSlice] = []
    current: List[FilterDecision] = []

    def flush() -> None:
        nonlocal current
        if current:
            slices.append(
                DialogueSlice(pairs=list(current), session_id=session_id, scenario=scenario)
            )
            current = []

    for item in kept:
        pair_chars = len(item.user_text) + len(item.preceding_assistant)
        if pair_chars >= char_budget and not current:
            slices.append(
                DialogueSlice(pairs=[item], session_id=session_id, scenario=scenario)
            )
            continue
        prospective = sum(
            len(p.user_text) + len(p.preceding_assistant) for p in current
        ) + pair_chars
        if current and (
            len(current) >= max_user_turns_per_slice or prospective > char_budget
        ):
            flush()
        current.append(item)
    flush()
    return slices
