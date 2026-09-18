from __future__ import annotations

from typing import Any, Dict, List, Sequence


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def _contains(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    return needle in haystack or _norm(needle) in _norm(haystack)


def admit_case(
    case: Dict[str, Any],
    user_texts: Sequence[str],
    assistant_texts: Sequence[str],
) -> List[str]:
    violations: List[str] = []
    users = list(user_texts)
    assistants = list(assistant_texts)
    joined_user = "\n".join(users)
    joined_assistant = "\n".join(assistants)

    self_check = case.get("self_check") if isinstance(case.get("self_check"), dict) else {}
    extra = self_check.get("violations") if isinstance(self_check, dict) else None
    if extra:
        violations.append("model_self_check_not_empty")

    for question in case.get("questions") or []:
        original = str(question.get("original_distractor") or "")
        extras = question.get("additional_distractors") or []
        if not original:
            violations.append("missing_original_distractor")
            continue
        if not _contains(joined_user, original):
            violations.append("original_distractor_not_in_user")
        if _contains(joined_assistant, original) and not _contains(joined_user, original):
            violations.append("original_distractor_from_assistant")
        if len(extras) != 2:
            violations.append("need_exactly_two_additional_distractors")
        stem = str(question.get("stem") or "")
        if "<blank>" not in stem:
            violations.append("stem_missing_blank")
        if not str(question.get("correct_answer") or "").strip():
            violations.append("missing_correct_answer")

    if not (case.get("questions") or []) and not (case.get("sentence_pairs") or []):
        violations.append("empty_case")

    for pair in case.get("sentence_pairs") or []:
        original_sentence = str(pair.get("original_sentence") or "")
        if original_sentence and not _contains(joined_user, original_sentence):
            # allow punctuation drift: require some overlap of first 8 words
            head = " ".join(original_sentence.split()[:6])
            if head and not _contains(joined_user, head):
                violations.append("sentence_pair_not_from_user")
        if not str(pair.get("polished_sentence") or "").strip():
            violations.append("missing_polished_sentence")

    return violations
