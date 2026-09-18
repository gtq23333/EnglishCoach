"""Map known English TTS text onto THA3 mouth channels.

v1 is a coarse grapheme schedule stretched to playback duration — the same
approach as the PoC hello clip. Forced alignment can replace this later
without changing AvatarEngine.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

MouthKey = Dict[str, float]
Timeline = List[Tuple[float, MouthKey]]

_VOWEL_MOUTH: Dict[str, MouthKey] = {
    "a": {"mouth_aaa": 0.85},
    "e": {"mouth_eee": 0.8},
    "i": {"mouth_iii": 0.8},
    "o": {"mouth_ooo": 0.85},
    "u": {"mouth_uuu": 0.85},
    "y": {"mouth_iii": 0.55},
}

# PoC timeline for the frozen demo sentence (seconds).
HELLO_TIMELINE: Timeline = [
    (0.00, {}),
    (0.08, {"mouth_eee": 0.85}),
    (0.26, {"mouth_eee": 0.7}),
    (0.34, {}),
    (0.42, {"mouth_ooo": 0.9}),
    (0.60, {}),
    (0.82, {}),
    (0.90, {}),
    (0.98, {"mouth_aaa": 0.85}),
    (1.14, {"mouth_iii": 0.8}),
    (1.28, {}),
    (1.38, {}),
    (1.46, {"mouth_ooo": 0.7}),
    (1.58, {}),
    (1.68, {}),
    (1.78, {"mouth_iii": 0.9}),
    (1.98, {}),
    (2.08, {"mouth_iii": 0.6}),
    (2.20, {"mouth_uuu": 0.85}),
    (2.40, {}),
    (2.85, {}),
]

HELLO_PHRASE = "hello, nice to meet you."


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _stretch(timeline: Timeline, duration_s: float) -> Timeline:
    if not timeline:
        return [(0.0, {}), (max(duration_s, 0.12), {})]
    src = max(timeline[-1][0], 1e-6)
    scale = max(duration_s, 0.12) / src
    return [(t * scale, dict(keys)) for t, keys in timeline]


def _from_letters(text: str) -> Timeline:
    letters = [ch for ch in text if ch.isalpha() or ch.isspace() or ch in ",.!?"]
    if not letters:
        return [(0.0, {}), (0.4, {})]
    keys: Timeline = [(0.0, {})]
    t = 0.0
    for ch in letters:
        if ch in "aeiouy":
            t += 0.09
            keys.append((t, dict(_VOWEL_MOUTH[ch])))
            t += 0.07
            keys.append((t, dict(_VOWEL_MOUTH[ch])))
        elif ch.isalpha():
            t += 0.05
            keys.append((t, {}))
        else:
            t += 0.08
            keys.append((t, {}))
    keys.append((t + 0.12, {}))
    return keys


def plan_visemes(text: str, duration_s: float) -> Timeline:
    """Return a mouth timeline in seconds, ending at ``duration_s``."""
    normalized = _normalize(text)
    if not normalized:
        return [(0.0, {}), (max(duration_s, 0.12), {})]
    if normalized.rstrip(".!?") == HELLO_PHRASE.rstrip("."):
        base = HELLO_TIMELINE
    else:
        base = _from_letters(normalized)
    return _stretch(base, duration_s)
