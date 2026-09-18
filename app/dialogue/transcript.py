from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Literal


@dataclass
class TranscriptEntry:
    role: Literal["user", "assistant"]
    text: str
    source: str
    phase: str
    timestamp: str


class Transcript:
    def __init__(self) -> None:
        self.entries: List[TranscriptEntry] = []

    def add(
        self,
        role: Literal["user", "assistant"],
        text: str,
        source: str = "model",
        phase: str = "practice",
    ) -> TranscriptEntry:
        entry = TranscriptEntry(
            role=role,
            text=text,
            source=source,
            phase=phase,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.entries.append(entry)
        return entry
