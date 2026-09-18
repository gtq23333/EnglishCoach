from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Union

from app.dialogue.transcript import TranscriptEntry
from app.prompts.base import SceneSpec


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionLog:
    """Append-only jsonl transcript. Each record is flushed to disk immediately."""

    def __init__(
        self,
        directory: Union[str, Path],
        session_id: str,
        mode: str,
    ):
        self.session_id = session_id
        self.mode = mode
        self.path = Path(directory) / f"{session_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._closed = False
        self._write(
            {
                "type": "session_start",
                "mode": mode,
                "created_at": utc_now(),
            }
        )

    def append_utterance(self, entry: TranscriptEntry) -> None:
        self._write(
            {
                "type": "utterance",
                "role": entry.role,
                "text": entry.text,
                "source": entry.source,
                "phase": entry.phase,
                "timestamp": entry.timestamp,
            }
        )

    def append_scene(self, scene: SceneSpec) -> None:
        self._write(
            {
                "type": "scene",
                "scene": scene.to_dict(),
                "timestamp": utc_now(),
            }
        )

    def close(self) -> None:
        if self._closed:
            return
        self._write({"type": "session_end", "timestamp": utc_now()})
        self._closed = True

    def _write(self, record: Dict[str, Any]) -> None:
        self._seq += 1
        payload = {
            "v": 1,
            "session_id": self.session_id,
            "seq": self._seq,
            **record,
        }
        line = json.dumps(payload, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def load_records(path: Union[str, Path]) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]
