from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, DefaultDict, Dict, List, Optional, Tuple


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CoachEvent:
    type: str
    session_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "session_id": self.session_id,
            "ts": self.ts,
            "payload": self.payload,
        }


class EventBus:
    def __init__(self, history_limit: int = 200):
        self._history_limit = history_limit
        self._history: DefaultDict[str, List[CoachEvent]] = defaultdict(list)
        self._subs: List[Tuple[Optional[str], asyncio.Queue]] = []

    def emit(self, event: CoachEvent) -> None:
        bucket = self._history[event.session_id]
        bucket.append(event)
        if len(bucket) > self._history_limit:
            del bucket[: len(bucket) - self._history_limit]
        stale = []
        for index, (session_id, queue) in enumerate(self._subs):
            if session_id is not None and session_id != event.session_id:
                continue
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                stale.append(index)
        for index in reversed(stale):
            self._subs.pop(index)

    def subscribe(
        self, session_id: Optional[str] = None, maxsize: int = 256
    ) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._subs.append((session_id, queue))
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subs = [item for item in self._subs if item[1] is not queue]

    def history(self, session_id: str, last_n: int = 50) -> List[CoachEvent]:
        events = self._history.get(session_id) or []
        if last_n <= 0:
            return list(events)
        return events[-last_n:]
