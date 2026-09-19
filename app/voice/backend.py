from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from app.voice.events import ToolResult, VoiceEvent


class VoiceBackend(ABC):
    """Pluggable speech+dialogue transport.

    Current implementation: DuplexVoiceBackend (Volcengine end-to-end API).
    Later: a pipeline backend that wires separate ASR + LLM + TTS to the
    same event/method surface so MicSession does not change.
    """

    @abstractmethod
    async def start(self, instructions: str, tools: List[Dict[str, Any]]) -> None:
        """Connect and open a dialogue session."""

    @abstractmethod
    async def update(self, instructions: str, tools: List[Dict[str, Any]]) -> None:
        """Hot-swap instructions and tools (e.g. after set_scene)."""

    @abstractmethod
    async def send_audio(self, pcm: bytes) -> None:
        """Push captured microphone PCM."""

    @abstractmethod
    async def speak_text(self, text: str) -> None:
        """Speak this text through the backend TTS path."""

    @abstractmethod
    async def inject_assistant_text(self, text: str) -> None:
        """Insert assistant text into dialogue context (may not speak)."""

    @abstractmethod
    async def submit_tool_results(self, results: List[ToolResult]) -> None:
        """Return function-call outputs to the model."""

    @abstractmethod
    async def recv_event(self) -> VoiceEvent:
        """Wait for the next backend-agnostic voice event."""

    async def commit_audio(self) -> None:
        """Force endpoint detection. Duplex implements this; others may no-op."""

    async def reconnect(self, instructions: str, tools: List[Dict[str, Any]]) -> None:
        """Replace a dead transport and open a new dialogue session."""
        await self.close()
        await self.start(instructions, tools)

    @abstractmethod
    async def close(self) -> None:
        """Tear down the session and transport."""
