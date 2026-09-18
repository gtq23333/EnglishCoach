from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional


Phase = Literal["bootstrap", "practice"]


@dataclass
class SceneSpec:
    raw: str
    scene: str = ""
    user_role: str = ""
    assistant_role: str = ""
    setting: str = ""
    goals: str = ""

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Dict[str, Any], raw_fallback: str = "") -> "SceneSpec":
        raw = str(data.get("raw") or raw_fallback or data.get("scene") or "")
        return cls(
            raw=raw,
            scene=str(data.get("scene") or raw),
            user_role=str(data.get("user_role") or ""),
            assistant_role=str(data.get("assistant_role") or ""),
            setting=str(data.get("setting") or ""),
            goals=str(data.get("goals") or ""),
        )

    @classmethod
    def from_cli(cls, text: str) -> "SceneSpec":
        return cls(raw=text, scene=text)


@dataclass
class AssemblyContext:
    phase: Phase
    scene: Optional[SceneSpec] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptBundle:
    instructions: str
    tools: List[Dict[str, Any]]
    opening_text: Optional[str] = None
    injected_user_prompt: Optional[str] = None


class PromptAssembler(ABC):
    name: str

    @abstractmethod
    def assemble(self, ctx: AssemblyContext) -> PromptBundle:
        raise NotImplementedError
