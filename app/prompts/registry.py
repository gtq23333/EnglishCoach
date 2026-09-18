from __future__ import annotations

from app.prompts.base import PromptAssembler
from app.prompts.scene_practice import ScenePracticeAssembler

_ASSEMBLERS = {
    ScenePracticeAssembler.name: ScenePracticeAssembler,
}


def get_assembler(name: str) -> PromptAssembler:
    cls = _ASSEMBLERS.get(name)
    if cls is None:
        known = ", ".join(sorted(_ASSEMBLERS))
        raise ValueError(f"unknown prompt assembler {name!r}; known: {known}")
    return cls()
