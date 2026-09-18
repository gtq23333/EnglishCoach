from __future__ import annotations

import json
from typing import Any, Callable, Dict, Literal, Optional, Tuple

from app.dialogue.session_log import SessionLog
from app.dialogue.transcript import Transcript
from app.events import CoachEvent
from app.prompts.base import AssemblyContext, PromptAssembler, PromptBundle, SceneSpec
from app.tools import TOOL_NAME_EXIT, TOOL_NAME_SET_SCENE


EXIT_PHRASES = {
    "quit",
    "exit",
    "q",
    "stop",
    "退出",
    "结束",
    "不聊了",
    "再见",
    "拜拜",
    "结束对话",
}

EmitFn = Callable[[CoachEvent], None]


def is_exit_phrase(text: str) -> bool:
    stripped = text.strip().lower()
    if stripped in {item.lower() for item in EXIT_PHRASES}:
        return True
    return False


def parse_json_args(arguments: str) -> Dict[str, Any]:
    if not arguments:
        return {}
    try:
        data = json.loads(arguments)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


class DialogueOrchestrator:
    def __init__(
        self,
        assembler: PromptAssembler,
        session_log: SessionLog,
        initial_scene: Optional[SceneSpec] = None,
        emit: Optional[EmitFn] = None,
        session_id: str = "",
    ):
        self.assembler = assembler
        self.session_log = session_log
        self.transcript = Transcript()
        self.should_exit = False
        self.scene: Optional[SceneSpec] = None
        self._emit = emit
        self.session_id = session_id or session_log.session_id
        if initial_scene is not None:
            self.phase = "practice"
            self.bundle = self.lock_scene(initial_scene)
        else:
            self.phase = "bootstrap"
            self.bundle = assembler.assemble(AssemblyContext(phase="bootstrap"))

    def emit(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if self._emit is None:
            return
        self._emit(
            CoachEvent(
                type=event_type,
                session_id=self.session_id,
                payload=payload or {},
            )
        )

    @property
    def tools(self):
        return self.bundle.tools

    def scene_dict(self) -> Optional[dict]:
        return self.scene.to_dict() if self.scene else None

    def lock_scene(self, scene: SceneSpec) -> PromptBundle:
        self.scene = scene
        self.phase = "practice"
        self.bundle = self.assembler.assemble(
            AssemblyContext(phase="practice", scene=scene)
        )
        self.session_log.append_scene(scene)
        self.emit("scene.locked", scene.to_dict())
        return self.bundle

    def on_injected_prompt(self, text: str) -> None:
        self._record("assistant", text, source="injected", phase="bootstrap")

    def on_opening(self, text: str) -> None:
        self._record("assistant", text, source="opening", phase="practice")

    def on_user_text(self, text: str) -> None:
        self._record("user", text, source="user", phase=self.phase)

    def on_assistant_text(self, text: str, source: str = "model") -> None:
        if source == "injected":
            self.on_injected_prompt(text)
            return
        if source == "opening":
            self.on_opening(text)
            return
        self._record("assistant", text, source=source, phase=self.phase)

    def _record(
        self,
        role: Literal["user", "assistant"],
        text: str,
        source: str,
        phase: str,
    ) -> None:
        if not text:
            return
        entry = self.transcript.add(role, text, source=source, phase=phase)
        self.session_log.append_utterance(entry)
        self.emit(
            "utterance",
            {
                "role": entry.role,
                "text": entry.text,
                "source": entry.source,
                "phase": entry.phase,
            },
        )

    def request_exit(self) -> None:
        self.should_exit = True

    def close(self) -> None:
        self.session_log.close()

    def handle_tool(
        self, name: str, arguments: str, raw_fallback: str = ""
    ) -> Tuple[str, Optional[PromptBundle]]:
        if name == TOOL_NAME_SET_SCENE:
            scene = SceneSpec.from_mapping(parse_json_args(arguments), raw_fallback)
            bundle = self.lock_scene(scene)
            return "场景已锁定，开始角色扮演。", bundle
        if name == TOOL_NAME_EXIT:
            self.request_exit()
            return "退出成功", None
        return f"{name}工具未注册", None
