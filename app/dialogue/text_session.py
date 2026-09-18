from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional

from app.dialogue.orchestrator import DialogueOrchestrator, is_exit_phrase
from app.events import CoachEvent
from app.llm import ChatResult, LLMClient
from app.prompts.base import PromptBundle
from app.tools import to_openai_tools


EmitFn = Callable[[CoachEvent], None]


async def default_input(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


def default_output(text: str) -> None:
    print(text, flush=True)


def _assistant_message(result: ChatResult) -> Dict[str, Any]:
    message: Dict[str, Any] = {
        "role": "assistant",
        "content": result.content or None,
    }
    if result.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in result.tool_calls
        ]
    return message


class TextSession:
    def __init__(
        self,
        orchestrator: DialogueOrchestrator,
        llm: LLMClient,
        emit: Optional[EmitFn] = None,
        output_fn: Optional[Callable[[str], None]] = None,
    ):
        self.orchestrator = orchestrator
        self.llm = llm
        self._emit = emit
        self._output_fn = output_fn
        self._inbox: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self.messages: List[Dict[str, Any]] = []
        self._busy = asyncio.Lock()

    def emit(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if self._emit is None:
            return
        self._emit(
            CoachEvent(
                type=event_type,
                session_id=self.orchestrator.session_id,
                payload=payload or {},
            )
        )

    def _say(self, text: str) -> None:
        if self._output_fn is not None:
            self._output_fn(text)

    async def submit(self, text: str) -> None:
        await self._inbox.put(text)

    async def wait_idle(self) -> None:
        for _ in range(500):
            if self._inbox.empty() and not self._busy.locked():
                return
            await asyncio.sleep(0.01)

    async def stop(self) -> None:
        await self._inbox.put(None)

    async def start(self) -> None:
        self.messages = [
            {"role": "system", "content": self.orchestrator.bundle.instructions}
        ]
        if self.orchestrator.phase == "bootstrap":
            hint = self.orchestrator.bundle.injected_user_prompt or ""
            if hint:
                self._say(f"[助手] {hint}")
                self.orchestrator.on_injected_prompt(hint)
                self.messages.append({"role": "assistant", "content": hint})
        elif self.orchestrator.bundle.opening_text:
            opening = self.orchestrator.bundle.opening_text
            self._say(f"[助手] {opening}")
            self.orchestrator.on_opening(opening)
            self.messages.append({"role": "assistant", "content": opening})
        try:
            while True:
                user = await self._inbox.get()
                if user is None:
                    break
                await self.handle_user(user)
        finally:
            self.orchestrator.close()
            self._say("[session ended]")
            self.emit("session.ended")

    async def handle_user(self, user: str) -> None:
        user = (user or "").strip()
        if not user:
            return
        if is_exit_phrase(user):
            self.orchestrator.request_exit()
            return
        async with self._busy:
            await self._turn(user)

    async def _turn(self, user: str) -> None:
        self.orchestrator.on_user_text(user)
        self.messages.append({"role": "user", "content": user})
        result = await self.llm.chat(
            messages=self.messages,
            tools=to_openai_tools(self.orchestrator.tools),
        )
        reset = await self._apply_result(result, last_user=user)
        if reset is not None:
            self.messages = reset
            return
        while result.tool_calls and not self.orchestrator.should_exit:
            result = await self.llm.chat(
                messages=self.messages,
                tools=to_openai_tools(self.orchestrator.tools),
            )
            reset = await self._apply_result(result, last_user=user)
            if reset is not None:
                self.messages = reset
                break

    async def _apply_result(
        self, result: ChatResult, last_user: str
    ) -> Optional[List[Dict[str, Any]]]:
        if result.tool_calls:
            self.messages.append(_assistant_message(result))
            new_bundle: Optional[PromptBundle] = None
            for call in result.tool_calls:
                output, bundle = self.orchestrator.handle_tool(
                    call.name, call.arguments, raw_fallback=last_user
                )
                self.messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": output}
                )
                if bundle is not None:
                    new_bundle = bundle
            if new_bundle is not None:
                return self._practice_messages(new_bundle)
            return None

        if result.content:
            self._say(f"[助手] {result.content}")
            self.emit("assistant.text_done", {"text": result.content})
            self.orchestrator.on_assistant_text(result.content)
            self.messages.append({"role": "assistant", "content": result.content})
        return None

    def _practice_messages(self, bundle: PromptBundle) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": bundle.instructions}
        ]
        if bundle.opening_text:
            self._say(f"[助手] {bundle.opening_text}")
            self.emit("assistant.text_done", {"text": bundle.opening_text})
            self.orchestrator.on_opening(bundle.opening_text)
            messages.append({"role": "assistant", "content": bundle.opening_text})
        return messages


async def run_text_session(
    orchestrator: DialogueOrchestrator,
    llm: LLMClient,
    *,
    input_fn: Callable[[str], Any] = default_input,
    output_fn: Callable[[str], None] = default_output,
) -> None:
    session = TextSession(orchestrator, llm, output_fn=output_fn)
    runner = asyncio.create_task(session.start())
    await asyncio.sleep(0)
    try:
        while not orchestrator.should_exit and not runner.done():
            raw = input_fn("> ")
            user = (await raw if hasattr(raw, "__await__") else raw)
            user = (user or "").strip()
            if not user:
                continue
            if is_exit_phrase(user):
                await session.stop()
                break
            await session.submit(user)
            await session.wait_idle()
    finally:
        if not runner.done():
            await session.stop()
        await runner
