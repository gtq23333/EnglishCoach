from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class ChatResult:
    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)


class LLMClient(Protocol):
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        ...


class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str, model: str):
        from openai import AsyncOpenAI

        if not api_key:
            raise RuntimeError("llm.api_key is required in config.toml")
        self.model = model
        self._client = AsyncOpenAI(base_url=base_url or None, api_key=api_key)

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        kwargs: Dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if response_format:
            kwargs["response_format"] = response_format
        resp = await self._client.chat.completions.create(**kwargs)
        message = resp.choices[0].message
        tool_calls: List[ToolCall] = []
        for call in message.tool_calls or []:
            function = call.function
            tool_calls.append(
                ToolCall(
                    id=call.id,
                    name=function.name if function else "",
                    arguments=function.arguments if function else "{}",
                )
            )
        return ChatResult(content=message.content, tool_calls=tool_calls)
