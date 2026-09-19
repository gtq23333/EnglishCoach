from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)

LLM_TIMEOUT_SECONDS = 120.0


def describe_llm_exception(exc: BaseException, *, model: str, timeout_s: float) -> str:
    """Flatten the exception chain so the UI is not stuck with 'Request timed out'."""
    parts: list[str] = []
    current: Optional[BaseException] = exc
    seen = 0
    while current is not None and seen < 4:
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or (
            current.__context__ if current.__cause__ is None else current.__cause__
        )
        if current is not None and current in {exc}:
            break
        seen += 1
    joined = " <- ".join(parts)
    blob = joined.lower()
    if "timeout" in blob or "timed out" in blob:
        return (
            f"{joined} | model={model} timeout={timeout_s:.0f}s | "
            "百炼在超时时间内未返回 HTTP 响应（不是 JSON 解析失败）"
        )
    return f"{joined} | model={model}"


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
        self._client = AsyncOpenAI(
            base_url=base_url or None,
            api_key=api_key,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=0,
        )

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            # qwen3.8-flash defaults to thinking; itemgen JSON then exceeds the read timeout.
            "extra_body": {"enable_thinking": False},
        }
        if tools:
            kwargs["tools"] = tools
        if response_format:
            kwargs["response_format"] = response_format
        logger.info(
            "llm chat model=%s messages=%s thinking=off timeout=%s",
            self.model,
            len(messages),
            LLM_TIMEOUT_SECONDS,
        )
        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            logger.exception("llm chat failed model=%s", self.model)
            raise RuntimeError(
                describe_llm_exception(
                    exc, model=self.model, timeout_s=LLM_TIMEOUT_SECONDS
                )
            ) from exc
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
