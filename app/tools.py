from __future__ import annotations

from typing import Any, Dict, List


TOOL_NAME_SET_SCENE = "set_scene"
TOOL_NAME_EXIT = "detect_exit_intent"


def set_scene_tool() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": TOOL_NAME_SET_SCENE,
        "description": (
            "Lock the English speaking-practice scene after the user describes it. "
            "Call this only when setting, learner role, and your role are clear enough "
            "to start role-play. Do not call it to begin acting yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "scene": {
                    "type": "string",
                    "description": "Short name of the scenario, e.g. ordering coffee.",
                },
                "setting": {
                    "type": "string",
                    "description": "Where the dialogue happens.",
                },
                "user_role": {
                    "type": "string",
                    "description": "Role the learner will play.",
                },
                "assistant_role": {
                    "type": "string",
                    "description": "Role you will play.",
                },
                "goals": {
                    "type": "string",
                    "description": "What the learner wants to practice or achieve.",
                },
                "raw": {
                    "type": "string",
                    "description": "The user's original scene description, unmodified.",
                },
            },
            "required": ["scene", "user_role", "assistant_role", "raw"],
        },
    }


def exit_intent_tool() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": TOOL_NAME_EXIT,
        "description": (
            "Call only when the user clearly wants to end this session "
            "(exit, quit, stop, 退出, 结束, 不聊了, 再见, etc.). "
            "Do not call when they only change topic or pause."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
            },
            "required": ["reason"],
        },
    }


def bootstrap_tools() -> List[Dict[str, Any]]:
    return [set_scene_tool(), exit_intent_tool()]


def practice_tools() -> List[Dict[str, Any]]:
    return [exit_intent_tool()]


def to_openai_tools(duplex_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for tool in duplex_tools:
        if tool.get("function"):
            converted.append(tool)
            continue
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters") or {"type": "object"},
                },
            }
        )
    return converted
