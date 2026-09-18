from app.prompts.base import AssemblyContext, SceneSpec
from app.prompts.scene_practice import (
    BOOTSTRAP_INJECTED_PROMPT,
    ScenePracticeAssembler,
)
from app.tools import TOOL_NAME_EXIT, TOOL_NAME_SET_SCENE


def test_bootstrap_injects_scene_prompt_and_forbids_roleplay():
    assembler = ScenePracticeAssembler()
    bundle = assembler.assemble(AssemblyContext(phase="bootstrap"))
    assert bundle.injected_user_prompt == BOOTSTRAP_INJECTED_PROMPT
    assert "当前是第一轮对话" in bundle.injected_user_prompt
    instructions = bundle.instructions.lower()
    assert "do not start the role-play" in instructions
    assert "do not speak in character" in instructions
    tool_names = {tool["name"] for tool in bundle.tools}
    assert TOOL_NAME_SET_SCENE in tool_names
    assert TOOL_NAME_EXIT in tool_names
    assert bundle.opening_text is None


def test_practice_prompt_forbids_correction_and_has_no_inject():
    assembler = ScenePracticeAssembler()
    scene = SceneSpec(
        raw="咖啡店点单",
        scene="ordering coffee",
        user_role="customer",
        assistant_role="barista",
        setting="a coffee shop",
        goals="order a drink politely",
    )
    bundle = assembler.assemble(AssemblyContext(phase="practice", scene=scene))
    assert bundle.injected_user_prompt is None
    assert bundle.opening_text
    assert "barista" in bundle.opening_text.lower()
    instructions = bundle.instructions
    assert "NEVER correct" in instructions
    assert "NEVER explain English" in instructions
    assert "Stay in character" in instructions
    assert "teach" not in instructions.lower() or "NEVER" in instructions
    tool_names = {tool["name"] for tool in bundle.tools}
    assert TOOL_NAME_SET_SCENE not in tool_names
    assert TOOL_NAME_EXIT in tool_names
