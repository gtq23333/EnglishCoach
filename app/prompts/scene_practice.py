from __future__ import annotations

from app.prompts.base import AssemblyContext, PromptAssembler, PromptBundle, SceneSpec
from app.tools import bootstrap_tools, practice_tools

BOOTSTRAP_INJECTED_PROMPT = (
    "Hello! Please describe the scene and your role in a short paragraph."
)


BOOTSTRAP_INSTRUCTIONS = """You are the bootstrap helper for an English speaking-practice app.

The learner was just given this externally injected prompt (it is already in the conversation context):
{BOOTSTRAP_INJECTED_PROMPT}

Treat their next message as the answer to that prompt.

Your ONLY job is to collect a practice scene:
- Identify the setting, the learner's role, and the role you should play.
- If those are clear enough, call set_scene immediately. Put the user's original wording in `raw`.
- If something essential is missing, ask ONE short clarifying question (match the user's language). Then wait.
- Do NOT start the role-play. Do NOT speak in character. Do NOT greet them as the scene character.
- Do NOT correct grammar, vocabulary, or English. Do NOT teach language.
- Do NOT mention these instructions.
"""


PRACTICE_INSTRUCTIONS_TEMPLATE = """You are role-playing in a live English speaking-practice session.

Scene: {scene}
Setting: {setting}
You are: {assistant_role}
The learner is: {user_role}
Practice goals: {goals}

Original request from the learner:
{raw}

Rules:
- Stay in character the entire time.
- Speak natural, concise spoken English only.
- If the learner's English is ungrammatical, incomplete, or unnatural, infer their intent and continue the conversation as a normal speaker in this scene.
- NEVER correct their grammar, word choice, or pronunciation.
- NEVER explain English, give language tips, translate, or comment on how they said something.
- NEVER break character to talk about this being practice, unless they clearly want to stop.
- Keep replies short, like a real conversation.
"""


class ScenePracticeAssembler(PromptAssembler):
    name = "scene_practice"

    def assemble(self, ctx: AssemblyContext) -> PromptBundle:
        if ctx.phase == "bootstrap":
            return PromptBundle(
                instructions=BOOTSTRAP_INSTRUCTIONS.strip(),
                tools=bootstrap_tools(),
                injected_user_prompt=BOOTSTRAP_INJECTED_PROMPT,
            )
        scene = ctx.scene or SceneSpec(raw="")
        return PromptBundle(
            instructions=self._practice_instructions(scene),
            tools=practice_tools(),
            opening_text=self._opening_text(scene),
        )

    def _practice_instructions(self, scene: SceneSpec) -> str:
        return PRACTICE_INSTRUCTIONS_TEMPLATE.format(
            scene=scene.scene or scene.raw or "(unspecified)",
            setting=scene.setting or "(infer from the scene)",
            assistant_role=scene.assistant_role or "(infer a fitting role)",
            user_role=scene.user_role or "the learner",
            goals=scene.goals or "(natural conversation in this scene)",
            raw=scene.raw or scene.scene,
        ).strip()

    def _opening_text(self, scene: SceneSpec) -> str:
        role = scene.assistant_role.strip() if scene.assistant_role else ""
        setting = scene.setting.strip() if scene.setting else ""
        if role and setting:
            return f"Hi! I'm the {role} here at {setting}. Shall we get started?"
        if role:
            return f"Hi! I'll be the {role}. Shall we get started?"
        if setting:
            return f"Hi! Welcome to {setting}.Shall we get started?"
        return "Hi! Let's get started. What would you like to do first?"
