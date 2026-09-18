from app.items.filter import filter_user_turn, strip_echo_prefix


ASSISTANT_ALIGN = (
    "Alright. Let’s dive deeper. For cross-border delivery, how do you usually "
    "align priorities between our local dev team and the overseas client side?"
)
USER_ALIGN_ECHO = (
    "Alright let's dive deep water delivery how do you usually align priorities "
    "between our local 项目经理 and the overseas client so well actually i didn't "
    "have the real uh scenarios and experience about it my opinion i will keep "
    "connecting and keep contact with my domestic team and customers"
)

ASSISTANT_TRANSPARENCY = (
    "Sure. Overseas clients usually demand high transparency for software delivery. "
    "What is one small step you would take to meet that need"
)
USER_TRANSPARENCY_ECHO = (
    "Sure are overseas clients usually demand high trust transparency for software "
    "delivery what is one small step you would take to meet that need um i think "
    "we talk enough today let's uh let's do this tomorrow"
)


def test_drop_tts_artifact_and_meta():
    artifact = filter_user_turn(
        source_seq=3,
        user_text="The voice you selected does not support this language.",
        phase="practice",
        preceding_assistant="hi",
    )
    assert artifact.action == "drop"
    assert artifact.reason == "tts_artifact"

    bootstrap = filter_user_turn(
        source_seq=5,
        user_text="I will play as a candidate",
        phase="bootstrap",
        preceding_assistant="",
    )
    assert bootstrap.action == "drop"
    assert bootstrap.reason == "not_practice"

    meta = filter_user_turn(
        source_seq=12,
        user_text="Keep going.",
        phase="practice",
        preceding_assistant="What experience do you have?",
    )
    assert meta.action == "drop"
    assert meta.reason in {"meta", "too_short"}


def test_strip_echo_keeps_real_answer():
    remainder, matched = strip_echo_prefix(ASSISTANT_ALIGN, USER_ALIGN_ECHO)
    assert matched >= 6
    assert "keep connecting" in remainder.lower() or "didn't have the real" in remainder.lower()
    decision = filter_user_turn(
        source_seq=14,
        user_text=USER_ALIGN_ECHO,
        phase="practice",
        preceding_assistant=ASSISTANT_ALIGN,
    )
    assert decision.action == "strip_echo_prefix"
    assert "align priorities" not in decision.user_text.lower() or "keep connecting" in decision.user_text.lower()


def test_echo_then_exit_is_dropped():
    remainder, matched = strip_echo_prefix(ASSISTANT_TRANSPARENCY, USER_TRANSPARENCY_ECHO)
    assert matched >= 6
    decision = filter_user_turn(
        source_seq=20,
        user_text=USER_TRANSPARENCY_ECHO,
        phase="practice",
        preceding_assistant=ASSISTANT_TRANSPARENCY,
    )
    assert decision.action == "drop"


def test_short_assistant_echo_dropped():
    decision = filter_user_turn(
        source_seq=5,
        user_text="Could you tell me what.",
        phase="practice",
        preceding_assistant=(
            "Could you tell me what specific speaking practice scene you want to simulate, "
            "including the place, your role and my role?"
        ),
    )
    assert decision.action == "drop"
