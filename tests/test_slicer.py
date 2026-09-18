from app.items.filter import FilterDecision
from app.items.slicer import slice_keep_turns


def _keep(seq: int, text: str, assistant: str = "question") -> FilterDecision:
    return FilterDecision(
        action="keep",
        reason="ok",
        user_text=text,
        source_seq=seq,
        preceding_assistant=assistant,
        phase="practice",
    )


def test_slicer_ignores_drops_and_respects_turn_budget():
    decisions = [
        FilterDecision("drop", "meta", "Hello.", 1, "hi", "practice"),
        _keep(2, "I will check the logs and find the issue."),
        _keep(3, "Then I will talk with the team about the delay."),
        _keep(4, "Finally I report to the client."),
    ]
    slices = slice_keep_turns(decisions, max_user_turns_per_slice=2, char_budget=800)
    assert len(slices) == 2
    assert [p.source_seq for p in slices[0].pairs] == [2, 3]
    assert [p.source_seq for p in slices[1].pairs] == [4]


def test_slicer_does_not_split_oversized_pair():
    long_user = "word " * 200
    decisions = [_keep(1, long_user, "short q")]
    slices = slice_keep_turns(decisions, max_user_turns_per_slice=2, char_budget=80)
    assert len(slices) == 1
    assert slices[0].pairs[0].source_seq == 1
