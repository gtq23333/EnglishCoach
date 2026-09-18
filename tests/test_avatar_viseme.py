from __future__ import annotations

from app.avatar.viseme import HELLO_PHRASE, HELLO_TIMELINE, plan_visemes


def test_hello_phrase_keeps_poc_keys():
    planned = plan_visemes("Hello, nice to meet you.", HELLO_TIMELINE[-1][0])
    assert planned[0][0] == 0.0
    assert abs(planned[-1][0] - HELLO_TIMELINE[-1][0]) < 1e-6
    assert planned[1][1].get("mouth_eee") == HELLO_TIMELINE[1][1].get("mouth_eee")


def test_plan_visemes_stretches_to_duration():
    planned = plan_visemes("Hello, nice to meet you.", 1.0)
    assert abs(planned[-1][0] - 1.0) < 1e-6


def test_unknown_sentence_has_vowels_and_rest():
    planned = plan_visemes("cat", 0.8)
    assert abs(planned[-1][0] - 0.8) < 1e-6
    mouths = [keys for _, keys in planned if keys]
    assert any("mouth_aaa" in keys for keys in mouths)


def test_empty_text_is_rest():
    planned = plan_visemes("   ", 0.5)
    assert abs(planned[-1][0] - 0.5) < 1e-6
    assert planned[0][1] == {}
