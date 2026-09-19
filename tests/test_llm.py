from __future__ import annotations

from app.llm import describe_llm_exception


def test_describe_llm_timeout_includes_cause_and_model():
    root = TimeoutError("read timed out")
    mid = RuntimeError("http failed")
    mid.__cause__ = root
    outer = RuntimeError("Request timed out")
    outer.__cause__ = mid
    text = describe_llm_exception(outer, model="qwen3.8-flash", timeout_s=120)
    assert "qwen3.8-flash" in text
    assert "TimeoutError" in text
    assert "JSON" in text
