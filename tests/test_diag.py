from __future__ import annotations

from app.diag import classify_exception


def test_classify_voice_timeout():
    info = classify_exception(TimeoutError("timed out"), component="voice")
    assert info["cause"] == "voice_api_timeout"
    assert "语音" in info["hint"]


def test_classify_voice_stream_reset():
    from app.diag import classify_voice_error

    info = classify_voice_error(
        "{'type': 'Internal Server Error', 'code': '55000000', "
        "'message': 'sami error: codes=50700000, desc=rpc error: code = 13 "
        "desc = stream terminated by RST_STREAM with error code: NO_ERROR'}"
    )
    assert info["cause"] == "voice_stream_reset"
    assert info["recoverable"] is True
    assert "太久" in info["hint"] or "掐断" in info["hint"]


def test_classify_hf_cache_missing():
    info = classify_exception(
        OSError("We couldn't connect to 'https://huggingface.co' because local_files_only=True"),
        component="rvc",
    )
    assert info["cause"] == "hf_cache_missing"


def test_classify_missing_avatar_weights():
    info = classify_exception(FileNotFoundError("editor.pt"), component="avatar")
    assert info["cause"] == "missing_local_file"
