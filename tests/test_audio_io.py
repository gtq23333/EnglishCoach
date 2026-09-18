from app.realtime.audio_io import extract_audio_b64, resample_s16le


def test_extract_audio_prefers_delta():
    assert extract_audio_b64({"delta": "abc", "audio": "zzz"}) == "abc"
    assert extract_audio_b64({"audio": "xyz"}) == "xyz"
    assert extract_audio_b64({"data": "d"}) == "d"
    assert extract_audio_b64({"type": "response.output_audio.delta"}) == ""


def test_resample_s16le_doubles_24k_to_48k():
    leftover = bytearray()
    # two int16 samples: 1, -2
    pcm = (1).to_bytes(2, "little", signed=True) + (-2).to_bytes(2, "little", signed=True)
    out = resample_s16le(pcm, 24000, 48000, leftover)
    assert leftover == b""
    assert len(out) == 8


def test_resample_keeps_odd_trailing_byte():
    leftover = bytearray()
    pcm = b"\x01\x00\x02"
    out = resample_s16le(pcm, 24000, 24000, leftover)
    assert out == b"\x01\x00"
    assert leftover == b"\x02"
