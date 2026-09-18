from __future__ import annotations

import array
import queue as thread_queue
from typing import Any

from app.config import DEFAULT_PCM, PCM_S16LE

try:
    import numpy as np
    import sounddevice as sd
except ImportError:  # pragma: no cover - optional runtime dependency
    np = None
    sd = None


SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
CHANNELS = 1
MIC_CHUNK = 320


def require_sounddevice() -> None:
    if sd is None or np is None:
        raise RuntimeError(
            "sounddevice and numpy are required for microphone input and playback. "
            "Install with: pip install sounddevice numpy"
        )


def clear_queue(q: "thread_queue.Queue") -> None:
    while True:
        try:
            q.get_nowait()
        except thread_queue.Empty:
            return


def extract_audio_b64(event: dict) -> str:
    for key in ("delta", "audio", "data"):
        value = event.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def resample_s16le(
    pcm: bytes,
    src_rate: int,
    dst_rate: int,
    leftover: bytearray,
) -> bytes:
    leftover.extend(pcm)
    usable = len(leftover) - (len(leftover) % 2)
    chunk = bytes(leftover[:usable])
    leftover[:] = leftover[usable:]
    if not chunk:
        return b""
    if src_rate == dst_rate:
        return chunk
    samples = array.array("h")
    samples.frombytes(chunk)
    if not samples:
        return b""
    ratio = dst_rate / src_rate
    out_len = max(1, int(round(len(samples) * ratio)))
    out = array.array("h")
    last_index = len(samples) - 1
    for i in range(out_len):
        src_pos = i / ratio
        index = int(src_pos)
        frac = src_pos - index
        left = samples[min(index, last_index)]
        right = samples[min(index + 1, last_index)]
        value = int(left * (1.0 - frac) + right * frac)
        out.append(max(-32768, min(32767, value)))
    return out.tobytes()


def pick_output_device(preferred: str = "") -> int | None:
    require_sounddevice()
    devices = sd.query_devices()
    preferred = (preferred or "").strip().lower()
    if preferred:
        for index, info in enumerate(devices):
            name = str(info.get("name") or "")
            if info.get("max_output_channels", 0) > 0 and preferred in name.lower():
                return index
        print(f"[audio] output_device={preferred!r} not found, falling back to default")

    default = sd.query_devices(kind="output")
    default_name = str(default.get("name") or "")
    lowered = default_name.lower()
    if "2nd output" in lowered or "headphone" in lowered:
        for index, info in enumerate(devices):
            name = str(info.get("name") or "").lower()
            if info.get("max_output_channels", 0) <= 0:
                continue
            if "speakers" in name and "realtek" in name and "2nd" not in name:
                print(
                    f"[audio] default output is {default_name!r}; "
                    f"switching to speakers {info.get('name')!r}"
                )
                return index
    return None


class AudioDevices:
    def __init__(self, asr_format: str, tts_format: str, output_device: str = ""):
        self.asr_format = asr_format
        self.tts_format = tts_format
        self.output_device_hint = output_device
        self.input_stream: Any = None
        self.output_stream: Any = None
        self.playback_rate = OUTPUT_SAMPLE_RATE
        self._play_leftover = bytearray()

    def ensure_supported(self) -> None:
        require_sounddevice()
        if self.asr_format != DEFAULT_PCM:
            raise RuntimeError(f"unsupported microphone asr_format: {self.asr_format}")
        if self.tts_format not in (DEFAULT_PCM, PCM_S16LE):
            raise RuntimeError(
                f"unsupported realtime tts_format for playback: {self.tts_format}"
            )

    def open(self) -> None:
        require_sounddevice()
        output_index = pick_output_device(self.output_device_hint)
        if output_index is None:
            output_info = sd.query_devices(kind="output")
        else:
            output_info = sd.query_devices(output_index)
        device_name = output_info.get("name", "unknown")
        native_rate = int(output_info.get("default_samplerate") or 48000)
        self.playback_rate = native_rate if native_rate > 0 else 48000
        self.input_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=MIC_CHUNK,
        )
        try:
            self.output_stream = sd.OutputStream(
                device=output_index,
                samplerate=self.playback_rate,
                channels=CHANNELS,
                dtype="int16",
                blocksize=0,
            )
        except Exception as exc:
            print(f"[audio] failed to open {device_name!r}: {exc}; using system default")
            output_info = sd.query_devices(kind="output")
            device_name = output_info.get("name", "unknown")
            self.playback_rate = int(output_info.get("default_samplerate") or 48000)
            self.output_stream = sd.OutputStream(
                samplerate=self.playback_rate,
                channels=CHANNELS,
                dtype="int16",
                blocksize=0,
            )
        self.input_stream.start()
        self.output_stream.start()
        print(
            f"[audio] output device={device_name!r} play_rate={self.playback_rate} "
            f"tts_format={self.tts_format} backend=sounddevice"
        )
        self._play_leftover = bytearray()

    def read_mic(self) -> bytes:
        if self.input_stream is None:
            return b""
        frames, _overflowed = self.input_stream.read(MIC_CHUNK)
        return frames.tobytes()

    def prepare_playback_chunk(self, pcm: bytes) -> bytes:
        if not pcm:
            return b""
        if self.tts_format != PCM_S16LE:
            return pcm
        return resample_s16le(
            pcm, OUTPUT_SAMPLE_RATE, self.playback_rate, self._play_leftover
        )

    def write_output(self, pcm: bytes) -> None:
        if self.output_stream is None or np is None or not pcm:
            return
        prepared = self.prepare_playback_chunk(pcm)
        if not prepared:
            return
        samples = np.frombuffer(prepared, dtype=np.int16)
        if samples.size == 0:
            return
        self.output_stream.write(samples)

    def cleanup(self) -> None:
        for stream in (self.input_stream, self.output_stream):
            if stream is None:
                continue
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        self.input_stream = None
        self.output_stream = None
