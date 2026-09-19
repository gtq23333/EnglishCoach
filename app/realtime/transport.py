from __future__ import annotations

import queue as thread_queue
import time
from typing import Optional, Protocol

from app.realtime.audio_io import (
    MIC_CHUNK,
    OUTPUT_SAMPLE_RATE,
    AudioDevices,
)

BYTES_PER_MIC_CHUNK = MIC_CHUNK * 2


class AudioTransport(Protocol):
    playback_rate: int

    def ensure_supported(self) -> None: ...

    def open(self) -> None: ...

    def read_mic(self) -> bytes: ...

    def write_output(self, pcm: bytes) -> None: ...

    def cleanup(self) -> None: ...


class SoundDeviceTransport:
    def __init__(self, asr_format: str, tts_format: str, output_device: str = ""):
        self.devices = AudioDevices(asr_format, tts_format, output_device)
        self.playback_rate = OUTPUT_SAMPLE_RATE

    def ensure_supported(self) -> None:
        self.devices.ensure_supported()

    def open(self) -> None:
        self.devices.open()
        self.playback_rate = self.devices.playback_rate

    def read_mic(self) -> bytes:
        return self.devices.read_mic()

    def write_output(self, pcm: bytes) -> None:
        self.devices.write_output(pcm)

    def cleanup(self) -> None:
        self.devices.cleanup()


class WebQueueTransport:
    """Browser PCM bridge. Does not open sounddevice."""

    def __init__(self):
        self.playback_rate = OUTPUT_SAMPLE_RATE
        self.uplink: "thread_queue.Queue[bytes]" = thread_queue.Queue(maxsize=256)
        self.downlink: "thread_queue.Queue[Optional[bytes]]" = thread_queue.Queue(maxsize=256)
        self._up_buf = bytearray()
        self._closed = False

    def ensure_supported(self) -> None:
        return

    def open(self) -> None:
        self._closed = False

    def push_uplink(self, pcm: bytes) -> None:
        if self._closed or not pcm:
            return
        try:
            self.uplink.put_nowait(pcm)
        except thread_queue.Full:
            try:
                self.uplink.get_nowait()
            except thread_queue.Empty:
                pass
            try:
                self.uplink.put_nowait(pcm)
            except thread_queue.Full:
                pass

    def read_mic(self) -> bytes:
        deadline = time.time() + 0.02
        while len(self._up_buf) < BYTES_PER_MIC_CHUNK and not self._closed:
            remain = deadline - time.time()
            if remain <= 0:
                break
            try:
                chunk = self.uplink.get(timeout=remain)
            except thread_queue.Empty:
                break
            if chunk:
                self._up_buf.extend(chunk)
        if len(self._up_buf) < BYTES_PER_MIC_CHUNK:
            return b""
        out = bytes(self._up_buf[:BYTES_PER_MIC_CHUNK])
        del self._up_buf[:BYTES_PER_MIC_CHUNK]
        return out

    def write_output(self, pcm: bytes) -> None:
        if self._closed or not pcm:
            return
        try:
            self.downlink.put_nowait(pcm)
        except thread_queue.Full:
            try:
                self.downlink.get_nowait()
            except thread_queue.Empty:
                pass
            try:
                self.downlink.put_nowait(pcm)
            except thread_queue.Full:
                pass

    def pull_downlink(self, timeout: float = 0.1) -> Optional[bytes]:
        if self._closed:
            return None
        try:
            item = self.downlink.get(timeout=timeout)
        except thread_queue.Empty:
            return None
        return item

    def cleanup(self) -> None:
        self._closed = True
        try:
            self.downlink.put_nowait(None)
        except thread_queue.Full:
            pass
