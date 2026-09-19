from __future__ import annotations

import io
import logging
import queue as thread_queue
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def rgb_to_jpeg(rgb: np.ndarray, quality: int = 72) -> bytes:
    array = np.asarray(rgb)
    try:
        import cv2

        bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            return buf.tobytes()
    except Exception:
        pass
    from PIL import Image

    image = Image.fromarray(array.astype("uint8"))
    bio = io.BytesIO()
    image.save(bio, format="JPEG", quality=quality)
    return bio.getvalue()


class JpegQueueSink:
    """FrameSink that JPEG-encodes RGB and keeps a small drop-old queue."""

    def __init__(self, maxsize: int = 2):
        self.queue: "thread_queue.Queue[Optional[bytes]]" = thread_queue.Queue(maxsize=maxsize)
        self._closed = False

    def push_frame(self, rgb) -> None:
        if self._closed:
            return
        try:
            payload = rgb_to_jpeg(np.asarray(rgb))
        except Exception:
            logger.exception("avatar jpeg encode failed")
            return
        try:
            self.queue.put_nowait(payload)
        except thread_queue.Full:
            try:
                self.queue.get_nowait()
            except thread_queue.Empty:
                pass
            try:
                self.queue.put_nowait(payload)
            except thread_queue.Full:
                pass

    def pull(self, timeout: float = 0.2) -> Optional[bytes]:
        if self._closed:
            return None
        try:
            return self.queue.get(timeout=timeout)
        except thread_queue.Empty:
            return None

    def close(self) -> None:
        self._closed = True
        try:
            self.queue.put_nowait(None)
        except thread_queue.Full:
            pass
