"""Serialize CUDA model loads so RVC and THA3 do not contend."""

from __future__ import annotations

import threading

LOAD_LOCK = threading.Lock()
