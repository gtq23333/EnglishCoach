"""Standalone 1x avatar preview (no coach session).

python -m app.avatar
"""

from __future__ import annotations

import argparse
import sys
import time

from app.avatar.driver import AvatarDriver, attach_sink_loop
from app.avatar.engine import DEFAULT_CHARACTER, DEFAULT_WEIGHTS, AvatarEngine
from app.avatar.preview import TkFrameSink
from app.avatar.viseme import HELLO_PHRASE, HELLO_TIMELINE


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="1x THA3 avatar preview")
    parser.add_argument("--character", default=str(DEFAULT_CHARACTER))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--loop", action="store_true", help="Repeat the hello clip")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    engine = AvatarEngine(character=args.character, weights_dir=args.weights)
    driver = AvatarDriver(engine, fps=15)
    duration = HELLO_TIMELINE[-1][0]
    driver.speak(HELLO_PHRASE, duration)
    sink = TkFrameSink()
    next_loop = {"t": time.perf_counter() + duration}

    def should_stop() -> bool:
        if sink.closed:
            return True
        if time.perf_counter() >= next_loop["t"]:
            if args.loop:
                driver.speak(HELLO_PHRASE, duration)
                next_loop["t"] = time.perf_counter() + duration
            else:
                driver.rest()
                next_loop["t"] = time.perf_counter() + 3600
        return False

    print(
        f"native={engine.native_size} device={engine.device} 1x preview",
        flush=True,
    )
    attach_sink_loop(driver, sink, should_stop)
    print("preview closed", flush=True)


if __name__ == "__main__":
    main()
