from __future__ import annotations

import argparse
import asyncio
import signal
import sys

from app.config import load_config
from app.events import CoachEvent
from app.service import CoachService


def print_event(event: CoachEvent) -> None:
    payload = event.payload or {}
    kind = event.type
    if kind == "session.started":
        print(f"session_id={event.session_id}")
        print(f"jsonl={payload.get('jsonl_path')}")
    elif kind == "session.ready":
        print(f"[session ready] dialog_id={payload.get('dialog_id')}")
    elif kind == "session.ended":
        print("[session ended]")
    elif kind == "session.error":
        print(f"[error] {payload.get('message')}")
    elif kind == "scene.locked":
        print(
            f"[scene locked] scene={payload.get('scene')!r} "
            f"user={payload.get('user_role')!r} assistant={payload.get('assistant_role')!r}"
        )
    elif kind == "utterance":
        role = payload.get("role")
        text = payload.get("text") or ""
        if role == "assistant":
            print(f"[助手] {text}")
        elif role == "user":
            print(f"[学员] {text}")
    elif kind == "asr.started":
        print("[transcription.started]")
    elif kind == "asr.delta":
        print(f"[ASR delta] {payload.get('text')}")
    elif kind == "asr.completed":
        print(f"[ASR completed] {payload.get('text')}")
    elif kind == "assistant.text_delta":
        print(f"[Chat delta] {payload.get('text')}")
    elif kind == "assistant.text_done":
        pass
    elif kind == "tts.started":
        print(f"[TTS start] tts_type={payload.get('tts_type')}")
    elif kind == "tts.done":
        print(f"[TTS done] status={payload.get('status_code')}")
    elif kind == "rvc.converting":
        print("[RVC] converting full utterance...")
    elif kind == "uplink.muted":
        print(f"[uplink muted] reason={payload.get('reason')}")
    elif kind == "uplink.unmuted":
        print("[uplink unmuted]")
    elif kind == "rvc.loading":
        print("[RVC] loading Kafka voice...")
    elif kind == "rvc.ready":
        print(
            f"[RVC] ready mode={payload.get('mode')} "
            f"pad={payload.get('pad_seconds')}s (full-utterance convert)"
        )
    elif kind == "rvc.failed":
        print(f"[RVC failed] {payload.get('message')}")
    elif kind == "playback.started":
        print(
            f"[playback start] duration={payload.get('duration_s')}s "
            f"bytes={payload.get('bytes')}"
        )
    elif kind == "playback.finished":
        print("[playback finished]")
    elif kind == "avatar.started":
        print(
            f"[avatar] preview={payload.get('show_preview')} "
            f"scale={payload.get('display_scale')}"
        )
    elif kind.startswith("itemgen."):
        print(f"[{kind}] {payload}")


async def _watch_events(service: CoachService, session_id: str) -> None:
    for event in service.bus.history(session_id, last_n=200):
        print_event(event)
    queue = service.bus.subscribe(session_id)
    try:
        while True:
            event = await queue.get()
            print_event(event)
            if event.type == "session.ended":
                return
    finally:
        service.bus.unsubscribe(queue)


async def _text_cli(service: CoachService, session_id: str) -> None:
    while True:
        try:
            line = await asyncio.to_thread(input, "> ")
        except EOFError:
            break
        text = (line or "").strip()
        if not text:
            continue
        await service.send_text(session_id, text)
        if text.lower() in {"quit", "exit", "q", "stop", "退出", "结束"}:
            break


async def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.no_avatar:
        cfg.avatar_enabled = False
        cfg.avatar_show_preview = False
    elif args.avatar:
        cfg.avatar_enabled = True
        cfg.avatar_show_preview = True
    service = CoachService(cfg)
    info = await service.create_session(args.mode, scene=args.scene or None)
    print(f"session_id={info['session_id']}")
    print(f"jsonl={info['jsonl_path']}")
    watcher = asyncio.create_task(_watch_events(service, info["session_id"]))
    stop_event = asyncio.Event()

    def request_stop() -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, request_stop)
        except NotImplementedError:
            pass

    try:
        if args.mode == "text":
            text_task = asyncio.create_task(_text_cli(service, info["session_id"]))
            stop_task = asyncio.create_task(stop_event.wait())
            done, pending = await asyncio.wait(
                {text_task, stop_task, service._active.task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
        else:
            wait_task = asyncio.create_task(service.wait_until_stopped(info["session_id"]))
            stop_task = asyncio.create_task(stop_event.wait())
            done, pending = await asyncio.wait(
                {wait_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
    finally:
        await service.stop_session(info["session_id"])
        watcher.cancel()
        try:
            await watcher
        except asyncio.CancelledError:
            pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="English scene speaking-practice assistant")
    parser.add_argument("--config", default="config.toml", help="Path to config.toml")
    parser.add_argument("--mode", choices=("mic", "text"), default="mic")
    parser.add_argument(
        "--scene",
        default="",
        help="Skip bootstrap and start role-play with this scene description.",
    )
    parser.add_argument(
        "--avatar",
        action="store_true",
        help="Show the 1x THA3 preview window (stand-in for the future UI corner).",
    )
    parser.add_argument("--no-avatar", action="store_true", help="Disable the avatar even if config enables it.")
    return parser.parse_args(argv)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\n[interrupted]")


if __name__ == "__main__":
    main()
