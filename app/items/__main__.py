from __future__ import annotations

import argparse
import asyncio
import json
import sys

from app.config import load_config
from app.items.generator import generate_for_session
from app.llm import OpenAICompatClient


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate practice items from saved session jsonl files."
    )
    parser.add_argument("session_ids", nargs="+", help="Session UUID(s); jsonl in data/sessions, items in data/items")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--max-user-turns-per-slice", type=int, default=2)
    parser.add_argument("--char-budget", type=int, default=800)
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    llm = OpenAICompatClient(cfg.llm_base_url, cfg.llm_api_key, cfg.llm_model)
    failed = 0
    for session_id in args.session_ids:
        try:
            result = await generate_for_session(
                session_id=session_id,
                session_dir=cfg.session_dir,
                items_dir=cfg.items_dir,
                llm=llm,
                max_user_turns_per_slice=args.max_user_turns_per_slice,
                char_budget=args.char_budget,
                parse_retries=cfg.itemgen_parse_retries,
            )
        except Exception as exc:
            print(f"{session_id}: error {exc}", file=sys.stderr)
            failed += 1
            continue
        print(
            json.dumps(
                {
                    "session_id": session_id,
                    "cases": len(result.get("cases") or []),
                    "skipped": len(result.get("skipped") or []),
                    "slices": result.get("slices"),
                },
                ensure_ascii=False,
            )
        )
    return 1 if failed else 0


def main() -> None:
    args = parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
