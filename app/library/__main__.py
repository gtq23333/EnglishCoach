from __future__ import annotations

import argparse
import asyncio
import json
import sys

from app.config import load_config
from app.library.ingest import ingest_session_json
from app.store.db import Store


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest generated items JSON into the local SQLite library.")
    parser.add_argument("command", choices=["ingest"], help="ingest existing data/items/<session_id>.json")
    parser.add_argument("session_ids", nargs="+", help="Session UUID(s)")
    parser.add_argument("--config", default="config.toml")
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    store = Store.from_config(cfg)
    await store.create_tables()
    failed = 0
    try:
        for session_id in args.session_ids:
            try:
                stats = await ingest_session_json(store, session_id, cfg.items_dir)
            except FileNotFoundError:
                print(f"{session_id}: items json not found", file=sys.stderr)
                failed += 1
                continue
            print(json.dumps(stats, ensure_ascii=False))
    finally:
        await store.dispose()
    return 1 if failed else 0


def main() -> None:
    args = parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
