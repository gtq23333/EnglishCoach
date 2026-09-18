"""Fetch RVC source + public inference weights. Does not launch Gradio.

Layout (all under english_coach/):
  third_party/rvc/              official WebUI checkout (code only)
  model_cache/rvc/hubert_base.pt
  model_cache/rvc/rmvpe.pt
  model_cache/KFK_V2_500.pth/   speaker pack (already present)

Public G/D pretrained and UVR weights are training/UVR-only and are skipped.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RVC_DIR = ROOT / "third_party" / "rvc"
ASSET_DIR = ROOT / "model_cache" / "rvc"
GITEE = "https://gitee.com/tizi8/Retrieval-based-Voice-Conversion-WebUI.git"
GITHUB = "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git"

HUBERT = "hubert_base.pt"
RMVPE = "rmvpe.pt"
MIRRORS = [
    "https://hf-mirror.com/lj1995/VoiceConversionWebUI/resolve/main/{name}",
    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/{name}",
]


def _download(name: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"skip existing {dest} ({dest.stat().st_size} bytes)")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_error = None
    for template in MIRRORS:
        url = template.format(name=name)
        print(f"downloading {name} from {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "english-coach-rvc-setup"})
            with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as handle:
                shutil.copyfileobj(resp, handle)
            if tmp.stat().st_size < 1_000_000:
                raise RuntimeError(f"file too small: {tmp.stat().st_size}")
            tmp.replace(dest)
            print(f"saved {dest} ({dest.stat().st_size} bytes)")
            return
        except Exception as exc:
            last_error = exc
            print(f"failed: {exc}")
            if tmp.exists():
                tmp.unlink()
    raise RuntimeError(f"could not download {name}: {last_error}") from last_error


def _link_into_assets() -> None:
    mapping = {
        ASSET_DIR / HUBERT: RVC_DIR / "assets" / "hubert" / HUBERT,
        ASSET_DIR / RMVPE: RVC_DIR / "assets" / "rmvpe" / RMVPE,
    }
    for src, dst in mapping.items():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            continue
        try:
            os.link(src, dst)
            print(f"hardlinked {dst}")
        except OSError:
            shutil.copy2(src, dst)
            print(f"copied {dst}")


def ensure_repo() -> None:
    if (RVC_DIR / "infer" / "modules" / "vc" / "modules.py").exists():
        print(f"RVC repo already at {RVC_DIR}")
        return
    RVC_DIR.parent.mkdir(parents=True, exist_ok=True)
    if RVC_DIR.exists():
        shutil.rmtree(RVC_DIR)
    for url in (GITEE, GITHUB):
        print(f"cloning {url}")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(RVC_DIR)],
                check=True,
            )
            return
        except subprocess.CalledProcessError as exc:
            print(f"clone failed: {exc}")
    raise RuntimeError("could not clone RVC repository")


def main() -> None:
    ensure_repo()
    _download(HUBERT, ASSET_DIR / HUBERT)
    _download(RMVPE, ASSET_DIR / RMVPE)
    _link_into_assets()
    print("done. Use app.voice.rvc_sdk.RvcConverter (no WebUI).")


if __name__ == "__main__":
    main()
