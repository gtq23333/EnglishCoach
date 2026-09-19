from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple

from app.config import AppConfig

SECRET_LEAVES = {
    ("auth", "api_key"),
    ("llm", "api_key"),
}

WRITE_ONLY_HINTS = ("api_key",)


def _set_path(data: dict, path: tuple[str, ...], value: Any) -> None:
    cursor = data
    for key in path[:-1]:
        nested = cursor.get(key)
        if not isinstance(nested, dict):
            nested = {}
            cursor[key] = nested
        cursor = nested
    cursor[path[-1]] = value


def _get_path(data: dict, path: tuple[str, ...]) -> Any:
    cursor: Any = data
    for key in path:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(key)
    return cursor


def public_settings(cfg: AppConfig) -> dict[str, Any]:
    raw = deepcopy(cfg.raw) if isinstance(cfg.raw, dict) else {}
    for path in SECRET_LEAVES:
        parent = _get_path(raw, path[:-1])
        if isinstance(parent, dict) and path[-1] in parent:
            present = bool(parent.get(path[-1]))
            parent.pop(path[-1], None)
            parent[f"has_{path[-1]}"] = present
    auth = raw.setdefault("auth", {})
    if isinstance(auth, dict):
        auth["has_api_key"] = bool(cfg.api_key)
        if "app_id" in auth:
            auth["has_app_id"] = bool(auth.get("app_id"))
            auth.pop("app_id", None)
    llm = raw.setdefault("llm", {})
    if isinstance(llm, dict):
        llm["has_api_key"] = bool(cfg.llm_api_key)
    return raw


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in WRITE_ONLY_HINTS) or lowered == "app_id"


def merge_settings_patch(raw: dict, patch: dict) -> dict:
    merged = deepcopy(raw) if isinstance(raw, dict) else {}

    def walk(dest: dict, src: dict) -> None:
        for key, value in src.items():
            if key.startswith("has_"):
                continue
            if isinstance(value, dict) and isinstance(dest.get(key), dict):
                walk(dest[key], value)
            elif isinstance(value, dict):
                dest[key] = deepcopy(value)
            else:
                dest[key] = value

    walk(merged, patch)
    return merged


def apply_to_config(cfg: AppConfig, merged: dict) -> list[str]:
    """Copy known fields onto the live AppConfig. Returns notes about restart."""
    notes: list[str] = []
    session = merged.get("session") if isinstance(merged.get("session"), dict) else {}
    if "voice" in session:
        cfg.speaker = str(session["voice"])
    elif "speaker" in session:
        cfg.speaker = str(session["speaker"])
    if "speed" in session:
        cfg.tts_speed = int(session["speed"])
    if "loudness" in session:
        cfg.tts_loudness = int(session["loudness"])
    if "output_device" in session:
        cfg.output_device = str(session["output_device"] or "")
    if "model" in session:
        cfg.duplex_model = str(session["model"])
        notes.append("session.model applies on next call")
    avatar = merged.get("avatar") if isinstance(merged.get("avatar"), dict) else {}
    if "enabled" in avatar:
        cfg.avatar_enabled = bool(avatar["enabled"])
    if "show_preview" in avatar:
        cfg.avatar_show_preview = bool(avatar["show_preview"])
    if "character" in avatar:
        cfg.avatar_character = str(avatar["character"])
    if "weights_dir" in avatar:
        cfg.avatar_weights_dir = str(avatar["weights_dir"])
    if "fps" in avatar:
        cfg.avatar_fps = float(avatar["fps"])
    if "display_scale" in avatar:
        cfg.avatar_display_scale = int(avatar["display_scale"])
    voice = merged.get("voice") if isinstance(merged.get("voice"), dict) else {}
    rvc = voice.get("rvc") if isinstance(voice.get("rvc"), dict) else {}
    if "enabled" in rvc:
        cfg.rvc_enabled = bool(rvc["enabled"])
        notes.append("voice.rvc.enabled applies on next call")
    for key, attr, caster in (
        ("f0_up_key", "rvc_f0_up_key", int),
        ("index_rate", "rvc_index_rate", float),
        ("protect", "rvc_protect", float),
        ("pad_seconds", "rvc_pad_seconds", float),
        ("rms_mix_rate", "rvc_rms_mix_rate", float),
        ("mode", "rvc_mode", str),
    ):
        if key in rvc:
            setattr(cfg, attr, caster(rvc[key]))
            notes.append(f"voice.rvc.{key} applies on next call")
    gate = voice.get("uplink_gate") if isinstance(voice.get("uplink_gate"), dict) else {}
    if "delta_idle_seconds" in gate:
        cfg.delta_idle_seconds = float(gate["delta_idle_seconds"])
    if "commit_timeout_seconds" in gate:
        cfg.commit_timeout_seconds = float(gate["commit_timeout_seconds"])
    if "mute_during_tts" in gate:
        cfg.mute_during_tts = bool(gate["mute_during_tts"])
    if "unmute_holdoff_seconds" in gate:
        cfg.unmute_holdoff_seconds = float(gate["unmute_holdoff_seconds"])
    itemgen = merged.get("itemgen") if isinstance(merged.get("itemgen"), dict) else {}
    if "parse_retries" in itemgen:
        cfg.itemgen_parse_retries = int(itemgen["parse_retries"])
    llm = merged.get("llm") if isinstance(merged.get("llm"), dict) else {}
    if "base_url" in llm:
        cfg.llm_base_url = str(llm["base_url"])
        notes.append("llm.base_url applies on next LLM call")
    if "model" in llm:
        cfg.llm_model = str(llm["model"])
        notes.append("llm.model applies on next LLM call")
    if "api_key" in llm and llm["api_key"]:
        cfg.llm_api_key = str(llm["api_key"])
    auth = merged.get("auth") if isinstance(merged.get("auth"), dict) else {}
    if "api_key" in auth and auth["api_key"]:
        cfg.api_key = str(auth["api_key"])
        notes.append("auth.api_key applies on next call")
    if "resource_id" in auth:
        cfg.resource_id = str(auth["resource_id"])
    prompt = merged.get("prompt") if isinstance(merged.get("prompt"), dict) else {}
    if "assembler" in prompt:
        cfg.assembler = str(prompt["assembler"])
    return notes


def save_settings(cfg: AppConfig, merged: dict) -> None:
    path = Path(cfg.config_path) if cfg.config_path else None
    if path is None or not path.exists():
        cfg.raw = merged
        return
    try:
        import tomli_w
    except ImportError as exc:
        raise RuntimeError("tomli-w is required to write config.toml") from exc
    path.write_bytes(tomli_w.dumps(merged).encode("utf-8"))
    cfg.raw = merged


def patch_settings(cfg: AppConfig, patch: Dict[str, Any]) -> Tuple[dict, list[str]]:
    merged = merge_settings_patch(cfg.raw if isinstance(cfg.raw, dict) else {}, patch)
    notes = apply_to_config(cfg, merged)
    save_settings(cfg, merged)
    return public_settings(cfg), notes
