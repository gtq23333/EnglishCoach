from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Union

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


DEFAULT_PCM = "pcm"
PCM_S16LE = "pcm_s16le"
OGG_OPUS = "ogg_opus"
DEFAULT_ENDPOINT = "wss://openspeech.bytedance.com/api/v3/duplex/realtime/dialogue"
DEFAULT_LLM_BASE = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_SESSION_DIR = "data/sessions"
DEFAULT_ITEMS_DIR = "data/items"
DEFAULT_VOICE_BACKEND = "duplex"


@dataclass
class AppConfig:
    api_key: str = ""
    llm_base_url: str = DEFAULT_LLM_BASE
    llm_api_key: str = ""
    llm_model: str = "doubao-seed-1-6-251015"
    duplex_model: str = "1.2.6.1"
    speaker: str = "en_female_dacey_uranus_bigtts"
    asr_format: str = DEFAULT_PCM
    tts_format: str = PCM_S16LE
    output_device: str = ""
    endpoint_url: str = DEFAULT_ENDPOINT
    session_dir: str = DEFAULT_SESSION_DIR
    items_dir: str = DEFAULT_ITEMS_DIR
    assembler: str = "scene_practice"
    resource_id: str = "volc.speech.dialog"
    app_id: str = ""
    voice_backend: str = DEFAULT_VOICE_BACKEND
    delta_idle_seconds: float = 6.0
    commit_timeout_seconds: float = 3.0
    mute_during_tts: bool = True
    unmute_holdoff_seconds: float = 0.08
    max_speech_seconds: float = 0.0
    tts_speed: int = 0
    tts_loudness: int = 0
    itemgen_parse_retries: int = 2
    rvc_enabled: bool = False
    rvc_pth: str = ""
    rvc_index: str = ""
    rvc_hubert: str = ""
    rvc_rmvpe: str = ""
    rvc_f0_up_key: int = 0
    rvc_index_rate: float = 0.0
    rvc_hop_seconds: float = 0.25
    rvc_extra_seconds: float = 2.5
    rvc_mode: str = "utterance"
    rvc_protect: float = 0.33
    rvc_pad_seconds: float = 1.0
    rvc_rms_mix_rate: float = 1.0
    avatar_enabled: bool = True
    avatar_show_preview: bool = False
    avatar_character: str = "data/avatar/kafka_512.png"
    avatar_weights_dir: str = "model_cache/tha3/separable_float"
    avatar_fps: float = 15.0
    avatar_display_scale: int = 1
    db_path: str = "data/coach.sqlite"
    owner_id: str = "local"
    raw: Dict[str, Any] = field(default_factory=dict)
    config_path: str = ""


def _section(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = data.get(name)
    if isinstance(value, dict):
        return value
    return {}


def load_config(path: Union[str, Path] = "config.toml") -> AppConfig:
    config_path = Path(path)
    with config_path.open("rb") as f:
        data = tomllib.load(f)

    auth = _section(data, "auth")
    llm = _section(data, "llm")
    session = _section(data, "session")
    endpoint = _section(data, "endpoint")
    logging = _section(data, "logging")
    analyzer = _section(data, "analyzer")
    prompt = _section(data, "prompt")
    voice = _section(data, "voice")

    cfg = AppConfig(raw=data, config_path=str(config_path.resolve()))
    cfg.api_key = auth.get("api_key") or cfg.api_key
    cfg.resource_id = auth.get("resource_id") or cfg.resource_id
    cfg.app_id = auth.get("app_id") or cfg.app_id
    cfg.llm_base_url = llm.get("base_url") or cfg.llm_base_url
    cfg.llm_api_key = llm.get("api_key") or cfg.llm_api_key
    cfg.llm_model = llm.get("model") or cfg.llm_model
    cfg.duplex_model = session.get("model") or cfg.duplex_model
    cfg.speaker = session.get("voice") or session.get("speaker") or cfg.speaker
    cfg.asr_format = session.get("asr_format") or cfg.asr_format
    cfg.tts_format = session.get("tts_format") or cfg.tts_format
    if "speed" in session:
        cfg.tts_speed = int(session["speed"])
    elif "speech_rate" in session:
        cfg.tts_speed = int(session["speech_rate"])
    if "loudness" in session:
        cfg.tts_loudness = int(session["loudness"])
    elif "loudness_rate" in session:
        cfg.tts_loudness = int(session["loudness_rate"])
    cfg.output_device = session.get("output_device") or cfg.output_device
    cfg.endpoint_url = endpoint.get("url") or cfg.endpoint_url
    cfg.session_dir = (
        logging.get("session_dir")
        or analyzer.get("jsonl_dir")
        or cfg.session_dir
    )
    cfg.items_dir = logging.get("items_dir") or _section(data, "itemgen").get("items_dir") or cfg.items_dir
    cfg.assembler = prompt.get("assembler") or cfg.assembler
    cfg.voice_backend = voice.get("backend") or cfg.voice_backend
    gate = voice.get("uplink_gate") if isinstance(voice.get("uplink_gate"), dict) else {}
    if "delta_idle_seconds" in gate:
        cfg.delta_idle_seconds = float(gate["delta_idle_seconds"])
    if "commit_timeout_seconds" in gate:
        cfg.commit_timeout_seconds = float(gate["commit_timeout_seconds"])
    if "mute_during_tts" in gate:
        cfg.mute_during_tts = bool(gate["mute_during_tts"])
    if "unmute_holdoff_seconds" in gate:
        cfg.unmute_holdoff_seconds = float(gate["unmute_holdoff_seconds"])
    elif "unmute_holdoff_ms" in gate:
        cfg.unmute_holdoff_seconds = float(gate["unmute_holdoff_ms"]) / 1000.0
    if "max_speech_seconds" in gate:
        cfg.max_speech_seconds = float(gate["max_speech_seconds"])
    itemgen = _section(data, "itemgen")
    if "parse_retries" in itemgen:
        cfg.itemgen_parse_retries = int(itemgen["parse_retries"])
    rvc = voice.get("rvc") if isinstance(voice.get("rvc"), dict) else _section(data, "rvc")
    if rvc:
        if "enabled" in rvc:
            cfg.rvc_enabled = bool(rvc["enabled"])
        cfg.rvc_pth = str(rvc.get("pth") or rvc.get("model") or cfg.rvc_pth)
        cfg.rvc_index = str(rvc.get("index") or cfg.rvc_index)
        cfg.rvc_hubert = str(rvc.get("hubert") or cfg.rvc_hubert)
        cfg.rvc_rmvpe = str(rvc.get("rmvpe") or cfg.rvc_rmvpe)
        if "f0_up_key" in rvc:
            cfg.rvc_f0_up_key = int(rvc["f0_up_key"])
        if "index_rate" in rvc:
            cfg.rvc_index_rate = float(rvc["index_rate"])
        if "hop_seconds" in rvc:
            cfg.rvc_hop_seconds = float(rvc["hop_seconds"])
        if "extra_seconds" in rvc:
            cfg.rvc_extra_seconds = float(rvc["extra_seconds"])
        if "mode" in rvc:
            cfg.rvc_mode = str(rvc["mode"]).strip().lower() or cfg.rvc_mode
        if "protect" in rvc:
            cfg.rvc_protect = float(rvc["protect"])
        if "pad_seconds" in rvc:
            cfg.rvc_pad_seconds = float(rvc["pad_seconds"])
        if "rms_mix_rate" in rvc:
            cfg.rvc_rms_mix_rate = float(rvc["rms_mix_rate"])
    avatar = _section(data, "avatar")
    if avatar:
        if "enabled" in avatar:
            cfg.avatar_enabled = bool(avatar["enabled"])
        if "show_preview" in avatar:
            cfg.avatar_show_preview = bool(avatar["show_preview"])
        if "character" in avatar:
            cfg.avatar_character = str(avatar["character"])
        if "weights_dir" in avatar:
            cfg.avatar_weights_dir = str(avatar["weights_dir"])
        elif "weights" in avatar:
            cfg.avatar_weights_dir = str(avatar["weights"])
        if "fps" in avatar:
            cfg.avatar_fps = float(avatar["fps"])
        if "display_scale" in avatar:
            cfg.avatar_display_scale = int(avatar["display_scale"])
    store = _section(data, "store")
    if store:
        if "db_path" in store:
            cfg.db_path = str(store["db_path"])
        if "owner_id" in store:
            cfg.owner_id = str(store["owner_id"]) or cfg.owner_id
    return cfg
