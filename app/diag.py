from __future__ import annotations

VOICE_STREAM_RESET_MARKERS = (
    "55000000",
    "50700000",
    "rst_stream",
    "stream recv timeout",
    "callwithtimeout",
    "connectionclosed",
    "going away",
)


def classify_voice_error(text: str) -> dict:
    """Classify a duplex ERROR payload or exception string."""
    raw = (text or "").strip() or "unknown voice error"
    low = raw.lower()
    if any(token in low for token in VOICE_STREAM_RESET_MARKERS):
        return {
            "component": "voice",
            "cause": "voice_stream_reset",
            "hint": (
                "火山语音流被服务端掐断（50700000 接收超时）。"
                "常见原因是一句连续说太久，识别一直不收尾。"
            ),
            "error": raw[:800],
            "recoverable": True,
        }
    if "timeout" in low or "timed out" in low:
        return {
            "component": "voice",
            "cause": "voice_api_timeout",
            "hint": "火山引擎语音对话接口超时，API 不可达或网络过慢。",
            "error": raw[:800],
            "recoverable": True,
        }
    return {
        "component": "voice",
        "cause": "voice_api_error",
        "hint": "语音对话 API 失败（密钥、资源授权、线路或服务端内部错误）。",
        "error": raw[:800],
        "recoverable": False,
    }


def classify_exception(exc: BaseException, *, component: str) -> dict:
    """Turn a load/connect error into a frontend-facing cause code."""
    text = f"{type(exc).__name__}: {exc}"
    low = text.lower()
    cause = "other"
    hint = "未分类错误，见 error 字段。"
    if component == "voice":
        return classify_voice_error(text)
    elif "cuda" in low or "out of memory" in low or "oom" in low:
        cause = "gpu"
        hint = "GPU 显存不足或 CUDA 错误。"
    elif (
        "local_files_only" in low
        or "is not a local folder" in low
        or "can't load files" in low
        or "does not appear to have" in low
    ):
        cause = "hf_cache_missing"
        hint = "本地 HuggingFace 缓存没有该模型。通话中不会联网下载。"
    elif any(
        token in low
        for token in ("huggingface", "hf-mirror", "hf.co", "hf-cdn")
    ):
        cause = "hf_unreachable"
        hint = "HuggingFace / hf-mirror 连不上。"
    elif isinstance(exc, FileNotFoundError) or "filenotfound" in low or "no such file" in low:
        cause = "missing_local_file"
        hint = "本地权重或角色图不存在。"
    return {
        "component": component,
        "cause": cause,
        "hint": hint,
        "error": text[:800],
    }
