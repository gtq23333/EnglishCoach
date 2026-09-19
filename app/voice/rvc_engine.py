"""Short-circuit RVC inference for coach TTS playback.

Imports only ``infer.lib`` pieces from ``third_party/rvc``. Never starts
Gradio / WebUI and never imports ``configs.config``.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RVC_ROOT = ROOT / "third_party" / "rvc"
DEFAULT_ASSET_DIR = ROOT / "model_cache" / "rvc"
DEFAULT_PTH = ROOT / "model_cache" / "KFK_V2_500.pth" / "KFK_V2_500.pth"
DEFAULT_INDEX = (
    ROOT
    / "model_cache"
    / "KFK_V2_500.pth"
    / "added_IVF3491_Flat_nprobe_1_KFK_V2_500_v2.index"
)
PLAYBACK_SR = 24000
HUBERT_SR = 16000
FRAME = 160

logger = logging.getLogger(__name__)


class RvcNotReadyError(RuntimeError):
    pass


@dataclass
class RvcPaths:
    rvc_root: Path = DEFAULT_RVC_ROOT
    hubert: Path = DEFAULT_ASSET_DIR / "hubert_base.pt"
    rmvpe: Path = DEFAULT_ASSET_DIR / "rmvpe.pt"
    model_pth: Path = DEFAULT_PTH
    index: Path = DEFAULT_INDEX

    def missing(self) -> list[str]:
        needed = {
            "rvc_root": self.rvc_root / "infer" / "lib" / "jit" / "get_synthesizer.py",
            "hubert": self.hubert,
            "rmvpe": self.rmvpe,
            "model_pth": self.model_pth,
            "index": self.index,
        }
        return [name for name, path in needed.items() if not path.exists()]


def _pcm_to_float(pcm: bytes) -> np.ndarray:
    if not pcm:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def _float_to_pcm(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def _ensure_rvc_path(rvc_root: Path) -> None:
    root = str(rvc_root)
    if root not in sys.path:
        sys.path.insert(0, root)


@contextmanager
def _allow_full_checkpoints():
    import torch

    orig = torch.load

    def wrapped(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        try:
            return orig(*args, **kwargs)
        except TypeError:
            kwargs.pop("weights_only", None)
            return orig(*args, **kwargs)

    torch.load = wrapped
    try:
        yield
    finally:
        torch.load = orig


class PlaybackRvc:
    """Convert 24 kHz s16le TTS through RVC and emit 24 kHz s16le.

    Default ``mode='utterance'`` buffers a full assistant turn then runs the
    official offline infer path (pad + protect, no skip_head). Interview
    Q&A does not need hop streaming.
    """

    def __init__(
        self,
        paths: Optional[RvcPaths] = None,
        *,
        f0_up_key: int = 0,
        index_rate: float = 0.0,
        hop_seconds: float = 0.25,
        extra_seconds: float = 2.5,
        crossfade_seconds: float = 0.04,
        playback_sr: int = PLAYBACK_SR,
        mode: str = "utterance",
        protect: float = 0.33,
        pad_seconds: float = 1.0,
        rms_mix_rate: float = 1.0,
    ):
        self.paths = paths or RvcPaths()
        missing = self.paths.missing()
        if missing:
            raise RvcNotReadyError(
                "RVC assets missing: "
                + ", ".join(missing)
                + ". Run: python -m scripts.setup_rvc"
            )
        self.mode = (mode or "utterance").strip().lower()
        if self.mode not in {"utterance", "stream"}:
            self.mode = "utterance"
        self.f0_up_key = int(f0_up_key)
        self.index_rate = float(index_rate)
        self.protect = float(protect)
        self.pad_seconds = max(0.0, float(pad_seconds))
        self.rms_mix_rate = float(rms_mix_rate)
        self.playback_sr = int(playback_sr)
        hop = max(0.08, float(hop_seconds))
        extra = max(hop, float(extra_seconds))
        self.hop_16 = int(round(hop * HUBERT_SR / FRAME)) * FRAME
        self.extra_16 = int(round(extra * HUBERT_SR / FRAME)) * FRAME
        self.skip_head = self.extra_16 // FRAME
        self.return_length = self.hop_16 // FRAME
        self.crossfade_24 = max(
            1, int(round(float(crossfade_seconds) * self.playback_sr))
        )
        self._lock = threading.Lock()
        self._loaded = False
        self._device = None
        self._is_half = False
        self._hubert = None
        self._net_g = None
        self._rmvpe = None
        self._index = None
        self._big_npy = None
        self._tgt_sr = 40000
        self._if_f0 = 1
        self._version = "v2"
        self._to16 = None
        self._to_play = None
        self._cache_pitch = None
        self._cache_pitchf = None
        self._ring16 = None
        self._pending16 = None
        self._fade_tail: Optional[np.ndarray] = None
        self._flushed = True

    def load(self) -> None:
        if self._loaded:
            return
        import torch
        import torch.nn.functional as F
        from torchaudio.transforms import Resample

        _ensure_rvc_path(self.paths.rvc_root)
        self._F = F
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._is_half = self._device.type == "cuda"
        logger.info("Loading RVC on %s (half=%s)", self._device, self._is_half)

        with _allow_full_checkpoints():
            from infer.lib.jit.get_synthesizer import get_synthesizer
            from infer.lib.rmvpe import RMVPE

            self._net_g, cpt = get_synthesizer(str(self.paths.model_pth), self._device)
            self._tgt_sr = int(cpt["config"][-1])
            self._if_f0 = int(cpt.get("f0", 1))
            self._version = str(cpt.get("version", "v1"))
            if self._is_half:
                self._net_g = self._net_g.half()
            else:
                self._net_g = self._net_g.float()
            # Hubert stays fp32: fp16 transformers features smear consonants.
            self._hubert = _load_hubert(
                self.paths.hubert, self._device, is_half=False
            )
            self._rmvpe = RMVPE(
                str(self.paths.rmvpe),
                is_half=self._is_half,
                device=self._device,
                use_jit=False,
            )

        if self.index_rate > 0 and self.paths.index.exists():
            try:
                import faiss

                self._index = faiss.read_index(str(self.paths.index))
                try:
                    self._index.nprobe = max(8, int(getattr(self._index, "nprobe", 1) or 1))
                except Exception:
                    pass
                self._big_npy = self._index.reconstruct_n(0, self._index.ntotal)
            except ImportError:
                logger.warning(
                    "faiss is not installed; RVC will run without the retrieval index. "
                    "Install with: pip install faiss-cpu"
                )
                self._index = None
                self._big_npy = None

        self._to16 = Resample(
            orig_freq=self.playback_sr, new_freq=HUBERT_SR, dtype=torch.float32
        ).to(self._device)
        if self._tgt_sr != self.playback_sr:
            self._to_play = Resample(
                orig_freq=self._tgt_sr, new_freq=self.playback_sr, dtype=torch.float32
            ).to(self._device)
        else:
            self._to_play = None

        self._cache_pitch = torch.zeros(
            1024, device=self._device, dtype=torch.long
        )
        self._cache_pitchf = torch.zeros(
            1024, device=self._device, dtype=torch.float32
        )
        self.begin()
        try:
            dummy = torch.zeros(HUBERT_SR // 2, device=self._device, dtype=torch.float32)
            self._infer_utterance(dummy)
        except Exception:
            logger.warning("RVC GPU warmup skipped", exc_info=True)
        self.begin()
        self._loaded = True
        logger.info(
            "RVC ready tgt_sr=%s version=%s mode=%s pad=%.1fs protect=%.2f",
            self._tgt_sr,
            self._version,
            self.mode,
            self.pad_seconds,
            self.protect,
        )

    def attach_thread(self) -> None:
        """Pin CUDA to the playback thread after loading on a worker thread."""
        if not self._loaded:
            return
        import torch

        if self._device is not None and self._device.type == "cuda":
            torch.cuda.set_device(self._device)
            torch.zeros(1, device=self._device)
            torch.cuda.synchronize()

    def begin(self) -> None:
        import torch

        with self._lock:
            width = self.extra_16 + self.hop_16
            device = self._device or torch.device("cpu")
            self._ring16 = torch.zeros(width, device=device, dtype=torch.float32)
            self._pending16 = torch.zeros(0, device=device, dtype=torch.float32)
            if self._cache_pitch is not None:
                self._cache_pitch.zero_()
                self._cache_pitchf.zero_()
            self._fade_tail = None
            self._flushed = False

    def has_pending(self) -> bool:
        with self._lock:
            if self._flushed:
                return False
            pending = 0 if self._pending16 is None else int(self._pending16.numel())
            return pending > 0

    def push_pcm24(self, pcm: bytes) -> List[bytes]:
        self.load()
        import torch

        wav24 = _pcm_to_float(pcm)
        if wav24.size == 0:
            return []
        with self._lock:
            chunk16 = self._to16(
                torch.from_numpy(wav24).to(self._device)
            ).detach()
            self._pending16 = torch.cat((self._pending16, chunk16), dim=0)
            if self.mode == "utterance":
                return []
            outs: List[bytes] = []
            while int(self._pending16.numel()) >= self.hop_16:
                hop = self._pending16[: self.hop_16]
                self._pending16 = self._pending16[self.hop_16 :]
                pcm_out = self._convert_hop(hop)
                if pcm_out:
                    outs.append(pcm_out)
            return outs

    def flush(self) -> List[bytes]:
        self.load()
        import torch

        with self._lock:
            if self._flushed:
                return []
            self._flushed = True
            leftover = self._pending16
            self._pending16 = torch.zeros(0, device=self._device, dtype=torch.float32)
            n = 0 if leftover is None else int(leftover.numel())
            if n < FRAME:
                self._fade_tail = None
                return []
            if self.mode == "utterance":
                audio = self._infer_utterance(leftover)
                self._fade_tail = None
                if audio.size == 0:
                    return []
                return [_float_to_pcm(audio)]
            if n < self.hop_16:
                leftover = torch.nn.functional.pad(leftover, (0, self.hop_16 - n))
            pcm_out = self._convert_hop(leftover[: self.hop_16], valid_16=n)
            self._fade_tail = None
            return [pcm_out] if pcm_out else []

    def convert_pcm_s16le(self, pcm: bytes, sample_rate: int) -> Tuple[bytes, int]:
        """Whole-utterance helper (offline)."""
        self.begin()
        if sample_rate != self.playback_sr:
            import torch
            from torchaudio.transforms import Resample

            wav = torch.from_numpy(_pcm_to_float(pcm)).float()
            wav = Resample(sample_rate, self.playback_sr)(wav)
            pcm = _float_to_pcm(wav.numpy())
        pieces = self.push_pcm24(pcm)
        pieces.extend(self.flush())
        return b"".join(pieces), self.playback_sr

    def convert_file(self, input_path: str | Path, output_path: str | Path) -> Path:
        import soundfile as sf

        data, sr = sf.read(str(input_path), always_2d=False)
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)
        pcm = _float_to_pcm(np.asarray(data, dtype=np.float32))
        out_pcm, out_sr = self.convert_pcm_s16le(pcm, int(sr))
        out = Path(output_path)
        sf.write(str(out), _pcm_to_float(out_pcm), out_sr)
        return out

    def _convert_hop(self, hop16, valid_16: Optional[int] = None) -> bytes:
        import torch

        self._ring16[:- self.hop_16] = self._ring16[self.hop_16 :].clone()
        self._ring16[-self.hop_16 :] = hop16
        converted = self._infer_window(self._ring16)
        if converted.size == 0:
            return b""
        if valid_16 is not None and valid_16 < self.hop_16:
            keep = max(1, int(round(converted.size * (valid_16 / float(self.hop_16)))))
            converted = converted[:keep]
        if self._fade_tail is not None and converted.size and self._fade_tail.size:
            n = min(self.crossfade_24, converted.size, self._fade_tail.size)
            if n > 0:
                fade_in = np.linspace(0.0, 1.0, n, dtype=np.float32)
                fade_out = 1.0 - fade_in
                converted[:n] = converted[:n] * fade_in + self._fade_tail[-n:] * fade_out
        if converted.size >= self.crossfade_24:
            self._fade_tail = converted[-self.crossfade_24 :].copy()
        else:
            self._fade_tail = converted.copy()
        return _float_to_pcm(converted)

    def _infer_window(self, input_wav):
        import torch

        F = self._F
        with torch.no_grad():
            feats = input_wav.float().view(1, -1)
            padding_mask = torch.zeros(feats.shape, dtype=torch.bool, device=self._device)
            inputs = {
                "source": feats,
                "padding_mask": padding_mask,
                "output_layer": 9 if self._version == "v1" else 12,
            }
            logits = self._hubert.extract_features(**inputs)
            feats = (
                self._hubert.final_proj(logits[0])
                if self._version == "v1"
                else logits[0]
            )
            feats = torch.cat((feats, feats[:, -1:, :]), 1)
            feats = self._blend_index(feats, skip_frames=self.skip_head // 2)
            p_len = int(input_wav.shape[0] // FRAME)
            if self._if_f0 == 1:
                f0_extractor_frame = self.hop_16 + 800
                f0_extractor_frame = 5120 * ((f0_extractor_frame - 1) // 5120 + 1) - FRAME
                f0_wav = input_wav[-f0_extractor_frame:]
                f0 = self._rmvpe.infer_from_audio(f0_wav, thred=0.03)
                f0 = f0 * pow(2, self.f0_up_key / 12)
                pitch, pitchf = self._f0_post(f0)
                shift = self.hop_16 // FRAME
                self._cache_pitch[:-shift] = self._cache_pitch[shift:].clone()
                self._cache_pitchf[:-shift] = self._cache_pitchf[shift:].clone()
                src_p = pitch[3:-1] if pitch.shape[0] > 4 else pitch
                src_f = pitchf[3:-1] if pitchf.shape[0] > 4 else pitchf
                n = int(src_p.shape[0])
                if n > 0:
                    self._cache_pitch[-n:] = src_p[-n:]
                    self._cache_pitchf[-n:] = src_f[-n:]
                cache_pitch = self._cache_pitch[None, -p_len:]
                cache_pitchf = self._cache_pitchf[None, -p_len:]
            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
            feats = feats[:, :p_len, :]
            if self._is_half:
                feats = feats.half()
            p_len_t = torch.LongTensor([p_len]).to(self._device)
            sid = torch.LongTensor([0]).to(self._device)
            skip_head = torch.LongTensor([self.skip_head])
            return_length = torch.LongTensor([self.return_length])
            if self._if_f0 == 1:
                audio, _, _ = self._net_g.infer(
                    feats,
                    p_len_t,
                    cache_pitch,
                    cache_pitchf,
                    sid,
                    skip_head,
                    return_length,
                    return_length,
                )
            else:
                audio, _, _ = self._net_g.infer(
                    feats, p_len_t, sid, skip_head, return_length, return_length
                )
            audio = audio.squeeze(1).float()
            if self._to_play is not None:
                audio = self._to_play(audio)
            return audio.squeeze().detach().cpu().numpy().astype(np.float32)

    def _infer_utterance(self, wav16) -> np.ndarray:
        """Official-style whole-utterance infer: pad, protect, no skip_head."""
        import torch
        from scipy.signal import butter, filtfilt

        F = self._F
        audio = wav16.detach().float().cpu().numpy().astype(np.float32).reshape(-1)
        if audio.size < FRAME:
            return np.zeros(0, dtype=np.float32)
        bh, ah = butter(N=5, Wn=48, btype="high", fs=HUBERT_SR)
        audio = np.asarray(filtfilt(bh, ah, audio), dtype=np.float32)
        pad = int(round(self.pad_seconds * HUBERT_SR))
        audio_pad = np.pad(audio, (pad, pad), mode="reflect") if pad else audio
        wav = torch.from_numpy(np.ascontiguousarray(audio_pad)).to(self._device)
        with torch.no_grad():
            feats_in = wav.float().view(1, -1)
            padding_mask = torch.zeros(
                feats_in.shape, dtype=torch.bool, device=self._device
            )
            logits = self._hubert.extract_features(
                source=feats_in,
                padding_mask=padding_mask,
                output_layer=9 if self._version == "v1" else 12,
            )
            feats = (
                self._hubert.final_proj(logits[0])
                if self._version == "v1"
                else logits[0]
            )
            feats0 = feats.clone()
            feats = self._blend_index(feats, skip_frames=0)
            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
            p_len = int(audio_pad.shape[0] // FRAME)
            if feats.shape[1] < p_len:
                p_len = int(feats.shape[1])
            feats = feats[:, :p_len, :]
            feats0 = feats0[:, :p_len, :]
            if self._if_f0 == 1:
                f0 = self._rmvpe.infer_from_audio(wav, thred=0.03)
                f0 = np.asarray(f0, dtype=np.float64) * pow(2, self.f0_up_key / 12)
                pitch, pitchf = self._f0_post(f0)
                if int(pitch.shape[0]) < p_len:
                    pitch = torch.nn.functional.pad(
                        pitch, (0, p_len - int(pitch.shape[0]))
                    )
                    pitchf = torch.nn.functional.pad(
                        pitchf, (0, p_len - int(pitchf.shape[0]))
                    )
                pitch = pitch[:p_len]
                pitchf = pitchf[:p_len]
                if self.protect < 0.5:
                    pitchff = pitchf.clone()
                    pitchff[pitchf > 0] = 1
                    pitchff[pitchf < 1] = self.protect
                    mix = pitchff.view(1, p_len, 1).to(feats.dtype)
                    feats = feats * mix + feats0 * (1.0 - mix)
                cache_pitch = pitch.view(1, -1)
                cache_pitchf = pitchf.view(1, -1)
            if self._is_half:
                feats = feats.half()
            p_len_t = torch.LongTensor([p_len]).to(self._device)
            sid = torch.LongTensor([0]).to(self._device)
            if self._if_f0 == 1:
                converted, _, _ = self._net_g.infer(
                    feats, p_len_t, cache_pitch, cache_pitchf, sid
                )
            else:
                converted, _, _ = self._net_g.infer(feats, p_len_t, sid)
            converted = converted.squeeze().float()
            tgt_pad = int(round(self.pad_seconds * self._tgt_sr))
            if tgt_pad > 0 and converted.numel() > 2 * tgt_pad:
                converted = converted[tgt_pad:-tgt_pad]
            if self._to_play is not None:
                converted = self._to_play(converted.unsqueeze(0)).squeeze()
            out = converted.detach().cpu().numpy().astype(np.float32)
        if 0.0 < self.rms_mix_rate < 1.0:
            out = _mix_rms(audio, HUBERT_SR, out, self.playback_sr, self.rms_mix_rate)
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 0.99:
            out = out * (0.99 / peak)
        return out

    def _blend_index(self, feats, skip_frames: int = 0):
        import torch

        if self._index is None or self.index_rate == 0:
            return feats
        start = max(0, int(skip_frames))
        npy = feats[0][start:].float().cpu().numpy()
        if npy.size == 0:
            return feats
        score, ix = self._index.search(npy, k=8)
        valid = (
            (ix >= 0).all(axis=1)
            & np.isfinite(score).all(axis=1)
            & (score > 1e-8).all(axis=1)
        )
        if not np.any(valid):
            return feats
        score_v = np.maximum(score[valid], 1e-6)
        weight = np.square(1.0 / score_v)
        weight /= weight.sum(axis=1, keepdims=True)
        blended = np.sum(
            self._big_npy[ix[valid]] * np.expand_dims(weight, axis=2), axis=1
        )
        npy = npy.copy()
        npy[valid] = blended * self.index_rate + npy[valid] * (1.0 - self.index_rate)
        feats[0][start:] = torch.from_numpy(npy).to(
            device=self._device, dtype=feats.dtype
        )
        return feats

    def _f0_post(self, f0):
        import torch

        if not torch.is_tensor(f0):
            f0 = torch.from_numpy(np.asarray(f0))
        f0 = f0.float().to(self._device).squeeze()
        f0_min, f0_max = 50.0, 1100.0
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        f0_mel = 1127 * torch.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        return torch.round(f0_mel).long(), f0


def _mix_rms(source_16k: np.ndarray, sr1: int, converted: np.ndarray, sr2: int, rate: float) -> np.ndarray:
    import librosa
    import torch
    import torch.nn.functional as F

    if converted.size == 0 or source_16k.size == 0:
        return converted
    rms1 = librosa.feature.rms(
        y=source_16k, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )
    rms2 = librosa.feature.rms(
        y=converted, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2
    )
    rms1_t = F.interpolate(
        torch.from_numpy(rms1).unsqueeze(0), size=converted.shape[0], mode="linear"
    ).squeeze()
    rms2_t = F.interpolate(
        torch.from_numpy(rms2).unsqueeze(0), size=converted.shape[0], mode="linear"
    ).squeeze()
    rms2_t = torch.clamp(rms2_t, min=1e-6)
    scale = torch.pow(rms1_t, 1 - rate) * torch.pow(rms2_t, rate - 1)
    return (torch.from_numpy(converted) * scale).numpy().astype(np.float32)


def _load_hubert(path: Path, device, is_half):
    try:
        from fairseq import checkpoint_utils
    except Exception:
        return _load_hubert_contentvec(device, is_half)
    with _allow_full_checkpoints():
        try:
            models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
                [str(path)],
                suffix="",
            )
            hubert = models[0]
            hubert = hubert.to(device)
            hubert = hubert.half() if is_half else hubert.float()
            logger.info("Loaded fairseq Hubert from %s", path)
            return hubert.eval()
        except Exception as fairseq_exc:
            logger.warning(
                "fairseq Hubert load failed (%s); using contentvec",
                fairseq_exc,
            )
            return _load_hubert_contentvec(device, is_half)


class _HubertAdapter:
    """Match fairseq Hubert extract_features used by RVC."""

    def __init__(self, model):
        self.model = model
        self.final_proj = getattr(model, "final_proj", torch_identity_proj)

    def extract_features(self, source, padding_mask=None, output_layer=12):
        attention_mask = None
        if padding_mask is not None:
            attention_mask = (~padding_mask).long()
        out = self.model(
            source.float(),
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = out.hidden_states or ()
        if output_layer is not None and output_layer < len(hidden_states):
            hidden = hidden_states[output_layer]
        else:
            hidden = out.last_hidden_state
        return (hidden, None)


def torch_identity_proj(x):
    return x


def _load_hubert_contentvec(device, is_half):
    """RVC hubert_base.pt is ContentVec, not facebook/hubert-base-ls960."""
    import torch.nn as nn
    from transformers import HubertModel

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
    except Exception:
        pass

    class HubertModelWithFinalProj(HubertModel):
        def __init__(self, config):
            super().__init__(config)
            proj = getattr(config, "classifier_proj_size", None) or config.hidden_size
            self.final_proj = nn.Linear(config.hidden_size, proj)

    model_id = "lengyue233/content-vec-best"
    cache_dir = DEFAULT_ASSET_DIR / "content-vec-best"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        model = HubertModelWithFinalProj.from_pretrained(
            model_id, cache_dir=str(cache_dir), local_files_only=True
        )
    except Exception as exc:
        raise RuntimeError(
            "ContentVec 本地缓存缺失（lengyue233/content-vec-best）。"
            "通话中不会访问 HuggingFace。请把模型放到 "
            f"{cache_dir} 后再开变声。"
        ) from exc
    model = model.to(device)
    model = model.half() if is_half else model.float()
    model.eval()
    logger.info("Using ContentVec Hubert (%s)", model_id)
    return _HubertAdapter(model)


_PLAYBACK_CACHE: Optional[PlaybackRvc] = None
_PLAYBACK_CACHE_KEY: Optional[tuple] = None


def reset_playback_rvc_cache() -> None:
    global _PLAYBACK_CACHE, _PLAYBACK_CACHE_KEY
    _PLAYBACK_CACHE = None
    _PLAYBACK_CACHE_KEY = None


def load_playback_rvc(
    *,
    enabled: bool,
    pth: str = "",
    index: str = "",
    hubert: str = "",
    rmvpe: str = "",
    f0_up_key: int = 0,
    index_rate: float = 0.0,
    hop_seconds: float = 0.25,
    extra_seconds: float = 2.5,
    mode: str = "utterance",
    protect: float = 0.33,
    pad_seconds: float = 1.0,
    rms_mix_rate: float = 1.0,
) -> Optional[PlaybackRvc]:
    if not enabled:
        return None
    from app.gpu import LOAD_LOCK

    paths = RvcPaths()
    if pth:
        paths.model_pth = Path(pth)
        if not paths.model_pth.is_absolute():
            paths.model_pth = ROOT / paths.model_pth
    if index:
        paths.index = Path(index)
        if not paths.index.is_absolute():
            paths.index = ROOT / paths.index
    if hubert:
        paths.hubert = Path(hubert)
        if not paths.hubert.is_absolute():
            paths.hubert = ROOT / paths.hubert
    if rmvpe:
        paths.rmvpe = Path(rmvpe)
        if not paths.rmvpe.is_absolute():
            paths.rmvpe = ROOT / paths.rmvpe
    key = (
        str(paths.model_pth),
        str(paths.index),
        str(paths.hubert),
        str(paths.rmvpe),
        f0_up_key,
        index_rate,
        hop_seconds,
        extra_seconds,
        mode,
        protect,
        pad_seconds,
        rms_mix_rate,
    )
    global _PLAYBACK_CACHE, _PLAYBACK_CACHE_KEY
    with LOAD_LOCK:
        if _PLAYBACK_CACHE is not None and _PLAYBACK_CACHE_KEY == key:
            return _PLAYBACK_CACHE
        engine = PlaybackRvc(
            paths,
            f0_up_key=f0_up_key,
            index_rate=index_rate,
            hop_seconds=hop_seconds,
            extra_seconds=extra_seconds,
            mode=mode,
            protect=protect,
            pad_seconds=pad_seconds,
            rms_mix_rate=rms_mix_rate,
        )
        engine.load()
        _PLAYBACK_CACHE = engine
        _PLAYBACK_CACHE_KEY = key
        return engine
