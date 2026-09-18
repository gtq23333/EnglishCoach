"""Talking Head Anime 3 runtime. Renders native-cropped RGB frames (1x).

Weights live in ``model_cache/tha3``. The ``tha3`` package lives in
``third_party/tha3``. This module does not create windows — UI later paints
``render_rgb`` into the coach corner.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
THA3_PARENT = ROOT / "third_party"
DEFAULT_WEIGHTS = ROOT / "model_cache" / "tha3" / "separable_float"
DEFAULT_CHARACTER = ROOT / "data" / "avatar" / "kafka_512.png"
CROP_PAD = 12
VIEW_BG = (255, 255, 255)

WEIGHT_FILES = {
    "eyebrow_decomposer": "eyebrow_decomposer.pt",
    "eyebrow_morphing_combiner": "eyebrow_morphing_combiner.pt",
    "face_morpher": "face_morpher.pt",
    "two_algo_face_body_rotator": "two_algo_face_body_rotator.pt",
    "editor": "editor.pt",
}


def _ensure_tha3_path() -> None:
    parent = str(THA3_PARENT)
    if parent not in sys.path:
        sys.path.insert(0, parent)


def _patch_torch_load() -> None:
    import torch

    original = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = _load  # type: ignore[method-assign]


def character_crop_box(rgba: Image.Image, pad: int = CROP_PAD) -> Tuple[int, int, int, int]:
    alpha = np.asarray(rgba.split()[-1])
    ys, xs = np.where(alpha > 16)
    if xs.size == 0:
        return (0, 0, rgba.width, rgba.height)
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(rgba.width, int(xs.max()) + 1 + pad)
    y1 = min(rgba.height, int(ys.max()) + 1 + pad)
    return x0, y0, x1, y1


def lerp_mouth(
    t: float,
    timeline: list[tuple[float, dict[str, float]]],
    n_params: int,
    name_to_index: Dict[str, int],
) -> np.ndarray:
    if not timeline:
        return np.zeros(n_params, dtype=np.float32)
    duration = max(timeline[-1][0], 1e-6)
    t = max(0.0, min(t, duration))
    left = timeline[0]
    right = timeline[-1]
    for i, key in enumerate(timeline):
        if key[0] <= t:
            left = key
            right = timeline[min(i + 1, len(timeline) - 1)]
        else:
            break
    t0, a = left
    t1, b = right
    mix = 0.0 if t1 <= t0 else max(0.0, min(1.0, (t - t0) / (t1 - t0)))
    pose = np.zeros(n_params, dtype=np.float32)
    for name in set(a) | set(b):
        index = name_to_index.get(name)
        if index is None:
            continue
        va = float(a.get(name, 0.0))
        vb = float(b.get(name, 0.0))
        pose[index] = va * (1.0 - mix) + vb * mix
    return pose


class AvatarEngine:
    """GPU poser. Always returns 1x cropped RGB (H, W, 3) uint8."""

    def __init__(
        self,
        character: Path | str = DEFAULT_CHARACTER,
        weights_dir: Path | str = DEFAULT_WEIGHTS,
        device: Optional[str] = None,
    ):
        import torch

        _ensure_tha3_path()
        _patch_torch_load()
        from tha3.poser.modes.separable_float import create_poser
        from tha3.util import extract_pytorch_image_from_PIL_image, rgba_to_numpy_image

        self.weights_dir = Path(weights_dir)
        missing = [name for name, file in WEIGHT_FILES.items() if not (self.weights_dir / file).exists()]
        if missing:
            raise FileNotFoundError(f"THA3 weights missing in {self.weights_dir}: {missing}")
        character_path = Path(character)
        if not character_path.exists():
            raise FileNotFoundError(character_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        module_file_names = {
            name: str(self.weights_dir / file) for name, file in WEIGHT_FILES.items()
        }
        self.poser = create_poser(self.device, module_file_names=module_file_names)
        self.name_to_index: Dict[str, int] = {}
        index = 0
        for group in self.poser.get_pose_parameter_groups():
            for name in group.get_parameter_names():
                self.name_to_index[name] = index
                index += 1
        self.n_params = self.poser.get_num_parameters()

        pil = Image.open(character_path).convert("RGBA").resize((512, 512), Image.Resampling.LANCZOS)
        self.crop_box = character_crop_box(pil)
        self.native_size = (
            self.crop_box[2] - self.crop_box[0],
            self.crop_box[3] - self.crop_box[1],
        )
        self._image = extract_pytorch_image_from_PIL_image(pil).to(self.device)
        self._to_numpy = rgba_to_numpy_image
        self._latest: Optional[np.ndarray] = None
        self.render_rgb(self.rest_pose())

    def rest_pose(self) -> np.ndarray:
        return np.zeros(self.n_params, dtype=np.float32)

    def pose_from_timeline(
        self, t: float, timeline: list[tuple[float, dict[str, float]]]
    ) -> np.ndarray:
        return lerp_mouth(t, timeline, self.n_params, self.name_to_index)

    def render_rgb(self, pose: Optional[np.ndarray] = None) -> np.ndarray:
        import torch

        if pose is None:
            pose = self.rest_pose()
        pose_t = torch.from_numpy(pose).to(self.device, dtype=self._image.dtype)
        with torch.no_grad():
            out = self.poser.pose(self._image, pose_t, output_index=0)[0]
        np_img = self._to_numpy(out.detach().cpu())
        rgba = np.clip(np_img * 255.0, 0, 255).astype(np.uint8)
        pil = Image.fromarray(rgba, mode="RGBA").crop(self.crop_box)
        rgb = Image.new("RGB", pil.size, VIEW_BG)
        rgb.paste(pil, mask=pil.split()[-1])
        frame = np.asarray(rgb, dtype=np.uint8)
        self._latest = frame
        return frame

    def latest_frame(self) -> Optional[np.ndarray]:
        return self._latest
