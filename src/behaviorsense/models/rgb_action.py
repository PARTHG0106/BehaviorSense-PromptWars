"""Offline pretrained Video Swin inference and its frozen RGB preprocessing.

Original paper/code: https://github.com/SwinTransformer/Video-Swin-Transformer
Official port/weights: https://github.com/pytorch/vision/tree/v0.26.0/torchvision/models/video
Spatial sampling uses explicitly recorded OpenCV bilinear interpolation. This is
not claimed to be byte-identical to torchvision's tensor resize implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import cv2
import numpy as np


ARCHITECTURES = {
    "swin3d_b": {"feature_dim": 1024,
        "weights_name": "Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1",
        "weights_url": "https://download.pytorch.org/models/swin3d_b_22k-7c6ae6fa.pth"},
    "swin3d_t": {"feature_dim": 768,
        "weights_name": "Swin3D_T_Weights.KINETICS400_V1",
        "weights_url": "https://download.pytorch.org/models/swin3d_t-7615ae03.pth"},
}
RGB_FRAMES = 32
RGB_FPS = 15.0
PREPROCESSING = {
    "schema": "behaviorsense.video_swin_rgb_preprocessing.v1",
    "views": ["scene", "person"],
    "frames": RGB_FRAMES, "sample_fps": RGB_FPS,
    "center": "existing_source_frames_column_15 / existing_source_fps",
    "sample_offsets_seconds": "(arange(32)-15.5)/15",
    "rounding": "numpy.rint_ties_to_even_then_clamp_to_source_bounds",
    "color": "OpenCV_BGR_to_RGB", "input_dtype": "uint8",
    "scene": "centered_square_side_ceil(max(width,height)*256/224)+2; preserve_whole_frame_with_integer_padding_margin",
    "person": "square_union_of_primary_slot_joints_score_at_least_0.3_over_30_center_frames",
    "person_margin_each_side": 0.25, "person_min_side_pixels": 32,
    "person_bounds": "clip_reliable_joint_coordinates_to_image_bounds_then_pad_square_ROI",
    "person_missing": "copy_scene_embedding; roi_observed_false",
    "roi_coordinates": "integer_xyxy_right_bottom_exclusive; may_extend_outside_image",
    "padding_rgb": [124, 116, 104],
    "resize_size": [256, 256], "resize_backend": "cv2.INTER_LINEAR_on_uint8",
    "center_crop_size": [224, 224],
    "rescale": "float32 / 255", "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225], "model_layout": "N,C,T,H,W",
    "feature": "final_normalization_and_global_average_pool_before_classifier",
}


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


PREPROCESSING_SHA256 = canonical_sha256(PREPROCESSING)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _dimensions(width: int, height: int) -> tuple[int, int]:
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 1
           for x in (width, height)):
        raise ValueError("image dimensions must be positive integers")
    return int(width), int(height)


def scene_roi(width: int, height: int) -> np.ndarray:
    """Pad enough that the later 256-to-224 crop retains the original frame."""
    width, height = _dimensions(width, height)
    # Two extra pixels protect both edges when centering an odd-sized image
    # requires asymmetric integer padding before the 16-pixel center crop.
    side = int(math.ceil(max(width, height) * 256 / 224)) + 2
    left, top = (width - side) // 2, (height - side) // 2
    return np.array([left, top, left + side, top + side], dtype=np.int64)


def person_roi(skeleton: np.ndarray, width: int, height: int) -> tuple[np.ndarray, bool]:
    """One fixed square per clip, using no activity labels or secondary person."""
    width, height = _dimensions(width, height)
    pose = np.asarray(skeleton)
    if pose.shape != (30, 2, 17, 3) or not np.isfinite(pose).all():
        raise ValueError("center skeleton must be finite [30,2,17,3] pixel poses")
    if np.any((pose[..., 2] < 0) | (pose[..., 2] > 1)):
        raise ValueError("joint confidence must be in [0,1]")
    primary = pose[:, 0]
    points = primary[..., :2][primary[..., 2] >= .3]
    if not len(points):
        return scene_roi(width, height), False
    # Boolean indexing gives a copy; original skeletons are never modified.
    points = points.astype(np.float64, copy=True)
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    side = int(math.ceil(max(32., float((high - low + 1).max()) * 1.5)))
    left, top = np.floor(center - side / 2).astype(np.int64)
    return np.array([left, top, left + side, top + side], dtype=np.int64), True


def spatial_view_reference(rgb: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Diagnostic-only 12 October NumPy canvas implementation for exact A/B checks."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("spatial preprocessing requires an RGB uint8 [H,W,3] frame")
    roi = np.asarray(roi)
    if roi.shape != (4,) or roi.dtype.kind not in "iu":
        raise ValueError("ROI must be four integer xyxy coordinates")
    left, top, right, bottom = map(int, roi)
    side = right - left
    if side < 1 or bottom - top != side:
        raise ValueError("ROI must be a nonempty square")
    height, width = rgb.shape[:2]
    if side > 2 * max(width, height) + 32:
        raise ValueError("ROI is implausibly larger than the source frame")
    canvas = np.empty((side, side, 3), dtype=np.uint8)
    canvas[:] = PREPROCESSING["padding_rgb"]
    x0, y0, x1, y1 = max(0, left), max(0, top), min(width, right), min(height, bottom)
    if x1 > x0 and y1 > y0:
        canvas[y0 - top:y1 - top, x0 - left:x1 - left] = rgb[y0:y1, x0:x1]
    resized = cv2.resize(canvas, (256, 256), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(resized[16:240, 16:240])


def spatial_view(rgb: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Pad an integer square ROI, resize to 256, and take the center 224 pixels."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("spatial preprocessing requires an RGB uint8 [H,W,3] frame")
    roi = np.asarray(roi)
    if roi.shape != (4,) or roi.dtype.kind not in "iu":
        raise ValueError("ROI must be four integer xyxy coordinates")
    left, top, right, bottom = map(int, roi)
    side = right - left
    if side < 1 or bottom - top != side:
        raise ValueError("ROI must be a nonempty square")
    height, width = rgb.shape[:2]
    if side > 2 * max(width, height) + 32:
        raise ValueError("ROI is implausibly larger than the source frame")
    x0, y0, x1, y1 = max(0, left), max(0, top), min(width, right), min(height, bottom)
    if x1 > x0 and y1 > y0:
        canvas = rgb[y0:y1, x0:x1]
        if (x0, y0, x1, y1) != (left, top, right, bottom):
            # Native constant padding avoids a slow NumPy RGB broadcast over the
            # entire square, including pixels that would immediately be replaced.
            # ISOLATED keeps a NumPy view's parent pixels outside this crop.
            canvas = cv2.copyMakeBorder(canvas, y0 - top, bottom - y1, x0 - left, right - x1,
                cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, value=PREPROCESSING["padding_rgb"])
    else:
        # OpenCV requires a nonempty source. A padding-colored pixel produces the
        # same all-padding canvas even for a square wholly outside the image.
        pixel = np.asarray(PREPROCESSING["padding_rgb"], dtype=np.uint8).reshape(1, 1, 3)
        canvas = cv2.copyMakeBorder(pixel, 0, side - 1, 0, side - 1,
            cv2.BORDER_CONSTANT, value=PREPROCESSING["padding_rgb"])
    resized = cv2.resize(canvas, (256, 256), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(resized[16:240, 16:240])


class RGBActionEncoder:
    """Callable accepting RGB uint8 [views,32,224,224,3] and returning [views,D]."""

    def __init__(self, model, device, feature_dim):
        import torch
        self.model = model.requires_grad_(False).to(device).eval()
        self.device, self.feature_dim = torch.device(device), feature_dim

    def __call__(self, views: np.ndarray, profile=None) -> np.ndarray:
        from contextlib import nullcontext
        import torch
        if (views.ndim != 5 or tuple(views.shape[1:]) != (32, 224, 224, 3)
                or views.dtype != np.uint8 or len(views) < 1):
            raise ValueError("encoder views must be RGB uint8 [N,32,224,224,3]")
        with torch.inference_mode():
            self.model.eval()
            with profile.cuda("transfer") if profile is not None else nullcontext():
                x = torch.from_numpy(np.ascontiguousarray(views)).to(self.device)
            with profile.cuda("preprocess") if profile is not None else nullcontext():
                x = x.permute(0, 4, 1, 2, 3).float().div_(255)
                mean = x.new_tensor(PREPROCESSING["mean"]).view(1, 3, 1, 1, 1)
                std = x.new_tensor(PREPROCESSING["std"]).view(1, 3, 1, 1, 1)
                x.sub_(mean).div_(std)
            with profile.cuda("forward") if profile is not None else nullcontext():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    features = self.model(x)
            with profile.cuda("readback") if profile is not None else nullcontext():
                features = features.float().cpu().numpy()
        if features.shape != (len(views), self.feature_dim) or not np.isfinite(features).all():
            raise RuntimeError("pretrained encoder returned invalid or nonfinite pooled features")
        return features


def load_rgb_encoder(checkpoint_path, expected_sha256, arch, device="cuda"):
    """Load a staged official checkpoint strictly; this function never downloads."""
    if arch not in ARCHITECTURES:
        raise ValueError(f"unsupported RGB architecture {arch!r}")
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
    if file_sha256(checkpoint_path) != expected_sha256:
        raise ValueError("pretrained RGB checkpoint SHA-256 mismatch")
    import torch
    import torchvision
    if not str(torch.__version__).startswith("2.11.") or not str(torchvision.__version__).startswith("0.26."):
        raise RuntimeError("RGB extraction requires the pinned Torch 2.11 / torchvision 0.26 runtime")
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("RGB extraction requires CUDA; CPU fallback is forbidden")
    with torch.cuda.device(target):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("RGB extraction requires CUDA bfloat16 support")
    constructor = getattr(torchvision.models.video, arch)
    model = constructor(weights=None)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=True)
    feature_dim = ARCHITECTURES[arch]["feature_dim"]
    if model.head.in_features != feature_dim:
        raise ValueError("official model feature dimension changed")
    model.head = torch.nn.Identity()
    encoder = RGBActionEncoder(model, target, feature_dim)
    info = {"arch": arch, "weights_sha256": expected_sha256, "feature_dim": feature_dim,
        "weights_name": ARCHITECTURES[arch]["weights_name"],
        "weights_url": ARCHITECTURES[arch]["weights_url"],
        "preprocessing": json.loads(json.dumps(PREPROCESSING)),
        "preprocessing_sha256": PREPROCESSING_SHA256,
        "torch_version": str(torch.__version__), "torchvision_version": str(torchvision.__version__),
        "opencv_version": str(cv2.__version__),
        "device": str(target), "precision": "bfloat16"}
    return encoder, info


__all__ = ["ARCHITECTURES", "RGB_FRAMES", "RGB_FPS", "PREPROCESSING", "PREPROCESSING_SHA256",
           "RGBActionEncoder", "load_rgb_encoder", "scene_roi", "person_roi", "spatial_view",
           "spatial_view_reference", "canonical_sha256", "file_sha256"]
