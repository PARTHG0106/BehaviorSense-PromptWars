"""Inference for the frozen 15 October RGB hybrid, with a separate T4 encoder port.

The original benchmark used BF16 RGB extraction. This serving implementation uses
FP32 on T4; it preserves weights, preprocessing and thresholds, but does not claim
that the historical benchmark measures this hardware/streaming adapter.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from behaviorsense.data.pose_contract_v2 import normalize_visible_pose, to_bone_visible
from behaviorsense.data.skeleton_dataset import normalise
from behaviorsense.data.temporal_context import build_context_neighbors
from behaviorsense.data.toyota import COARSE_V11
from behaviorsense.models.context_fusion import STGCNContextFusion
from behaviorsense.models.rgb_action import PREPROCESSING, PREPROCESSING_SHA256
from behaviorsense.models.stgcnpp import make_stream
from behaviorsense.models.temporal_fusion import TemporalMultilabelFusion


MODEL_ID = "rgb-hybrid-20261015-t4-fp32-v1"
BUNDLE_SCHEMA = "behaviorsense.rgb_hybrid_serving_bundle.v1"
FROZEN_FILES = {
    "context_seed2.pt": "32f7c33e0234b0931fb28b26144fef9ce3b9e3a096c50d1112009b901a7ef384",
    "temporal_asl_seed1.pt": "8a4bc8f7052d552cf6ce9d539d7b68a0cfb18c6b39ac6ccc3dfdbbb323a4c49c",
    "rgb_head.pt": "f5f75b889387eecb9eb564a10e34311f2e8fa6a6aa6742e32640d17d655b0a25",
    "blend_thresholds.json": "b3b3cd848f347f6a041706ef41ffa54b16fd1e471d952b361918e9da2273bba8",
    "rgb_thresholds.json": "5923b9e24a81947baff02fce79b6180df8f3157504bdbb3bbbe87bdd678631f8",
    "swin3d_t-7615ae03.pth": "7615ae035996b65eb38dad437ae533d2dfcd36f9f89d28c0f0fa7bfb8e6b3130",
}


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def validate_bundle(directory):
    directory = Path(directory)
    manifest = read_json(directory / "hybrid_manifest.json")
    if manifest.get("schema") != BUNDLE_SCHEMA or manifest.get("model_id") != MODEL_ID:
        raise ValueError("Attach the prepared RGB hybrid serving bundle.")
    if manifest.get("preprocessing_sha256") != PREPROCESSING_SHA256:
        raise ValueError("RGB preprocessing differs from the frozen model.")
    for name, expected in FROZEN_FILES.items():
        if manifest.get("files", {}).get(name, {}).get("sha256") != expected:
            raise ValueError(f"Unexpected model identity: {name}")
        if file_sha256(directory / name) != expected:
            raise ValueError(f"Model file checksum differs: {name}")
    return manifest


def pose_control_features(skeletons, object_context):
    """Frozen RGB-head feature recipe; the context/temporal heads retain legacy normalization."""
    normalized = np.stack([normalize_visible_pose(window) for window in skeletons])
    joint_tensor = torch.from_numpy(normalized.transpose(0, 4, 1, 3, 2).copy())
    bones = to_bone_visible(joint_tensor).numpy().transpose(0, 2, 4, 3, 1)
    features = []
    for values in (normalized, bones):
        flat = values.reshape(len(values), values.shape[1], -1)
        features.extend([flat.mean(axis=1), flat.std(axis=1), flat[:, 0], flat[:, -1]])
        valid = (values[:, 1:, ..., 2] >= .3) & (values[:, :-1, ..., 2] >= .3)
        velocity = (values[:, 1:, ..., :2] - values[:, :-1, ..., :2]) * np.float32(20.)
        velocity = np.where(valid[..., None], velocity, 0.)
        count = valid.sum(axis=1)[..., None].clip(1)
        mean = velocity.sum(axis=1) / count
        variance = ((velocity - mean[:, None]) ** 2 * valid[..., None]).sum(axis=1) / count
        features.extend([mean.reshape(len(values), -1), np.sqrt(variance).reshape(len(values), -1)])
    result = np.concatenate([*features, np.asarray(object_context, dtype=np.float32)], axis=1).astype(np.float32)
    if result.shape != (len(skeletons), 1489) or not np.isfinite(result).all():
        raise ValueError("Invalid corrected-pose/object features")
    return result


def decode_hybrid(context_scores, temporal_scores, rgb_scores, thresholds, rgb_thresholds, starts, groups):
    """Frozen blend and pair rule; no recalibration, softmax or single-label smoothing."""
    values = [np.asarray(value, dtype=np.float32) for value in
              (context_scores, temporal_scores, rgb_scores)]
    n = len(values[0])
    if any(value.shape != (n, 22) or not np.isfinite(value).all()
           or np.any((value < 0) | (value > 1)) for value in values):
        raise ValueError("All three components must supply finite [N,22] probabilities.")
    thresholds, rgb_thresholds = np.asarray(thresholds), np.asarray(rgb_thresholds)
    if any(value.shape != (22,) or not np.isfinite(value).all()
           or np.any((value < 0) | (value > 1)) for value in (thresholds, rgb_thresholds)):
        raise ValueError("Invalid frozen thresholds")
    if not (0 < thresholds[13] < 1 and np.all((rgb_thresholds[[10, 13]] > 0)
                                                           & (rgb_thresholds[[10, 13]] < 1))):
        raise ValueError("The frozen pair thresholds must be inside (0,1).")
    starts, groups = np.asarray(starts), np.asarray(groups, dtype=str)
    if starts.shape != (n,) or starts.dtype.kind not in "iu" or groups.shape != (n,):
        raise ValueError("Invalid window clocks or recording/track identities")
    build_context_neighbors(groups, groups, starts, radius=0)
    baseline = .5 * values[0] + .5 * values[1]
    scores = (.75 * baseline + .25 * values[2]).astype(np.float64)
    confidence = (values[2].astype(np.float64)[:, [10, 13]] / rgb_thresholds[[10, 13]]).min(axis=1)
    supported = np.empty(n, dtype=np.float64)
    witnesses = np.empty(n, dtype=np.int64)
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        rows = rows[np.argsort(starts[rows], kind="stable")]
        times = starts[rows]
        for index, row in enumerate(rows):
            lo = np.searchsorted(times, times[index] - 30, side="left")
            hi = np.searchsorted(times, times[index] + 30, side="right")
            neighbors = rows[lo:hi]
            witness = neighbors[int(np.argmax(confidence[neighbors]))]
            witnesses[row], supported[row] = witness, confidence[witness]
    pair_confidence = np.minimum(supported, scores[:, 13] / thresholds[13])
    pair = pair_confidence > 1.
    decisions = scores > thresholds
    collisions = decisions[:, [10, 13]].all(axis=1) & ~pair
    margins = (scores[:, [10, 13]] - thresholds[[10, 13]]) / np.maximum(1 - thresholds[[10, 13]], 1e-8)
    suppress = np.where(margins[:, 0] <= margins[:, 1], 10, 13)
    decisions[np.flatnonzero(collisions), suppress[collisions]] = False
    decisions[np.ix_(pair, [10, 13])] = True
    return {"scores": scores, "decisions": decisions, "pair_confidence": pair_confidence,
            "pair_witness": witnesses}


class RGBHead(nn.Module):
    def __init__(self, input_dim=3025, hidden=256, dropout=.3):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(hidden, 22))

    def forward(self, features):
        return self.layers(features)


class FrozenRGBHybrid:
    """Three immutable heads. CPU is supported for local verification, never automatic fallback."""
    def __init__(self, directory, device="cuda:0"):
        self.directory, self.device = Path(directory), torch.device(device)
        self.manifest = validate_bundle(self.directory)
        self.class_names = tuple(COARSE_V11)
        context = torch.load(self.directory / "context_seed2.pt", map_location="cpu", weights_only=False)
        temporal = torch.load(self.directory / "temporal_asl_seed1.pt", map_location="cpu", weights_only=False)
        rgb = torch.load(self.directory / "rgb_head.pt", map_location="cpu", weights_only=False)
        self.context = STGCNContextFusion(n_classes=22, base_channels=64, context_hidden=128)
        self.temporal = TemporalMultilabelFusion(n_classes=22, base_channels=64, context_hidden=128,
                                                context_radius=6, dropout=.2)
        self.rgb = RGBHead(**{key: rgb["config"][key] for key in ("input_dim", "hidden", "dropout")})
        for model, checkpoint in ((self.context, context), (self.temporal, temporal), (self.rgb, rgb)):
            if tuple(checkpoint["manifest"]["class_names"]) != self.class_names:
                raise ValueError("Checkpoint class order does not match the serving taxonomy.")
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.float().eval().requires_grad_(False).to(self.device)
        self.mean = np.asarray(rgb["normalization"]["mean"])
        self.std = np.asarray(rgb["normalization"]["std"])
        if self.mean.shape != (3025,) or self.std.shape != (3025,) or np.any(self.std < 1e-4):
            raise ValueError("Invalid frozen RGB normalization")
        self.thresholds = np.asarray(read_json(self.directory / "blend_thresholds.json")["thresholds"])
        self.rgb_thresholds = np.asarray(read_json(self.directory / "rgb_thresholds.json")["thresholds"])

    @torch.inference_mode()
    def component_logits(self, skeletons, context, rgb_features, starts, groups, batch_size=64):
        skeletons, context = np.asarray(skeletons), np.asarray(context, dtype=np.float32)
        rgb_features = np.asarray(rgb_features)
        n = len(skeletons)
        if (skeletons.shape != (n, 30, 2, 17, 3) or context.shape != (n, 401)
                or rgb_features.shape != (n, 2, 768) or rgb_features.dtype != np.float16
                or any(not np.isfinite(value).all() for value in (skeletons, context, rgb_features))):
            raise ValueError("Expected finite pixel poses, 401-D context and float16 [N,2,768] RGB features.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        neighbors = build_context_neighbors(groups, groups, starts, radius=6, stride=15)
        outputs = {name: np.empty((n, 22), dtype=np.float32) for name in ("context", "temporal", "rgb")}
        for lo in range(0, n, batch_size):
            hi = min(n, lo + batch_size)
            raw = skeletons[lo:hi].astype(np.float32)
            normalized = np.stack([normalise(window) for window in raw])
            joints = torch.from_numpy(normalized.transpose(0, 4, 1, 3, 2).copy()).to(self.device)
            center = torch.from_numpy(context[lo:hi]).to(self.device)
            lookup = neighbors[lo:hi]
            sequence = np.zeros((hi - lo, 13, 401), dtype=np.float32)
            observed = lookup >= 0
            sequence[observed] = context[lookup[observed]]
            control = pose_control_features(raw, context[lo:hi])
            combined = np.concatenate((control, rgb_features[lo:hi].reshape(hi - lo, -1)), axis=1)
            standardized = np.clip((combined - self.mean) / self.std, -8., 8.).astype(np.float32)
            with torch.autocast(self.device.type, enabled=False):
                predictions = {
                    "context": self.context(make_stream(joints, "bone"), center),
                    "temporal": self.temporal(joints, torch.from_numpy(sequence).to(self.device)),
                    "rgb": self.rgb(torch.from_numpy(standardized).to(self.device)),
                }
            for name, value in predictions.items():
                outputs[name][lo:hi] = value.cpu().numpy()
        return outputs

    def predict(self, skeletons, context, rgb_features, starts, groups):
        logits = self.component_logits(skeletons, context, rgb_features, starts, groups)
        probabilities = {name: torch.from_numpy(value).sigmoid().numpy() for name, value in logits.items()}
        return decode_hybrid(probabilities["context"], probabilities["temporal"], probabilities["rgb"],
                             self.thresholds, self.rgb_thresholds, starts, groups)


class T4RGBEncoder:
    """Explicit FP32 port: T4 has no native BF16; never silently changes precision."""
    def __init__(self, checkpoint, device="cuda:0"):
        import torchvision
        if file_sha256(checkpoint) != FROZEN_FILES["swin3d_t-7615ae03.pth"]:
            raise ValueError("Unexpected Swin3D-T checkpoint")
        if not torch.__version__.startswith("2.11.") or not torchvision.__version__.startswith("0.26."):
            raise RuntimeError("Use torch 2.11 and torchvision 0.26 for this serving package.")
        self.device = torch.device(device)
        self.model = torchvision.models.video.swin3d_t(weights=None)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        self.model.head = nn.Identity()
        self.model.float().eval().requires_grad_(False).to(self.device)

    @torch.inference_mode()
    def __call__(self, views):
        views = np.asarray(views)
        if views.dtype != np.uint8 or views.ndim != 5 or views.shape[1:] != (32, 224, 224, 3):
            raise ValueError("RGB encoder expects uint8 [N,32,224,224,3].")
        values = torch.from_numpy(views).permute(0, 4, 1, 2, 3).to(self.device, dtype=torch.float32)
        values.div_(255.)
        mean = values.new_tensor(PREPROCESSING["mean"]).view(1, 3, 1, 1, 1)
        std = values.new_tensor(PREPROCESSING["std"]).view(1, 3, 1, 1, 1)
        values.sub_(mean).div_(std)
        with torch.autocast(self.device.type, enabled=False):
            result = self.model(values).float().cpu().numpy()
        if result.shape != (len(views), 768) or not np.isfinite(result).all():
            raise RuntimeError("Swin3D-T returned invalid features.")
        return result.astype(np.float16)
