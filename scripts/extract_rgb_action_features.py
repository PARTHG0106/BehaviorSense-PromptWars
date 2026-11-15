"""Decode each source video once and cache aligned pretrained scene/person features.

Only complete per-video outputs are exported. A timeout raises TimeoutError; it
never turns a processed prefix into a complete feature shard. Existing outputs
must pass identity, content and array-shape checks before they can be reused.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import behaviorsense.models.rgb_action as rgb_action
from behaviorsense.models.rgb_action import (ARCHITECTURES, PREPROCESSING, PREPROCESSING_SHA256,
    RGB_FPS, RGB_FRAMES, canonical_sha256, load_rgb_encoder, person_roi, scene_roi, spatial_view)

SCHEMA = "behaviorsense.rgb_action_features.v1"
SKELETON_FPS = 20.0


def _deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("RGB extraction reached its finite session deadline")


def _file_sha256(path, deadline=None):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            _deadline(deadline)
            h.update(block)
    return h.hexdigest()


def _array_sha256(value):
    array = np.ascontiguousarray(value)
    h = hashlib.sha256(json.dumps([array.dtype.str, list(array.shape)]).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def source_frame_indices(center_frames, source_fps, source_frame_count):
    """Exact 32-frame, 15 Hz requests around the existing source-center frames."""
    centers = np.asarray(center_frames)
    if (centers.ndim != 1 or centers.dtype.kind not in "iu" or (centers < 0).any()
            or not isinstance(source_frame_count, (int, np.integer)) or source_frame_count < 1
            or (centers >= source_frame_count).any()):
        raise ValueError("center frames must be integer source positions inside the video")
    if not math.isfinite(source_fps) or source_fps < SKELETON_FPS:
        raise ValueError("source_fps must be finite and at least the original 20 Hz clock")
    offsets = (np.arange(RGB_FRAMES, dtype=np.float64) - (RGB_FRAMES - 1) / 2) / RGB_FPS
    requested = np.rint(centers[:, None].astype(np.float64) + offsets * source_fps).astype(np.int64)
    clamped = np.clip(requested, 0, int(source_frame_count) - 1)
    return clamped, requested != clamped


def _load_shard(path):
    with np.load(path, allow_pickle=False) as z:
        required = {"skeletons", "starts", "source_frames", "source_fps", "clips", "subjects",
                    "datasets", "extraction_metadata"}
        if not required <= set(z.files):
            raise ValueError(f"source shard lacks required fields: {sorted(required - set(z.files))}")
        meta = json.loads(str(z["extraction_metadata"]))
        data = {key: np.array(z[key], copy=True) for key in required - {"extraction_metadata"}}
    starts, frames, poses = data["starts"], data["source_frames"], data["skeletons"]
    n = len(starts) if starts.ndim == 1 else -1
    if (starts.ndim != 1 or starts.dtype.kind not in "iu" or (starts < 0).any()
            or (starts > np.iinfo(np.int64).max).any() or len(np.unique(starts)) != n):
        raise ValueError("source starts must be unique nonnegative int64-compatible integers")
    if poses.shape != (n, 30, 2, 17, 3) or poses.dtype.kind != "f" or not np.isfinite(poses).all():
        raise ValueError("source skeletons must be finite [N,30,2,17,3] pixel poses")
    if np.any((poses[..., 2] < 0) | (poses[..., 2] > 1)):
        raise ValueError("source joint confidence must be in [0,1]")
    if frames.shape != (n, 30) or frames.dtype.kind not in "iu" or (frames < 0).any():
        raise ValueError("source_frames must be integer [N,30] positions")
    video_id, subject = meta.get("video_id"), meta.get("subject")
    if (not isinstance(video_id, str) or not re.fullmatch(r"P\d{2}T\d{2}C\d{2}", video_id)
            or subject != "p" + video_id[1:3]):
        raise ValueError("source shard has inconsistent video/subject identity")
    for field in ("clips", "subjects", "datasets"):
        if data[field].shape != (n,) or data[field].dtype.kind not in "US":
            raise ValueError(f"source {field} must be a string array [N]")
    if (any(Path(str(clip)).stem != video_id for clip in data["clips"])
            or not np.all(data["subjects"] == subject)
            or not np.all(data["datasets"] == "toyota_untrimmed")):
        raise ValueError("source rows disagree with video/subject identity")
    fps, count = meta.get("source_fps"), meta.get("source_frame_count")
    if (not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps < SKELETON_FPS
            or isinstance(count, bool) or not isinstance(count, int) or count < 1):
        raise ValueError("source shard has an invalid FPS/frame-count contract")
    if (data["source_fps"].shape != (n,) or not np.all(data["source_fps"] == fps)
            or not np.isfinite(data["source_fps"]).all() or (frames >= count).any()):
        raise ValueError("source FPS/frame positions disagree with extraction metadata")
    expected = np.rint((starts[:, None].astype(np.float64) + np.arange(30)) * fps / SKELETON_FPS)
    if not np.array_equal(frames, expected):
        raise ValueError("source frames disagree with starts and the original 20 Hz sampling clock")
    if (meta.get("retained_windows") != n or meta.get("retained_starts") != starts.tolist()
            or not re.fullmatch(r"[0-9a-f]{64}", str(meta.get("source_sha256", "")))):
        raise ValueError("source retained-window audit/hash is inconsistent")
    for value in data.values():
        value.setflags(write=False)
    return data, meta


def _encoder_contract(info):
    if (info.get("arch") not in ARCHITECTURES
            or info.get("feature_dim") != ARCHITECTURES[info["arch"]]["feature_dim"]
            or not re.fullmatch(r"[0-9a-f]{64}", str(info.get("weights_sha256", "")))
            or info.get("preprocessing") != PREPROCESSING
            or info.get("preprocessing_sha256") != PREPROCESSING_SHA256):
        raise ValueError("encoder identity or frozen preprocessing contract is invalid")
    # Validate that the entire recorded identity is finite and serializable.
    canonical_sha256(info)


def _report(out, metadata, began, reused, deadline=None):
    return {key: metadata[key] for key in ("status", "n_windows", "video_id", "subject", "arch",
        "feature_dim", "source_video_sha256", "source_shard_sha256", "encoder_sha256",
        "preprocessing_sha256", "decoded_frames", "retrieved_frames", "encoder_views", "peak_cached_frames")} | {
        "reused": reused, "elapsed_seconds": round(time.monotonic() - began, 6),
        "output_path": str(out), "output_sha256": _file_sha256(out, deadline)}


def _reuse(out, identity, expected, began, deadline=None, spatial_backend="native", prefetch_batches=0):
    with np.load(out, allow_pickle=False) as z:
        if not {"features", "metadata", *expected} <= set(z.files):
            raise ValueError("existing RGB cache lacks required arrays")
        meta = json.loads(str(z["metadata"]))
        if meta.get("spatial_backend", "native") != spatial_backend:
            raise ValueError("existing RGB cache spatial backend differs from requested extraction")
        if meta.get("prefetch_batches", 0) != prefetch_batches:
            raise ValueError("existing RGB cache prefetch configuration differs from requested extraction")
        if (meta.get("status") != "complete" or meta.get("fingerprint") != canonical_sha256(identity)
                or any(meta.get(key) != value for key, value in identity.items())):
            raise ValueError("existing RGB cache identity differs from video/shard/encoder/implementation")
        for key, value in expected.items():
            actual = z[key]
            if actual.dtype != value.dtype or not np.array_equal(actual, value):
                raise ValueError(f"existing RGB cache row/clock identity changed: {key}")
        features = z["features"]
        if (features.shape != (identity["n_windows"], 2, identity["feature_dim"])
                or features.dtype != np.float16 or not np.isfinite(features).all()):
            raise ValueError("existing RGB cache contains invalid features")
        arrays = {**{key: z[key] for key in expected}, "features": features}
        if meta.get("array_sha256") != {key: _array_sha256(value) for key, value in arrays.items()}:
            raise ValueError("existing RGB cache array checksum mismatch")
    return _report(out, meta, began, True, deadline)


def extract_video_features(video_path, shard_path, out_path, encoder, encoder_info,
                           batch_size=8, deadline=None, *, profile=None, spatial_backend="native", prefetch_batches=0):
    """Export all retained rows for one video, or raise without publishing a prefix.

    ``encoder`` accepts RGB uint8 [views,32,224,224,3] and returns [views,D].
    ``deadline`` is an absolute time.monotonic() deadline. No per-window seeking
    occurs; each requested source frame is retrieved and scene-preprocessed once.
    The rolling cache is bounded by the 32-frame clip's source-time span, not by
    recording length. Output rows preserve the source shard's original order.
    """
    began = time.monotonic()
    if spatial_backend not in ("native", "reference"):
        raise ValueError("spatial_backend must be native or reference")
    if type(prefetch_batches) is not int or prefetch_batches not in (0, 2):
        raise ValueError("prefetch_batches must be 0 or the bounded two-batch queue")
    transform = spatial_view if spatial_backend == "native" else rgb_action.spatial_view_reference
    def timing(name):
        return profile.cpu(name) if profile is not None else nullcontext()
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive number of center windows")
    if deadline is not None and (not isinstance(deadline, (int, float)) or not math.isfinite(deadline)):
        raise ValueError("deadline must be a finite absolute monotonic timestamp")
    _deadline(deadline)
    _encoder_contract(encoder_info)
    video, shard, out = map(Path, (video_path, shard_path, out_path))
    if out.resolve() in (video.resolve(), shard.resolve()):
        raise ValueError("RGB output cannot replace its input video or source shard")
    with timing("input_validation"):
        source_shard_sha256 = _file_sha256(shard, deadline)
        data, source = _load_shard(shard)
        source_video_sha256 = _file_sha256(video, deadline)
    if source_video_sha256 != source["source_sha256"]:
        raise ValueError("raw video SHA-256 differs from the verified source shard")
    n, dim = len(data["starts"]), encoder_info["feature_dim"]
    fps, count = float(source["source_fps"]), source["source_frame_count"]
    rgb_frames, boundary = source_frame_indices(data["source_frames"][:, 15], fps, count)
    cap = cv2.VideoCapture(str(video))
    executor = None
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("cannot open raw video for sequential decoding")
    try:
        actual_fps, actual_count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (not math.isfinite(actual_fps) or abs(actual_fps - fps) > 1e-6
                or not math.isfinite(actual_count) or actual_count != count or width < 1 or height < 1):
            raise ValueError("decoded video FPS/frame-count differs from the source clock")
        global_roi = scene_roi(width, height)
        rois, roi_observed = np.empty((n, 4), np.int64), np.empty(n, bool)
        with timing("roi"):
            for row in range(n):
                _deadline(deadline)
                rois[row], roi_observed[row] = person_roi(data["skeletons"][row], width, height)
        implementation = canonical_sha256({"extractor": _file_sha256(__file__),
                                           "rgb_action": _file_sha256(rgb_action.__file__)})
        identity = {"schema": SCHEMA, "video_id": source["video_id"], "subject": source["subject"],
            "n_windows": n, "arch": encoder_info["arch"], "feature_dim": dim,
            "source_video_sha256": source_video_sha256, "source_shard_sha256": source_shard_sha256,
            "encoder_sha256": encoder_info["weights_sha256"], "encoder_info": encoder_info,
            "preprocessing_sha256": PREPROCESSING_SHA256,
            "extraction_implementation_sha256": implementation,
            "source_fps": fps, "source_frame_count": count, "source_width": width, "source_height": height,
            "views": ["scene", "person"]}
        expected = {"starts": data["starts"].astype(np.int64),
            "video_id": np.asarray(source["video_id"]), "subject": np.asarray(source["subject"]),
            "video_ids": np.full(n, source["video_id"]), "subjects": np.full(n, source["subject"]),
            "source_frames_rgb": rgb_frames, "center_source_frames": data["source_frames"][:, 15].astype(np.int64),
            "boundary_duplicate_flags": boundary, "roi_observed": roi_observed,
            "person_rois": rois, "source_fps": np.asarray(fps, dtype=np.float64)}
        if out.exists():
            _deadline(deadline)
            return _reuse(out, identity, expected, began, deadline, spatial_backend, prefetch_batches)
        features = np.empty((n, 2, dim), np.float16)
        order = np.argsort(data["source_frames"][:, 15], kind="stable")
        needed = set(map(int, rgb_frames.ravel()))
        cache, pending_views, pending_rows = {}, [], []
        decoded = retrieved = peak = encoded_views = 0
        peak_inflight = 0
        inflight = deque()
        if prefetch_batches:
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rgb-inference")

        def encode(batch):
            return encoder(batch, profile=profile) if profile is not None else encoder(batch)

        def store_output(values, rows, n_views):
            output = np.asarray(values)
            _deadline(deadline)
            if output.shape != (n_views, dim) or not np.isfinite(output).all():
                raise RuntimeError("encoder output shape/nonfinite features invalidate complete extraction")
            with np.errstate(over="ignore", invalid="ignore"):
                output = output.astype(np.float16)
            if not np.isfinite(output).all():
                raise RuntimeError("encoder features overflow float16 storage")
            for row, scene_index, person_index in rows:
                features[row, 0], features[row, 1] = output[scene_index], output[person_index]

        def complete_oldest():
            _deadline(deadline)
            future, rows, n_views = inflight.popleft()
            remaining = None if deadline is None else max(0., deadline - time.monotonic())
            # Exactly one worker owns the model; publication stays on this thread.
            with timing("encoder_wait"):
                values = future.result(timeout=remaining)
            store_output(values, rows, n_views)

        def flush():
            nonlocal encoded_views, peak_inflight
            if not pending_rows:
                return
            _deadline(deadline)
            if executor is not None and len(inflight) >= prefetch_batches:
                complete_oldest()
            with timing("pack"):
                batch = np.stack(pending_views)
            if executor is None:
                store_output(encode(batch), pending_rows, len(pending_views))
            else:
                inflight.append((executor.submit(encode, batch), tuple(pending_rows), len(pending_views)))
                peak_inflight = max(peak_inflight, len(inflight))
            encoded_views += len(pending_views)
            pending_views.clear()
            pending_rows.clear()

        for position, row in enumerate(order):
            _deadline(deadline)
            indices = rgb_frames[row]
            while decoded <= int(indices[-1]):
                _deadline(deadline)
                with timing("decode"):
                    if not cap.grab():
                        raise RuntimeError(f"video decoder ended before requested source frame {decoded}")
                    if decoded in needed:
                        ok, frame = cap.retrieve()
                        if not ok or frame is None or frame.shape != (height, width, 3) or frame.dtype != np.uint8:
                            raise RuntimeError(f"invalid decoded RGB source frame {decoded}")
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if decoded in needed:
                    with timing("scene_crop"):
                        cache[decoded] = (rgb, transform(rgb, global_roi))
                    retrieved += 1
                    peak = max(peak, len(cache))
                decoded += 1
            if any(int(frame) not in cache for frame in indices):
                raise RuntimeError("rolling source-frame cache lost a required timestamp")
            scene_index = len(pending_views)
            with timing("pack"):
                pending_views.append(np.stack([cache[int(frame)][1] for frame in indices]))
            person_index = scene_index
            if roi_observed[row]:
                person_index = len(pending_views)
                with timing("person_crop"):
                    pending_views.append(np.stack([transform(cache[int(frame)][0], rois[row]) for frame in indices]))
            pending_rows.append((int(row), scene_index, person_index))
            if len(pending_rows) >= batch_size:
                flush()
            next_min = int(rgb_frames[order[position + 1], 0]) if position + 1 < n else count
            for old in [frame for frame in cache if frame < next_min]:
                del cache[old]
        flush()
        while inflight:
            complete_oldest()
        if retrieved != len(needed):
            raise RuntimeError("sequential extraction did not retrieve each requested frame exactly once")
        _deadline(deadline)
        arrays = {**expected, "features": features}
        metadata = {**identity, "fingerprint": canonical_sha256(identity), "status": "complete",
            "spatial_backend": spatial_backend,
            "prefetch_batches": prefetch_batches,
            "encoder_peak_inflight_batches": peak_inflight,
            "decoded_frames": decoded, "retrieved_frames": retrieved, "encoder_views": encoded_views,
            "peak_cached_frames": peak, "seeks": 0, "batch_size_windows": batch_size,
            "boundary_clamped_samples": int(boundary.sum()), "missing_person_windows": int((~roi_observed).sum()),
            "array_sha256": {key: _array_sha256(value) for key, value in arrays.items()},
            "elapsed_seconds": round(time.monotonic() - began, 6)}
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_suffix(out.suffix + ".partial")
        try:
            with timing("serialize"):
                with temporary.open("wb") as handle:
                    np.savez_compressed(handle, **arrays, metadata=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))
            _deadline(deadline)
            temporary.replace(out)
        finally:
            if temporary.exists():
                temporary.unlink()
        return _report(out, metadata, began, False, deadline)
    finally:
        if executor is not None:
            # Cancel queued work after errors. The parent supervisor bounds a
            # native GPU call that cannot cooperate with the deadline.
            executor.shutdown(wait=True, cancel_futures=True)
        cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--expected-sha", required=True)
    ap.add_argument("--arch", choices=tuple(ARCHITECTURES), required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--deadline-seconds", type=float)
    args = ap.parse_args()
    if args.deadline_seconds is not None and (not math.isfinite(args.deadline_seconds) or args.deadline_seconds <= 0):
        ap.error("--deadline-seconds must be positive and finite")
    deadline = time.monotonic() + args.deadline_seconds if args.deadline_seconds is not None else None
    encoder, info = load_rgb_encoder(args.checkpoint, args.expected_sha, args.arch, args.device)
    report = extract_video_features(args.video, args.shard, args.out, encoder, info, args.batch_size, deadline)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
