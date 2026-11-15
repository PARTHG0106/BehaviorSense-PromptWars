"""Extract full annotated TSU coverage with exact source-frame provenance.

This script never installs dependencies or downloads weights. Stage rtmlib, OpenCV,
onnxruntime-gpu and rtmo-l.onnx first. Existing per-video shards are resumed only when
their extraction fingerprint matches. Production defaults to every annotation-eligible
30-frame window on a 15-frame grid at 20 Hz. Explicit stratified selection retains the
bounded pilot workflow. Deterministic video partitions and read-only resume mounts allow
the complete corpus to span multiple GPU sessions without dropping videos or windows.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from behaviorsense.data.toyota import (  # noqa: E402
    CS_TEST_SUBJECTS, CS_TRAIN_SUBJECTS, TSU_CLASSES, TSU_FPS,
    TsuVideo, load_tsu_annotations, parse_tsu_filename,
)
from prepare_skeletons import (  # noqa: E402
    MAX_PERSONS, MIN_VISIBLE_KEYPOINTS, WINDOW_FRAMES, assign_slots, window_clip,
)

SAMPLE_FPS = 20.0
STRATA = ("tv_drinking", "tv_only", "drinking_only", "other_activity", "context")
TV_ID = TSU_CLASSES.index("Watch_TV")
DRINK_IDS = tuple(i for i, name in enumerate(TSU_CLASSES) if name.startswith("Drink."))
SCHEMA_VERSION = 2
CONTEXT_SAMPLE_OFFSET = WINDOW_FRAMES // 2
# Official CS inventory from the released smarthome_CS_51.json. A truncated
# annotation file must not turn its own shorter inventory into a full-corpus claim.
OFFICIAL_SUBSET_VIDEO_COUNTS = {"training": 351, "testing": 185, "all": 536}
OFFICIAL_SUBSET_INVENTORY_SHA256 = {
    "training": "f1764bd2f3f1d24c5482f8153738905302ea0b5870153aae35e8b700a36e0199",
    "testing": "e0667e544593250b72f90036d3a541d15088a4fce0dc3d62752eceeb1d0a1686",
    "all": "717c51345d590fc6da7726db975568fc4076fa392d3ababc523d3da79b0b0802",
}


@dataclass(frozen=True)
class WindowSelection:
    starts: np.ndarray
    source_frames: np.ndarray
    strata: np.ndarray
    coverage: np.ndarray  # TV, drinking, simultaneous, any-annotation frame fractions
    dominant_fine: np.ndarray
    candidate_counts: dict[str, int]
    candidate_windows: int


def source_timeline(total_frames: int, source_fps: float, annotation_duration: int) -> np.ndarray:
    """Map every 20 Hz sample to its original frame, with no duration rescaling."""
    if total_frames < 1 or annotation_duration < 1:
        raise ValueError("source frame count and annotation duration must be positive")
    if not math.isfinite(source_fps) or not SAMPLE_FPS <= source_fps <= 240:
        raise ValueError(f"source fps {source_fps!r} cannot supply real 20 Hz samples")
    # round(), including ties to even, matches notebook 07. There are no duplicated
    # source frames because rates below 20 Hz are rejected instead of upsampled.
    count = int(math.ceil(total_frames * SAMPLE_FPS / source_fps)) + 1
    frames = np.rint(np.arange(count, dtype=np.float64) * source_fps / SAMPLE_FPS).astype(np.int64)
    return frames[(frames < total_frames) & (frames * (TSU_FPS / source_fps) <= annotation_duration)]


def _window_sums(values: np.ndarray) -> np.ndarray:
    cumulative = np.concatenate((np.zeros((1,) + values.shape[1:], np.int64),
                                 np.cumsum(values, axis=0, dtype=np.int64)), axis=0)
    return cumulative[WINDOW_FRAMES:] - cumulative[:-WINDOW_FRAMES]


def select_windows(video: TsuVideo, total_frames: int, source_fps: float, *,
                   budget: int = 64, seed: int = 20261007,
                   candidate_stride: int = 15, min_overlap: float = 0.5,
                   selection: str = "full") -> WindowSelection:
    """All eligible grid windows, or explicit bounded round-robin pilot sampling.

    Full selection depends only on total annotation coverage, never on activity class.
    Descriptive strata are recorded for the coverage audit in both modes. Candidates
    must contain real annotation for at least half the window. No TV interval is
    extended through a drinking interval or an annotation gap.
    """
    if selection not in ("full", "stratified"):
        raise ValueError("selection must be full or stratified")
    if budget < 1 or candidate_stride < 1 or not 0 < min_overlap <= 1:
        raise ValueError("budget/stride must be positive and min_overlap must be in (0,1]")
    timeline = source_timeline(total_frames, source_fps, video.duration)
    if len(timeline) < WINDOW_FRAMES:
        return WindowSelection(np.empty(0, np.int64), np.empty((0, WINDOW_FRAMES), np.int64),
                               np.empty(0, "<U20"), np.empty((0, 4), np.float32),
                               np.empty(0, np.int16), {name: 0 for name in STRATA}, 0)
    positions = timeline * (TSU_FPS / source_fps)
    labels = np.zeros((len(timeline), len(TSU_CLASSES)), dtype=bool)
    for cls, start, end in video.actions:
        if start < 0 or end < start or not 0 <= cls < len(TSU_CLASSES):
            raise ValueError(f"{video.vid}: invalid annotation {(cls, start, end)}")
        labels[(positions >= start) & (positions <= end), cls] = True
    tv = labels[:, TV_ID]
    drink = labels[:, DRINK_IDS].any(axis=1)
    flags = np.column_stack((tv, drink, tv & drink, labels.any(axis=1)))
    starts = np.arange(0, len(timeline) - WINDOW_FRAMES + 1, candidate_stride, dtype=np.int64)
    counts = _window_sums(flags)[starts]
    fine_counts = _window_sums(labels)[starts]
    dominant = fine_counts.argmax(axis=1).astype(np.int16)
    needed = WINDOW_FRAMES * min_overlap
    eligible = counts[:, 3] >= needed
    tags = np.full(len(starts), "context", dtype="<U20")
    tags[(counts[:, 0] == 0) & (counts[:, 1] == 0)] = "other_activity"
    tags[(counts[:, 0] >= needed) & (counts[:, 1] == 0)] = "tv_only"
    tags[(counts[:, 1] >= needed) & (counts[:, 0] == 0)] = "drinking_only"
    tags[(counts[:, 0] >= needed) & (counts[:, 1] >= needed) & (counts[:, 2] > 0)] = "tv_drinking"
    candidate_counts = {name: int(((tags == name) & eligible).sum()) for name in STRATA}
    if selection == "full":
        ix = np.flatnonzero(eligible)
        chosen_starts = starts[ix]
        return WindowSelection(chosen_starts,
                               timeline[chosen_starts[:, None] + np.arange(WINDOW_FRAMES)],
                               tags[ix], counts[ix].astype(np.float32) / WINDOW_FRAMES,
                               dominant[ix], candidate_counts, int(len(starts)))
    # Use a stable per-video seed, independent of corpus iteration and process hash seed.
    salt = int.from_bytes(hashlib.sha256(video.vid.encode()).digest()[:8], "little")
    rng = np.random.default_rng((seed + salt) % (2**64))
    pools: dict[str, deque] = {}
    for name in STRATA:
        indices = np.flatnonzero((tags == name) & eligible)
        if name == "other_activity":
            class_ids = np.unique(dominant[indices])
            rng.shuffle(class_ids)
            classes = [deque(rng.permutation(indices[dominant[indices] == cls]).tolist())
                       for cls in class_ids]
            balanced = []
            while any(classes):
                for group in classes:
                    if group:
                        balanced.append(group.popleft())
            pools[name] = deque(balanced)
        else:
            pools[name] = deque(rng.permutation(indices).tolist())
    selected = []
    while len(selected) < budget and any(pools.values()):
        for name in STRATA:
            if pools[name] and len(selected) < budget:
                selected.append(pools[name].popleft())
    # Storage is temporal even though selection is random; overlaps share inference.
    ix = np.asarray(sorted(selected, key=lambda i: starts[i]), dtype=np.int64)
    chosen_starts = starts[ix]
    source_frames = timeline[chosen_starts[:, None] + np.arange(WINDOW_FRAMES)]
    return WindowSelection(chosen_starts, source_frames, tags[ix],
                           counts[ix].astype(np.float32) / WINDOW_FRAMES,
                           dominant[ix], candidate_counts, int(len(starts)))


def validate_pose(kpts, scores) -> tuple[np.ndarray, np.ndarray]:
    points, confidence = np.asarray(kpts), np.asarray(scores)
    if points.size == confidence.size == 0:
        return np.empty((0, 17, 2), np.float32), np.empty((0, 17), np.float32)
    if points.ndim != 3 or points.shape[1:] != (17, 2) or confidence.shape != points.shape[:2]:
        raise ValueError(f"RTMO must return COCO-17 detections; got {points.shape}/{confidence.shape}")
    if not np.isfinite(points).all() or not np.isfinite(confidence).all():
        raise ValueError("RTMO returned non-finite detections")
    return points, confidence


def require_cuda_session(body, *, name: str = "RTMO"):
    session = getattr(body, "session", None)
    if session is None or not hasattr(session, "get_providers"):
        raise RuntimeError(f"cannot verify {name}'s ONNX Runtime CUDA session")
    providers = session.get_providers()
    if not providers or providers[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"{name} requires CUDA, got providers {providers}; CPU fallback is refused")
    if not hasattr(session, "disable_fallback"):
        raise RuntimeError("ONNX session cannot disable runtime provider fallback")
    session.disable_fallback()
    return session


def load_detector(model: Path):
    if not model.is_file() or model.suffix.lower() != ".onnx":
        raise FileNotFoundError(f"a staged local RTMO .onnx file is required: {model}")
    # Importing torch first exposes its staged CUDA/cuDNN libraries to ORT, matching
    # notebook 07's process. preload_dlls also searches staged NVIDIA wheel libraries
    # on current ORT releases. Both operations are local; neither installs anything.
    import torch  # noqa: F401
    import onnxruntime as ort
    preload = getattr(ort, "preload_dlls", None)
    if callable(preload):
        preload()
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("onnxruntime-gpu with CUDAExecutionProvider is required; no CPU fallback")
    # RTMO is notebook 07's direct model interface. Body's preset selection can resolve
    # models remotely; passing the checked local path to RTMO avoids that mechanism.
    from rtmlib import RTMO
    body = RTMO(onnx_model=str(model.resolve()), model_input_size=(640, 640),
                backend="onnxruntime", device="cuda")
    require_cuda_session(body)
    return body


def decode_selected(path: Path, selection: WindowSelection, body, *,
                    decode_mode: str = "sequential", seek_gap: int = 250,
                    capture_factory=None, context_detector=None
                    ) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray | None]:
    """Infer only selected unique frames; reset identity association across sample gaps.

    sequential uses grab/retrieve, avoiding BGR conversion for unused frames. seek is
    opt-in: large gaps use decoder frame-index checks; a failed check reopens the video
    and reaches the target by sequential grabs. Neither mode changes requested indices.
    Pose/frame storage is a rolling 30-sample cache. RGB is never retained. Output
    arrays are allocated for one video only, and context uses its exact center frame
    and the corresponding tracked pixel-coordinate pose. No annotation labels enter
    the object detector.
    """
    import cv2
    if decode_mode not in ("sequential", "seek"):
        raise ValueError(f"unknown decode mode {decode_mode!r}")
    factory = capture_factory or cv2.VideoCapture
    cap = factory(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"decoder refused {path}")
    if (selection.starts.ndim != 1 or selection.starts.dtype.kind not in "iu" or
            (selection.starts < 0).any() or (np.diff(selection.starts) <= 0).any() or
            selection.source_frames.shape != (len(selection.starts), WINDOW_FRAMES) or
            selection.source_frames.dtype.kind not in "iu" or
            (selection.source_frames < 0).any() or
            (np.diff(selection.source_frames, axis=1) <= 0).any()):
        cap.release()
        raise ValueError("selection must contain sorted unique starts and increasing source-frame windows")
    for index in range(1, len(selection.starts)):
        advance = int(selection.starts[index] - selection.starts[index - 1])
        if advance < WINDOW_FRAMES and not np.array_equal(
                selection.source_frames[index - 1, advance:],
                selection.source_frames[index, :WINDOW_FRAMES - advance]):
            cap.release()
            raise ValueError("overlapping windows disagree on exact source frames")
        if advance >= WINDOW_FRAMES and selection.source_frames[index, 0] <= selection.source_frames[index - 1, -1]:
            cap.release()
            raise ValueError("different sampled positions map to the same or decreasing source frame")
    stats = {"unique_inference_frames": 0, "grabbed_frames": 0, "seeks": 0,
             "seek_fallbacks": 0, "pose_seconds": 0.0, "decode_mode": decode_mode,
             "context_inference_frames": 0, "context_seconds": 0.0, "peak_cached_pose_frames": 0}
    # Adjacent windows share pose inference without a dictionary of the full video.
    def required_frames():
        previous = -1
        for start, frames in zip(selection.starts, selection.source_frames):
            for offset in range(max(0, previous + 1 - int(start)), WINDOW_FRAMES):
                previous = int(start) + offset
                yield int(frames[offset]), previous

    poses = {}
    cached_samples = deque()
    skeletons = np.empty((len(selection.starts), WINDOW_FRAMES, MAX_PERSONS, 17, 3), np.float16)
    context_values = None
    if context_detector is not None:
        from behaviorsense.models.visual_context import CONTEXT_DIM
        context_values = np.empty((len(selection.starts), CONTEXT_DIM), np.float32)
    kept, ready_window, context_window = [], 0, 0
    pending_context = {}
    next_frame = 0
    seeks_enabled = decode_mode == "seek"
    decoder_used_seek = False
    previous_sample, previous_pose = None, None
    try:
        for frame_index, sample in required_frames():
            if seeks_enabled and frame_index - next_frame >= seek_gap:
                moved = cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                position = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if moved and abs(position - frame_index) < 0.25:
                    next_frame = frame_index
                    decoder_used_seek = True
                    stats["seeks"] += 1
                else:
                    cap.release()
                    cap = factory(str(path))
                    if not cap.isOpened():
                        raise RuntimeError(f"decoder refused reopening {path}")
                    next_frame = 0
                    seeks_enabled = decoder_used_seek = False
                    stats["seek_fallbacks"] += 1
            while next_frame <= frame_index:
                if not cap.grab():
                    raise RuntimeError(f"{path.name}: decode ended before required source frame {frame_index}")
                next_frame += 1
                stats["grabbed_frames"] += 1
            reported_position = cap.get(cv2.CAP_PROP_POS_FRAMES) if decoder_used_seek else next_frame
            if decoder_used_seek and (not math.isfinite(reported_position) or
                                      abs(reported_position - next_frame) >= 0.25):
                # Some backends accept set() but land incorrectly only on the next
                # decode. Discard that frame and all further seeking for this video.
                cap.release()
                cap = factory(str(path))
                if not cap.isOpened():
                    raise RuntimeError(f"decoder refused reopening {path}")
                next_frame = 0
                seeks_enabled = decoder_used_seek = False
                stats["seek_fallbacks"] += 1
                while next_frame <= frame_index:
                    if not cap.grab():
                        raise RuntimeError(f"{path.name}: decode ended before required source frame {frame_index}")
                    next_frame += 1
                    stats["grabbed_frames"] += 1
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                raise RuntimeError(f"{path.name}: failed to retrieve source frame {frame_index}")
            began = time.monotonic()
            kpts, scores = validate_pose(*body(frame))
            stats["pose_seconds"] += time.monotonic() - began
            # A long unsampled gap is not adjacent motion. Within each continuous
            # sampled interval this is precisely notebook 07's slot assignment.
            prev = previous_pose if previous_sample is not None and sample == previous_sample + 1 else None
            pose = assign_slots(kpts, scores, prev, max_persons=MAX_PERSONS)
            poses[frame_index] = pose
            cached_samples.append((sample, frame_index))
            while cached_samples and cached_samples[0][0] < sample - WINDOW_FRAMES + 1:
                _, old_frame = cached_samples.popleft()
                del poses[old_frame]
            stats["peak_cached_pose_frames"] = max(stats["peak_cached_pose_frames"], len(poses))
            previous_sample, previous_pose = sample, pose
            stats["unique_inference_frames"] += 1
            if context_detector is not None and context_window < len(selection.starts) and sample == (
                    int(selection.starts[context_window]) + CONTEXT_SAMPLE_OFFSET):
                began = time.monotonic()
                values = np.asarray(context_detector.features(frame, pose), dtype=np.float32)
                if values.shape != (context_values.shape[1],) or not np.isfinite(values).all():
                    raise ValueError("context detector returned malformed or non-finite features")
                pending_context[context_window] = values
                context_window += 1
                stats["context_inference_frames"] += 1
                stats["context_seconds"] += time.monotonic() - began
            if ready_window < len(selection.starts) and sample == int(selection.starts[ready_window]) + WINDOW_FRAMES - 1:
                window = np.stack([poses[int(frame)] for frame in selection.source_frames[ready_window]])
                values = pending_context.pop(ready_window) if context_detector is not None else None
                if window_clip(window, window=WINDOW_FRAMES, min_visible=MIN_VISIBLE_KEYPOINTS):
                    skeletons[len(kept)] = window
                    if values is not None:
                        context_values[len(kept)] = values
                    kept.append(ready_window)
                ready_window += 1
    finally:
        cap.release()
    if ready_window != len(selection.starts) or pending_context:
        raise RuntimeError("decoder did not complete every selected window")
    stats["pose_frames_per_second"] = stats["unique_inference_frames"] / max(stats["pose_seconds"], 1e-9)
    return (skeletons[:len(kept)], np.asarray(kept, np.int64), stats,
            context_values[:len(kept)] if context_values is not None else None)


def subset_videos(annotations: dict[str, TsuVideo], subset: str) -> list[TsuVideo]:
    """Refuse mislabelled CS annotations, then ignore videos on the unrequested side."""
    if subset not in ("training", "testing", "all"):
        raise ValueError("subset must be training, testing or all")
    for video in annotations.values():
        expected = "training" if video.subject in CS_TRAIN_SUBJECTS else (
            "testing" if video.subject in CS_TEST_SUBJECTS else None)
        if video.subset != expected:
            raise ValueError(f"{video.vid}: annotation subset {video.subset!r} violates official CS partition")
    return sorted((v for v in annotations.values() if subset == "all" or v.subset == subset),
                  key=lambda video: video.vid)


def inventory_sha256(videos: list[TsuVideo]) -> str:
    """Stable digest of sorted official IDs; JSON whitespace does not affect it."""
    return hashlib.sha256("\n".join(sorted(video.vid for video in videos)).encode()).hexdigest()


def partition_videos(videos: list[TsuVideo], count: int) -> list[list[TsuVideo]]:
    """Deterministic longest-duration-first balancing; every video appears once.

    Duration is the class-independent upper bound on 20 Hz extraction work. Partition
    membership does not depend on filesystem order, model outputs, labels, or seed.
    """
    if count < 1 or count > len(videos):
        raise ValueError("partition-count must be between 1 and the number of subset videos")
    if len({video.vid for video in videos}) != len(videos):
        raise ValueError("duplicate video IDs in partition input")
    parts: list[list[TsuVideo]] = [[] for _ in range(count)]
    durations = [0] * count
    for video in sorted(videos, key=lambda value: (-value.duration, value.vid)):
        if video.duration <= 0:
            raise ValueError(f"{video.vid}: annotation duration must be positive")
        index = min(range(count), key=lambda i: (durations[i], i))
        parts[index].append(video)
        durations[index] += video.duration
    return [sorted(part, key=lambda video: video.vid) for part in parts]


def completion_status(work: list[TsuVideo], scope: list[TsuVideo], reports: list[dict],
                      selection: str) -> dict:
    """A completed partition or pilot cannot claim complete subset coverage."""
    expected = {video.vid for video in work}
    scoped = {video.vid for video in scope}
    completed_list = [report["video_id"] for report in reports]
    completed = set(completed_list)
    if len(completed) != len(completed_list) or not completed <= scoped or not scoped <= expected:
        raise ValueError("completion reports contain duplicate or out-of-scope videos")
    subset_complete = selection == "full" and completed == expected
    return {"complete": subset_complete, "subset_complete": subset_complete,
            "scope_complete": completed == scoped, "subset_videos_complete": completed == expected,
            "expected_videos": len(expected), "scope_expected_videos": len(scoped),
            "completed_videos": len(completed), "expected_video_ids": sorted(expected),
            "scope_video_ids": sorted(scoped), "completed_video_ids": sorted(completed),
            "missing_scope_video_ids": sorted(scoped - completed),
            "missing_subset_video_ids": sorted(expected - completed)}


def video_index(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"video directory does not exist: {root}")
    index = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in (".mp4", ".avi", ".mkv", ".mov"):
            continue
        try:
            vid = parse_tsu_filename(path).tsu
        except ValueError:
            continue
        if vid in index:
            raise ValueError(f"duplicate source video {vid}: {index[vid]} and {path}")
        index[vid] = path
    return index


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shard_fingerprint(config_fingerprint: str, video: TsuVideo, source_sha256: str) -> str:
    """Bind resume eligibility to actual source bytes, not a filename or file size."""
    identity = {"config_fingerprint": config_fingerprint, "video_id": video.vid,
                "subset": video.subset, "source_sha256": source_sha256}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def save_video_shard(path: Path, *, video: TsuVideo, source_path: Path, source_fps: float,
                     selection: WindowSelection, skeletons: np.ndarray, kept: np.ndarray,
                     fingerprint: str, report: dict, visual_context: np.ndarray | None = None,
                     context_metadata: dict | None = None) -> None:
    n = len(kept)
    if skeletons.shape != (n, WINDOW_FRAMES, MAX_PERSONS, 17, 3):
        raise ValueError(f"invalid output skeleton shape {skeletons.shape}")
    if (kept.ndim != 1 or kept.dtype.kind not in "iu" or (kept < 0).any() or
            (kept >= len(selection.starts)).any() or (np.diff(kept) <= 0).any()):
        raise ValueError("kept indices must be sorted unique selected-window indices")
    context_fields = {}
    if visual_context is not None:
        from behaviorsense.models.visual_context import CONTEXT_DIM, FEATURE_NAMES
        if (visual_context.shape != (n, CONTEXT_DIM) or not np.isfinite(visual_context).all() or
                not context_metadata):
            raise ValueError("visual context needs finite [N,CONTEXT_DIM] features and model metadata")
        context_fields = {"visual_context": visual_context.astype(np.float32, copy=False),
                          "context_source_frames": selection.source_frames[kept, CONTEXT_SAMPLE_OFFSET, None],
                          "context_feature_names": np.asarray(FEATURE_NAMES),
                          "context_metadata": np.asarray(json.dumps(context_metadata, sort_keys=True))}
    elif context_metadata is not None:
        raise ValueError("context metadata was supplied without extracted features")
    payload = dict(report, fingerprint=fingerprint, schema_version=SCHEMA_VERSION,
                   context_present=visual_context is not None)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle, skeletons=skeletons, clips=np.full(n, source_path.name, dtype="<U128"),
            subjects=np.full(n, f"p{video.subject:02d}", dtype="<U8"),
            datasets=np.full(n, "toyota_untrimmed", dtype="<U32"),
            starts=selection.starts[kept], source_frames=selection.source_frames[kept],
            source_fps=np.full(n, source_fps, np.float64),
            sample_fps=np.asarray(SAMPLE_FPS), annotation_fps=np.asarray(TSU_FPS),
            selection_strata=selection.strata[kept], selection_coverage=selection.coverage[kept],
            dominant_fine=selection.dominant_fine[kept],
            subset=np.full(n, video.subset, dtype="<U16"),
            fingerprint=np.asarray(fingerprint), extraction_metadata=np.asarray(json.dumps(payload)),
            **context_fields)
    temporary.replace(path)
    atomic_json(path.with_suffix(".json"), payload)


def resume_report(path: Path, fingerprint: str) -> dict | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        if str(data["fingerprint"]) != fingerprint:
            raise ValueError(f"{path}: extraction settings changed; use a new output directory")
        report = json.loads(str(data["extraction_metadata"]))
        if report.get("schema_version") != SCHEMA_VERSION or report.get("fingerprint") != fingerprint:
            raise ValueError(f"{path}: incompatible or inconsistent extraction metadata")
        n = report["retained_windows"]
        skeletons = data["skeletons"]
        if skeletons.shape != (n, WINDOW_FRAMES, MAX_PERSONS, 17, 3) or not np.isfinite(skeletons).all():
            raise ValueError(f"{path}: shard shape does not match saved report")
        frames, starts, rates = data["source_frames"], data["starts"], data["source_fps"]
        if (frames.shape != (n, WINDOW_FRAMES) or frames.dtype.kind not in "iu" or
                starts.shape != (n,) or starts.dtype.kind not in "iu" or
                (starts < 0).any() or (np.diff(starts) <= 0).any() or
                rates.shape != (n,) or not np.isfinite(rates).all() or (rates < SAMPLE_FPS).any() or
                not np.array_equal(frames, np.rint((starts[:, None] + np.arange(WINDOW_FRAMES)) *
                                                 rates[:, None] / SAMPLE_FPS)) or
                float(data["sample_fps"]) != SAMPLE_FPS or float(data["annotation_fps"]) != TSU_FPS):
            raise ValueError(f"{path}: missing or malformed original-frame provenance")
        if "retained_starts" in report and report["retained_starts"] != starts.tolist():
            raise ValueError(f"{path}: retained-start audit disagrees with shard")
        for field in ("clips", "subjects", "datasets", "subset", "selection_strata", "dominant_fine"):
            if data[field].shape != (n,):
                raise ValueError(f"{path}: malformed row metadata {field}")
        if data["selection_coverage"].shape != (n, 4) or not np.isfinite(data["selection_coverage"]).all():
            raise ValueError(f"{path}: malformed selection coverage")
        if report.get("context_present"):
            from behaviorsense.models.visual_context import CONTEXT_DIM, CONTEXT_SCHEMA, FEATURE_NAMES
            values = data["visual_context"]
            metadata = json.loads(str(data["context_metadata"]))
            if (values.shape != (n, CONTEXT_DIM) or not np.isfinite(values).all() or
                    not np.all(values[:, 0] == 1) or
                    not np.array_equal(data["context_feature_names"], FEATURE_NAMES) or
                    not np.array_equal(data["context_source_frames"], frames[:, CONTEXT_SAMPLE_OFFSET, None]) or
                    not isinstance(metadata, dict) or metadata.get("schema") != CONTEXT_SCHEMA or
                    metadata.get("feature_dim") != CONTEXT_DIM or metadata.get("feature_names") != list(FEATURE_NAMES) or
                    not metadata.get("model_sha256") or
                    ("context_model_sha256" in report and metadata["model_sha256"] != report["context_model_sha256"])):
                raise ValueError(f"{path}: missing or malformed exact-frame visual context")
        elif any(key in data for key in ("visual_context", "context_source_frames", "context_metadata")):
            raise ValueError(f"{path}: unexpected context without model provenance")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-root", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--rtmo-model", required=True, type=Path)
    parser.add_argument("--context-model", type=Path,
                        help="staged local RT-DETR ONNX; extracts object features at each window's exact center frame")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--subset", choices=("training", "testing", "all"), default="training")
    parser.add_argument("--selection", choices=("full", "stratified"), default="full",
                        help="full keeps every annotation-eligible grid window; stratified is the bounded pilot")
    parser.add_argument("--windows-per-video", type=int, default=64,
                        help="pilot cap used only with --selection stratified; full mode has no window cap")
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--candidate-stride", type=int, default=15,
                        help="window-grid stride at 20 Hz; production default 15, legacy pilot used 5")
    parser.add_argument("--min-overlap", type=float, default=0.5)
    parser.add_argument("--partition-count", type=int, default=1,
                        help="number of deterministic duration-balanced video partitions")
    parser.add_argument("--partition-index", type=int, default=0,
                        help="zero-based partition to extract in this GPU session")
    parser.add_argument("--resume-from", type=Path, action="append", default=[],
                        help="read-only directory of previous per-video shards; repeat for earlier sessions")
    parser.add_argument("--max-videos", type=int, default=None, help="smoke-run cap; manifest marks incomplete coverage")
    parser.add_argument("--decode-mode", choices=("sequential", "seek"), default="sequential")
    parser.add_argument("--seek-gap", type=int, default=250)
    parser.add_argument("--stop-after-seconds", type=float, default=None,
                        help="stop between videos; completed shards can be resumed")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.windows_per_video < 1 or args.candidate_stride < 1 or not 0 < args.min_overlap <= 1:
        raise ValueError("invalid window selection settings")
    if args.max_videos is not None and args.max_videos < 1:
        raise ValueError("max-videos must be positive")
    if args.seek_gap < 1 or (args.stop_after_seconds is not None and args.stop_after_seconds <= 0):
        raise ValueError("seek-gap and stop-after-seconds must be positive")
    if args.partition_count < 1 or not 0 <= args.partition_index < args.partition_count:
        raise ValueError("partition-index must be within [0, partition-count)")
    for root in args.resume_from:
        if not root.is_dir():
            raise FileNotFoundError(f"resume directory does not exist: {root}")
    import cv2
    annotations = load_tsu_annotations(args.annotations)
    work = subset_videos(annotations, args.subset)
    if not work:
        raise ValueError(f"no annotated {args.subset} videos")
    if args.selection == "full" and len(work) != OFFICIAL_SUBSET_VIDEO_COUNTS[args.subset]:
        raise ValueError(f"full {args.subset} extraction requires the complete official annotation inventory: "
                         f"{OFFICIAL_SUBSET_VIDEO_COUNTS[args.subset]} videos, got {len(work)}")
    if args.selection == "full" and inventory_sha256(work) != OFFICIAL_SUBSET_INVENTORY_SHA256[args.subset]:
        raise ValueError(f"full {args.subset} extraction video IDs do not match the official annotation inventory")
    partitions = partition_videos(work, args.partition_count)
    scope = partitions[args.partition_index]
    selected_work = scope[:args.max_videos] if args.max_videos else scope
    index = video_index(args.videos_root)
    missing = [v.vid for v in scope if v.vid not in index]
    if missing:
        raise FileNotFoundError(f"{len(missing)} annotated videos in this {args.subset} partition are absent: {missing[:12]}")
    config = {"schema_version": SCHEMA_VERSION, "sample_fps": SAMPLE_FPS,
              "annotation_fps": TSU_FPS, "window_frames": WINDOW_FRAMES,
              "selection_mode": args.selection,
              "official_subset_video_counts": OFFICIAL_SUBSET_VIDEO_COUNTS,
              "official_subset_inventory_sha256": OFFICIAL_SUBSET_INVENTORY_SHA256,
              "windows_per_video": args.windows_per_video if args.selection == "stratified" else None,
              "seed": args.seed if args.selection == "stratified" else None,
              "candidate_stride": args.candidate_stride, "min_overlap": args.min_overlap,
              "decode_mode": args.decode_mode, "seek_gap": args.seek_gap,
              "annotations_sha256": file_sha256(args.annotations),
              "rtmo_sha256": file_sha256(args.rtmo_model),
              "extractor_sha256": file_sha256(Path(__file__)),
              "slot_window_helpers_sha256": file_sha256(ROOT / "scripts" / "prepare_skeletons.py"),
              "selection": ("every grid window with annotation coverage >= min_overlap; no class-based sampling"
                            if args.selection == "full" else
                            "fixed-seed round-robin strata; class-balanced other_activity; annotation coverage >= min_overlap"),
              "strata_used_for_selection": args.selection == "stratified",
              "strata": list(STRATA), "selection_coverage_columns": ["tv", "drinking", "simultaneous", "annotated"],
              "gap_only_windows": "excluded; labels and masks must be rebuilt from original annotations",
              "evaluation_population": ("dense annotation-eligible windows after visibility gate; frame metrics require explicit reconstruction"
                                        if args.selection == "full" else
                                        "annotation-stratified sampled windows, not full-video temporal evaluation"),
              "tracking": "notebook07 assign_slots, reset between noncontiguous sampled intervals",
              "source_identity": "SHA-256 of complete video bytes included in each shard fingerprint",
              "visibility": "primary slot has >=8 joints confidence>0.3 in >=50% of frames"}
    context_config = None
    if args.context_model is not None:
        from behaviorsense.models.visual_context import CONTEXT_DIM, CONTEXT_SCHEMA, FEATURE_NAMES
        context_config = {"schema": CONTEXT_SCHEMA, "dimension": CONTEXT_DIM,
                          "feature_names": list(FEATURE_NAMES), "sample_offsets": [CONTEXT_SAMPLE_OFFSET],
                          "model_sha256": file_sha256(args.context_model),
                          "implementation_sha256": file_sha256(ROOT / "src" / "behaviorsense" / "models" / "visual_context.py"),
                          "source": "exact source_frames offset, with corresponding tracked pixel pose",
                          "label_source": "none; pretrained COCO object detections only"}
    config["visual_context"] = context_config
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    partition_suffix = (f"_part{args.partition_index:03d}-of{args.partition_count:03d}"
                        if args.partition_count > 1 else "")
    manifest_path = args.out / f"manifest_{args.subset}{partition_suffix}.json"
    partition_info = {"index": args.partition_index, "count": args.partition_count,
                      "strategy": "longest_annotation_duration_first; ties by video ID then partition index",
                      "assignments": [{"index": i, "video_ids": [v.vid for v in part],
                                       "annotation_seconds": sum(v.duration for v in part) / TSU_FPS}
                                      for i, part in enumerate(partitions)]}
    frame_bound = sum(math.ceil(v.duration * SAMPLE_FPS / TSU_FPS) for v in selected_work)
    if args.selection == "stratified":
        frame_bound = min(frame_bound, len(selected_work) * args.windows_per_video * WINDOW_FRAMES)
    print(f"{args.selection} selection, stride {args.candidate_stride}: "
          f"{len(selected_work)}/{len(scope)} partition videos, {len(work)} {args.subset} videos overall; "
          f"at most {frame_bound:,} inference frames in this invocation", flush=True)
    reports, resumed, detector, context_detector = [], 0, None, None
    began = time.monotonic()
    stopped = False
    extracted_duration = 0.0
    observed_seconds_per_frame = []
    budget_stop = None

    def write_manifest() -> dict:
        manifest = {"config": config, "fingerprint": fingerprint, "subset": args.subset,
                    "partition": partition_info, **completion_status(work, scope, reports, args.selection),
                    "max_videos": args.max_videos, "stopped_for_budget": stopped,
                    "budget_stop": budget_stop,
                    "resumed_videos": resumed, "resume_directories": [str(p.resolve()) for p in args.resume_from],
                    "windows": sum(r["retained_windows"] for r in reports),
                    "unique_inference_frames": sum(r["unique_inference_frames"] for r in reports),
                    "seconds_this_invocation": round(time.monotonic() - began, 3), "videos": reports}
        atomic_json(manifest_path, manifest)
        return manifest

    write_manifest()
    for position, video in enumerate(selected_work):
        path = index[video.vid]
        source_sha256 = file_sha256(path)
        video_fingerprint = shard_fingerprint(fingerprint, video, source_sha256)
        shard = args.out / f"toyota_multilabel_20hz_{video.subset}_{video.vid}.npz"
        resume_path = next((p for p in [shard, *(root / shard.name for root in args.resume_from)] if p.is_file()), shard)
        previous = resume_report(resume_path, video_fingerprint)
        if previous is not None:
            if previous.get("video_id") != video.vid or previous.get("subset") != video.subset:
                raise ValueError(f"{resume_path}: resume metadata identifies a different video/subset")
            if (previous.get("source_bytes") != path.stat().st_size or
                    previous.get("source_sha256") != source_sha256 or
                    previous.get("config_fingerprint") != fingerprint):
                raise ValueError(f"{path}: source video identity changed since extraction")
            reports.append(dict(previous, artifact_path=str(resume_path.resolve())))
            resumed += 1
            write_manifest()
            continue
        if args.stop_after_seconds is not None and time.monotonic() - began >= args.stop_after_seconds:
            stopped = True
            budget_stop = {"video_id": video.vid, "reason": "elapsed extraction budget"}
            break
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise RuntimeError(f"decoder refused {path}")
            source_fps = float(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        # Annotation bounds are at 25 Hz, not proportional to whatever the decoder
        # happens to return. Refuse the wrong video/version instead of stretching labels.
        source_timeline(total_frames, source_fps, video.duration)
        duration_delta = abs(total_frames / source_fps - video.duration / TSU_FPS)
        if duration_delta > max(2.0, 0.01 * video.duration / TSU_FPS):
            raise ValueError(f"{video.vid}: video/annotation duration differs by {duration_delta:.3f}s")
        selection = select_windows(video, total_frames, source_fps, budget=args.windows_per_video,
                                   seed=args.seed, candidate_stride=args.candidate_stride,
                                   min_overlap=args.min_overlap, selection=args.selection)
        if args.stop_after_seconds is not None and observed_seconds_per_frame:
            required_count = int(np.unique(selection.source_frames).size)
            # Use the slowest observed rate plus margin. Checkpointing is atomic after
            # every video; avoid starting a video likely to cross the session budget.
            estimated_seconds = required_count * max(observed_seconds_per_frame) * 1.15
            if time.monotonic() - began + estimated_seconds >= args.stop_after_seconds:
                stopped = True
                budget_stop = {"video_id": video.vid, "reason": "next video exceeds estimated remaining budget",
                               "estimated_next_video_seconds": round(estimated_seconds, 3),
                               "remaining_seconds": round(args.stop_after_seconds - (time.monotonic() - began), 3)}
                break
        video_began = time.monotonic()
        if detector is None:
            detector = load_detector(args.rtmo_model)
        if context_config is not None and context_detector is None:
            from behaviorsense.models.visual_context import load_context_detector
            context_detector = load_context_detector(args.context_model, require_cuda=True)
        skeletons, kept, statistics, visual_context = decode_selected(
            path, selection, detector, decode_mode=args.decode_mode, seek_gap=args.seek_gap,
            context_detector=context_detector)
        require_cuda_session(detector)
        if context_detector is not None:
            require_cuda_session(context_detector, name="RT-DETR")
        context_metadata = (dict(context_detector.metadata, sampling=context_config)
                            if context_detector is not None else None)
        report = {"video_id": video.vid, "subject": f"p{video.subject:02d}", "subset": video.subset,
                  "source_filename": path.name, "source_bytes": path.stat().st_size,
                  "source_sha256": source_sha256, "config_fingerprint": fingerprint,
                  "source_fps": source_fps, "source_frame_count": total_frames,
                  "annotation_duration_frames": video.duration, "duration_delta_seconds": duration_delta,
                  "selection_mode": args.selection, "candidate_stride": args.candidate_stride,
                  "min_overlap": args.min_overlap,
                  "window_cap": args.windows_per_video if args.selection == "stratified" else None,
                  "full_dense_coverage": args.selection == "full" and len(selection.starts) == sum(selection.candidate_counts.values()),
                  "candidate_windows": selection.candidate_windows,
                  "candidate_strata": selection.candidate_counts,
                  "excluded_annotation_coverage": selection.candidate_windows - sum(selection.candidate_counts.values()),
                  "selected_windows": len(selection.starts), "retained_windows": len(kept),
                  "selected_strata": dict(Counter(selection.strata.tolist())),
                  "retained_strata": dict(Counter(selection.strata[kept].tolist())),
                  "dropped_visibility": len(selection.starts) - len(kept),
                  "selected_starts": selection.starts.tolist(), "retained_starts": selection.starts[kept].tolist(),
                  "context_present": context_metadata is not None,
                  "context_schema": context_config["schema"] if context_config else None,
                  "context_dim": context_config["dimension"] if context_config else 0,
                  "context_sample_offsets": [CONTEXT_SAMPLE_OFFSET] if context_config else [],
                  "context_model_sha256": context_config["model_sha256"] if context_config else None,
                  "artifact_path": str(shard.resolve()),
                  "seconds": round(time.monotonic() - video_began, 3), **statistics}
        save_video_shard(shard, video=video, source_path=path, source_fps=source_fps,
                         selection=selection, skeletons=skeletons, kept=kept,
                         fingerprint=video_fingerprint, report=report, visual_context=visual_context,
                         context_metadata=context_metadata)
        report = dict(report, fingerprint=video_fingerprint, schema_version=SCHEMA_VERSION)
        reports.append(report)
        if statistics["unique_inference_frames"]:
            observed_seconds_per_frame.append((time.monotonic() - video_began) / statistics["unique_inference_frames"])
        elapsed = time.monotonic() - began
        extracted_duration += video.duration / TSU_FPS
        remaining_duration = sum(v.duration for v in selected_work[position + 1:]) / TSU_FPS
        eta = elapsed / max(extracted_duration, 1e-9) * remaining_duration
        print(f"{position + 1}/{len(selected_work)} {video.vid}: {len(kept)}/{len(selection.starts)} windows, "
              f"{statistics['unique_inference_frames']} pose frames at {statistics['pose_frames_per_second']:.1f}/s, "
              f"elapsed {elapsed / 60:.1f}m, estimated remaining {eta / 60:.1f}m", flush=True)
        write_manifest()
    manifest = write_manifest()
    print(json.dumps({key: manifest[key] for key in ("subset", "scope_complete", "subset_complete",
          "expected_videos", "scope_expected_videos", "completed_videos", "windows", "resumed_videos",
          "stopped_for_budget", "seconds_this_invocation")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
