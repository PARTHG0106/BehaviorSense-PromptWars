"""Time-aligned multilabel targets for Toyota's untrimmed RTMO windows.

The legacy ``labels`` and ``fine_labels`` arrays contain one argmax class and cannot
represent overlapping actions. This loader ignores them and reads the original annotation
spans. A retained window must carry its real sampled ``starts`` index: visibility filtering
makes its row number an invalid substitute. Trimmed clips have no untrimmed source offset
and are refused.

New extraction also records ``source_frames`` and ``source_fps``. Those fields preserve
the decoder's rounded source-frame selection exactly. Timestamped older shards without
these fields use the declared sampling clock; provenance distinguishes the two cases.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from behaviorsense.data.toyota import (
    COARSE_V11,
    CS_TEST_SUBJECTS,
    CS_TRAIN_SUBJECTS,
    TSU_CLASSES,
    TSU_TO_COARSE,
    TsuVideo,
    load_tsu_annotations,
    parse_tsu_filename,
    supervised_coarse,
)

WINDOW_FRAMES = 30
SKELETON_SHAPE = (WINDOW_FRAMES, 2, 17, 3)
_TV = COARSE_V11.index("watching_tv")
_DRINK = COARSE_V11.index("drinking")
_FINE_TO_COARSE = np.asarray(
    [COARSE_V11.index(TSU_TO_COARSE[name]) for name in TSU_CLASSES], dtype=np.int64
)
_RATE_IN_NAME = re.compile(r"_(\d+(?:\.\d+)?)hz(?:_|\.|$)", re.IGNORECASE)
OFFICIAL_VIDEO_COUNTS = {"training": 351, "testing": 185}
OFFICIAL_INVENTORY_SHA256 = {
    "training": "f1764bd2f3f1d24c5482f8153738905302ea0b5870153aae35e8b700a36e0199",
    "testing": "e0667e544593250b72f90036d3a541d15088a4fce0dc3d62752eceeb1d0a1686",
}


def _verify_full_audit(report, starts, video, *, sample_fps, annotation_fps, min_overlap):
    """Reconstruct the eligible grid instead of trusting a 'complete' status bit."""
    if (report.get("schema_version") != 2 or report.get("selection_mode") != "full"
            or report.get("window_cap", "missing") is not None
            or report.get("full_dense_coverage") is not True):
        raise ValueError(f"{video.vid}: full coverage requires an uncapped schema-2 extraction audit")
    if report.get("subset") != video.subset or report.get("subject") != f"p{video.subject:02d}":
        raise ValueError(f"{video.vid}: extraction identity/subset disagrees with annotations")
    for key in ("source_sha256", "config_fingerprint"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(report.get(key, ""))):
            raise ValueError(f"{video.vid}: extraction audit lacks {key}")
    identity = {key: report[key] for key in ("config_fingerprint", "video_id", "subset", "source_sha256")}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if report.get("fingerprint") != fingerprint:
        raise ValueError(f"{video.vid}: extraction fingerprint disagrees with source/configuration identity")
    stride = report.get("candidate_stride")
    if not isinstance(stride, int) or not 1 <= stride <= WINDOW_FRAMES // 2:
        raise ValueError(f"{video.vid}: full coverage requires a grid stride of at most 15 samples")
    rate, frame_count = report.get("source_fps"), report.get("source_frame_count")
    if (not isinstance(frame_count, int) or frame_count < 1 or not isinstance(rate, (int, float))
            or not np.isfinite(rate) or rate < sample_fps):
        raise ValueError(f"{video.vid}: invalid full-coverage source clock")
    delta = abs(frame_count / rate - video.duration / annotation_fps)
    if delta > max(2., .01 * video.duration / annotation_fps):
        raise ValueError(f"{video.vid}: source duration disagrees with annotations")
    count = int(np.ceil(frame_count * sample_fps / rate)) + 1
    timeline = np.rint(np.arange(count, dtype=np.float64) * rate / sample_fps).astype(np.int64)
    timeline = timeline[(timeline < frame_count) & (timeline * annotation_fps / rate <= video.duration)]
    candidates = np.arange(0, max(0, len(timeline) - WINDOW_FRAMES + 1), stride, dtype=np.int64)
    _, _, coverage = _window_supervision(video, candidates, sample_fps=sample_fps,
        annotation_fps=annotation_fps, min_overlap=min_overlap,
        source_frames=timeline[candidates[:, None] + np.arange(WINDOW_FRAMES)], source_fps=rate)
    selected = candidates[coverage >= WINDOW_FRAMES * min_overlap]
    if (report.get("candidate_windows") != len(candidates)
            or report.get("selected_windows") != len(selected)
            or report.get("excluded_annotation_coverage") != len(candidates) - len(selected)
            or report.get("selected_starts") != selected.tolist()):
        raise ValueError(f"{video.vid}: extraction omitted eligible full-coverage windows")
    if (report.get("retained_windows") != len(starts)
            or report.get("retained_starts") != starts.tolist()
            or report.get("dropped_visibility") != len(selected) - len(starts)
            or len(np.unique(starts)) != len(starts) or not np.isin(starts, selected).all()):
        raise ValueError(f"{video.vid}: retained windows disagree with the full-coverage audit")


def _allocate_array(shape, dtype, cache_dir, cache_key, name):
    if cache_dir is None:
        return np.empty(shape, dtype=dtype)
    directory = Path(cache_dir).expanduser().resolve() / cache_key
    directory.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(directory / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_options(sample_fps: float, annotation_fps: float, min_overlap: float) -> None:
    if not np.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("sample_fps must be positive and finite")
    if not np.isfinite(annotation_fps) or annotation_fps <= 0:
        raise ValueError("annotation_fps must be positive and finite")
    if not np.isfinite(min_overlap) or not 0 < min_overlap <= 1:
        raise ValueError("min_overlap must be in (0, 1]")


def _starts_array(starts: np.ndarray) -> np.ndarray:
    values = np.asarray(starts)
    if values.ndim != 1 or values.dtype.kind not in "iu" or (values < 0).any():
        raise ValueError("starts must be a one-dimensional array of nonnegative integers")
    return values.astype(np.int64, copy=False)


def _source_clock(source_frames, source_fps, starts, sample_fps):
    """Validate the actual decoder clock, including agreement with sampled starts."""
    if (source_frames is None) != (source_fps is None):
        raise ValueError("source_frames and source_fps must be supplied together")
    if source_frames is None:
        return None, None
    frames = np.asarray(source_frames)
    if frames.shape != (len(starts), WINDOW_FRAMES) or frames.dtype.kind not in "iu":
        raise ValueError("source_frames must be integer [N,30] original-frame indices")
    if (frames < 0).any() or (np.diff(frames, axis=1) <= 0).any():
        raise ValueError("source_frames must be nonnegative and strictly increasing per window")
    rates = np.asarray(source_fps, dtype=np.float64)
    if rates.ndim == 0:
        rates = np.full(len(starts), float(rates))
    if rates.shape != (len(starts),) or not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError("source_fps must be a positive finite scalar or [N] array")
    if (rates < sample_fps).any():
        raise ValueError("source_fps is below the declared sample_fps")
    desired = (starts[:, None] + np.arange(WINDOW_FRAMES)) * rates[:, None] / sample_fps
    if not np.array_equal(frames, np.rint(desired)):
        raise ValueError("source_frames disagree with starts and the declared sampling clock")
    return frames.astype(np.int64, copy=False), rates


def _window_supervision(
    video: TsuVideo,
    starts: np.ndarray,
    *,
    sample_fps: float = 20.0,
    annotation_fps: float = 25.0,
    min_overlap: float = 0.5,
    source_frames: np.ndarray | None = None,
    source_fps: np.ndarray | float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return coarse multi-hot targets and the true TV/drinking intersection frame count.

    An action is positive when its union of annotation spans covers at least
    ``min_overlap`` of the 30 sampled positions. Fine classes mapping to one coarse
    class are ORed per frame before counting, so overlapping ``Drink.From_*`` spans
    never count a frame twice. Boundaries are inclusive, matching Toyota training's
    ``official=False`` convention. This is a window target, not official frame mAP.

    Pair intersection counts remain separate: two classes can each occupy half a
    window without occurring at the same sampled position.
    """
    _validate_options(sample_fps, annotation_fps, min_overlap)
    starts = _starts_array(starts)
    frames, rates = _source_clock(source_frames, source_fps, starts, sample_fps)
    if video.duration <= 0:
        raise ValueError(f"{video.vid}: annotation duration must be positive")
    for fine_id, start, end in video.actions:
        if not 0 <= fine_id < len(TSU_CLASSES) or start < 0 or end < start:
            raise ValueError(f"{video.vid}: invalid annotation span {(fine_id, start, end)}")

    targets = np.zeros((len(starts), len(COARSE_V11)), dtype=np.float32)
    intersections = np.zeros(len(starts), dtype=np.int16)
    annotated_frames = np.zeros(len(starts), dtype=np.int16)
    # Bound temporary allocations even when a caller supplies every window of a long video.
    for offset in range(0, len(starts), 4096):
        take = slice(offset, offset + 4096)
        if frames is None:
            positions = (starts[take, None] + np.arange(WINDOW_FRAMES)) * (
                annotation_fps / sample_fps
            )
        else:
            positions = frames[take] * (annotation_fps / rates[take, None])
        if (positions > video.duration).any():
            raise ValueError(f"{video.vid}: window extends beyond the annotated video duration")
        active = np.zeros((*positions.shape, len(COARSE_V11)), dtype=bool)
        for fine_id, start, end in video.actions:
            active[:, :, _FINE_TO_COARSE[fine_id]] |= (positions >= start) & (positions <= end)
        counts = active.sum(axis=1)
        targets[take] = counts >= WINDOW_FRAMES * min_overlap
        intersections[take] = (active[:, :, _TV] & active[:, :, _DRINK]).sum(axis=1)
        annotated_frames[take] = active.any(axis=2).sum(axis=1)
    return targets, intersections, annotated_frames


def targets_for_windows(
    video: TsuVideo,
    starts: np.ndarray,
    *,
    sample_fps: float = 20.0,
    annotation_fps: float = 25.0,
    min_overlap: float = 0.5,
    source_frames: np.ndarray | None = None,
    source_fps: np.ndarray | float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Coarse multi-hot targets and true TV/drinking overlap counts, on the decoder clock.

    Annotation boundaries are inclusive. Fine spans are ORed per coarse class and each
    target requires at least ``min_overlap`` of 30 sampled positions. Overlap counts
    distinguish actual simultaneous frames from actions occupying separate window halves.
    """
    targets, intersections, _ = _window_supervision(
        video, starts, sample_fps=sample_fps, annotation_fps=annotation_fps,
        min_overlap=min_overlap, source_frames=source_frames, source_fps=source_fps,
    )
    return targets, intersections


def load_multilabel_shards(
    paths: Sequence[str | Path] | str | Path,
    annotations_path: str | Path,
    sample_fps: float = 20.0,
    annotation_fps: float = 25.0,
    min_overlap: float = 0.5,
    *,
    cache_dir: str | Path | None = None,
    require_context: bool = False,
    require_full_coverage: bool = False,
    expected_subset: str | None = None,
) -> dict:
    """Load timestamped untrimmed windows; the trainer owns fit/validation/test splitting.

    Returns skeletons [N,30,2,17,3], targets/target_mask [N,22], pNN subjects,
    canonical video_ids, starts, sample_fps, tv_drinking_overlap_frames, and provenance.
    Both official CS sides may be present, but their identities and annotation subsets
    are validated before anything is returned. Unsupported posture/fall/idle classes
    have a false loss mask, rather than invented negative labels.
    """
    _validate_options(sample_fps, annotation_fps, min_overlap)
    if isinstance(paths, (str, Path)):
        paths = [paths]
    files = [Path(path).expanduser().resolve() for path in paths]
    if not files:
        raise ValueError("no multilabel shards supplied")
    if len(set(files)) != len(files):
        raise ValueError("the same shard was supplied more than once")
    annotation_path = Path(annotations_path).expanduser().resolve()
    annotations = load_tsu_annotations(annotation_path)
    if expected_subset is not None and expected_subset not in OFFICIAL_VIDEO_COUNTS:
        raise ValueError("expected_subset must be training or testing")
    if require_full_coverage:
        if expected_subset is None:
            raise ValueError("full coverage requires an explicit official subset")
        expected_videos = {key for key, video in annotations.items() if video.subset == expected_subset}
        if len(expected_videos) != OFFICIAL_VIDEO_COUNTS[expected_subset]:
            raise ValueError(f"full coverage requires all {OFFICIAL_VIDEO_COUNTS[expected_subset]} official {expected_subset} videos")
        inventory = hashlib.sha256("\n".join(sorted(expected_videos)).encode()).hexdigest()
        if inventory != OFFICIAL_INVENTORY_SHA256[expected_subset]:
            raise ValueError("full coverage annotation inventory differs from the official video IDs")
    audited_videos, context_contract, context_shape = set(), None, None

    records, subjects_all, videos_all, starts_all = [], [], [], []
    targets_all, intersections_all, coverage_all, seen_windows = [], [], [], set()
    n_total = 0
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        rates_in_name = _RATE_IN_NAME.findall(path.name)
        if any(not np.isclose(float(rate), sample_fps) for rate in rates_in_name):
            raise ValueError(f"{path.name}: filename rate disagrees with sample_fps={sample_fps}")
        with np.load(path, allow_pickle=False) as shard:
            required = {"skeletons", "clips", "starts", "subjects", "datasets"}
            missing = required - set(shard.files)
            if missing:
                raise ValueError(
                    f"{path.name}: missing {sorted(missing)}; exact starts are required, "
                    "and row indices cannot reconstruct filtered windows"
                )
            starts = _starts_array(shard["starts"])
            n_rows = len(starts)
            report = json.loads(str(shard["extraction_metadata"])) if "extraction_metadata" in shard.files else None
            if report and expected_subset and report.get("subset") not in (None, expected_subset):
                raise ValueError(f"{path.name}: extraction belongs to a different official subset")
            if require_full_coverage:
                if not report or report.get("video_id") not in expected_videos:
                    raise ValueError(f"{path.name}: missing or out-of-subset full extraction audit")
                vid = report["video_id"]
                if vid in audited_videos:
                    raise ValueError(f"duplicate full-coverage video {vid}")
                _verify_full_audit(report, starts, annotations[vid], sample_fps=sample_fps,
                                  annotation_fps=annotation_fps, min_overlap=min_overlap)
                if "fingerprint" not in shard.files or str(shard["fingerprint"]) != report["fingerprint"]:
                    raise ValueError(f"{path.name}: stored shard fingerprint disagrees with extraction audit")
                audited_videos.add(vid)
            context_keys = {"visual_context", "context_source_frames", "context_feature_names", "context_metadata"}
            has_context = context_keys <= set(shard.files)
            if require_context and not has_context:
                raise ValueError(f"{path.name}: actual visual context and its provenance are required")
            if has_context and require_context:
                from behaviorsense.models.visual_context import CONTEXT_DIM, CONTEXT_SCHEMA, FEATURE_NAMES
                values = shard["visual_context"]
                metadata = json.loads(str(shard["context_metadata"]))
                if (values.shape != (n_rows, CONTEXT_DIM) or values.dtype.kind != "f"
                        or not np.isfinite(values).all() or (n_rows and not (values[:, 0] == 1).all())):
                    raise ValueError(f"{path.name}: visual context must be finite observed [N,{CONTEXT_DIM}] detector features")
                if (metadata.get("schema") != CONTEXT_SCHEMA
                        or shard["context_feature_names"].tolist() != list(FEATURE_NAMES)
                        or not re.fullmatch(r"[0-9a-f]{64}", str(metadata.get("model_sha256", "")))):
                    raise ValueError(f"{path.name}: invalid visual context schema or detector provenance")
                contract = {key: value for key, value in metadata.items() if key not in ("provider", "providers")}
                if context_contract is not None and context_contract != contract:
                    raise ValueError(f"{path.name}: visual context detector/feature contract changed between shards")
                context_contract, context_shape = contract, values.shape[1:]
                if ("source_frames" not in shard.files or shard["context_source_frames"].shape != (n_rows, 1)
                        or not np.array_equal(shard["context_source_frames"], shard["source_frames"][:, [WINDOW_FRAMES // 2]])):
                    raise ValueError(f"{path.name}: context frame is not the real skeleton-window center")
            if not n_rows:
                # A completed zero-visibility extraction is an intentional resume marker.
                # Only the audited extraction schema may contribute an empty shard.
                empty_fields = {"source_frames", "source_fps", "extraction_metadata"}
                if not empty_fields <= set(shard.files):
                    raise ValueError(f"{path.name}: empty shard has no extraction audit")
                report = json.loads(str(shard["extraction_metadata"]))
                if (report.get("schema_version") not in (1, 2) or report.get("retained_windows") != 0
                        or shard["skeletons"].shape != (0, *SKELETON_SHAPE)
                        or shard["source_frames"].shape != (0, WINDOW_FRAMES)
                        or shard["source_fps"].shape != (0,)
                        or any(shard[key].shape != (0,) for key in ("clips", "subjects", "datasets"))):
                    raise ValueError(f"{path.name}: empty shard disagrees with extraction audit")
                records.append({"path": str(path), "sha256": _sha256(path), "n_windows": 0,
                                "clock": "source_frames", "excluded": "empty_extraction", "extraction": report})
                continue
            subjects, clips, datasets = (np.asarray(shard[key]) for key in
                                         ("subjects", "clips", "datasets"))
            for key, array in (("subjects", subjects), ("clips", clips), ("datasets", datasets)):
                if array.shape != (n_rows,) or array.dtype.kind not in "US":
                    raise ValueError(f"{path.name}: {key} must be a string [N] array")
            subjects, clips, datasets = (array.astype(str) for array in (subjects, clips, datasets))
            if set(datasets.tolist()) != {"toyota_untrimmed"}:
                raise ValueError(f"{path.name}: only toyota_untrimmed source windows can be relabelled")
            if "sample_fps" in shard.files and not np.allclose(shard["sample_fps"], sample_fps):
                raise ValueError(f"{path.name}: stored sample_fps disagrees with declared rate")
            if "annotation_fps" in shard.files and not np.allclose(shard["annotation_fps"], annotation_fps):
                raise ValueError(f"{path.name}: stored annotation_fps disagrees with declared rate")
            source_frames = shard["source_frames"] if "source_frames" in shard.files else None
            source_fps = shard["source_fps"] if "source_fps" in shard.files else None
            source_frames, source_fps = _source_clock(source_frames, source_fps, starts, sample_fps)
            exact_clock = source_frames is not None
            if require_full_coverage and not exact_clock:
                raise ValueError(f"{path.name}: full coverage requires exact source-frame clocks")
            if require_full_coverage and (not np.all(source_fps == report["source_fps"])
                                         or (source_frames >= report["source_frame_count"]).any()):
                raise ValueError(f"{path.name}: row source clock disagrees with audited source video")

        targets = np.zeros((n_rows, len(COARSE_V11)), dtype=np.float32)
        intersections = np.zeros(n_rows, dtype=np.int16)
        coverage = np.zeros(n_rows, dtype=np.float32)
        video_ids = np.empty(n_rows, dtype="<U16")
        for clip in np.unique(clips):
            ident = parse_tsu_filename(str(clip))
            video_id = ident.tsu
            if video_id not in annotations:
                raise ValueError(f"{path.name}: no original annotations for {video_id}")
            video = annotations[video_id]
            if expected_subset and video.subset != expected_subset:
                raise ValueError(f"{path.name}: input contains videos outside the requested official {expected_subset} subset")
            if require_full_coverage and video_id != report["video_id"]:
                raise ValueError(f"{path.name}: clip identity disagrees with extraction audit")
            rows = np.flatnonzero(clips == clip)
            if not np.all(subjects[rows] == f"p{ident.subject:02d}"):
                raise ValueError(f"{path.name}: subject disagrees with {video_id}")
            if video.subject != ident.subject:
                raise ValueError(f"{video_id}: annotation subject disagrees with filename")
            expected = ("training" if ident.subject in CS_TRAIN_SUBJECTS else
                        "testing" if ident.subject in CS_TEST_SUBJECTS else None)
            if expected is None or video.subset != expected:
                raise ValueError(f"{video_id}: annotations disagree with official CS subject/subset")
            for start in starts[rows]:
                window_key = (video_id, int(start))
                if window_key in seen_windows:
                    raise ValueError(f"duplicate window {video_id} at sampled start {start}")
                seen_windows.add(window_key)
            target, intersection, annotated = _window_supervision(
                video, starts[rows], sample_fps=sample_fps, annotation_fps=annotation_fps,
                min_overlap=min_overlap,
                source_frames=source_frames[rows] if exact_clock else None,
                source_fps=source_fps[rows] if exact_clock else None,
            )
            targets[rows], intersections[rows], video_ids[rows] = target, intersection, video_id
            coverage[rows] = annotated / WINDOW_FRAMES
        records.append({"path": str(path), "sha256": _sha256(path), "n_windows": n_rows,
                        "clock": "source_frames" if exact_clock else "declared_sampling_clock", "extraction": report})
        subjects_all.append(subjects)
        videos_all.append(video_ids)
        starts_all.append(starts)
        targets_all.append(targets)
        intersections_all.append(intersections)
        coverage_all.append(coverage)
        n_total += n_rows

    if not n_total:
        raise ValueError("all supplied shards are empty; no windows available for supervision")
    if require_full_coverage and audited_videos != expected_videos:
        raise ValueError(f"full coverage is missing {len(expected_videos - audited_videos)} official videos: "
                         f"{sorted(expected_videos - audited_videos)[:10]}")
    if require_full_coverage and len({record["extraction"]["config_fingerprint"] for record in records}) != 1:
        raise ValueError("full-coverage extraction settings changed between shards")

    # Allocate one output and materialise at most one compressed skeleton shard at a time.
    # Concatenating a list of all loaded skeleton arrays would double peak memory.
    cache_key = hashlib.sha256(json.dumps([(record["sha256"], record["n_windows"]) for record in records]).encode()).hexdigest()[:24]
    skeletons = _allocate_array((n_total, *SKELETON_SHAPE), np.float16, cache_dir, cache_key, "skeletons")
    context = (_allocate_array((n_total, *context_shape), np.float32, cache_dir, cache_key, "visual_context")
               if require_context else None)
    offset = 0
    for path, record in zip(files, records):
        with np.load(path, allow_pickle=False) as shard:
            values = shard["skeletons"]
            if values.shape != (record["n_windows"], *SKELETON_SHAPE):
                raise ValueError(f"{path.name}: expected skeletons [N,30,2,17,3], got {values.shape}")
            if values.dtype.kind != "f" or not np.isfinite(values).all():
                raise ValueError(f"{path.name}: skeletons must contain finite floating-point values")
            if require_full_coverage:
                # Compare float32 scores to the same threshold used before float16
                # serialization; numpy otherwise rounds the threshold down to float16.
                visible_joints = (values[:, :, 0, :, 2].astype(np.float32) > .3).sum(axis=2)
                if ((visible_joints >= 8).sum(axis=1) < WINDOW_FRAMES // 2).any():
                    raise ValueError(f"{path.name}: retained skeleton fails the audited primary-person visibility gate")
            end = offset + len(values)
            skeletons[offset:end] = values
            if context is not None:
                context[offset:end] = shard["visual_context"]
            if not np.isfinite(skeletons[offset:end]).all():
                raise ValueError(f"{path.name}: skeleton values overflow float16 storage")
            offset = end
    for array in (skeletons, context):
        if isinstance(array, np.memmap):
            array.flush()

    targets = np.concatenate(targets_all)
    target_mask = np.zeros_like(targets, dtype=bool)
    supported = supervised_coarse(TSU_TO_COARSE)
    target_mask[:, list(supported)] = True
    coverage = np.concatenate(coverage_all)
    target_mask[coverage < min_overlap] = False
    intersections = np.concatenate(intersections_all)
    provenance = {
        "schema_version": 1,
        "target_source": "original_toyota_untrimmed_annotations",
        "class_names": list(COARSE_V11),
        "annotations_path": str(annotation_path),
        "annotations_sha256": _sha256(annotation_path),
        "source_shards": records,
        "sample_fps": float(sample_fps),
        "annotation_fps": float(annotation_fps),
        "window_frames": WINDOW_FRAMES,
        "min_overlap": float(min_overlap),
        "interval_boundary": "inclusive",
        "aggregation": "framewise_coarse_union_then_occupancy_threshold",
        "supervised_class_ids": list(supported),
        "positive_windows_per_class": targets.sum(axis=0).astype(int).tolist(),
        "n_tv_drinking_intersection_windows": int((intersections > 0).sum()),
        "annotation_coverage_mask": "abstain_when_annotated_fraction_below_min_overlap",
        "n_masked_gap_windows": int((coverage < min_overlap).sum()),
        "n_empty_shards_skipped": sum(record["n_windows"] == 0 for record in records),
        "n_windows": n_total,
        "full_coverage_verified": bool(require_full_coverage),
        "official_subset": expected_subset,
        "audited_video_count": len(audited_videos),
        "visual_context": context_contract,
        "storage": "disk_backed_float16_skeletons_float32_context" if cache_dir else "memory",
        "legacy_single_label_targets_used": False,
        "coverage": "retained_windows_only; extraction may omit gaps or low-visibility windows",
    }
    result = {
        "skeletons": skeletons,
        "targets": targets,
        "target_mask": target_mask,
        "subjects": np.concatenate(subjects_all),
        "video_ids": np.concatenate(videos_all),
        "starts": np.concatenate(starts_all),
        "sample_fps": float(sample_fps),
        "tv_drinking_overlap_frames": intersections,
        "annotation_coverage": coverage,
        "provenance": provenance,
    }
    if context is not None:
        result["visual_context"] = context
    return result
