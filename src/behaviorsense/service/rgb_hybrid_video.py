"""Bounded raw-video adapter for the frozen RGB hybrid's local Kaggle notebook.

Each tracked person occupies the primary skeleton slot. A single co-present
person occupies the secondary slot within each window. This serving track adapter
is separate from the historical benchmark's prerecorded primary-subject setup.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
import math
import time

import cv2
import numpy as np

from behaviorsense.agents.perception import PerceptionAgent, PerceptionConfig
from behaviorsense.models.rgb_action import person_roi, scene_roi, spatial_view
from behaviorsense.models.visual_context import context_features
from behaviorsense.schemas import ActivitySegment, Role
from behaviorsense.service.staging import assert_subject


SAMPLE_FPS = 20.
WINDOW = 30
STRIDE = 15
BASE = datetime(2026, 1, 1, 9, 0, 0)


def decode_tracks(path, detector, *, embedder=None, gallery=None, max_frames=36000,
                  progress=lambda **event: None, capture_factory=cv2.VideoCapture):
    cap = capture_factory(str(path))
    if not cap.isOpened():
        raise ValueError("The video decoder could not open this upload.")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(source_fps) or source_fps < SAMPLE_FPS:
        cap.release()
        raise ValueError("The frozen RGB hybrid requires source video at 20 FPS or higher.")
    if max_frames < WINDOW:
        cap.release()
        raise ValueError("max_frames must allow at least one complete 30-frame window.")
    # Preserve partial poses for the model's window-level visibility test. The
    # display's usability gate must not delete the low-visibility training inputs.
    agent = PerceptionAgent(detector, embedder=embedder, gallery=gallery,
                            config=PerceptionConfig(min_visible_keypoints=0), fps=SAMPLE_FPS)
    raw, wire, source_indices = [], [], []
    counts, spans, boxes, roles = Counter(), {}, {}, {}
    total, next_source, width, height = 0, 0, 0, 0
    try:
        while cap.grab():
            index = total
            total += 1
            if index != next_source:
                continue
            if len(raw) >= max_frames:
                raise ValueError(f"Video exceeds {max_frames / SAMPLE_FPS / 60:g} minutes; split it into shorter clips.")
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                raise ValueError(f"Could not decode source frame {index}.")
            h, w = frame.shape[:2]
            if width and (w, h) != (width, height):
                raise ValueError("Decoded frame dimensions change within the video.")
            width, height = w, h
            observation = agent.process_frame(frame, BASE + timedelta(seconds=len(raw) / SAMPLE_FPS))
            sampled = len(raw)
            people, poses = [], {}
            for person in observation.persons:
                tid = int(person.track_id)
                counts[tid] += 1
                spans[tid] = (spans.get(tid, (sampled, sampled))[0], sampled)
                box = [float(person.box.x1), float(person.box.y1), float(person.box.x2), float(person.box.y2)]
                boxes[tid] = (boxes.get(tid, (box, box))[0], box)
                roles.setdefault(tid, []).append(person.role.value)
                pose = np.zeros((17, 3), dtype=np.float32)
                if person.keypoints is not None:
                    pose[:, :2] = person.keypoints.xy
                    pose[:, 2] = person.keypoints.scores
                poses[tid] = pose
                people.append({"track_id": tid, "role": person.role.value,
                    "role_confidence": round(person.role_confidence, 3), "box": box,
                    "kp": [[round(float(x), 2), round(float(y), 2), round(float(s), 3)] for x, y, s in pose]})
            raw.append(poses)
            wire.append({"i": sampled, "t": sampled / SAMPLE_FPS, "people": people})
            source_indices.append(index)
            next_source = int(np.rint(len(raw) * source_fps / SAMPLE_FPS))
            if len(raw) % 100 == 0:
                progress(step="pose", frames=len(raw), max_frames=max_frames)
    finally:
        cap.release()
    if not raw:
        raise ValueError("Decoded zero sampled frames.")
    resolved_roles = {}
    for tid, values in roles.items():
        known = [value for value in values if value != "unknown"]
        resolved_roles[tid] = Counter(known).most_common(1)[0][0] if known else "unknown"
    return {"raw": raw, "frames": wire, "source_indices": np.asarray(source_indices, dtype=np.int64),
        "source_fps": source_fps, "source_frame_count": total, "fps": SAMPLE_FPS,
        "width": width, "height": height, "frames_kept": len(raw), "truncated": False,
        "providers": list(getattr(detector, "providers", [])), "reid": embedder is not None,
        "frames_per_track": dict(counts), "track_spans": spans, "track_boxes": boxes,
        "roles": resolved_roles, "track_names": dict(agent._track_names),
        "enrolment": {tid: [value.tolist() for value in vectors]
                       for tid, vectors in agent.enrolment_samples().items()} if embedder else {}}


def build_windows(decoded):
    raw = decoded["raw"]
    windows, centers, starts, tracks = [], [], [], []
    zero = np.zeros((17, 3), dtype=np.float32)
    for start in range(0, len(raw) - WINDOW + 1, STRIDE):
        interval = raw[start:start + WINDOW]
        candidates = sorted({tid for frame in interval for tid in frame})
        for tid in candidates:
            primary = np.stack([frame.get(tid, zero) for frame in interval])
            if np.count_nonzero((primary[..., 2] > .3).sum(axis=1) >= 8) < 15:
                continue
            other_ids = [other for other in candidates if other != tid]
            secondary = (max(other_ids, key=lambda other: (
                sum(float(frame.get(other, zero)[..., 2].sum()) for frame in interval), -other))
                         if other_ids else None)
            secondary_values = np.stack([frame.get(secondary, zero) for frame in interval])
            windows.append(np.stack((primary, secondary_values), axis=1).astype(np.float16))
            # Detector geometry uses the unrounded center poses, exactly as during extraction.
            centers.append(np.stack((primary[15], secondary_values[15])))
            starts.append(start)
            tracks.append(tid)
    n = len(starts)
    return {"skeletons": np.asarray(windows, dtype=np.float16).reshape(n, 30, 2, 17, 3),
            "center_poses": np.asarray(centers, dtype=np.float32).reshape(n, 2, 17, 3),
            "starts": np.asarray(starts, dtype=np.int64), "track_ids": np.asarray(tracks, dtype=np.int64),
            "groups": np.asarray([f"upload/track-{tid}" for tid in tracks], dtype=str)}


def rgb_source_indices(centers, source_fps, frame_count):
    offsets = (np.arange(32, dtype=np.float64) - 15.5) / 15.
    requested = np.rint(np.asarray(centers)[:, None] + offsets * source_fps).astype(np.int64)
    return np.clip(requested, 0, frame_count - 1)


def extract_features(path, decoded, windows, encoder, object_detector, *,
                     progress=lambda **event: None, capture_factory=cv2.VideoCapture):
    n = len(windows["starts"])
    rgb_features = np.empty((n, 2, 768), dtype=np.float16)
    contexts = np.empty((n, 401), dtype=np.float32)
    if not n:
        return contexts, rgb_features
    centers = decoded["source_indices"][windows["starts"] + 15]
    requests = rgb_source_indices(centers, decoded["source_fps"], decoded["source_frame_count"])
    needed = set(requests.reshape(-1).tolist()) | set(centers.tolist())
    order = np.argsort(centers, kind="stable")
    cap = capture_factory(str(path))
    if not cap.isOpened():
        raise ValueError("Could not reopen the video for RGB and object features.")
    current, cache, last_center, last_detections, last_scene = -1, {}, None, None, None
    scene = scene_roi(decoded["width"], decoded["height"])
    try:
        for position, row in enumerate(order):
            end = max(int(requests[row].max()), int(centers[row]))
            while current < end:
                if not cap.grab():
                    raise ValueError(f"Video ended before required source frame {end}.")
                current += 1
                if current in needed:
                    ok, frame = cap.retrieve()
                    if not ok or frame is None:
                        raise ValueError(f"Could not decode source frame {current}.")
                    if frame.shape[:2] != (decoded["height"], decoded["width"]):
                        raise ValueError("RGB/pose decoded dimensions differ.")
                    cache[current] = frame
            center = int(centers[row])
            new_center = center != last_center
            if new_center:
                last_detections = object_detector.detect(cache[center])
            contexts[row] = context_features(last_detections, windows["center_poses"][row],
                                             cache[center].shape)
            roi, observed = person_roi(windows["skeletons"][row], decoded["width"], decoded["height"])
            views = []
            if new_center:
                views.append(np.stack([spatial_view(cv2.cvtColor(cache[int(index)], cv2.COLOR_BGR2RGB), scene)
                                       for index in requests[row]]))
            if observed:
                views.append(np.stack([spatial_view(cv2.cvtColor(cache[int(index)], cv2.COLOR_BGR2RGB), roi)
                                       for index in requests[row]]))
            if views:
                features = encoder(np.stack(views))
                if new_center:
                    last_scene = features[0]
                person_features = features[-1] if observed else last_scene
            else:
                person_features = last_scene
            rgb_features[row, 0], rgb_features[row, 1] = last_scene, person_features
            last_center = center
            if position + 1 < n:
                following = order[position + 1]
                minimum = min(int(requests[following].min()), int(centers[following]))
                cache = {index: frame for index, frame in cache.items() if index >= minimum}
            if position % 8 == 0 or position + 1 == n:
                progress(step="rgb_and_objects", windows=position + 1, total_windows=n)
    finally:
        cap.release()
    return contexts, rgb_features


def decision_segments(windows, prediction, class_names, roles):
    """Use disjoint 0.75-second center bins so different windows cannot invent pair overlap."""
    decisions, scores = prediction["decisions"], prediction["scores"]
    result = []
    for tid in sorted(set(windows["track_ids"].tolist())):
        rows = np.flatnonzero(windows["track_ids"] == tid)
        rows = rows[np.argsort(windows["starts"][rows], kind="stable")]
        for label, name in enumerate(class_names):
            run_start, previous, probabilities = None, None, []

            def finish():
                if run_start is not None:
                    start_s = (run_start + 15 - STRIDE / 2) / SAMPLE_FPS
                    end_s = (previous + 15 + STRIDE / 2) / SAMPLE_FPS
                    result.append(ActivitySegment(segment_id=f"track-{tid}-{label}-{run_start}",
                        track_id=tid, role=Role(roles.get(tid, "unknown")), activity_id=label,
                        activity_name=name, start_time=BASE + timedelta(seconds=start_s),
                        end_time=BASE + timedelta(seconds=end_s), confidence=float(np.mean(probabilities))))

            for row in rows:
                start = int(windows["starts"][row])
                if not decisions[row, label] or (previous is not None and start != previous + STRIDE):
                    finish()
                    run_start, previous, probabilities = None, None, []
                if decisions[row, label]:
                    if run_start is None:
                        run_start = start
                    previous = start
                    probabilities.append(float(scores[row, label]))
            finish()
    return sorted(result, key=lambda item: (item.track_id, item.start_time, item.activity_id))


def make_payload(decoded, windows, prediction, model, timings):
    segments = decision_segments(windows, prediction, model.class_names, decoded["roles"])
    reid_roles = dict(decoded["roles"])
    segments, subject, provenance, subject_tracks = assert_subject(segments, decoded["frames_per_track"],
        spans=decoded["track_spans"], boxes=decoded["track_boxes"], fps=SAMPLE_FPS)
    by_track = {}
    for segment in segments:
        tid = segment.track_id
        record = by_track.setdefault(tid, {"track_id": tid, "role": segment.role.value,
            "reid_role": reid_roles.get(tid, "unknown"), "is_subject": tid in ([subject] + list(subject_tracks)),
            "matched_name": decoded["track_names"].get(tid), "segments": []})
        record["segments"].append({"label": segment.activity_id, "name": segment.activity_name,
            "t0": (segment.start_time - BASE).total_seconds(), "t1": (segment.end_time - BASE).total_seconds(),
            "confidence": segment.confidence, "room": segment.room})
    tracks = list(by_track.values())
    for record in tracks:
        record["n_segments"] = len(record["segments"])
        record["fall_segments"] = sum(item["label"] in (7, 8) for item in record["segments"])
    top = []
    for score in prediction["scores"][:40]:
        top.append([[model.class_names[index], round(float(score[index]), 3)] for index in np.argsort(-score)[:3]])
    metadata = {"model_id": model.manifest["model_id"], "task_type": "multilabel",
        "rgb_precision": "float32", "thresholds": "frozen_20261015", "runtime_benchmark_verified": False,
        "source_benchmark_map": .6920715053917367,
        "source_benchmark_pair_f1": .28119293974437004,
        "source_benchmark_quality_gate_passed": False,
        "scope": "Offline uploaded-video research demo; T4 FP32 and tracked-person port are not the original benchmark."}
    stage1 = {"agent": 1, "name": "perception", "status": "done", "elapsed_s": timings["pose_s"],
        "payload": {"fps": SAMPLE_FPS, "frames_kept": decoded["frames_kept"], "truncated": False,
            "providers": decoded["providers"], "n_people": len(decoded["frames_per_track"]),
            "reid": decoded["reid"], "source_fps": decoded["source_fps"], "trained_fps": SAMPLE_FPS,
            "window_seconds": 1.5, "rate_matches_shards": True,
            "tracks": [{"track_id": tid, "role": role} for tid, role in reid_roles.items()]}}
    stage2 = {"agent": 2, "name": "activity", "status": "done" if len(windows["starts"]) else "skipped",
        "elapsed_s": timings["classify_s"], "payload": {"window_top3": top, "n_windows": len(windows["starts"]),
            "streams_served": ["context", "temporal", "rgb"], "n_segments": len(segments), "per_track": tracks,
            "decoding": "frozen sigmoid blend and temporal TV/drinking rule; 0.75-second center bins",
            "subject_track": subject, "subject_tracks": subject_tracks, "subject_provenance": provenance,
            "model": metadata}}
    return {"fps": SAMPLE_FPS, "width": decoded["width"], "height": decoded["height"],
        "frames_kept": decoded["frames_kept"], "truncated": False, "reid": decoded["reid"],
        "providers": decoded["providers"], "frames": decoded["frames"], "tracks": tracks,
        "n_people": len(decoded["frames_per_track"]), "stages": [stage1, stage2], "checks": [], "stages_run": "12",
        "frames_per_track": {str(key): value for key, value in decoded["frames_per_track"].items()},
        "track_spans": {str(key): list(value) for key, value in decoded["track_spans"].items()},
        "track_boxes": {str(key): list(value) for key, value in decoded["track_boxes"].items()},
        "subject_track": subject, "subject_tracks": subject_tracks, "subject_provenance": provenance,
        "gallery_updates": {"subject_track": subject, "matched": provenance == "reid_matched",
                            "embeddings": decoded["enrolment"].get(subject, [])},
        "observed_hours": decoded["frames_kept"] / SAMPLE_FPS / 3600., "day": BASE.date().isoformat(),
        "timing": timings, "model": metadata}


def process_video(path, model, encoder, detector, object_detector, *, embedder=None, gallery=None,
                  max_frames=36000, progress=lambda **event: None):
    began = time.monotonic()
    decoded = decode_tracks(path, detector, embedder=embedder, gallery=gallery,
                             max_frames=max_frames, progress=progress)
    pose_done = time.monotonic()
    windows = build_windows(decoded)
    contexts, rgb_features = extract_features(path, decoded, windows, encoder, object_detector, progress=progress)
    progress(step="hybrid_heads", windows=len(windows["starts"]))
    prediction = model.predict(windows["skeletons"], contexts, rgb_features, windows["starts"], windows["groups"])
    timings = {"pose_s": round(pose_done - began, 2), "classify_s": round(time.monotonic() - pose_done, 2)}
    return make_payload(decoded, windows, prediction, model, timings)
