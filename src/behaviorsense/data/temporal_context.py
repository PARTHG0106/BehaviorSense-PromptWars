"""Exact-time context neighborhoods over the retained multilabel windows.

Extraction can omit annotation gaps and low-visibility windows. A neighboring
array row is therefore not necessarily a neighboring time, even within a video.
This module uses only recording identity and sampled timestamps, never labels.
"""

from __future__ import annotations

import numpy as np


def _integer_option(value: int, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _identities(values, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or result.dtype.kind not in "US":
        raise ValueError(f"{name} must be a one-dimensional string array")
    result = result.astype(str, copy=False)
    if np.any(np.char.strip(result) == "") or np.any(np.char.strip(result) != result):
        raise ValueError(f"{name} must contain nonempty identities without surrounding whitespace")
    return result


def build_context_neighbors(video_ids, subjects, starts, radius=6, stride=15) -> np.ndarray:
    """Return int64 ``[N, 2 * radius + 1]`` row indices, with -1 for missing times.

    Column ``radius`` always contains the row itself. Other columns require the
    exact same video's ``start + offset * stride``; no closest-row substitution,
    temporal interpolation, or recording/subject crossing is permitted. Input
    rows need not be sorted. Empty, well-typed input arrays are supported.
    """
    radius = _integer_option(radius, "radius", 0)
    stride = _integer_option(stride, "stride", 1)
    videos = _identities(video_ids, "video_ids")
    people = _identities(subjects, "subjects")
    times = np.asarray(starts)
    if (times.ndim != 1 or times.dtype.kind not in "iu"
            or np.any(times < 0) or np.any(times > np.iinfo(np.int64).max)):
        raise ValueError("starts must be a one-dimensional array of nonnegative int64-compatible integers")
    if not (len(videos) == len(people) == len(times)):
        raise ValueError("video_ids, subjects and starts must have the same length")
    if radius * stride > np.iinfo(np.int64).max:
        raise ValueError("radius * stride exceeds the supported timestamp range")

    records: dict[str, tuple[str, dict[int, int]]] = {}
    for index, (video, subject, start) in enumerate(zip(videos, people, times)):
        if video not in records:
            records[video] = (subject, {})
        expected_subject, timeline = records[video]
        if subject != expected_subject:
            raise ValueError(f"video {video!r} contains mixed subjects")
        start = int(start)
        if start in timeline:
            raise ValueError(f"duplicate window identity: {video!r} at start {start}")
        timeline[start] = index

    neighbors = np.full((len(times), 2 * radius + 1), -1, dtype=np.int64)
    offsets = range(-radius, radius + 1)
    for _, timeline in records.values():
        for start, index in timeline.items():
            # Python integer arithmetic cannot wrap near an int64 boundary.
            neighbors[index] = [timeline.get(start + offset * stride, -1) for offset in offsets]
    return neighbors


__all__ = ["build_context_neighbors"]
