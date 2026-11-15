"""Opt-in visibility-safe pose inputs for newly trained models.

Archived/serving checkpoints retain their original normalise(), to_bone(), and
build_adjacency() behavior. These functions are a different input contract and
must be recorded in new feature/checkpoint manifests; swapping them underneath
an existing checkpoint is not a compatible inference change. They cannot repair
person identity switches already serialized in old skeleton shards.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from behaviorsense.models.stgcnpp import COCO_EDGES, N_JOINTS

POSE_CONTRACT_VERSION = "behaviorsense.pose-visible.v2"


def _threshold(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError("threshold must be finite and in (0,1]")
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError("threshold must be finite and in (0,1]")
    return value


def normalize_visible_pose(skeleton: np.ndarray, threshold: float = .3) -> np.ndarray:
    """Return float32 ``[T,2,17,3]`` normalized coordinates and confidence.

    Scores below ``threshold`` are missing: all their output channels are zero,
    and their input coordinates cannot affect any other joint. Each person uses
    one fixed origin from the first frame with two reliable hips, preserving root
    motion. Without a reliable hip pair, the first reliable-frame joint centroid
    is used. Scale is median torso length on fully observed torso frames, then
    median reliable anatomical-edge length when the torso is unavailable. A
    degenerate person with no measurable length uses scale one.

    Translation and uniform scale are removed when a length is measurable.
    Rotation is preserved rather than making upright and horizontal poses equal.
    Nonfinite coordinates on missing joints are ignored; nonfinite reliable
    coordinates and invalid confidence values are refused.
    """
    threshold = _threshold(threshold)
    values = np.asarray(skeleton)
    if values.ndim != 4 or values.shape[0] < 1 or values.shape[1:] != (2, N_JOINTS, 3):
        raise ValueError("skeleton must have shape [T,2,17,3] with T >= 1")
    if values.dtype.kind not in "fiu":
        raise ValueError("skeleton must contain numeric coordinates and confidence")
    values = values.astype(np.float64, copy=False)
    scores = values[..., 2]
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("joint confidence must be finite and in [0,1]")
    reliable = scores >= threshold
    if not np.isfinite(values[..., :2][reliable]).all():
        raise ValueError("reliable joint coordinates must be finite")
    clean = np.where(reliable[..., None], values, 0.)
    result = np.zeros(values.shape, dtype=np.float32)
    for person in range(2):
        visible = reliable[:, person]
        active_frames = visible.any(axis=1)
        if not active_frames.any():
            continue
        points = clean[:, person, :, :2]
        hip_valid = visible[:, 11] & visible[:, 12]
        shoulder_valid = visible[:, 5] & visible[:, 6]
        hips = (points[:, 11] + points[:, 12]) * .5
        shoulders = (points[:, 5] + points[:, 6]) * .5
        if hip_valid.any():
            origin = hips[np.flatnonzero(hip_valid)[0]]
        else:
            first = np.flatnonzero(active_frames)[0]
            origin = points[first, visible[first]].mean(axis=0)
        lengths = np.linalg.norm(shoulders - hips, axis=1)[hip_valid & shoulder_valid]
        lengths = lengths[lengths > 1e-12]
        if not len(lengths):
            edges = []
            for a, b in COCO_EDGES:
                valid = visible[:, a] & visible[:, b]
                edges.extend(np.linalg.norm(points[valid, a] - points[valid, b], axis=1).tolist())
            lengths = np.asarray(edges)
            lengths = lengths[lengths > 1e-12]
        scale = float(np.median(lengths)) if len(lengths) else 1.
        normalized = (points[visible] - origin) / scale
        if not np.isfinite(normalized).all() or np.any(np.abs(normalized) > np.finfo(np.float32).max):
            raise ValueError("normalized reliable coordinates exceed finite float32 range")
        result[:, person, :, :2][visible] = normalized.astype(np.float32)
        result[:, person, :, 2][visible] = scores[:, person][visible]
    return result


def to_bone_visible(joints: torch.Tensor, threshold: float = .3) -> torch.Tensor:
    """Convert ``[N,3,T,17,2]`` joints to masked bone xy/min-endpoint-confidence.

    A bone exists only when both endpoints meet the confidence threshold. Missing
    endpoints yield three exact zeros, including when their coordinates contain
    NaN. The root has no parent and its bone is zero. Dtype/device and gradients
    through observed coordinates are preserved.
    """
    threshold = _threshold(threshold)
    if (joints.ndim != 5 or joints.shape[0] < 1 or joints.shape[1] != 3
            or joints.shape[2] < 1 or tuple(joints.shape[3:]) != (N_JOINTS, 2)
            or not joints.is_floating_point()):
        raise ValueError("joints must be a floating tensor [N,3,T,17,2]")
    scores = joints[:, 2]
    if not bool(torch.isfinite(scores).all()) or bool(((scores < 0) | (scores > 1)).any()):
        raise ValueError("joint confidence must be finite and in [0,1]")
    reliable = scores >= threshold
    if not bool((torch.isfinite(joints[:, :2]) | ~reliable[:, None]).all()):
        raise ValueError("reliable joint coordinates must be finite")
    clean = joints.masked_fill(~reliable[:, None], 0)
    bones = torch.zeros_like(joints)
    parents = {}
    for a, b in COCO_EDGES:
        parents.setdefault(max(a, b), min(a, b))
    for child, parent in parents.items():
        valid = reliable[:, :, child] & reliable[:, :, parent]
        delta = clean[:, :2, :, child] - clean[:, :2, :, parent]
        bones[:, :2, :, child] = delta.masked_fill(~valid[:, None], 0)
        confidence = torch.minimum(scores[:, :, child], scores[:, :, parent])
        bones[:, 2, :, child] = confidence.masked_fill(~valid, 0)
    return bones


def build_directed_adjacency(strategy: str = "spatial", *, edges=COCO_EDGES,
                             n_nodes: int = N_JOINTS, centre: int = 0) -> np.ndarray:
    """Return source→target spatial partitions with nonempty columns summing to one.

    Partition order matches the legacy convention: equal distance/self, outward,
    inward. Column normalization averages the incoming messages at each target
    for ``einsum('nkctv,kvw->nctw', features, A)``. It never multiplies a source by
    its *incoming* degree, which used to delete directed edges at roots/leaves.
    Graph layout arguments match build_adjacency(), but the numerical contract
    is opt-in v2 and cannot replace a checkpoint's stored adjacency silently.
    """
    if strategy != "spatial":
        raise ValueError("only the spatial adjacency strategy is supported")
    if (isinstance(n_nodes, bool) or not isinstance(n_nodes, (int, np.integer)) or n_nodes < 1
            or isinstance(centre, bool) or not isinstance(centre, (int, np.integer))
            or not 0 <= centre < n_nodes):
        raise ValueError("n_nodes and centre must identify a nonempty graph")
    hop = np.full((n_nodes, n_nodes), np.inf)
    np.fill_diagonal(hop, 0.)
    for edge in edges:
        if len(edge) != 2:
            raise ValueError("each edge must contain two valid integer nodes")
        a, b = edge
        if any(isinstance(node, bool) or not isinstance(node, (int, np.integer))
               or not 0 <= node < n_nodes for node in (a, b)):
            raise ValueError("each edge must contain two valid integer nodes")
        if a != b:
            hop[a, b] = hop[b, a] = 1.
    for node in range(n_nodes):
        hop = np.minimum(hop, hop[:, node, None] + hop[None, node, :])
    distance = hop[:, centre]
    if not np.isfinite(distance).all():
        raise ValueError(f"joints {np.flatnonzero(~np.isfinite(distance)).tolist()} are not connected to centre")
    adjacency = np.zeros((3, n_nodes, n_nodes), dtype=np.float32)
    for source in range(n_nodes):
        for target in range(n_nodes):
            if hop[source, target] > 1:
                continue
            partition = 0 if distance[target] == distance[source] else 1 if distance[target] > distance[source] else 2
            adjacency[partition, source, target] = 1.
    for partition in adjacency:
        degree = partition.sum(axis=0, keepdims=True)
        np.divide(partition, degree, out=partition, where=degree > 0)
    return adjacency


__all__ = ["POSE_CONTRACT_VERSION", "normalize_visible_pose", "to_bone_visible", "build_directed_adjacency"]
