"""Regression witnesses for opt-in pose v2 and future-extraction tracking."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from behaviorsense.data.pose_contract_v2 import (
    POSE_CONTRACT_VERSION, build_directed_adjacency, normalize_visible_pose, to_bone_visible,
)
from behaviorsense.data.skeleton_dataset import normalise
from behaviorsense.models.stgcnpp import COCO_EDGES, FLIP_INDEX, build_adjacency, to_bone
from prepare_skeletons import POSE_SLOT_CONTRACT, assign_slots


def pose():
    values = np.zeros((4, 2, 17, 3), dtype=np.float32)
    values[:, 0, :, 0] = np.linspace(80, 120, 17)
    values[:, 0, :, 1] = np.linspace(40, 150, 17)
    values[:, 0, :, 2] = .8
    values[:, 0, [5, 6], :2] = [[90, 60], [110, 60]]
    values[:, 0, [11, 12], :2] = [[90, 100], [110, 100]]
    values[:, 0, :, 0] += np.arange(4, dtype=np.float32)[:, None] * 4
    return values


def tensor_pose(values):
    return torch.from_numpy(values.transpose(3, 0, 2, 1)[None].copy())


def test_missing_coordinate_invariance_and_known_legacy_failure():
    original = pose()
    original[:, 0, 11, 2] = 0
    corrupted = original.copy()
    corrupted[:, 0, 11, :2] = [10000, 10000]
    assert not np.allclose(normalise(original), normalise(corrupted))
    expected = normalize_visible_pose(original)
    np.testing.assert_array_equal(normalize_visible_pose(corrupted), expected)
    corrupted[:, 0, 11, :2] = np.nan
    np.testing.assert_array_equal(normalize_visible_pose(corrupted), expected)
    assert (expected[:, 0, 11] == 0).all()


def test_low_confidence_is_missing_and_confidence_is_preserved():
    original = pose()
    original[:, 0, 9, 2] = .1
    original[:, 0, 9, :2] = 100000
    normalized = normalize_visible_pose(original)
    assert (normalized[:, 0, 9] == 0).all()
    np.testing.assert_array_equal(normalized[..., 2][original[..., 2] >= .3],
                                  original[..., 2][original[..., 2] >= .3])
    assert np.isfinite(normalized).all()


def test_normalization_removes_translation_scale_but_preserves_root_motion_and_rotation():
    original = pose()
    expected = normalize_visible_pose(original)
    transformed = original.copy()
    transformed[..., :2] = transformed[..., :2] * 2.5 + [550, -320]
    np.testing.assert_allclose(normalize_visible_pose(transformed), expected, atol=2e-6)
    assert expected[3, 0, 11, 0] - expected[0, 0, 11, 0] == pytest.approx(12 / 40)
    rotation = np.array([[0., -1.], [1., 0.]])
    rotated = original.copy()
    rotated[..., :2] = rotated[..., :2] @ rotation.T
    np.testing.assert_allclose(normalize_visible_pose(rotated)[..., :2],
                               expected[..., :2] @ rotation.T, atol=2e-6)


def test_normalization_has_finite_fallbacks_and_all_missing_is_exact_zero():
    missing = np.full((4, 2, 17, 3), np.nan)
    missing[..., 2] = 0
    np.testing.assert_array_equal(normalize_visible_pose(missing), np.zeros_like(missing))
    partial = pose()
    partial[..., 2] = 0
    partial[:, 0, [7, 9], 2] = .8
    assert np.isfinite(normalize_visible_pose(partial)).all()
    degenerate = pose()
    degenerate[..., 2] = 0
    degenerate[:, 0, 9, 2] = .8
    assert np.isfinite(normalize_visible_pose(degenerate)).all()
    assert not np.any(normalize_visible_pose(degenerate)[:, 1])


def test_bone_confidence_is_minimum_and_missing_endpoint_cannot_invent_geometry():
    joints = tensor_pose(pose())
    joints[:, 2, :, 7] = .9
    joints[:, 2, :, 9] = .6
    bones = to_bone_visible(joints)
    torch.testing.assert_close(bones[:, :2, :, 9], joints[:, :2, :, 9] - joints[:, :2, :, 7])
    torch.testing.assert_close(bones[:, 2, :, 9], torch.full_like(bones[:, 2, :, 9], .6))
    joints[:, 2, :, 9] = 0
    assert to_bone(joints)[:, :, :, 9].abs().sum() > 0
    joints[:, :2, :, 9] = float("nan")
    bones = to_bone_visible(joints)
    assert torch.isfinite(bones).all() and (bones[:, 2] >= 0).all()
    assert not bones[:, :, :, 9].any()
    assert not bones[:, :, :, 0].any()


def test_bones_are_translation_invariant_flip_equivariant_and_differentiable():
    joints = tensor_pose(normalize_visible_pose(pose())).requires_grad_(True)
    original = to_bone_visible(joints)
    translated = joints.detach().clone()
    translated[:, :2] += 123
    torch.testing.assert_close(to_bone_visible(translated), original.detach(), atol=1e-5, rtol=0)
    flipped = joints.detach().clone()[:, :, :, list(FLIP_INDEX)]
    flipped[:, 0] *= -1
    expected = original.detach().clone()[:, :, :, list(FLIP_INDEX)]
    expected[:, 0] *= -1
    torch.testing.assert_close(to_bone_visible(flipped), expected)
    original[:, :2].square().sum().backward()
    assert torch.isfinite(joints.grad).all()
    assert joints.grad[:, :2].abs().sum() > 0
    empty = torch.zeros_like(joints)
    empty[:, :2] = float("nan")
    assert not to_bone_visible(empty).any()


def test_directed_normalization_preserves_every_anatomical_edge_and_incoming_mass():
    old, new = build_adjacency(), build_directed_adjacency()
    missing_before = [(a, b) for i, j in COCO_EDGES for a, b in ((i, j), (j, i)) if not old[:, a, b].any()]
    assert len(missing_before) == 10
    for i, j in COCO_EDGES:
        assert new[:, i, j].sum() > 0 and new[:, j, i].sum() > 0
    assert new[:, 9, 7].sum() > 0 and new[:, 10, 8].sum() > 0
    for partition in new:
        totals = partition.sum(axis=0)
        np.testing.assert_allclose(totals[totals > 0], 1.)
    assert np.isfinite(new).all() and (new >= 0).all()
    # A source wrist impulse can actually reach its elbow under the model's einsum convention.
    wrist = np.eye(17, dtype=np.float32)[9]
    assert (wrist @ new.sum(axis=0))[7] > 0
    assert POSE_CONTRACT_VERSION.endswith("v2")


def test_directed_graph_arguments_and_disconnected_input():
    simple = build_directed_adjacency(edges=((0, 1), (1, 2)), n_nodes=3, centre=1)
    assert simple.shape == (3, 3, 3)
    np.testing.assert_array_equal(build_directed_adjacency(edges=(), n_nodes=1), [[[1.]], [[0.]], [[0.]]])
    with pytest.raises(ValueError, match="not connected"):
        build_directed_adjacency(edges=((0, 1),), n_nodes=3)
    with pytest.raises(ValueError):
        build_directed_adjacency(edges=((0, 17),))
    with pytest.raises(ValueError):
        build_directed_adjacency(strategy="unknown")


def test_rejected_tracking_jumps_leave_former_occupied_slots_blank_then_reseed():
    points = pose()[0, 0, :, :2]
    first = np.stack([points, points + [200, 0]])
    scores = np.full((2, 17), .9, dtype=np.float32)
    previous = assign_slots(first, scores, None)
    jumped = first + 10000
    blank = assign_slots(jumped, scores, previous)
    assert not blank.any()
    seeded = assign_slots(jumped, scores, blank)
    assert (seeded[:, :, 2] > 0).all()
    np.testing.assert_allclose(np.sort(seeded[:, :, 0].mean(axis=1)), np.sort(jumped[:, :, 0].mean(axis=1)))
    assert POSE_SLOT_CONTRACT.endswith("v2")


@pytest.mark.parametrize("threshold", [0., -1., 1.1, float("nan")])
def test_bad_thresholds_are_rejected(threshold):
    with pytest.raises(ValueError):
        normalize_visible_pose(pose(), threshold)
    with pytest.raises(ValueError):
        to_bone_visible(tensor_pose(pose()), threshold)


def test_bad_visible_values_are_rejected():
    bad = pose()
    bad[0, 0, 9, 0] = np.nan
    with pytest.raises(ValueError, match="reliable"):
        normalize_visible_pose(bad)
    with pytest.raises(ValueError, match="reliable"):
        to_bone_visible(tensor_pose(bad))
    bad = pose()
    bad[0, 0, 9, 2] = 1.1
    with pytest.raises(ValueError, match="confidence"):
        normalize_visible_pose(bad)
    with pytest.raises(ValueError, match="confidence"):
        to_bone_visible(tensor_pose(bad))


if __name__ == "__main__":
    class RunnerSummary:
        def pytest_terminal_summary(self, terminalreporter):
            stats = terminalreporter.stats
            passed = len(stats.get("passed", []))
            total = passed + len(stats.get("failed", [])) + len(stats.get("error", []))
            print(f"{passed}/{total} passed; {len(stats.get('skipped', []))} skipped")

    raise SystemExit(pytest.main([__file__, "-q"], plugins=[RunnerSummary()]))
