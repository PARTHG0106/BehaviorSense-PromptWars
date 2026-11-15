"""Temporal identity, masking and gradient tests; no synthetic accuracy claims."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from behaviorsense.data.temporal_context import build_context_neighbors
from behaviorsense.models.temporal_fusion import TEMPORAL_FORMAT, TemporalMultilabelFusion
from behaviorsense.models.visual_context import CONTEXT_DIM


def test_exact_time_neighbors_handle_gaps_unsorted_rows_and_other_recordings():
    videos = np.array(["video_a", "video_b", "video_a", "video_a", "video_b"])
    subjects = np.array(["p03", "p04", "p03", "p03", "p04"])
    starts = np.array([30, 0, 0, 45, 15])
    neighbors = build_context_neighbors(videos, subjects, starts, radius=1)
    assert neighbors.dtype == np.int64
    np.testing.assert_array_equal(neighbors, [[-1, 0, 3], [-1, 1, 4], [-1, 2, -1],
                                             [0, 3, -1], [1, 4, -1]])
    np.testing.assert_array_equal(neighbors[:, 1], np.arange(len(starts)))
    for row, indices in enumerate(neighbors):
        valid = indices >= 0
        assert (videos[indices[valid]] == videos[row]).all()
        assert (subjects[indices[valid]] == subjects[row]).all()
        np.testing.assert_array_equal(starts[indices[valid]], starts[row] + np.array([-15, 0, 15])[valid])


@pytest.mark.parametrize("videos,subjects,starts,message", [
    (["a", "a"], ["p03", "p03"], [0, 0], "duplicate"),
    (["a", "a"], ["p03", "p04"], [0, 15], "mixed subjects"),
    (["a"], ["p03", "p03"], [0], "same length"),
    ([["a"]], ["p03"], [0], "one-dimensional"),
    (["a"], ["p03"], [0.0], "integers"),
    (["a"], ["p03"], [-1], "nonnegative"),
    ([1], ["p03"], [0], "string array"),
    ([" "], ["p03"], [0], "nonempty"),
])
def test_neighbor_builder_rejects_ambiguous_or_malformed_identity(videos, subjects, starts, message):
    with pytest.raises(ValueError, match=message):
        build_context_neighbors(videos, subjects, starts)


@pytest.mark.parametrize("options", [{"radius": -1}, {"radius": True}, {"stride": 0}, {"stride": 1.5}])
def test_neighbor_options_are_validated(options):
    with pytest.raises(ValueError):
        build_context_neighbors(np.array(["a"]), np.array(["p03"]), np.array([0]), **options)


def test_neighbor_builder_empty_zero_radius_and_int64_boundary():
    empty = build_context_neighbors(np.array([], dtype=str), np.array([], dtype=str),
                                    np.array([], dtype=np.int64))
    assert empty.shape == (0, 13)
    limit = np.iinfo(np.int64).max
    one = build_context_neighbors(["a"], ["p03"], np.array([limit]), radius=0)
    np.testing.assert_array_equal(one, [[0]])
    boundary = build_context_neighbors(["a"], ["p03"], np.array([limit]), radius=1)
    np.testing.assert_array_equal(boundary, [[-1, 0, -1]])
    with pytest.raises(ValueError, match="int64-compatible"):
        build_context_neighbors(["a"], ["p03"], np.array([limit + 1], dtype=np.uint64))


@pytest.fixture
def model_and_inputs():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(123)
    model = TemporalMultilabelFusion(n_classes=22, base_channels=4, context_hidden=8, dropout=0.)
    joints = torch.randn(2, 3, 30, 17, 2)
    context = torch.randn(2, 13, CONTEXT_DIM)
    context[..., 0] = 1.
    yield model, joints, context
    torch.set_num_threads(previous_threads)


def _activate_residual(model):
    # At initialization context intentionally contributes zero. Set the final
    # trainable layer to a deterministic nonzero value to exercise its behavior.
    with torch.no_grad():
        torch.manual_seed(124)
        model.fusion_head[-1].weight.normal_(0., .2)
        model.fusion_head[-1].bias.fill_(.3)


def test_zero_initialized_residual_and_trained_all_missing_fallback(model_and_inputs):
    model, joints, context = model_and_inputs
    model.eval()
    assert TEMPORAL_FORMAT != "behaviorsense.stgcnpp.context-multilabel.v1"
    with torch.no_grad():
        baseline = model.forward_skeleton(joints)
        torch.testing.assert_close(model(joints, context), baseline, rtol=0, atol=0)
        _activate_residual(model)
        context[..., 0] = 0
        context[..., 1:] = float("nan")
        torch.testing.assert_close(model(joints, context), baseline, rtol=0, atol=0)


def test_padding_values_do_not_affect_logits_after_temporal_processing(model_and_inputs):
    model, joints, context = model_and_inputs
    model.eval()
    _activate_residual(model)
    context[:, [0, 2, 7, 12], 0] = 0
    changed = context.clone()
    changed[:, [0, 2, 7, 12], 1:] = float("nan")
    with torch.no_grad():
        torch.testing.assert_close(model(joints, changed), model(joints, context), rtol=0, atol=0)


def test_temporal_order_changes_prediction_with_center_observation_unchanged(model_and_inputs):
    model, joints, context = model_and_inputs
    model.eval()
    _activate_residual(model)
    # Time reversal preserves both the center observation and the observation
    # multiset. A mean-only context encoder would produce identical predictions.
    reverse = context.flip(dims=(1,))
    torch.testing.assert_close(context[:, 6], reverse[:, 6], rtol=0, atol=0)
    with torch.no_grad():
        forward_logits, reverse_logits = model(joints, context), model(joints, reverse)
    assert (forward_logits - reverse_logits).abs().max().item() > 1e-6


def test_both_skeleton_branches_learn_and_context_receives_gradients(model_and_inputs):
    model, joints, context = model_and_inputs
    model.train()
    target = torch.rand(2, 22)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(joints, context), target)
    loss.backward()
    for branch in (model.joint_backbone, model.bone_backbone):
        first = branch.blocks[0].gcn.conv.weight.grad
        assert first is not None and torch.isfinite(first).all() and first.abs().sum() > 0
    last = model.fusion_head[-1].weight.grad
    assert last is not None and torch.isfinite(last).all() and last.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    _activate_residual(model)
    torch.nn.functional.binary_cross_entropy_with_logits(model(joints, context), target).backward()
    gradient = model.context_encoder[1].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_model_rejects_wrong_stream_and_context_shapes(model_and_inputs):
    model, joints, context = model_and_inputs
    with pytest.raises(ValueError, match="joints"):
        model(joints[:, :, :29], context)
    with pytest.raises(ValueError, match="context"):
        model(joints, context[:, :12])


if __name__ == "__main__":
    class RunnerSummary:
        def pytest_terminal_summary(self, terminalreporter):
            stats = terminalreporter.stats
            passed = len(stats.get("passed", []))
            total = passed + len(stats.get("failed", [])) + len(stats.get("error", []))
            print(f"{passed}/{total} passed; {len(stats.get('skipped', []))} skipped")

    raise SystemExit(pytest.main([__file__, "-q"], plugins=[RunnerSummary()]))
