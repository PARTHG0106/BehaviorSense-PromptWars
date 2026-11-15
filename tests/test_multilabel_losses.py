"""Partial labels and numerical stability must survive asymmetric focusing."""

from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behaviorsense.models.multilabel_losses import masked_asymmetric_loss


def test_no_focusing_or_clipping_matches_masked_bce_and_gradients():
    logits = torch.tensor([[-1000., -4., 0., 4., 1000., 50.]], requires_grad=True)
    targets = torch.tensor([[1., 0., 1., 0., 0., 1.]])
    mask = torch.tensor([[True, True, True, True, True, False]])
    actual = masked_asymmetric_loss(logits, targets, mask, gamma_neg=0, gamma_pos=0, clip=0)
    expected = F.binary_cross_entropy_with_logits(logits[mask], targets[mask])
    torch.testing.assert_close(actual, expected)
    actual_grad, = torch.autograd.grad(actual, logits, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected, logits)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_unknown_entries_change_neither_value_nor_gradient():
    targets = torch.tensor([[1., 0., float("nan"), 99.]])
    mask = torch.tensor([[True, True, False, False]])
    logits = torch.tensor([[-2., 1., float("nan"), float("inf")]], requires_grad=True)
    actual = masked_asymmetric_loss(logits, targets, mask)
    expected = masked_asymmetric_loss(logits[:, :2], targets[:, :2], mask[:, :2])
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] > 0
    torch.testing.assert_close(logits.grad[:, 2:], torch.zeros(1, 2))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("clip", [0., .05])
def test_extreme_logits_have_finite_loss_and_gradients_under_autocast(dtype, clip):
    logits = torch.tensor([[-1000., -100., 0., 100., 1000., -1000., 1000.]],
                          dtype=dtype, requires_grad=True)
    targets = torch.tensor([[1., 1., 0., 0., 0., 0., 1.]])
    mask = torch.ones_like(targets, dtype=torch.bool)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = masked_asymmetric_loss(logits, targets, mask, clip=clip)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0] < 0  # Saturated wrong positives can still recover.


def test_easy_negatives_are_removed_but_positive_labels_remain_independent():
    logits = torch.tensor([[0., 0., -10., 2.]], requires_grad=True)
    targets = torch.tensor([[1., 1., 0., 0.]])
    masked_asymmetric_loss(logits, targets, torch.ones_like(targets, dtype=torch.bool)).backward()
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] < 0
    assert logits.grad[0, 2] == 0 and logits.grad[0, 3] > 0


def test_forward_preserves_callers_gradient_mode():
    logits = torch.tensor([[0.]], requires_grad=True)
    targets = torch.ones_like(logits)
    mask = torch.ones_like(logits, dtype=torch.bool)
    assert torch.is_grad_enabled()
    masked_asymmetric_loss(logits, targets, mask)
    assert torch.is_grad_enabled()
    with torch.no_grad():
        loss = masked_asymmetric_loss(logits, targets, mask)
        assert not torch.is_grad_enabled() and not loss.requires_grad
    assert torch.is_grad_enabled()


@pytest.mark.parametrize("kwargs", [
    {"gamma_neg": -1}, {"gamma_pos": -1}, {"gamma_neg": float("nan")},
    {"gamma_pos": float("inf")}, {"clip": -.01}, {"clip": 1},
    {"clip": float("nan")}, {"eps": 0}, {"eps": 1}, {"eps": float("inf")},
])
def test_invalid_loss_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        masked_asymmetric_loss(torch.zeros(1, 2), torch.ones(1, 2),
                               torch.ones(1, 2, dtype=torch.bool), **kwargs)


def test_invalid_shapes_empty_mask_and_nonbinary_known_labels_are_rejected():
    logits, targets = torch.zeros(2, 3), torch.ones(2, 3)
    mask = torch.ones(2, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="identical shapes"):
        masked_asymmetric_loss(logits, targets[:, :1], mask)
    with pytest.raises(ValueError, match="no known labels"):
        masked_asymmetric_loss(logits, targets, torch.zeros_like(mask))
    with pytest.raises(ValueError, match="known targets must be binary"):
        masked_asymmetric_loss(logits, targets * .5, mask)
    with pytest.raises(ValueError, match="mask must be binary"):
        masked_asymmetric_loss(logits, targets, mask.float() * .5)
    with pytest.raises(ValueError, match="logits must be finite"):
        masked_asymmetric_loss(logits + float("nan"), targets, mask)


def test_reference_equations_match_for_moderate_logits():
    logits = torch.tensor([[-3., -.2, .7, 2.]], requires_grad=True)
    targets = torch.tensor([[1., 0., 1., 0.]])
    mask = torch.ones_like(targets, dtype=torch.bool)
    p = logits.sigmoid()
    negative = (1 - p + .05).clamp(max=1)
    likelihood = torch.where(targets.bool(), p, negative)
    exponent = torch.where(targets.bool(), 1., 4.)
    reference = -(likelihood.log() * (1 - likelihood.detach()).pow(exponent)).mean()
    actual = masked_asymmetric_loss(logits, targets, mask)
    torch.testing.assert_close(actual, reference)
    actual_gradient, = torch.autograd.grad(actual, logits, retain_graph=True)
    reference_gradient, = torch.autograd.grad(reference, logits)
    torch.testing.assert_close(actual_gradient, reference_gradient)


if __name__ == "__main__":
    class RunnerSummary:
        def pytest_terminal_summary(self, terminalreporter):
            stats = terminalreporter.stats
            passed = len(stats.get("passed", []))
            total = passed + len(stats.get("failed", [])) + len(stats.get("error", []))
            print(f"{passed}/{total} passed; {len(stats.get('skipped', []))} skipped")

    raise SystemExit(pytest.main([__file__, "-q"], plugins=[RunnerSummary()]))
