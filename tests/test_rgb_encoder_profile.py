"""Analytical CPU checks of encoder profiling; no GPU throughput/accuracy claims."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from behaviorsense.models import rgb_action


class RecordingProfile:
    def __init__(self):
        self.events = []
        self.active = None

    @contextmanager
    def cuda(self, name):
        assert self.active is None
        self.active = name
        self.events.append(("enter", name))
        try:
            yield
        except BaseException:
            self.events.append(("failed", name))
            raise
        finally:
            self.events.append(("exit", name))
            self.active = None


class AnalyticEncoder(torch.nn.Module):
    def __init__(self, profile=None, *, fail=False):
        super().__init__()
        self.register_parameter("weight", torch.nn.Parameter(torch.ones(1)))
        self.profile, self.fail = profile, fail

    def forward(self, values):
        assert not self.training and not torch.is_grad_enabled()
        assert torch.is_inference_mode_enabled() and not self.weight.requires_grad
        assert values.dtype == torch.float32 and values.shape[1:] == (3, 32, 224, 224)
        assert values.is_contiguous(memory_format=torch.channels_last_3d)
        if self.profile is not None:
            assert self.profile.active == "forward"
        if self.fail:
            raise RuntimeError("analytic forward failed")
        return values[:, :, 0, 0, 0] + values[:, :, -1, -1, -1]


@pytest.fixture
def cpu_autocast(monkeypatch):
    calls = []
    def autocast(device, *, dtype):
        assert device == "cuda" and dtype == torch.bfloat16
        calls.append((device, dtype))
        return nullcontext()
    # Keep actual tensor conversion and model invocation; replace CUDA-only AMP.
    monkeypatch.setattr(torch, "autocast", autocast)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: pytest.fail("unexpected real CUDA synchronization"))
    return calls


def legacy_reference(model, views):
    """Original encoder arithmetic, independent of the optional timing path."""
    with torch.inference_mode():
        model.eval()
        values = torch.from_numpy(np.ascontiguousarray(views)).to("cpu")
        values = values.permute(0, 4, 1, 2, 3).float().div_(255)
        mean = values.new_tensor(rgb_action.PREPROCESSING["mean"]).view(1, 3, 1, 1, 1)
        std = values.new_tensor(rgb_action.PREPROCESSING["std"]).view(1, 3, 1, 1, 1)
        values.sub_(mean).div_(std)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(values)
        return result.float().cpu().numpy()


@pytest.mark.parametrize("strided", [False, True])
def test_profile_preserves_original_encoder_values_layout_and_input(cpu_autocast, strided):
    views = np.random.default_rng(42).integers(0, 256, (2, 32, 224, 224, 3), dtype=np.uint8)
    if strided:
        views = views[:, :, ::-1, :, ::-1]
    original = views.copy()
    model = AnalyticEncoder()
    encoder = rgb_action.RGBActionEncoder(model, "cpu", 3)
    expected = legacy_reference(model, views)
    model.train()
    ordinary = encoder(views)
    profile = RecordingProfile()
    model.profile = profile
    profiled = encoder(views, profile=profile)
    np.testing.assert_array_equal(ordinary, expected)
    np.testing.assert_array_equal(profiled, expected)
    np.testing.assert_array_equal(views, original)
    assert len(cpu_autocast) == 3
    assert profile.events == [(phase, name) for name in ("transfer", "preprocess", "forward", "readback")
                              for phase in ("enter", "exit")]


def test_forward_failure_closes_timing_context_and_never_records_readback(cpu_autocast):
    profile = RecordingProfile()
    encoder = rgb_action.RGBActionEncoder(AnalyticEncoder(profile, fail=True), "cpu", 3)
    views = np.zeros((1, 32, 224, 224, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="analytic forward failed"):
        encoder(views, profile=profile)
    assert profile.active is None
    assert profile.events[-3:] == [("enter", "forward"), ("failed", "forward"), ("exit", "forward")]
    assert not any(name == "readback" for _, name in profile.events)


def test_invalid_views_are_rejected_before_profiling_or_model_work(cpu_autocast):
    profile = RecordingProfile()
    encoder = rgb_action.RGBActionEncoder(AnalyticEncoder(profile, fail=True), "cpu", 3)
    with pytest.raises(ValueError, match="encoder views"):
        encoder(np.zeros((1, 16, 224, 224, 3), dtype=np.uint8), profile=profile)
    assert profile.events == [] and cpu_autocast == []


if __name__ == "__main__":
    class RunnerSummary:
        def pytest_terminal_summary(self, terminalreporter):
            stats = terminalreporter.stats
            passed = len(stats.get("passed", []))
            total = passed + len(stats.get("failed", [])) + len(stats.get("error", []))
            print(f"{passed}/{total} passed; {len(stats.get('skipped', []))} skipped")

    raise SystemExit(pytest.main([__file__, "-q"], plugins=[RunnerSummary()]))
