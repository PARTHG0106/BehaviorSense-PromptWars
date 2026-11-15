"""Profiled extraction preserves every input pixel and published row."""
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from test_rgb_action import tiny_video, encoder_info
import extract_rgb_action_features as extraction
from extract_rgb_action_features import extract_video_features
from rgb_profile_stage import TimingProfile


class PixelRecorder:
    def __init__(self):
        self.hashes = []

    def __call__(self, values, profile=None):
        self.hashes.append(hashlib.sha256(values.tobytes()).hexdigest())
        return np.repeat(values[:, 15, 112, 112, 0:1].astype(np.float32), 768, axis=1)


def test_profiled_backends_preserve_entire_encoder_inputs_and_all_output_arrays(tiny_video):
    video, shard, out, _ = tiny_video
    encoders, outputs = [], []
    for backend in ("reference", "native"):
        encoder, profile = PixelRecorder(), TimingProfile()
        target = out.with_name(backend + ".npz")
        report = extract_video_features(video, shard, target, encoder, encoder_info(),
            batch_size=2, profile=profile, spatial_backend=backend)
        assert report["status"] == "complete" and report["n_windows"] == 3
        stages = profile.summary()["stages"]
        assert all(stages[name]["calls"] > 0 for name in
            ("input_validation", "decode", "scene_crop", "person_crop", "pack", "serialize"))
        assert not any(stage["failed_calls"] for stage in stages.values())
        encoders.append(encoder)
        with np.load(target, allow_pickle=False) as data:
            assert json.loads(str(data["metadata"]))["spatial_backend"] == backend
            outputs.append({name: data[name] for name in data.files if name != "metadata"})
    assert encoders[0].hashes == encoders[1].hashes
    assert outputs[0].keys() == outputs[1].keys()
    for name in outputs[0]:
        np.testing.assert_array_equal(outputs[0][name], outputs[1][name])


def test_reference_cache_cannot_impersonate_native_probe(tiny_video):
    video, shard, out, _ = tiny_video
    extract_video_features(video, shard, out, PixelRecorder(), encoder_info(), spatial_backend="reference")
    with pytest.raises(ValueError, match="spatial backend"):
        extract_video_features(video, shard, out, PixelRecorder(), encoder_info(), spatial_backend="native")


def test_unknown_spatial_backend_is_rejected_before_input_reads(tmp_path):
    with pytest.raises(ValueError, match="spatial_backend"):
        extract_video_features(tmp_path / "missing.mp4", tmp_path / "missing.npz",
            tmp_path / "output.npz", PixelRecorder(), encoder_info(), spatial_backend="unreviewed")


def test_prefetch_overlaps_decode_with_one_encoder_worker_and_preserves_exact_pixels(tiny_video, monkeypatch):
    video, shard, out, payload = tiny_video
    reference = PixelRecorder()
    sequential = out.with_name("sequential.npz")
    extract_video_features(video, shard, sequential, reference, encoder_info(), batch_size=1)
    started, overlapped = threading.Event(), threading.Event()
    real_capture = extraction.cv2.VideoCapture
    centers = payload["source_frames"][:, 15]
    first_frames, _ = extraction.source_frame_indices(np.asarray([centers.min()]), 25., 113)
    first_end = int(first_frames.max())

    class Capture:
        def __init__(self, path):
            self.inner = real_capture(path)
            self.position = 0
        def grab(self):
            if self.position > first_end:
                assert started.wait(2), "encoder worker never began"
                overlapped.set()
            self.position += 1
            return self.inner.grab()
        def __getattr__(self, name):
            return getattr(self.inner, name)

    class OverlapEncoder(PixelRecorder):
        def __init__(self):
            super().__init__()
            self.active = 0
            self.peak_active = 0
        def __call__(self, values, profile=None):
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
            try:
                started.set()
                assert overlapped.wait(2), "decode did not overlap inference"
                return super().__call__(values, profile)
            finally:
                self.active -= 1

    monkeypatch.setattr(extraction.cv2, "VideoCapture", Capture)
    encoder = OverlapEncoder()
    extract_video_features(video, shard, out, encoder, encoder_info(), batch_size=1, prefetch_batches=2)
    assert encoder.hashes == reference.hashes and encoder.peak_active == 1
    with np.load(out, allow_pickle=False) as actual, np.load(sequential, allow_pickle=False) as expected:
        for name in actual.files:
            if name != "metadata":
                np.testing.assert_array_equal(actual[name], expected[name])
        metadata = json.loads(str(actual["metadata"]))
        assert metadata["prefetch_batches"] == 2 and metadata["encoder_peak_inflight_batches"] <= 2
    assert not any(t.name.startswith("rgb-inference") for t in threading.enumerate())


def test_worker_failure_cannot_publish_a_partial_video_or_leave_threads(tiny_video):
    video, shard, out, _ = tiny_video
    def failing(values):
        raise RuntimeError("worker failed")
    with pytest.raises(RuntimeError, match="worker failed"):
        extract_video_features(video, shard, out, failing, encoder_info(), batch_size=1, prefetch_batches=2)
    assert not out.exists() and not out.with_suffix(".npz.partial").exists()
    assert not any(t.name.startswith("rgb-inference") for t in threading.enumerate())


def test_deadline_during_worker_execution_does_not_publish(tiny_video):
    video, shard, out, _ = tiny_video
    def slow(values):
        time.sleep(.3)
        return np.zeros((len(values), 768), np.float32)
    with pytest.raises(TimeoutError):
        extract_video_features(video, shard, out, slow, encoder_info(), batch_size=1,
            prefetch_batches=2, deadline=time.monotonic() + .1)
    assert not out.exists()
    assert not any(t.name.startswith("rgb-inference") for t in threading.enumerate())


@pytest.mark.parametrize("size", [-1, 1, 3, True])
def test_prefetch_queue_must_keep_its_fixed_bound(tmp_path, size):
    with pytest.raises(ValueError, match="prefetch_batches"):
        extract_video_features(tmp_path / "missing.mp4", tmp_path / "missing.npz",
            tmp_path / "out.npz", PixelRecorder(), encoder_info(), prefetch_batches=size)


if __name__ == "__main__":
    class RunnerSummary:
        def pytest_terminal_summary(self, terminalreporter):
            stats = terminalreporter.stats
            passed = len(stats.get("passed", []))
            total = passed + len(stats.get("failed", [])) + len(stats.get("error", []))
            print(f"{passed}/{total} passed; {len(stats.get('skipped', []))} skipped")

    raise SystemExit(pytest.main([__file__, "-q"], plugins=[RunnerSummary()]))
