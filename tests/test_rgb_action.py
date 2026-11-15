"""Real tiny-video decoder checks with a fake encoder; no GPU/accuracy claims."""

from contextlib import nullcontext
import copy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import extract_rgb_action_features as extraction
from behaviorsense.models import rgb_action


def encoder_info():
    return {"arch": "swin3d_t", "weights_sha256": "a" * 64, "feature_dim": 768,
            "preprocessing": copy.deepcopy(rgb_action.PREPROCESSING),
            "preprocessing_sha256": rgb_action.PREPROCESSING_SHA256,
            "torch_version": "fixture", "torchvision_version": "fixture",
            "device": "fake", "precision": "fixture"}


class FakeEncoder:
    def __init__(self):
        self.calls, self.views = 0, 0

    def __call__(self, values):
        assert values.dtype == np.uint8
        assert values.shape[1:] == (32, 224, 224, 3)
        self.calls += 1
        self.views += len(values)
        # Red encodes the video time in the fixture. This tests row ordering,
        # not recognition quality, and avoids retaining large frame batches.
        summary = values[..., 0].mean(axis=(1, 2, 3), dtype=np.float64).astype(np.float32)
        return np.repeat(summary[:, None], 768, axis=1)


@pytest.fixture
def tiny_video(tmp_path):
    video = tmp_path / "P03T01C01.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25., (64, 48))
    assert writer.isOpened(), "test environment must support OpenCV MJPG encoding"
    for index in range(113):
        frame = np.empty((48, 64, 3), np.uint8)
        frame[:] = [11, 40, index]
        writer.write(frame)
    writer.release()
    starts = np.array([60, 0, 15], np.int64)  # Deliberately unsorted source rows.
    poses = np.zeros((3, 30, 2, 17, 3), np.float32)
    poses[:, :, 0, :, 0] = np.linspace(20, 44, 17)
    poses[:, :, 0, :, 1] = np.linspace(10, 38, 17)
    poses[:, :, 0, :, 2] = .9
    metadata = {"video_id": "P03T01C01", "subject": "p03", "source_fps": 25.,
        "source_frame_count": 113, "source_sha256": rgb_action.file_sha256(video),
        "retained_windows": len(starts), "retained_starts": starts.tolist()}
    payload = {"skeletons": poses, "starts": starts,
        "source_frames": np.rint((starts[:, None] + np.arange(30)) * 25 / 20).astype(np.int64),
        "source_fps": np.full(len(starts), 25., np.float64),
        "clips": np.full(len(starts), video.name), "subjects": np.full(len(starts), "p03"),
        "datasets": np.full(len(starts), "toyota_untrimmed"),
        "extraction_metadata": np.asarray(json.dumps(metadata))}
    shard = tmp_path / "source.npz"
    np.savez_compressed(shard, **payload)
    return video, shard, tmp_path / "P03T01C01.npz", payload


def test_exact_source_clock_rounding_and_boundary_duplicate_flags():
    centers = np.array([0, 19, 112], np.int64)
    actual, flags = extraction.source_frame_indices(centers, 25., 113)
    requested = np.rint(centers[:, None] + (np.arange(32) - 15.5) * 25 / 15).astype(np.int64)
    np.testing.assert_array_equal(actual, np.clip(requested, 0, 112))
    np.testing.assert_array_equal(flags, actual != requested)
    assert actual[0, 0] == 0 and flags[0, 0]
    assert actual[-1, -1] == 112 and flags[-1, -1]
    assert not flags[1, -1]


@pytest.mark.parametrize("centers,fps,count", [
    (np.array([1.]), 25., 113), (np.array([-1]), 25., 113),
    (np.array([113]), 25., 113), (np.array([1]), float("nan"), 113),
    (np.array([1]), 15., 113), (np.array([1]), 25., 0),
])
def test_source_clock_refuses_malformed_inputs(centers, fps, count):
    with pytest.raises(ValueError):
        extraction.source_frame_indices(centers, fps, count)


def test_scene_padding_preserves_content_on_all_four_edges():
    rgb = np.zeros((48, 96, 3), np.uint8)
    rgb[:, :5] = [255, 0, 0]
    rgb[:, -5:] = [0, 255, 0]
    rgb[:5, 20:76] = [0, 0, 255]
    rgb[-5:, 20:76] = [255, 255, 255]
    output = rgb_action.spatial_view(rgb, rgb_action.scene_roi(96, 48))
    assert output.shape == (224, 224, 3)
    assert output[112, :10, 0].max() > 220
    assert output[112, -10:, 1].max() > 220
    assert output[:90, 112, 2].max() > 220
    assert output[134:, 112].min(axis=1).max() > 220


@pytest.mark.parametrize("width,height", [(5, 4), (641, 479), (640, 480), (101, 103)])
def test_scene_padding_retains_the_entire_continuous_image_rectangle(width, height):
    left, top, right, bottom = rgb_action.scene_roi(width, height)
    removed_border = (right - left) * 16 / 256
    assert left + removed_border <= 0 and top + removed_border <= 0
    assert right - removed_border >= width and bottom - removed_border >= height


@pytest.mark.parametrize("width,height", [(0, 0), (1, 1), (5, 4), (101, 103), (640, 480), (641, 479)])
@pytest.mark.parametrize("layout", ["contiguous", "spatial_stride", "reversed_rows", "reversed_channels"])
def test_native_padding_is_byte_identical_to_frozen_spatial_sampling(width, height, layout):
    rng = np.random.default_rng(20481013)
    rgb = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    if layout == "spatial_stride":
        rgb = rng.integers(0, 256, (height * 2, width * 2, 3), dtype=np.uint8)[::2, ::2]
    elif layout == "reversed_rows":
        rgb = rgb[::-1]
    elif layout == "reversed_channels":
        rgb = rgb[..., ::-1]
    original = rgb.copy()
    rgb.setflags(write=False)
    sides = {1, 2, 31, 32, 33, 224, 256, 257, max(1, min(width, height)), max(width, height) + 1,
             2 * max(width, height) + 32}
    rois = []
    for side in sorted(side for side in sides if side <= 2 * max(width, height) + 32):
        # Exact edge contact, one-pixel intersection, wholly outside, and padding
        # on each side exercise all crop geometries, including odd coordinates.
        positions = [(0, 0), ((width-side)//2, (height-side)//2), (-side, -side),
            (width, height), (1-side, 1-side), (width-1, height-1), (-side//2, 0),
            (0, -side//2), (width-side//2, 0), (0, height-side//2)]
        rois.extend((left, top, left+side, top+side) for left, top in positions)
    if width and height:
        rois.append(rgb_action.scene_roi(width, height))
    for roi in rois:
        actual = rgb_action.spatial_view(rgb, np.asarray(roi, np.int64))
        expected = rgb_action.spatial_view_reference(rgb, np.asarray(roi, np.int64))
        np.testing.assert_array_equal(actual, expected, err_msg=f"ROI={tuple(roi)}, layout={layout}")
        assert actual.shape == (224, 224, 3) and actual.dtype == np.uint8
        assert actual.flags.c_contiguous and not np.shares_memory(actual, rgb)
    np.testing.assert_array_equal(rgb, original)


def test_person_roi_uses_only_primary_slot_and_never_mutates_poses():
    pose = np.zeros((30, 2, 17, 3), np.float32)
    pose[:, 0, :2] = [[-10., 2., .3], [63., 47., .9]]
    pose[:, 1] = [10000., 10000., 1.]
    original = pose.copy()
    pose.setflags(write=False)
    roi, observed = rgb_action.person_roi(pose, 64, 48)
    assert observed and roi[0] < 0 and roi[1] < 0
    assert roi[2] - roi[0] == roi[3] - roi[1] == 96
    np.testing.assert_array_equal(pose, original)
    missing = original.copy()
    missing[:, 0, :, 2] = 0
    fallback, observed = rgb_action.person_roi(missing, 64, 48)
    assert not observed
    np.testing.assert_array_equal(fallback, rgb_action.scene_roi(64, 48))


@pytest.mark.parametrize("confidence", [-.01, 1.01])
def test_person_roi_refuses_invalid_confidence_before_selecting_a_crop(confidence):
    pose = np.zeros((30, 2, 17, 3), np.float32)
    pose[0, 0, 0, 2] = confidence
    with pytest.raises(ValueError, match="confidence"):
        rgb_action.person_roi(pose, 64, 48)


def test_sequential_decoder_preserves_rows_and_retrieves_each_needed_frame_once(tiny_video, monkeypatch):
    video, shard, out, payload = tiny_video
    original_sha = rgb_action.file_sha256(shard)
    real_capture = cv2.VideoCapture
    captures = []

    class SpyCapture:
        def __init__(self, path):
            self.inner = real_capture(path)
            self.grabs = self.retrievals = 0
            captures.append(self)

        def grab(self):
            self.grabs += 1
            return self.inner.grab()

        def retrieve(self):
            self.retrievals += 1
            return self.inner.retrieve()

        def set(self, *args):
            raise AssertionError("per-window seeking is forbidden")

        def __getattr__(self, name):
            return getattr(self.inner, name)

    monkeypatch.setattr(extraction.cv2, "VideoCapture", SpyCapture)
    fake = FakeEncoder()
    report = extraction.extract_video_features(video, shard, out, fake, encoder_info(), batch_size=2)
    assert report["status"] == "complete" and not report["reused"]
    assert report["n_windows"] == 3 and fake.views == 6 and fake.calls == 2
    assert len(captures) == 1
    with np.load(out, allow_pickle=False) as result:
        np.testing.assert_array_equal(result["starts"], payload["starts"])
        np.testing.assert_array_equal(result["center_source_frames"], payload["source_frames"][:, 15])
        expected, flags = extraction.source_frame_indices(payload["source_frames"][:, 15], 25., 113)
        np.testing.assert_array_equal(result["source_frames_rgb"], expected)
        np.testing.assert_array_equal(result["boundary_duplicate_flags"], flags)
        assert result["features"].shape == (3, 2, 768) and result["features"].dtype == np.float16
        assert result["features"][0, 0, 0] > result["features"][2, 0, 0] > result["features"][1, 0, 0]
        assert str(result["video_id"]) == "P03T01C01" and str(result["subject"]) == "p03"
        assert report["retrieved_frames"] == len(np.unique(expected)) == captures[0].retrievals
        assert report["decoded_frames"] == captures[0].grabs == int(expected.max()) + 1
        assert report["peak_cached_frames"] <= 55  # A 2.07-second source span, not a whole video.
    assert rgb_action.file_sha256(shard) == original_sha
    assert not out.with_suffix(".npz.partial").exists()


def test_missing_person_copies_scene_embedding_without_duplicate_encoding(tiny_video):
    video, shard, out, payload = tiny_video
    payload["skeletons"][:, :, 0, :, 2] = 0
    np.savez_compressed(shard, **payload)
    fake = FakeEncoder()
    extraction.extract_video_features(video, shard, out, fake, encoder_info(), batch_size=2)
    assert fake.views == 3
    with np.load(out, allow_pickle=False) as result:
        assert not result["roi_observed"].any()
        np.testing.assert_array_equal(result["features"][:, 0], result["features"][:, 1])


def test_complete_cache_is_verified_and_does_not_call_encoder_again(tiny_video):
    video, shard, out, _ = tiny_video
    fake = FakeEncoder()
    info = encoder_info()
    first = extraction.extract_video_features(video, shard, out, fake, info)
    calls = fake.calls
    second = extraction.extract_video_features(video, shard, out, fake, info, batch_size=1)
    assert second["reused"] and second["output_sha256"] == first["output_sha256"]
    assert fake.calls == calls
    changed = {**info, "weights_sha256": "b" * 64}
    with pytest.raises(ValueError, match="cache identity differs"):
        extraction.extract_video_features(video, shard, out, fake, changed)


def test_cache_finite_feature_corruption_is_detected(tiny_video):
    video, shard, out, _ = tiny_video
    fake = FakeEncoder()
    extraction.extract_video_features(video, shard, out, fake, encoder_info())
    with np.load(out, allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    payload["features"][0, 0, 0] += 1
    np.savez_compressed(out, **payload)
    with pytest.raises(ValueError, match="checksum mismatch"):
        extraction.extract_video_features(video, shard, out, fake, encoder_info())


@pytest.mark.parametrize("mutation,message", [
    (lambda p: p.update(skeletons=p["skeletons"][:, :29]), "skeletons"),
    (lambda p: p["skeletons"].__setitem__((0, 0, 0, 0, 2), -1.), "confidence"),
    (lambda p: p["skeletons"].__setitem__((0, 0, 0, 0, 2), 1.1), "confidence"),
    (lambda p: p.update(source_fps=np.full(3, 24.)), "FPS"),
    (lambda p: p["source_frames"].__setitem__((0, 15), 2), "sampling clock"),
    (lambda p: p["starts"].__setitem__(0, 0), "unique"),
    (lambda p: p["subjects"].__setitem__(0, "p04"), "identity"),
])
def test_invalid_source_shard_fails_before_encoding(tiny_video, mutation, message):
    video, shard, out, payload = tiny_video
    mutation(payload)
    np.savez_compressed(shard, **payload)
    fake = FakeEncoder()
    with pytest.raises(ValueError, match=message):
        extraction.extract_video_features(video, shard, out, fake, encoder_info())
    assert fake.calls == 0 and not out.exists()


def test_source_video_hash_and_decoded_clock_are_both_verified(tiny_video):
    video, shard, out, payload = tiny_video
    metadata = json.loads(str(payload["extraction_metadata"]))
    metadata["source_sha256"] = "0" * 64
    payload["extraction_metadata"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(shard, **payload)
    with pytest.raises(ValueError, match="raw video SHA-256"):
        extraction.extract_video_features(video, shard, out, FakeEncoder(), encoder_info())
    metadata["source_sha256"] = rgb_action.file_sha256(video)
    metadata["source_fps"] = 24.
    payload["source_fps"][:] = 24.
    payload["source_frames"] = np.rint((payload["starts"][:, None] + np.arange(30)) * 24 / 20).astype(np.int64)
    payload["extraction_metadata"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(shard, **payload)
    with pytest.raises(ValueError, match="decoded video FPS"):
        extraction.extract_video_features(video, shard, out, FakeEncoder(), encoder_info())


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), 100000.])
def test_nonfinite_or_unstorable_encoder_output_never_exports(tiny_video, bad_value):
    video, shard, out, _ = tiny_video
    def broken(values):
        return np.full((len(values), 768), bad_value, np.float32)
    with pytest.raises(RuntimeError, match="features"):
        extraction.extract_video_features(video, shard, out, broken, encoder_info())
    assert not out.exists()


def test_deadline_does_not_publish_partial_video(tiny_video):
    video, shard, out, _ = tiny_video
    with pytest.raises(TimeoutError, match="deadline"):
        extraction.extract_video_features(video, shard, out, FakeEncoder(), encoder_info(),
                                           deadline=time.monotonic() - 1)
    assert not out.exists()
    def interrupted(values):
        raise TimeoutError("fixture encoder exceeded its deadline")
    with pytest.raises(TimeoutError, match="deadline"):
        extraction.extract_video_features(video, shard, out, interrupted, encoder_info())
    assert not out.exists() and not out.with_suffix(".npz.partial").exists()


@pytest.mark.parametrize("phase", ["encoder", "serialize"])
def test_deadline_crossed_inside_a_blocking_operation_never_publishes(tiny_video, monkeypatch, phase):
    video, shard, out, _ = tiny_video
    clock = [1.]
    monkeypatch.setattr(extraction.time, "monotonic", lambda: clock[0])
    fake = FakeEncoder()

    def encode(values):
        result = fake(values)
        if phase == "encoder":
            clock[0] = 3.
        return result

    save = extraction.np.savez_compressed

    def serialize(*args, **kwargs):
        save(*args, **kwargs)
        if phase == "serialize":
            clock[0] = 3.

    monkeypatch.setattr(extraction.np, "savez_compressed", serialize)
    with pytest.raises(TimeoutError, match="deadline"):
        extraction.extract_video_features(video, shard, out, encode, encoder_info(), deadline=2.)
    assert fake.calls > 0
    assert not out.exists() and not out.with_suffix(".npz.partial").exists()


def test_encoder_tensor_layout_normalization_and_frozen_inference_on_cpu(monkeypatch):
    import torch

    class InspectInput(torch.nn.Module):
        def forward(self, values):
            assert not self.training and not torch.is_grad_enabled()
            assert values.shape == (1, 3, 32, 224, 224)
            assert values.dtype == torch.float32
            return values[:, :, 16, 100, 100]

    # The production loader still requires CUDA. This isolates the actual tensor
    # conversion with an analytic channel reference, without a GPU accuracy claim.
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    model = InspectInput()
    encoder = rgb_action.RGBActionEncoder(model, "cpu", 3)
    model.train()
    views = np.empty((1, 32, 224, 224, 3), np.uint8)
    views[:] = [0, 128, 255]
    actual = encoder(views)
    expected = np.array([[-.485 / .229, (128 / 255 - .456) / .224, (1 - .406) / .225]])
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_zero_retained_rows_export_complete_empty_features_without_encoding(tiny_video):
    video, shard, out, payload = tiny_video
    for key in tuple(payload):
        if key != "extraction_metadata":
            payload[key] = payload[key][:0]
    metadata = json.loads(str(payload["extraction_metadata"]))
    metadata.update(retained_windows=0, retained_starts=[])
    payload["extraction_metadata"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(shard, **payload)
    fake = FakeEncoder()
    report = extraction.extract_video_features(video, shard, out, fake, encoder_info())
    assert report["status"] == "complete" and report["n_windows"] == 0 and fake.calls == 0
    with np.load(out) as z:
        assert z["features"].shape == (0, 2, 768)


def test_checkpoint_hash_checked_before_loading_any_framework(tmp_path):
    path = tmp_path / "weights.pth"
    path.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        rgb_action.load_rgb_encoder(path, "0" * 64, "swin3d_t")
    with pytest.raises(ValueError, match="unsupported"):
        rgb_action.load_rgb_encoder(path, "0" * 64, "unknown")


def test_encoder_loads_only_staged_weights_strictly_without_download(tmp_path, monkeypatch):
    path = tmp_path / "weights.pth"
    path.write_bytes(b"synthetic weight bytes")
    calls = {}
    class Device:
        type = "cuda"
        def __init__(self, value):
            pass
        def __str__(self):
            return "cuda:0"
    class Model:
        head = SimpleNamespace(in_features=768)
        def load_state_dict(self, state, strict):
            calls["strict"] = strict
            assert state == {"fixture": True}
        def requires_grad_(self, value):
            calls["grad"] = value
            return self
        def to(self, device):
            return self
        def eval(self):
            calls["eval"] = True
            return self
    def constructor(**kwargs):
        calls["constructor"] = kwargs
        return Model()
    def load(weights, **kwargs):
        calls["load"] = kwargs
        assert Path(weights) == path
        return {"fixture": True}
    torch_stub = SimpleNamespace(__version__="2.11.0+cu128", device=Device, load=load,
        cuda=SimpleNamespace(is_available=lambda: True, device=lambda x: nullcontext(),
                             is_bf16_supported=lambda: True), nn=SimpleNamespace(Identity=lambda: object()))
    vision_stub = SimpleNamespace(__version__="0.26.0+cu128",
        models=SimpleNamespace(video=SimpleNamespace(swin3d_t=constructor)))
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "torchvision", vision_stub)
    _, info = rgb_action.load_rgb_encoder(path, rgb_action.file_sha256(path), "swin3d_t")
    assert calls["constructor"] == {"weights": None}
    assert calls["load"] == {"map_location": "cpu", "weights_only": True}
    assert calls["strict"] and calls["eval"] and calls["grad"] is False
    assert info["feature_dim"] == 768 and info["precision"] == "bfloat16"
    vision_stub.__version__ = "0.25.0"
    with pytest.raises(RuntimeError, match="pinned Torch"):
        rgb_action.load_rgb_encoder(path, rgb_action.file_sha256(path), "swin3d_t")


if __name__ == "__main__":
    class RunnerSummary:
        def pytest_terminal_summary(self, terminalreporter):
            stats = terminalreporter.stats
            passed = len(stats.get("passed", []))
            total = passed + len(stats.get("failed", [])) + len(stats.get("error", []))
            print(f"{passed}/{total} passed; {len(stats.get('skipped', []))} skipped")

    raise SystemExit(pytest.main([__file__, "-q"], plugins=[RunnerSummary()]))
