"""Frozen-head parity, time boundaries, and real HTTP admission for the RGB serving port."""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
from behaviorsense.models.rgb_hybrid import decode_hybrid, pose_control_features
from behaviorsense.service.rgb_hybrid_backend import ServingConfig, create_app
from behaviorsense.service.rgb_hybrid_video import build_windows, decision_segments, rgb_source_indices
from behaviorsense.service.clip_workers import ClipWorker


def poses(count=4):
    rng = np.random.default_rng(20261016)
    result = np.zeros((count, 30, 2, 17, 3), dtype=np.float16)
    result[..., 0] = rng.uniform(100, 200, result.shape[:-1])
    result[..., 1] = rng.uniform(80, 300, result.shape[:-1])
    result[..., 2] = rng.uniform(.1, 1, result.shape[:-1])
    result[0, :8, 0] = 0
    return result


def test_rgb_control_features_reproduce_the_frozen_training_recipe():
    from train_rgb_multilabel import pose_control_features as original
    raw = poses()
    context = np.random.default_rng(7).normal(size=(len(raw), 401)).astype(np.float32)
    np.testing.assert_array_equal(pose_control_features(raw, context), original(raw, context))


def test_pair_rule_respects_track_boundaries_gaps_and_strict_cutoffs():
    a = np.full((5, 22), .1, dtype=np.float32)
    b = a.copy()
    rgb = a.copy()
    a[:, 13] = b[:, 13] = .9
    rgb[0, [10, 13]] = .9
    rgb[4, [10, 13]] = .5
    starts = np.array([0, 30, 75, 0, 0], dtype=np.int64)
    groups = np.array(["A", "A", "A", "B", "C"])
    cutoffs = np.full(22, .5)
    result = decode_hybrid(a, b, rgb, cutoffs, cutoffs, starts, groups)
    np.testing.assert_array_equal(result["decisions"][:, [10, 13]].all(1), [True, True, False, False, False])
    assert result["pair_witness"][1] == 0
    assert result["pair_witness"][3] == 3
    assert result["pair_confidence"][4] == 1.


def test_pair_collision_suppresses_drinking_on_an_exact_margin_tie():
    context = np.full((1, 22), .9, dtype=np.float32)
    rgb = np.full((1, 22), .1, dtype=np.float32)
    result = decode_hybrid(context, context, rgb, np.full(22, .5), np.full(22, .5),
                           np.array([0]), np.array(["one"]))
    assert result["decisions"][0, 13]
    assert not result["decisions"][0, 10]
    assert result["decisions"][0, [0, 1, 9, 14]].all()


def test_duplicate_window_identity_is_rejected():
    p = np.full((2, 22), .6, dtype=np.float32)
    with pytest.raises(ValueError, match="duplicate"):
        decode_hybrid(p, p, p, np.full(22, .5), np.full(22, .5), np.array([0, 0]), np.array(["A", "A"]))


def test_raw_windows_keep_full_precision_context_and_store_half_precision_poses():
    first = np.zeros((17, 3), dtype=np.float32)
    first[:, :2], first[:, 2] = [120.12345, 77.98765], .87654
    second = first.copy()
    second[:, 0] += 250
    decoded = {"raw": [{2: first.copy(), 8: second.copy()} for _ in range(60)]}
    result = build_windows(decoded)
    assert result["skeletons"].shape == (6, 30, 2, 17, 3)
    assert result["skeletons"].dtype == np.float16
    assert result["center_poses"].dtype == np.float32
    np.testing.assert_array_equal(result["center_poses"][0, 0], first)
    np.testing.assert_array_equal(result["skeletons"][0, 0, 0], first.astype(np.float16))
    np.testing.assert_array_equal(result["skeletons"][1, 0, 0], second.astype(np.float16))
    assert result["groups"][0] != result["groups"][1]


def test_short_and_low_visibility_clips_do_not_create_padding_windows():
    assert len(build_windows({"raw": [{} for _ in range(29)]})["starts"]) == 0
    person = np.ones((17, 3), np.float32)
    person[:, 2] = .29
    assert len(build_windows({"raw": [{1: person} for _ in range(60)]})["starts"]) == 0


def test_rgb_clock_matches_the_frozen_sampler_including_edges():
    from extract_rgb_action_features import source_frame_indices
    centers = np.array([0, 18, 100, 499], dtype=np.int64)
    expected, _ = source_frame_indices(centers, 29.97, 500)
    np.testing.assert_array_equal(rgb_source_indices(centers, 29.97, 500), expected)


def test_multilabel_segments_keep_temporal_gaps_and_do_not_invent_pair_overlap():
    windows = {"starts": np.array([0, 15, 60, 0]), "track_ids": np.array([1, 1, 1, 2])}
    decisions = np.zeros((4, 22), bool)
    decisions[0, 10], decisions[1, 13], decisions[2, 10], decisions[3, 10] = True, True, True, True
    predictions = {"decisions": decisions, "scores": np.full((4, 22), .8)}
    segments = decision_segments(windows, predictions, tuple(f"class{i}" for i in range(22)), {})
    first = [item for item in segments if item.track_id == 1]
    assert len(first) == 3
    assert first[0].end_time == first[1].start_time
    assert first[2].start_time > first[1].end_time
    assert len([item for item in segments if item.track_id == 2]) == 1


@pytest.fixture(scope="module")
def frozen_model():
    from behaviorsense.models.rgb_hybrid import FrozenRGBHybrid
    bundle = ROOT / "artifacts/rgb-serving-20261016/bundle"
    if not (bundle / "hybrid_manifest.json").is_file():
        pytest.skip("Private local model bundle is needed for frozen checkpoint parity.")
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield FrozenRGBHybrid(bundle, device="cpu")
    finally:
        torch.set_num_threads(previous)


def test_all_three_real_heads_match_existing_inference_implementations(frozen_model):
    import train_adl_multilabel as old
    import train_multilabel_temporal as temporal
    import train_rgb_multilabel as rgb
    raw = poses()
    context = np.random.default_rng(9).normal(size=(4, 401)).astype(np.float32)
    context[:, 0] = 1
    features = np.random.default_rng(8).normal(size=(4, 2, 768)).astype(np.float16)
    starts = np.array([0, 15, 45, 0], dtype=np.int64)
    groups = np.array(["A", "A", "A", "B"])
    data = {"skeletons": raw, "visual_context": context, "targets": np.zeros((4, 22)),
            "target_mask": np.ones((4, 22)), "starts": starts, "video_ids": groups, "subjects": groups}
    predictions = frozen_model.component_logits(raw, context, features, starts, groups, batch_size=4)
    context_data = old.MultilabelWindows(data, np.arange(4))
    temporal_data = temporal.TemporalWindows(data, np.arange(4), context_radius=6)
    with torch.inference_mode():
        expected_context = old.predict(frozen_model.context,
            torch.utils.data.DataLoader(context_data, batch_size=4), "cpu")
        expected_temporal = temporal.predict(frozen_model.temporal,
            torch.utils.data.DataLoader(temporal_data, batch_size=4), "cpu")
        combined = np.concatenate((rgb.pose_control_features(raw, context), features.reshape(4, -1)), axis=1)
        standardized = rgb.standardized_tensor(combined, frozen_model.mean, frozen_model.std, "cpu")
        expected_rgb = frozen_model.rgb(standardized).numpy()
    np.testing.assert_allclose(predictions["context"], expected_context, rtol=0, atol=1e-6)
    np.testing.assert_allclose(predictions["temporal"], expected_temporal, rtol=0, atol=1e-6)
    np.testing.assert_allclose(predictions["rgb"], expected_rgb, rtol=0, atol=1e-6)


def test_real_heads_support_empty_window_sets(frozen_model):
    result = frozen_model.predict(np.empty((0, 30, 2, 17, 3), np.float16), np.empty((0, 401), np.float32),
        np.empty((0, 2, 768), np.float16), np.empty(0, np.int64), np.empty(0, dtype=str))
    assert result["decisions"].shape == (0, 22)


def wait_until(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("Worker did not finish cleanup.")
        time.sleep(.01)


def test_real_upload_routes_run_two_distinct_workers_and_clean_files(tmp_path):
    barrier = threading.Barrier(2)
    observed, guard = [], threading.Lock()

    def process(config, worker, video, metadata, directory):
        value = Path(video).read_text()
        with guard:
            observed.append((worker.device, video, directory, metadata))
        barrier.wait(timeout=5)
        assert Path(video).is_file()
        yield json.dumps({"event": "result", "payload": {"source": value, "device": worker.device}}) + "\n"

    workers = [ClipWorker(f"cuda:{i}", f"cuda:{i}", None) for i in range(2)]
    config = ServingConfig(tmp_path, tmp_path, tmp_path, max_upload_mb=1)
    app = create_app(config, workers=workers, process=process, validate=False, upload_directory=tmp_path)
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["max_parallel_clips"] == 2
        assert health["upload_protocol"] == "chunks-v1"
        assert health["agent4"] is None
        uploads = []
        for name in ("first", "second"):
            created = client.post("/uploads", json={"filename": name + ".mp4", "size": len(name), "enrol": "Ada"}).json()
            url = "/uploads/" + created["upload_id"]
            headers = {"X-BS-Upload-Token": created["upload_token"]}
            assert client.put(url + "?offset=0", headers=headers, content=name.encode()).status_code == 200
            uploads.append((url, headers))

        def request(value):
            url, headers = value
            response = client.post(url + "/complete?stages=12", headers=headers)
            assert response.status_code == 200
            return [json.loads(line) for line in response.text.splitlines()]

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(request, uploads))
        assert {next(e["payload"]["device"] for e in result if e["event"] == "result")
                for result in results} == {"cuda:0", "cuda:1"}
        wait_until(lambda: not app.state.upload_store.uploads)
        assert all(not Path(video).exists() and not Path(directory).exists() for _, video, directory, _ in observed)
        assert all(meta["enrol"] == "Ada" for _, _, _, meta in observed)


def test_notebook_has_no_qwen_loading_and_checks_both_gpu_forwards():
    from package_rgb_hybrid_serving import notebook_document
    notebook = notebook_document()
    sources = [cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for source in sources:
        ast.parse(source)
        assert "qwen" not in source.lower()
        assert "transformers" not in source.lower()
        assert "kaggle_secrets" not in source
    assert any("range(2)" in source and "smoke_passed" in source for source in sources)
    assert any("source_manifest" in source for source in sources)


def test_complete_video_pipeline_preserves_two_people_and_bridge_contract(tmp_path):
    import cv2
    from behaviorsense.agents.perception import RawDetection
    from behaviorsense.schemas import BoundingBox, Keypoints
    from behaviorsense.data.toyota import COARSE_V11
    from behaviorsense.models.rgb_hybrid import MODEL_ID
    from behaviorsense.service.rgb_hybrid_video import process_video
    from web.local_backend import rebuild_segments

    path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30., (320, 240))
    assert writer.isOpened()
    for index in range(90):
        writer.write(np.full((240, 320, 3), index, dtype=np.uint8))
    writer.release()

    class Detector:
        providers = ["test"]

        def detect(self, frame):
            detections = []
            for offset in (30, 180):
                points = [(offset + (j % 3) * 10, 30 + (j // 3) * 22) for j in range(17)]
                detections.append(RawDetection(BoundingBox(x1=offset, y1=20, x2=offset+60, y2=190),
                    .9, Keypoints(xy=points, scores=[.9] * 17)))
            return detections

    class Objects:
        calls = 0

        def detect(self, frame):
            self.calls += 1
            return np.array([[10, 10, 50, 50, .8, 62]], np.float32)

    class Model:
        class_names = COARSE_V11
        manifest = {"model_id": MODEL_ID}

        def predict(self, skeletons, context, rgb_features, starts, groups):
            assert rgb_features.dtype == np.float16
            assert np.all(context[:, 0] == 1)
            scores = np.full((len(starts), 22), .1)
            scores[:, [10, 13]] = .9
            return {"scores": scores, "decisions": scores > .5}

    objects = Objects()
    encoder_calls = []

    def encoder(views):
        encoder_calls.append(len(views))
        assert views.shape[1:] == (32, 224, 224, 3)
        return np.full((len(views), 768), float(views.mean() / 255), np.float16)

    payload = process_video(path, Model(), encoder, Detector(), objects)
    assert payload["frames_kept"] == 60
    assert payload["n_people"] == 2
    assert objects.calls == 3
    assert encoder_calls == [2, 1, 2, 1, 2, 1]
    assert payload["stages_run"] == "12"
    assert payload["model"]["runtime_benchmark_verified"] is False
    assert payload["observed_hours"] == 3 / 3600
    assert len(rebuild_segments(payload["tracks"])) == 4
    for track in payload["tracks"]:
        assert {segment["label"] for segment in track["segments"]} == {10, 13}
        assert track["segments"][0]["t0"] == track["segments"][1]["t0"]


def test_child_cancellation_terminates_only_its_assigned_process_and_skips_empty_gallery(tmp_path, monkeypatch):
    import os
    import behaviorsense.service.rgb_hybrid_backend as backend
    observed = {}
    clock = iter(range(0, 1000, 6))

    class Process:
        returncode = None

        def __init__(self, command, **kwargs):
            observed.update(command=command, environment=kwargs["env"], terminated=False)

        def poll(self):
            return self.returncode

        def terminate(self):
            observed["terminated"] = True
            self.returncode = -15

        def wait(self, **kwargs):
            return self.returncode

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7")
    monkeypatch.setattr(backend.subprocess, "Popen", Process)
    monkeypatch.setattr(backend.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(backend.time, "sleep", lambda value: None)
    config = ServingConfig(tmp_path, tmp_path, tmp_path)
    events = backend.isolated_events(config, ClipWorker("cuda:1", "cuda:1", None), tmp_path / "clip.mp4",
                                    {"gallery": '{"residents":{}}'}, tmp_path)
    assert json.loads(next(events))["event"] == "heartbeat"
    events.close()
    assert observed["terminated"]
    assert observed["environment"]["CUDA_VISIBLE_DEVICES"] == "7"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3,7"
    assert json.loads((tmp_path / "worker.json").read_text())["use_osnet"] is False


if __name__ == "__main__":
    class CountSummary:
        def pytest_terminal_summary(self, terminalreporter):
            stats = terminalreporter.stats
            passed = len(stats.get("passed", []))
            failed = len(stats.get("failed", [])) + len(stats.get("error", []))
            skipped = len(stats.get("skipped", []))
            terminalreporter.write_line(f"{passed}/{passed + failed} passed; {skipped} skipped")

    raise SystemExit(pytest.main([str(Path(__file__).resolve()), "-q", "--color=no",
                                 "-p", "no:cacheprovider"], plugins=[CountSummary()]))
