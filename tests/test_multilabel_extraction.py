"""CPU checks for actual-frame extraction, real annotation strata, and resumable shards.

Run: python tests/test_multilabel_extraction.py
The decoder and detector doubles exercise indexing only; no synthetic data is published.
"""

from __future__ import annotations

from dataclasses import replace
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from extract_multilabel_windows import (  # noqa: E402
    WindowSelection, build_parser, completion_status, decode_selected, load_detector,
    main, partition_videos, require_cuda_session, shard_fingerprint, inventory_sha256, file_sha256,
    resume_report, save_video_shard, select_windows, source_timeline,
    subset_videos, validate_pose,
)
from behaviorsense.data.toyota import COARSE_V11, TSU_CLASSES, TsuVideo  # noqa: E402
from behaviorsense.data.multilabel_skeleton import targets_for_windows  # noqa: E402


def video(actions=None):
    if actions is None:
        actions = ((26, 50, 300), (13, 175, 220), (29, 182, 205),
                   (19, 400, 600), (13, 650, 750), (26, 800, 950))
    return TsuVideo("P03T01C01", 3, 1, 1, "training", 1000, tuple(actions))


def selection(starts):
    starts = np.asarray(starts, np.int64)
    frames = np.rint((starts[:, None] + np.arange(30)) * 25 / 20).astype(np.int64)
    return WindowSelection(starts, frames, np.full(len(starts), "other_activity", "<U20"),
                           np.ones((len(starts), 4), np.float32), np.ones(len(starts), np.int16),
                           {"other_activity": len(starts)}, len(starts))


def pilot_windows(*args, **kwargs):
    kwargs.setdefault("selection", "stratified")
    kwargs.setdefault("candidate_stride", 5)
    return select_windows(*args, **kwargs)


CV2 = types.SimpleNamespace(CAP_PROP_POS_FRAMES=1)


class Capture:
    """An indexed decoder with distinguishable image contents at every source frame."""
    def __init__(self, *, count=150, failed_seek=False):
        self.count, self.failed_seek = count, failed_seek
        self.position, self.released = 0, False

    def isOpened(self):
        return True

    def set(self, key, position):
        if self.failed_seek:
            self.position = 7
            return False
        self.position = int(position)
        return True

    def get(self, key):
        return self.position

    def grab(self):
        if self.position >= self.count:
            return False
        self.position += 1
        return True

    def retrieve(self):
        return True, np.full((2, 2, 3), self.position - 1, np.int64)

    def release(self):
        self.released = True


class Detector:
    def __init__(self, invisible_until=-1):
        self.calls = []
        self.invisible_until = invisible_until

    def __call__(self, frame):
        index = int(frame[0, 0, 0])
        self.calls.append(index)
        kpts = np.stack((np.arange(17) + index, np.arange(17) + 100), axis=-1)[None]
        scores = np.ones((1, 17)) if index > self.invisible_until else np.zeros((1, 17))
        return kpts, scores


class ExtractionTests(unittest.TestCase):
    def test_exact_frame_clock_and_no_upsampling(self):
        frames = source_timeline(1000, 25.0, 1000)
        expected = [round(k * 25 / 20) for k in range(800)]
        np.testing.assert_array_equal(frames, expected)
        self.assertEqual(frames[2], 2)  # Python round ties to even, not integer stride.
        self.assertTrue(np.all(np.diff(frames) > 0))
        for fps in (0, 19.9, float("nan"), 241):
            with self.assertRaises(ValueError):
                source_timeline(1000, fps, 1000)

    def test_strata_preserve_original_intersections(self):
        v = video()
        chosen = pilot_windows(v, 1000, 25.0, budget=40, seed=14)
        self.assertEqual(len(chosen.starts), 40)
        self.assertEqual(set(chosen.strata),
                         {"tv_drinking", "tv_only", "drinking_only", "other_activity", "context"})
        targets, intersections = targets_for_windows(v, chosen.starts,
                                                     source_frames=chosen.source_frames, source_fps=25.0)
        tv, drink = COARSE_V11.index("watching_tv"), COARSE_V11.index("drinking")
        np.testing.assert_array_equal(targets[:, tv], chosen.coverage[:, 0] >= 0.5)
        np.testing.assert_array_equal(targets[:, drink], chosen.coverage[:, 1] >= 0.5)
        np.testing.assert_array_equal(intersections, np.rint(chosen.coverage[:, 2] * 30))
        self.assertTrue(np.all(chosen.coverage[:, 3] >= 0.5))
        for tag, row, overlap in zip(chosen.strata, targets, intersections):
            if tag == "tv_drinking":
                self.assertEqual((row[tv], row[drink]), (1, 1))
                self.assertGreater(overlap, 0)
            elif tag == "tv_only":
                self.assertEqual((row[tv], row[drink]), (1, 0))
            elif tag == "drinking_only":
                self.assertEqual((row[tv], row[drink]), (0, 1))

    def test_no_tv_is_invented_through_drinking_or_gaps(self):
        v = video(((26, 0, 80), (13, 180, 230), (19, 600, 720)))
        chosen = pilot_windows(v, 1000, 25.0, budget=200, candidate_stride=1)
        self.assertEqual(chosen.candidate_counts["tv_drinking"], 0)
        for frames, coverage in zip(chosen.source_frames, chosen.coverage):
            np.testing.assert_allclose(coverage[0], np.mean((frames >= 0) & (frames <= 80)), atol=1e-7)
        empty = pilot_windows(video(()), 1000, 25.0)
        self.assertEqual(len(empty.starts), 0)
        self.assertEqual(empty.source_frames.shape, (0, 30))

    def test_reproducible_and_budget_prefix_selection(self):
        one = pilot_windows(video(), 1000, 25.0, budget=32, seed=4)
        same = pilot_windows(video(), 1000, 25.0, budget=32, seed=4)
        larger = pilot_windows(video(), 1000, 25.0, budget=64, seed=4)
        other = pilot_windows(video(), 1000, 25.0, budget=32, seed=5)
        np.testing.assert_array_equal(one.starts, same.starts)
        self.assertTrue(set(one.starts) <= set(larger.starts))
        self.assertFalse(np.array_equal(one.starts, other.starts))

    def test_defaults_only_official_training_subjects(self):
        testing = replace(video(), vid="P02T01C01", subject=2, subset="testing")
        selected = subset_videos({"train": video(), "test": testing}, "training")
        self.assertEqual([v.subject for v in selected], [3])
        with self.assertRaisesRegex(ValueError, "official CS"):
            subset_videos({"bad": replace(testing, subset="training")}, "training")
        parsed = build_parser().parse_args(["--videos-root", ".", "--annotations", "ann.json",
                                           "--rtmo-model", "rtmo.onnx", "--out", "out"])
        self.assertEqual(parsed.subset, "training")
        self.assertEqual(parsed.selection, "full")
        self.assertEqual(parsed.candidate_stride, 15)
        self.assertEqual((parsed.partition_index, parsed.partition_count), (0, 1))

    def test_full_selection_keeps_every_annotation_eligible_window_without_budget(self):
        v = video()
        chosen = select_windows(v, 1000, 25.0, budget=1, seed=12)
        timeline = source_timeline(1000, 25, 1000)
        annotated = np.zeros(len(timeline), bool)
        for _, begin, end in v.actions:
            annotated |= (timeline >= begin) & (timeline <= end)
        expected = [start for start in range(0, len(timeline) - 30 + 1, 15)
                    if annotated[start:start + 30].mean() >= 0.5]
        np.testing.assert_array_equal(chosen.starts, expected)
        self.assertGreater(len(chosen.starts), 1)
        self.assertEqual(len(chosen.starts), sum(chosen.candidate_counts.values()))
        self.assertEqual(chosen.candidate_windows, len(range(0, len(timeline) - 30 + 1, 15)))
        # Relabel the exact same intervals, removing TV/drinking altogether: neither
        # activity names, the seed, nor a pilot budget may alter full coverage.
        renamed = replace(v, actions=tuple((19, begin, end) for _, begin, end in v.actions))
        other = select_windows(renamed, 1000, 25.0, budget=64, seed=999)
        np.testing.assert_array_equal(chosen.starts, other.starts)
        np.testing.assert_array_equal(chosen.source_frames, other.source_frames)

    def test_partitions_are_balanced_deterministic_disjoint_and_exhaustive(self):
        videos = [replace(video(), vid=f"P03T{i:02d}C01", take=i, duration=duration)
                  for i, duration in enumerate((1000, 900, 800, 700, 600, 500, 400), 1)]
        parts = partition_videos(videos, 3)
        reverse = partition_videos(list(reversed(videos)), 3)
        self.assertEqual([[v.vid for v in p] for p in parts], [[v.vid for v in p] for p in reverse])
        flattened = [v.vid for part in parts for v in part]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), {v.vid for v in videos})
        durations = [sum(v.duration for v in part) for part in parts]
        self.assertLessEqual(max(durations) - min(durations), max(v.duration for v in videos))
        for count in (0, len(videos) + 1):
            with self.assertRaises(ValueError):
                partition_videos(videos, count)

    def test_partition_or_pilot_completion_cannot_claim_full_subset(self):
        work = [video(), replace(video(), vid="P04T01C01", subject=4)]
        scope = work[:1]
        finished = [{"video_id": work[0].vid}]
        partial = completion_status(work, scope, finished, "full")
        self.assertTrue(partial["scope_complete"])
        self.assertFalse(partial["complete"])
        self.assertFalse(partial["subset_complete"])
        full_reports = [{"video_id": v.vid} for v in work]
        pilot = completion_status(work, work, full_reports, "stratified")
        self.assertTrue(pilot["scope_complete"])
        self.assertFalse(pilot["subset_complete"])
        self.assertTrue(completion_status(work, work, full_reports, "full")["subset_complete"])
        capped = completion_status(work, work, finished, "full")
        self.assertFalse(capped["scope_complete"])
        self.assertEqual(capped["missing_scope_video_ids"], ["P04T01C01"])
        with self.assertRaisesRegex(ValueError, "duplicate or out-of-scope"):
            completion_status(work, scope, full_reports, "full")

    def test_decoder_only_infers_required_unique_frames_and_keeps_real_starts(self):
        chosen = selection([0, 1, 50])
        detector, capture = Detector(invisible_until=40), Capture()
        with patch.dict(sys.modules, {"cv2": CV2}):
            windows, kept, stats, context = decode_selected(Path("sample.mp4"), chosen, detector,
                                                   capture_factory=lambda _: capture)
        np.testing.assert_array_equal(detector.calls, np.unique(chosen.source_frames))
        np.testing.assert_array_equal(kept, [2])
        np.testing.assert_array_equal(chosen.starts[kept], [50])
        np.testing.assert_array_equal(windows[0, :, 0, 0, 0], chosen.source_frames[2])
        self.assertEqual(windows.shape, (1, 30, 2, 17, 3))
        self.assertEqual(stats["unique_inference_frames"], len(set(detector.calls)))
        self.assertTrue(capture.released)
        self.assertIsNone(context)
        self.assertLessEqual(stats["peak_cached_pose_frames"], 30)

    def test_seek_and_failed_seek_preserve_frame_contents(self):
        chosen = selection([50])
        outputs = []
        for mode, failed_seek in (("sequential", False), ("seek", False), ("seek", True)):
            with patch.dict(sys.modules, {"cv2": CV2}):
                windows, _, stats, _ = decode_selected(Path("sample.mp4"), chosen, Detector(),
                                                      decode_mode=mode, seek_gap=20,
                                                      capture_factory=lambda _: Capture(failed_seek=failed_seek))
            outputs.append(windows)
            self.assertEqual(stats["seek_fallbacks"], int(failed_seek))
        for output in outputs[1:]:
            np.testing.assert_array_equal(output, outputs[0])

    def test_dense_streaming_never_caches_more_than_one_pose_window(self):
        chosen = selection(np.arange(500))
        detector = Detector()
        with patch.dict(sys.modules, {"cv2": CV2}):
            windows, kept, stats, _ = decode_selected(
                Path("sample.mp4"), chosen, detector, capture_factory=lambda _: Capture(count=1000))
        self.assertEqual(len(kept), 500)
        self.assertEqual(stats["peak_cached_pose_frames"], 30)
        self.assertEqual(stats["unique_inference_frames"], 529)
        np.testing.assert_array_equal(windows[:, :, 0, 0, 0], chosen.source_frames)

    def test_context_uses_exact_center_rgb_and_pixel_pose_after_visibility_filter(self):
        from behaviorsense.models.visual_context import CONTEXT_DIM, CONTEXT_SCHEMA, FEATURE_NAMES

        class ContextDetector:
            def __init__(self):
                self.calls = []

            def features(self, frame, poses):
                index = int(frame[0, 0, 0])
                self.calls.append((index, float(poses[0, 0, 0])))
                self.last_shape = poses.shape
                features = np.zeros(CONTEXT_DIM, np.float32)
                features[0] = 1
                features[1] = index / 1000
                return features

        chosen = selection([0, 1, 50])
        context_detector = ContextDetector()
        with patch.dict(sys.modules, {"cv2": CV2}):
            _, kept, stats, context = decode_selected(
                Path("sample.mp4"), chosen, Detector(invisible_until=40),
                capture_factory=lambda _: Capture(), context_detector=context_detector)
        np.testing.assert_array_equal(kept, [2])
        self.assertEqual(context_detector.last_shape, (2, 17, 3))
        expected_centers = chosen.source_frames[:, 15]
        np.testing.assert_array_equal([item[0] for item in context_detector.calls], expected_centers)
        # Invisible skeletons carry no geometry; the retained center's pose carries
        # the image's exact original-frame x position, not normalized coordinates.
        self.assertEqual(context_detector.calls[-1][1], float(expected_centers[-1]))
        self.assertEqual(stats["context_inference_frames"], len(chosen.starts))
        self.assertEqual(context.shape, (1, CONTEXT_DIM))
        np.testing.assert_allclose(context[:, 1], expected_centers[kept] / 1000)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "context.npz"
            save_video_shard(path, video=video(), source_path=Path("P03T01C01.mp4"), source_fps=25,
                             selection=chosen, skeletons=np.zeros((1, 30, 2, 17, 3), np.float16),
                             kept=kept, fingerprint="context", report={"retained_windows": 1},
                             visual_context=context, context_metadata={"schema": CONTEXT_SCHEMA,
                                 "feature_dim": CONTEXT_DIM, "feature_names": list(FEATURE_NAMES),
                                 "model_sha256": "test-only-model-hash"})
            self.assertTrue(resume_report(path, "context")["context_present"])
            with np.load(path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["context_source_frames"], expected_centers[kept, None])
                np.testing.assert_array_equal(data["visual_context"], context)
                changed = {key: data[key] for key in data.files}
            changed["context_source_frames"] = changed["context_source_frames"] + 1
            np.savez_compressed(path, **changed)
            with self.assertRaisesRegex(ValueError, "exact-frame visual context"):
                resume_report(path, "context")

    def test_decoder_rejects_inconsistent_overlap_instead_of_silent_frame_reuse(self):
        chosen = selection([0, 1])
        frames = chosen.source_frames.copy()
        frames[1] += 2
        with patch.dict(sys.modules, {"cv2": CV2}):
            with self.assertRaisesRegex(ValueError, "overlapping windows disagree"):
                decode_selected(Path("sample.mp4"), replace(chosen, source_frames=frames), Detector(),
                                capture_factory=lambda _: Capture())

    def test_nonfinite_context_fails_and_releases_decoder(self):
        from behaviorsense.models.visual_context import CONTEXT_DIM
        capture = Capture()
        bad_context = types.SimpleNamespace(features=lambda frame, pose: np.full(CONTEXT_DIM, np.nan))
        with patch.dict(sys.modules, {"cv2": CV2}):
            with self.assertRaisesRegex(ValueError, "non-finite features"):
                decode_selected(Path("sample.mp4"), selection([0]), Detector(),
                                capture_factory=lambda _: capture, context_detector=bad_context)
        self.assertTrue(capture.released)

    def test_shard_fingerprint_binds_source_bytes_config_and_video_identity(self):
        initial = shard_fingerprint("config", video(), "source-a")
        self.assertNotEqual(initial, shard_fingerprint("config", video(), "source-b"))
        self.assertNotEqual(initial, shard_fingerprint("changed", video(), "source-a"))
        self.assertNotEqual(initial, shard_fingerprint("config", replace(video(), vid="P04T01C01"), "source-a"))
        self.assertEqual(initial, shard_fingerprint("config", video(), "source-a"))

    def test_truncated_video_fails_before_writing_misaligned_window(self):
        with patch.dict(sys.modules, {"cv2": CV2}):
            with self.assertRaisesRegex(RuntimeError, "before required source frame"):
                decode_selected(Path("sample.mp4"), selection([50]), Detector(),
                                capture_factory=lambda _: Capture(count=70))

    def test_seek_post_read_mismatch_reopens_sequentially(self):
        class MisreportingCapture(Capture):
            def __init__(self):
                super().__init__()
                self.seek_position = None

            def set(self, key, position):
                self.seek_position = position
                return super().set(key, position)

            def get(self, key):
                if self.seek_position is not None and self.position > self.seek_position:
                    return self.position + 10
                return self.position

        captures = []
        def factory(_):
            capture = MisreportingCapture()
            captures.append(capture)
            return capture

        with patch.dict(sys.modules, {"cv2": CV2}):
            windows, _, stats, _ = decode_selected(Path("sample.mp4"), selection([50]), Detector(),
                                                decode_mode="seek", seek_gap=20,
                                                capture_factory=factory)
        self.assertEqual(stats["seek_fallbacks"], 1)
        self.assertEqual(len(captures), 2)
        np.testing.assert_array_equal(windows[0, :, 0, 0, 0], selection([50]).source_frames[0])

    def test_rejects_wrong_joint_layout_and_unverified_cuda(self):
        with self.assertRaisesRegex(ValueError, "COCO-17"):
            validate_pose(np.zeros((1, 15, 2)), np.ones((1, 15)))
        with self.assertRaisesRegex(RuntimeError, "cannot verify"):
            require_cuda_session(object())
        cpu = types.SimpleNamespace(session=types.SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"]))
        with self.assertRaisesRegex(RuntimeError, "CPU fallback"):
            require_cuda_session(cpu)
        calls = []
        session = types.SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
                                        disable_fallback=lambda: calls.append("disabled"))
        self.assertIs(require_cuda_session(types.SimpleNamespace(session=session)), session)
        self.assertEqual(calls, ["disabled"])

    def test_shard_resume_carries_exact_metadata_and_rejects_changed_config(self):
        chosen = selection([0, 50])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toyota_multilabel_20hz_training_P03T01C01.npz"
            save_video_shard(path, video=video(), source_path=Path("P03T01C01.mp4"), source_fps=25.0,
                             selection=chosen, skeletons=np.zeros((1, 30, 2, 17, 3), np.float16),
                             kept=np.asarray([1]), fingerprint="same", report={"retained_windows": 1})
            self.assertEqual(resume_report(path, "same")["retained_windows"], 1)
            with np.load(path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["starts"], [50])
                np.testing.assert_array_equal(data["source_frames"], chosen.source_frames[[1]])
                self.assertEqual(data["datasets"][0], "toyota_untrimmed")
                self.assertEqual(data["subjects"][0], "p03")
                self.assertEqual(float(data["sample_fps"]), 20.0)
            with self.assertRaisesRegex(ValueError, "settings changed"):
                resume_report(path, "different")
            self.assertFalse(path.with_suffix(".npz.partial").exists())
            self.assertEqual(json.loads(path.with_suffix(".json").read_text())["schema_version"], 2)

    def test_local_cuda_libraries_preload_before_model_creation(self):
        calls = []
        session = types.SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider"],
                                        disable_fallback=lambda: calls.append("disable_fallback"))
        def rtmo(**kwargs):
            self.assertEqual(kwargs["device"], "cuda")
            self.assertTrue(Path(kwargs["onnx_model"]).is_file())
            calls.append("model")
            return types.SimpleNamespace(session=session)

        ort = types.SimpleNamespace(preload_dlls=lambda: calls.append("preload"),
                                    get_available_providers=lambda: ["CUDAExecutionProvider"])
        modules = {"torch": types.SimpleNamespace(), "onnxruntime": ort,
                   "rtmlib": types.SimpleNamespace(RTMO=rtmo)}
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules):
            model = Path(directory) / "rtmo-l.onnx"
            model.write_bytes(b"test-only local model double")
            load_detector(model)
        self.assertEqual(calls, ["preload", "model", "disable_fallback"])

    def test_full_partition_resume_from_read_only_mount_and_final_subset_completion(self):
        class MetadataCapture(Capture):
            def __init__(self):
                super().__init__(count=1000)

            def get(self, key):
                return 25.0 if key == 5 else 1000 if key == 7 else super().get(key)

        fake_cv2 = types.SimpleNamespace(CAP_PROP_POS_FRAMES=1, CAP_PROP_FPS=5,
                                         CAP_PROP_FRAME_COUNT=7, VideoCapture=lambda _: MetadataCapture())
        def make_detector(_):
            detector = Detector()
            detector.session = types.SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider"],
                                                      disable_fallback=lambda: None)
            return detector

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = root / "videos"
            videos.mkdir()
            annotations = {}
            for subject, subset in ((3, "training"), (4, "training"), (6, "training"), (2, "testing")):
                vid = f"P{subject:02d}T01C01"
                (videos / f"{vid}.mp4").write_bytes(b"test-only decoder input")
                annotations[vid] = {"subset": subset, "duration": 1000, "actions": [[19, 100, 800]]}
            annotation_path = root / "smarthome_CS_51.json"
            annotation_path.write_text(json.dumps(annotations))
            model = root / "rtmo-l.onnx"
            model.write_bytes(b"test-only local model double")
            original = root / "first_session"
            resumed = root / "second_session"
            aggregate = root / "aggregate_session"
            base = ["--videos-root", str(videos), "--annotations", str(annotation_path),
                    "--rtmo-model", str(model)]
            partition = ["--partition-count", "2", "--partition-index", "0"]
            manifest_name = "manifest_training_part000-of002.json"
            with patch.dict(sys.modules, {"cv2": fake_cv2}), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "complete official annotation inventory"):
                    main(base + ["--out", str(aggregate)])
                with patch("extract_multilabel_windows.OFFICIAL_SUBSET_VIDEO_COUNTS",
                           {"training": 3, "testing": 1, "all": 4}):
                    with self.assertRaisesRegex(ValueError, "video IDs do not match"):
                        main(base + ["--out", str(aggregate)])
            # The extraction/resume integration test substitutes a tiny official
            # inventory; production above correctly rejects these four test records.
            with patch.dict(sys.modules, {"cv2": fake_cv2}), redirect_stdout(io.StringIO()), \
                    patch("extract_multilabel_windows.OFFICIAL_SUBSET_VIDEO_COUNTS",
                          {"training": 3, "testing": 1, "all": 4}), \
                    patch("extract_multilabel_windows.OFFICIAL_SUBSET_INVENTORY_SHA256",
                          {"training": inventory_sha256([replace(video(), vid=vid) for vid, record in annotations.items()
                                                          if record["subset"] == "training"])}):
                with patch("extract_multilabel_windows.load_detector", side_effect=make_detector):
                    self.assertEqual(main(base + partition + ["--out", str(original), "--max-videos", "1"]), 0)
                    manifest = json.loads((original / manifest_name).read_text())
                    self.assertFalse(manifest["scope_complete"])
                    self.assertEqual(manifest["completed_videos"], 1)
                    self.assertEqual(main(base + partition + ["--out", str(original)]), 0)
                manifest = json.loads((original / manifest_name).read_text())
                self.assertTrue(manifest["scope_complete"])
                self.assertFalse(manifest["subset_complete"])
                self.assertEqual(manifest["resumed_videos"], 1)
                # Read-only carry-forward does not need to load a detector or copy shards.
                with patch("extract_multilabel_windows.load_detector", side_effect=AssertionError("must resume")):
                    self.assertEqual(main(base + partition + ["--out", str(resumed), "--resume-from", str(original),
                                                               "--seed", "99", "--windows-per-video", "1"]), 0)
                manifest = json.loads((resumed / manifest_name).read_text())
                self.assertTrue(manifest["scope_complete"])
                self.assertFalse(manifest["complete"])
                self.assertEqual(manifest["resumed_videos"], 2)
                self.assertEqual(list(resumed.glob("*.npz")), [])
                self.assertTrue(all(Path(item["artifact_path"]).parent == original for item in manifest["videos"]))
                # The next video is deferred before the budget is crossed using the
                # observed complete decode/inference rate of the first video.
                ticks = {"seconds": 0.0}
                class BudgetCapture(MetadataCapture):
                    def grab(self):
                        ticks["seconds"] += 0.01
                        return super().grab()
                budget_cv2 = types.SimpleNamespace(CAP_PROP_POS_FRAMES=1, CAP_PROP_FPS=5,
                    CAP_PROP_FRAME_COUNT=7, VideoCapture=lambda _: BudgetCapture())
                budget_out = root / "budget_session"
                with patch.dict(sys.modules, {"cv2": budget_cv2}), \
                        patch("extract_multilabel_windows.time.monotonic", side_effect=lambda: ticks["seconds"]), \
                        patch("extract_multilabel_windows.load_detector", side_effect=make_detector):
                    self.assertEqual(main(base + ["--out", str(budget_out), "--stop-after-seconds", "12"]), 0)
                budget_manifest = json.loads((budget_out / "manifest_training.json").read_text())
                self.assertEqual(budget_manifest["completed_videos"], 1)
                self.assertTrue(budget_manifest["stopped_for_budget"])
                self.assertFalse(budget_manifest["scope_complete"])
                self.assertLess(budget_manifest["seconds_this_invocation"], 12)
                self.assertIn("estimated", budget_manifest["budget_stop"]["reason"])
                # A final run that verifies both carried shards and extracts the remaining
                # official training video can truthfully mark complete subset coverage.
                with patch("extract_multilabel_windows.load_detector", side_effect=make_detector):
                    self.assertEqual(main(base + ["--out", str(aggregate), "--resume-from", str(original)]), 0)
                manifest = json.loads((aggregate / "manifest_training.json").read_text())
                self.assertTrue(manifest["complete"])
                self.assertTrue(manifest["subset_complete"])
                self.assertEqual(manifest["completed_videos"], 3)
                self.assertEqual(manifest["resumed_videos"], 2)
                self.assertNotIn("P02T01C01", manifest["completed_video_ids"])
                self.assertEqual(manifest["config"]["selection_mode"], "full")
                self.assertFalse(manifest["config"]["strata_used_for_selection"])
                self.assertTrue(all(item["full_dense_coverage"] and item["window_cap"] is None
                                    for item in manifest["videos"]))
                self.assertTrue(all(item["source_sha256"] and item["config_fingerprint"] == manifest["fingerprint"]
                                    for item in manifest["videos"]))
                from behaviorsense.models.visual_context import CONTEXT_DIM, CONTEXT_SCHEMA, FEATURE_NAMES
                context_model = root / "rtdetr-l.onnx"
                context_model.write_bytes(b"test-only context detector model")
                def context_features(frame, pose):
                    self.assertEqual(float(pose[0, 0, 0]), float(frame[0, 0, 0]))
                    values = np.zeros(CONTEXT_DIM, np.float32)
                    values[0] = 1
                    return values
                context_detector = types.SimpleNamespace(
                    features=context_features, session=make_detector(None).session,
                    metadata={"schema": CONTEXT_SCHEMA, "feature_dim": CONTEXT_DIM,
                              "feature_names": list(FEATURE_NAMES), "model_sha256": file_sha256(context_model),
                              "provider": "CUDAExecutionProvider"})
                context_args = base + ["--context-model", str(context_model), "--out", str(root / "context_session")]
                with patch("extract_multilabel_windows.load_detector", side_effect=make_detector), \
                        patch("behaviorsense.models.visual_context.load_context_detector", return_value=context_detector) as loader:
                    self.assertEqual(main(context_args), 0)
                    self.assertEqual(loader.call_count, 1)
                context_manifest = json.loads((root / "context_session" / "manifest_training.json").read_text())
                self.assertTrue(context_manifest["complete"])
                self.assertTrue(all(item["context_present"] and item["context_dim"] == CONTEXT_DIM
                                    for item in context_manifest["videos"]))
                with patch("extract_multilabel_windows.load_detector", side_effect=AssertionError("must resume")), \
                        patch("behaviorsense.models.visual_context.load_context_detector", side_effect=AssertionError("must resume")):
                    self.assertEqual(main(context_args), 0)
                # A same-sized changed video must never reuse the old detector output.
                changed_video = videos / (manifest["videos"][0]["video_id"] + ".mp4")
                changed_video.write_bytes(b"x" * changed_video.stat().st_size)
                with self.assertRaisesRegex(ValueError, "settings changed"):
                    main(base + ["--out", str(aggregate), "--resume-from", str(original)])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    total = result.testsRun - len(result.skipped)
    passed = total - len(result.failures) - len(result.errors)
    print(f"{passed}/{total} passed; {len(result.skipped)} skipped")
    sys.exit(0 if result.wasSuccessful() else 1)
