"""CPU-only tests of CUDA timing order and the bounded fitting-probe A/B guard."""
from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rgb_profile_stage as profile_stage
import test_rgb_launcher as fixtures
from extract_rgb_action_features import SCHEMA, _array_sha256, source_frame_indices


class FakeCUDA:
    def __init__(self):
        self.calls = []

    def is_available(self):
        return True

    def device(self, device):
        self.calls.append(("device", device))
        return nullcontext()

    def synchronize(self, device):
        self.calls.append(("synchronize", device))

    def Event(self, enable_timing):
        assert enable_timing is True
        parent = self

        class Event:
            def record(self):
                parent.calls.append(("record", self))

            def elapsed_time(self, other):
                parent.calls.append(("elapsed_time", self, other))
                return 2.5

        return Event()


class StageTimingTests(unittest.TestCase):
    def test_cpu_timing_never_imports_torch_or_accesses_cuda(self):
        clock = iter([1., 1.25])
        cuda = FakeCUDA()
        value = profile_stage.TimingProfile(clock=lambda: next(clock), cuda_module=cuda)
        with patch.dict(sys.modules, {"torch": None}), value.cpu("decode"):
            pass
        stage = value.summary()["stages"]["decode"]
        self.assertEqual(.25, stage["wall_seconds"])
        self.assertIsNone(stage["cuda_seconds"])
        self.assertEqual("cpu_wall_clock", stage["timing_mode"])
        self.assertEqual([], cuda.calls)

    def test_cuda_events_are_bracketed_by_synchronization_on_same_device(self):
        cuda = FakeCUDA()
        clock = iter([2., 2.01])
        value = profile_stage.TimingProfile("cuda:2", clock=lambda: next(clock), cuda_module=cuda)
        with value.cuda("forward"):
            cuda.calls.append(("work",))
        self.assertEqual(["device", "synchronize", "record", "work", "record", "synchronize", "elapsed_time"],
                         [call[0] for call in cuda.calls])
        self.assertEqual([("synchronize", "cuda:2")] * 2,
                         [call for call in cuda.calls if call[0] == "synchronize"])
        stage = value.summary()["stages"]["forward"]
        self.assertAlmostEqual(.01, stage["wall_seconds"])
        self.assertEqual(.0025, stage["cuda_seconds"])
        self.assertEqual(1, stage["calls"])
        self.assertEqual(0, stage["failed_calls"])

    def test_cuda_failure_preserves_original_exception_and_records_incomplete_call(self):
        cuda = FakeCUDA()
        value = profile_stage.TimingProfile(cuda_module=cuda)
        with self.assertRaisesRegex(RuntimeError, "original allocation failure"):
            with value.cuda("forward"):
                raise RuntimeError("original allocation failure")
        self.assertEqual(1, value.summary()["stages"]["forward"]["failed_calls"])
        self.assertEqual(1, len([call for call in cuda.calls if call[0] == "synchronize"]))

    def test_repeated_stage_totals_are_additive_but_summary_is_not_mutable_state(self):
        clock = iter([1., 1.1, 2., 2.2])
        value = profile_stage.TimingProfile(clock=lambda: next(clock))
        with value.cpu("person_crop"):
            pass
        with value.cpu("person_crop"):
            pass
        summary = value.summary()
        self.assertAlmostEqual(.3, summary["stages"]["person_crop"]["wall_seconds"])
        summary["stages"]["person_crop"]["calls"] = 99
        self.assertEqual(2, value.summary()["stages"]["person_crop"]["calls"])
        self.assertTrue(summary["totals_may_overlap"])


class ProbeComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rgb-profile-")
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.job = fixtures.fixture_job()
        self.records = {row["video_id"]: row for row in self.job["video_inventory"]}
        arch = "swin3d_t"
        self.info = {"arch": arch, "weights_sha256": self.job["assets"]["encoders"][arch]["sha256"],
            "feature_dim": 768, "preprocessing_sha256": "1" * 64, "device": "cuda",
            "preprocessing": self.job["assets"]["preprocessing"]}
        self.cuda = FakeCUDA()
        self.called = []
        self.change = None
        self.instrument = True
        self.fail_at = None
        self.started = time.monotonic()

    def callback(self, variant, vid, output, deadline, profile):
        self.called.append((variant, vid, output, deadline))
        if len(self.called) == self.fail_at:
            raise TimeoutError("synthetic existing probe deadline")
        began = time.monotonic()
        row = self.records[vid]
        n = row["n_windows"]
        starts = np.arange(n, dtype=np.int64) * 15
        centers = np.rint((starts + 15) * 1.5).astype(np.int64)
        frame_count = int(centers[-1]) + 100
        frames, boundary = source_frame_indices(centers, 30., frame_count)
        arrays = {"starts": starts, "video_id": np.asarray(vid), "subject": np.asarray(row["subject"]),
            "video_ids": np.full(n, vid), "subjects": np.full(n, row["subject"]),
            "center_source_frames": centers, "source_frames_rgb": frames,
            "boundary_duplicate_flags": boundary, "person_rois": np.tile(np.array([0, 0, 32, 32], np.int64), (n, 1)),
            "roi_observed": np.ones(n, bool), "source_fps": np.asarray(30., np.float64),
            "features": np.zeros((n, 2, self.info["feature_dim"]), np.float16)}
        metadata = {"schema": SCHEMA, "video_id": vid, "subject": row["subject"], "n_windows": n,
            "arch": self.info["arch"], "feature_dim": self.info["feature_dim"],
            "source_video_sha256": row["source_sha256"], "source_shard_sha256": row["shard_sha256"],
            "encoder_sha256": self.info["weights_sha256"], "encoder_info": self.info,
            "preprocessing_sha256": self.info["preprocessing_sha256"], "source_fps": 30.,
            "source_frame_count": frame_count, "source_width": 100, "source_height": 80,
            "views": ["scene", "person"], "status": "complete"}
        if variant == "optimized" and self.change is not None:
            self.change(arrays, metadata)
        metadata["array_sha256"] = {name: _array_sha256(value) for name, value in arrays.items()}
        if self.instrument:
            for name in ("decode", "scene_crop", "person_crop", "pack"):
                with profile.cpu(name):
                    pass
            for name in ("transfer", "preprocess", "forward", "readback"):
                with profile.cuda(name):
                    pass
        with profile.cpu("serialize"):
            np.savez_compressed(output, **arrays, metadata=np.asarray(json.dumps(metadata)))
        return {key: metadata[key] for key in ("status", "video_id", "subject", "n_windows",
            "source_video_sha256", "source_shard_sha256", "encoder_sha256", "preprocessing_sha256")} | {
            "output_path": str(output), "output_sha256": fixtures.sha256(output),
            "elapsed_seconds": time.monotonic() - began, "reused": False}

    def run_profile(self, **kwargs):
        return profile_stage.profile_probe_variants(self.job, self.work, self.info, self.callback,
            started_at=self.started, deadline=kwargs.get("deadline", self.started + 1200), cuda_module=self.cuda)

    def test_complete_identical_probes_return_only_three_fresh_optimized_reports(self):
        value = self.run_profile()
        self.assertEqual("complete", value["status"])
        self.assertEqual(6, len(value["runs"]))
        self.assertEqual(3, len(value["comparisons"]))
        self.assertEqual(self.job["pilot"]["probe_video_ids"], [r["video_id"] for r in value["optimized_reports"]])
        self.assertTrue(value["all_three_equivalence_checks_passed"])
        self.assertFalse(value["quality_gate_evaluated"])
        self.assertFalse(value["throughput_gate_evaluated"])
        self.assertFalse(value["test_features_loaded"])
        self.assertEqual(["baseline", "optimized", "optimized", "baseline", "baseline", "optimized"],
                         [call[0] for call in self.called])
        self.assertTrue(all(call[3] == self.started + 900 for call in self.called))
        self.assertTrue(all(self.records[call[1]]["subject"] in fixtures.stage.FIT_SUBJECTS for call in self.called))
        for report in value["optimized_reports"]:
            self.assertEqual(self.work / "rgb_features" / self.info["arch"] / (report["video_id"] + ".npz"), Path(report["output_path"]))

    def test_original_source_clock_arrays_cannot_change_even_with_fresh_valid_hash(self):
        self.change = lambda arrays, meta: arrays["source_frames_rgb"].__setitem__((0, 0), 99)
        with self.assertRaisesRegex(ValueError, "changed original rows, clocks or crops"):
            self.run_profile()
        value = fixtures.read_json(self.work / "profiling" / self.info["arch"] / "profile_report.json")
        self.assertEqual("failed", value["status"])
        self.assertNotIn("optimized_reports", value)

    def test_embedding_mismatch_blocks_optimized_forecast_reports(self):
        self.change = lambda arrays, meta: arrays["features"].__setitem__((0, 0, 0), .1)
        with self.assertRaisesRegex(ValueError, "frozen feature tolerance"):
            self.run_profile()
        value = fixtures.read_json(self.work / "profiling" / self.info["arch"] / "profile_report.json")
        self.assertNotIn("optimized_reports", value)

    def test_nonfinite_embedding_blocks_optimized_forecast_reports(self):
        self.change = lambda arrays, meta: arrays["features"].__setitem__((0, 0, 0), np.nan)
        with self.assertRaisesRegex(ValueError, "finite float16"):
            self.run_profile()

    def test_source_contract_change_blocks_comparison(self):
        self.change = lambda arrays, meta: meta.__setitem__("source_fps", 25.)
        with self.assertRaisesRegex(ValueError, "changed encoder, source or preprocessing"):
            self.run_profile()

    def test_incomplete_instrumentation_cannot_look_like_valid_profiling(self):
        self.instrument = False
        with self.assertRaisesRegex(ValueError, "instrumentation"):
            self.run_profile()

    def test_budget_expiry_returns_no_optimized_reports_or_new_deadline(self):
        value = self.run_profile(deadline=self.started - 1)
        self.assertEqual("incomplete_budget", value["status"])
        self.assertEqual([], self.called)
        self.assertNotIn("optimized_reports", value)

    def test_mid_comparison_timeout_cannot_promote_partial_optimized_coverage(self):
        self.fail_at = 4
        value = self.run_profile()
        self.assertEqual("incomplete_budget", value["status"])
        self.assertEqual(1, len(value["comparisons"]))
        self.assertNotIn("optimized_reports", value)
        self.assertFalse(value["all_three_equivalence_checks_passed"])

    def test_profile_does_not_resume_previous_comparison(self):
        self.run_profile(deadline=self.started - 1)
        with self.assertRaisesRegex(ValueError, "cannot resume"):
            self.run_profile()

    def test_test_video_cannot_enter_profile_and_budgets_cannot_expand(self):
        tests = [r["video_id"] for r in self.job["video_inventory"] if r["subset"] == "testing"]
        self.job["pilot"]["probe_video_ids"][0] = tests[0]
        with self.assertRaises(ValueError):
            self.run_profile()
        self.assertEqual([], self.called)
        self.job = fixtures.fixture_job()
        self.job["probe_seconds"] = 901
        with self.assertRaisesRegex(ValueError, "finite RGB experiment"):
            self.run_profile()


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    total = result.testsRun - len(result.skipped)
    passed = total - len(result.failures) - len(result.errors)
    print(f"{passed}/{total} passed; {len(result.skipped)} skipped")
    raise SystemExit(0 if result.wasSuccessful() else 1)
