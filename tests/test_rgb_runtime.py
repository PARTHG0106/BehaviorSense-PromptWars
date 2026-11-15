"""Offline regressions for runtime fallback, fixed gates and budget receipts."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import test_rgb_launcher as fixtures

fixture_job, stage = fixtures.fixture_job, fixtures.stage
read_json, write_json = fixtures.read_json, fixtures.write_json


class RGBRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.run = fixtures.RGBExecutionTests()
        self.run.setUp()
        self.addCleanup(self.run.doCleanups)

    def assert_no_test_score(self, result):
        self.assertNotIn("test", result)
        self.assertNotEqual("complete", result["status"])

    def test_gate_requirements_cannot_be_weakened_in_frozen_job(self):
        for path in (("pair_precision_min",), ("macro_ap_min",),
                     ("subjects", "p15", "macro_ap_min")):
            job = fixture_job()
            node = job["gate_criteria"]
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = 0.
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "frozen material-improvement"):
                stage.validate_job(job)

    def test_base_load_out_of_memory_tries_tiny_in_same_session(self):
        load = self.run.fake_load

        def failing_load(checkpoint, expected, arch, device):
            if arch == "swin3d_b":
                self.run.loads.append(arch)
                self.run.now += 17
                raise sys.modules["torch"].cuda.OutOfMemoryError("synthetic allocation failure")
            return load(checkpoint, expected, arch, device)

        self.run.fake_load = failing_load
        result = self.run.execute()
        self.assertEqual("complete", result["status"])
        self.assertEqual(["swin3d_b", "swin3d_t"], self.run.loads)
        probes = read_json(self.run.work / "probe_report.json")
        self.assertEqual(["cuda_out_of_memory", "passed"], [x["status"] for x in probes])
        self.assertLessEqual(probes[-1]["projection"]["projected_pilot_seconds"], 4500)

    def test_base_forward_out_of_memory_tries_tiny_without_reusing_base(self):
        extract = self.run.fake_extract

        def failing_extract(video, shard, output, encoder, info, batch_size, deadline):
            if info["arch"] == "swin3d_b":
                raise sys.modules["torch"].cuda.OutOfMemoryError("synthetic forward allocation failure")
            return extract(video, shard, output, encoder, info, batch_size, deadline)

        self.run.fake_extract = failing_extract
        result = self.run.execute()
        self.assertEqual("complete", result["status"])
        self.assertTrue(all(arch == "swin3d_t" for arch, _, _, _ in self.run.extractions))
        probes = read_json(self.run.work / "probe_report.json")
        self.assertEqual(["cuda_out_of_memory", "passed"], [x["status"] for x in probes])

    def test_both_encoder_allocations_failing_stops_before_heads_or_test(self):
        def failing_load(checkpoint, expected, arch, device):
            self.run.loads.append(arch)
            raise sys.modules["torch"].cuda.OutOfMemoryError("synthetic allocation failure")

        self.run.fake_load = failing_load
        result = self.run.execute()
        self.assertEqual("throughput_rejected", result["status"])
        self.assertEqual(["swin3d_b", "swin3d_t"], self.run.loads)
        self.assertEqual([], self.run.commands)
        self.assertEqual(["training"], self.run.binds)

    def test_identity_failure_does_not_silently_fall_back(self):
        def failing_load(checkpoint, expected, arch, device):
            self.run.loads.append(arch)
            raise ValueError("pretrained RGB checkpoint SHA-256 mismatch")

        self.run.fake_load = failing_load
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.run.execute()
        self.assertEqual(["swin3d_b"], self.run.loads)
        self.assertEqual([], self.run.commands)

    def child_reports_incomplete(self, wanted, status="incomplete_budget"):
        child = self.run.fake_child

        def incomplete_child(arguments, timeout):
            args = list(map(str, arguments))
            command = args[2] if args[2] == "evaluate-test" else args[args.index("--scope") + 1]
            if command != wanted:
                return child(arguments, timeout)
            self.run.commands.append(args)
            out = Path(args[args.index("--out") + 1])
            out.mkdir(parents=True)
            write_json(out / "training_status.json", {"status": status,
                "inference_checkpoint_exported": False, "test_evaluated": False})
            if wanted != "evaluate-test":
                gate = {"passed": False, "reason": status, "metrics": {}}
                write_json(out / "gate.json", gate)
                write_json(out / "frozen_selection.json", {"gate_passed": False, "gate": gate})

        self.run.fake_child = incomplete_child

    def test_pilot_cooperative_timeout_is_budget_exhaustion_and_stops(self):
        self.child_reports_incomplete("pilot")
        result = self.run.execute()
        self.assertEqual("pilot_budget_exhausted", result["status"])
        self.assertEqual(1, len(self.run.commands))
        self.assertEqual(["training"], self.run.binds)
        self.assert_no_test_score(result)

    def test_full_cooperative_timeout_is_budget_exhaustion_and_stops(self):
        self.child_reports_incomplete("full")
        result = self.run.execute()
        self.assertEqual("session_budget_exhausted", result["status"])
        self.assertEqual(2, len(self.run.commands))
        self.assertEqual(["training"], self.run.binds)
        self.assertFalse((self.run.work / "frozen_selection.json").exists())
        self.assert_no_test_score(result)

    def test_pilot_epoch_limit_stays_incomplete_and_stops_before_full_training(self):
        self.child_reports_incomplete("pilot", "incomplete_epoch_limit")
        result = self.run.execute()
        self.assertEqual("incomplete_epoch_limit", result["status"])
        self.assertEqual("pilot", result["training_scope"])
        self.assertIn("without a validation plateau", result["reason"])
        self.assertEqual(1, len(self.run.commands))
        self.assertEqual(["training"], self.run.binds)
        self.assertFalse(result["test_features_loaded"])
        self.assertFalse((self.run.work / "pilot_passed.json").exists())
        self.assertFalse((self.run.work / "full").exists())
        self.assertFalse((self.run.work / "frozen_selection.json").exists())
        self.assert_no_test_score(result)

    def test_full_epoch_limit_stays_incomplete_and_stops_before_test_binding(self):
        self.child_reports_incomplete("full", "incomplete_epoch_limit")
        result = self.run.execute()
        self.assertEqual("incomplete_epoch_limit", result["status"])
        self.assertEqual("full", result["training_scope"])
        self.assertIn("without a validation plateau", result["reason"])
        self.assertEqual(2, len(self.run.commands))
        self.assertEqual(["training"], self.run.binds)
        self.assertFalse(result["test_features_loaded"])
        self.assertFalse((self.run.work / "test").exists())
        self.assertFalse((self.run.work / "frozen_selection.json").exists())
        self.assert_no_test_score(result)

    def test_test_cooperative_timeout_does_not_expect_nonexistent_metrics(self):
        self.child_reports_incomplete("evaluate-test")
        result = self.run.execute()
        self.assertEqual("session_budget_exhausted", result["status"])
        self.assertEqual(3, len(self.run.commands))
        self.assertFalse((self.run.work / "test" / "test_metrics.json").exists())
        self.assert_no_test_score(result)

    def test_test_process_timeout_is_budget_exhaustion_without_score(self):
        self.run.child_timeout = "evaluate-test"
        result = self.run.execute()
        self.assertEqual("session_budget_exhausted", result["status"])
        self.assertEqual(3, len(self.run.commands))
        self.assert_no_test_score(result)


class RGBSupervisorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rgb-supervisor-")
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.job = fixture_job()
        write_json(self.work / "job.json", self.job)
        self.now = 0.
        self.process = Mock(pid=24601, args=["synthetic-worker"])
        self.process.poll.return_value = None
        self.spawn = Mock(return_value=self.process)
        self.terminate = Mock()
        self.sleeps = []

    def passage(self, seconds):
        self.assertGreater(seconds, 0)
        self.assertLessEqual(seconds, .25)
        self.sleeps.append(seconds)
        self.now += seconds

    def probe_receipt(self):
        write_json(self.work / "encoder_selection.json", {"job_id": self.job["job_id"],
            "source_sha256": self.job["source_sha256"], "completed_elapsed_seconds": 100.,
            "projection": {"passed": True}})

    def pilot_receipt(self, completed=1000., changed_gate=False):
        (self.work / "pilot").mkdir()
        gate_path = self.work / "pilot" / "gate.json"
        gate = {"passed": True, "criteria": self.job["gate_criteria"]}
        write_json(gate_path, gate)
        write_json(self.work / "pilot_passed.json", {"job_id": self.job["job_id"],
            "source_sha256": self.job["source_sha256"], "completed_elapsed_seconds": completed,
            "gate_sha256": fixtures.sha256(gate_path)})
        if changed_gate:
            write_json(gate_path, {**gate, "passed": False})

    def supervise(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(stage.sys, "platform", "linux"))
            stack.enter_context(patch.object(stage.time, "monotonic", side_effect=lambda: self.now))
            stack.enter_context(patch.object(stage.time, "sleep", side_effect=self.passage))
            stack.enter_context(patch.object(stage.subprocess, "Popen", self.spawn))
            stack.enter_context(patch.object(stage, "_terminate_process_group", self.terminate))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return stage.supervise(self.job, fixtures.ROOT, self.work, started_at=0.)

    def test_probe_hang_kills_only_worker_group_at_shared_probe_deadline(self):
        self.now = 899.9
        result = self.supervise()
        self.assertEqual("throughput_rejected", result["status"])
        self.assertAlmostEqual(900., result["elapsed_seconds"])
        self.assertEqual(900, result["supervisor_deadline_seconds"])
        self.terminate.assert_called_once_with(self.process)
        self.assertTrue(self.spawn.call_args.kwargs["start_new_session"])
        self.assertIn("'scripts'", self.spawn.call_args.args[0][3])
        self.assertIn("'src'", self.spawn.call_args.args[0][3])

    def test_timely_probe_without_pilot_pass_still_has_absolute_pilot_cap(self):
        self.now = 4499.9
        self.probe_receipt()
        result = self.supervise()
        self.assertEqual("pilot_budget_exhausted", result["status"])
        self.assertAlmostEqual(4500., result["elapsed_seconds"])
        self.terminate.assert_called_once_with(self.process)

    def test_changed_gate_cannot_extend_pilot_deadline(self):
        self.now = 4499.9
        self.probe_receipt()
        self.pilot_receipt(changed_gate=True)
        self.assertEqual("pilot_budget_exhausted", self.supervise()["status"])

    def test_late_pilot_pass_cannot_extend_pilot_deadline(self):
        self.now = 4499.9
        self.probe_receipt()
        self.pilot_receipt(completed=4500.01)
        self.assertEqual("pilot_budget_exhausted", self.supervise()["status"])

    def test_valid_pilot_pass_allows_only_remaining_single_session_time(self):
        self.now = 40499.9
        self.probe_receipt()
        self.pilot_receipt()
        write_json(self.work / "stage_result.json", {"job_id": self.job["job_id"],
            "source_sha256": self.job["source_sha256"], "status": "running",
            "test_features_loaded": True, "test": {"partial": "must not survive timeout"}})
        result = self.supervise()
        self.assertEqual("session_budget_exhausted", result["status"])
        self.assertAlmostEqual(40500., result["elapsed_seconds"])
        self.assertTrue(result["test_features_loaded"])
        self.assertNotIn("test", result)
        self.assertTrue(result["hard_timeout"])
        self.terminate.assert_called_once_with(self.process)

    def test_setup_that_already_spent_probe_budget_cannot_start_worker(self):
        self.now = 900.
        self.assertEqual("throughput_rejected", self.supervise()["status"])
        self.spawn.assert_not_called()
        self.terminate.assert_not_called()

    def test_empty_child_success_is_failure_and_cleans_up_group(self):
        self.process.poll.return_value = 0
        with self.assertRaisesRegex(RuntimeError, "valid terminal result"):
            self.supervise()
        self.terminate.assert_called_once_with(self.process)

    def test_child_failure_propagates_and_cleans_up_group(self):
        self.process.poll.return_value = 3
        with self.assertRaises(subprocess.CalledProcessError):
            self.supervise()
        self.terminate.assert_called_once_with(self.process)

    def test_valid_terminal_result_returns_without_retry(self):
        self.process.poll.return_value = 0
        value = {"job_id": self.job["job_id"], "source_sha256": self.job["source_sha256"],
                 "status": "throughput_rejected"}
        write_json(self.work / "stage_result.json", value)
        self.assertEqual(value, self.supervise())
        self.spawn.assert_called_once()
        self.terminate.assert_called_once_with(self.process)
        with self.assertRaisesRegex(ValueError, "cannot resume"):
            self.supervise()

    def test_epoch_limit_terminal_result_returns_without_retry_or_pilot_pass(self):
        self.process.poll.return_value = 0
        self.now = 1100.
        self.probe_receipt()
        value = {"job_id": self.job["job_id"], "source_sha256": self.job["source_sha256"],
                 "status": "incomplete_epoch_limit", "training_scope": "pilot",
                 "test_features_loaded": False}
        write_json(self.work / "stage_result.json", value)
        self.assertEqual(value, self.supervise())
        self.spawn.assert_called_once()
        self.terminate.assert_called_once_with(self.process)

    def test_termination_uses_unconditional_group_kill_and_bounded_reap(self):
        with patch.object(stage.os, "killpg", create=True) as kill, patch.object(stage.signal, "SIGKILL", 9, create=True):
            stage._terminate_process_group(self.process)
        kill.assert_called_once_with(self.process.pid, 9)
        self.process.wait.assert_called_once_with(timeout=10)

    @unittest.skipUnless(sys.platform == "linux", "real POSIX process-group cleanup requires Linux")
    def test_real_worker_and_descendant_processes_are_killed_together(self):
        child_pid_path = self.work / "descendant.pid"
        body = ("import subprocess,sys,time; from pathlib import Path; "
                "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                "Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)")
        process = subprocess.Popen([sys.executable, "-c", body, str(child_pid_path)], start_new_session=True)
        try:
            until = time.monotonic() + 5
            while not child_pid_path.exists() and time.monotonic() < until:
                time.sleep(.01)
            self.assertTrue(child_pid_path.exists())
            child_pid = int(child_pid_path.read_text())
            stage._terminate_process_group(process)
            self.assertEqual(-9, process.returncode)
            # Container PID 1 may reap adopted descendants asynchronously.
            # A zombie is terminated, even if its /proc entry remains briefly.
            until = time.monotonic() + 2
            while time.monotonic() < until:
                state = Path(f"/proc/{child_pid}/stat")
                if not state.exists() or state.read_text().split()[2] == "Z":
                    break
                time.sleep(.01)
            else:
                self.fail("worker's descendant survived process-group termination")
        finally:
            stage._terminate_process_group(process)


class RGBProfileIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.run = fixtures.RGBExecutionTests()
        self.run.setUp()
        self.addCleanup(self.run.doCleanups)
        self.run.job.update(rgb_protocol=stage.PROFILE_PROTOCOL, encoder_preference=["swin3d_t"], opencv_threads=1)
        original = self.run.fake_extract
        self.backends = []
        self.prefetch_settings = []
        def extraction(*args, profile=None, spatial_backend="native", prefetch_batches=0, **kwargs):
            self.backends.append((spatial_backend, profile is not None))
            self.prefetch_settings.append(prefetch_batches)
            return original(*args, **kwargs)
        self.run.fake_extract = extraction

    def profile(self, job, work, info, callback, *, started_at, deadline):
        self.assertEqual(started_at + 900, deadline)
        fresh = []
        for vid in job["pilot"]["probe_video_ids"]:
            for variant, folder in (("baseline", work / "profiling" / info["arch"] / "baseline"),
                                    ("optimized", work / "rgb_features" / info["arch"])):
                value = callback(variant, vid, folder / (vid + ".npz"), deadline, object())
                if variant == "optimized":
                    fresh.append(value)
        result = {"status": "complete", "all_three_equivalence_checks_passed": True,
                  "optimized_reports": fresh}
        (work / "profiling" / info["arch"]).mkdir(parents=True, exist_ok=True)
        write_json(work / "profiling" / info["arch"] / "profile_report.json", result)
        return result

    def execute(self, function):
        with patch.dict(sys.modules, {"rgb_profile_stage": SimpleNamespace(profile_probe_variants=function),
                                     "cv2": SimpleNamespace(setNumThreads=Mock())}):
            return self.run.execute()

    def test_tiny_profile_reuses_only_verified_optimized_probes_in_single_session(self):
        helper = Mock(side_effect=self.profile)
        result = self.execute(helper)
        self.assertEqual("complete", result["status"])
        helper.assert_called_once()
        self.assertEqual(["swin3d_t"], self.run.loads)
        self.assertEqual([("reference", True), ("native", True)] * 3, self.backends[:6])
        fresh = [x for x in self.run.extractions if not x[2]]
        self.assertEqual(539, len(fresh))  # 536 final outputs plus three diagnostic references.
        probes = read_json(self.run.work / "probe_report.json")
        self.assertTrue(probes[0]["all_three_equivalence_checks_passed"])
        self.assertEqual(stage.PROFILE_PROTOCOL, probes[0]["rgb_protocol"])
        self.assertEqual(3, len(self.run.commands))

    def test_incomplete_profile_stops_without_any_head_or_test(self):
        result = self.execute(lambda *a, **k: {"status": "incomplete_budget"})
        self.assertEqual("throughput_rejected", result["status"])
        self.assertEqual([], self.run.commands)
        self.assertEqual(["training"], self.run.binds)

    def test_prefetch_protocol_compares_native_sequential_then_bounded_overlap(self):
        self.run.job.update(rgb_protocol=stage.PREFETCH_PROTOCOL, prefetch_batches=2)
        result = self.execute(self.profile)
        self.assertEqual("complete", result["status"])
        self.assertEqual([("native", True)] * 6, self.backends[:6])
        self.assertEqual([0, 2] * 3, self.prefetch_settings[:6])
        self.assertTrue(all(value == 2 for value in self.prefetch_settings[6:]))

    def test_failed_equivalence_cannot_become_a_throughput_pass(self):
        with self.assertRaisesRegex(ValueError, "equivalence"):
            self.execute(lambda *a, **k: {"status": "complete", "all_three_equivalence_checks_passed": False})
        self.assertEqual([], self.run.commands)

    def test_profile_cannot_return_reused_reports_as_fresh_measurements(self):
        def reused(*args, **kwargs):
            result = self.profile(*args, **kwargs)
            result["optimized_reports"][0]["reused"] = True
            return result
        with self.assertRaisesRegex(ValueError, "fresh prescribed"):
            self.execute(reused)
        self.assertEqual([], self.run.commands)

    def test_tiny_only_protocol_must_be_explicit_and_preserve_budgets(self):
        job = self.run.job
        stage.validate_job(job)
        job.pop("rgb_protocol")
        with self.assertRaisesRegex(ValueError, "finite RGB"):
            stage.validate_job(job)
        job.update(rgb_protocol=stage.PROFILE_PROTOCOL, pilot_seconds=4501)
        with self.assertRaisesRegex(ValueError, "finite RGB"):
            stage.validate_job(job)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    total = result.testsRun - len(result.skipped)
    passed = total - len(result.failures) - len(result.errors)
    print(f"{passed}/{total} passed; {len(result.skipped)} skipped")
    raise SystemExit(0 if result.wasSuccessful() else 1)
