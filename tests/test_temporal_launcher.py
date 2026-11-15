"""Single-session launch/selection invariants without network, Torch or a GPU."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import launch_temporal_kaggle as launch
import temporal_kaggle_stage as stage
from multilabel_kaggle_stage import read_json, sha256, write_json


class HTTPError(Exception):
    """Offline requests-shaped exception; the CPU suite does not require Kaggle."""

    def __init__(self, message, response):
        super().__init__(message)
        self.response = response


class Response:
    def __init__(self):
        self.headers = {}

    def json(self):
        return json.loads(self._content)

    def __bool__(self):
        # requests treats an HTTP error response as false, even when it has a body.
        return self.status_code < 400


def baseline_plan():
    plan = {"source_sha256": "e" * 64, "annotation_sha256": "a" * 64, "stages": []}
    for subset, groups in (("training", [(0,), (1,), (2,), (3,), (4, 5, 6, 7)]),
                           ("testing", [(0,), (1,), (2,), (3,)])):
        counts = [44, 44, 44, 44, 175] if subset == "training" else [46, 47, 46, 46]
        for group, count in zip(groups, counts):
            key = f"{subset}-{group[0]}"
            receipt = {"job_id": f"fixture/{key}", "stage": "extract", "subset": subset,
                       "source_sha256": plan["source_sha256"], "status": "complete",
                       "scope_complete": True, "expected_videos": count, "completed_videos": count,
                       "partition_count": 8 if subset == "training" else 4,
                       "config_fingerprint": "f" * 64}
            receipt["partition_indices" if len(group) > 1 else "partition_index"] = list(group) if len(group) > 1 else group[0]
            plan["stages"].append({"key": key, "stage": "extract", "subset": subset,
                                   "complete": True, "result": receipt})
    plan["selection"] = {"job_id": "fixture/baseline", "model": "context-fusion", "seed": 2,
                         "checkpoint_relative_path": "candidates/seed2/best_multilabel.pt",
                         "checkpoint_sha256": "b" * 64, "validation_macro_ap": .5925371084457453}
    return plan


class TemporalLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="temporal-launch-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def build(self, include_test=True):
        baseline = self.root / "baseline" / "full-run"
        baseline.mkdir(parents=True, exist_ok=True)
        plan = baseline_plan()
        write_json(baseline / "plan.json", plan)
        write_json(baseline / "frozen_selection.json", plan["selection"])
        validation = baseline.parent / "completion-audit/baseline/candidates/seed2"
        validation.mkdir(parents=True, exist_ok=True)
        write_json(validation / "validation_metrics.json", {"macro_ap": .5925371084457453,
            "tv_drinking_colabeled_windows": {"f1": .14488188976377953, "precision": .10697674418604651}})
        repo = self.root / "source"
        for name in (*launch.SCRIPTS, "src/example.py", "configs/taxonomy.yaml"):
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# synthetic source\n")
        directory = self.root / "experiment"
        result = launch.build(directory, "fixture", baseline, include_test, repo)
        return directory, result

    def test_build_one_private_script_with_separate_source_hashes(self):
        directory, plan = self.build()
        metadata = read_json(directory / "submission/kernel-metadata.json")
        self.assertEqual(9, len(metadata["kernel_sources"]))
        self.assertEqual(["fixture/behaviorsense-toyota-meta"], metadata["dataset_sources"])
        self.assertTrue(metadata["is_private"])
        self.assertFalse(metadata["enable_internet"])
        self.assertEqual("NvidiaRtxPro6000", metadata["machine_shape"])
        self.assertEqual(43200, plan["timeout_seconds"])
        self.assertNotEqual(plan["job"]["source_sha256"], plan["job"]["extraction_source_sha256"])
        self.assertEqual(stage.CANDIDATES, plan["job"]["candidates"])
        compile((directory / "submission/training.py").read_text(), "training.py", "exec")

    def test_build_can_exclude_test_mounts(self):
        _, plan = self.build(False)
        self.assertEqual(5, len(plan["job"]["feature_sources"]))
        self.assertEqual({"training"}, {item["subset"] for item in plan["job"]["feature_sources"]})

    def test_no_overwrite_of_frozen_build(self):
        directory, _ = self.build()
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            launch.build(directory, "fixture")

    def test_changed_submission_refuses_push_before_network(self):
        directory, _ = self.build()
        with (directory / "submission/training.py").open("a") as out:
            out.write("\n# modified\n")
        api = Mock()
        with self.assertRaisesRegex(ValueError, "changed"):
            launch.push(api, directory)
        api.kernels_push.assert_not_called()

    def test_accepted_push_records_version_and_cannot_repeat(self):
        directory, plan = self.build()
        receipt = {"ref": "/code/" + plan["job"]["job_id"], "url": "https://www.kaggle.com/code/fixture/test",
                   "versionNumber": 1, "kernelId": 23, "error": None}
        api = Mock()
        api.kernels_push.return_value = receipt
        self.assertEqual(receipt, launch.push(api, directory))
        self.assertEqual("accepted", read_json(directory / "plan.json")["submission_attempt"]["state"])
        self.assertEqual(receipt, read_json(directory / "submission_receipt.json"))
        api.kernels_push.assert_called_once_with(str(directory / "submission"), timeout="43200", acc="NvidiaRtxPro6000")
        with self.assertRaisesRegex(ValueError, "already attempted"):
            launch.push(api, directory)

    def test_rejection_keeps_receipt_and_never_retries(self):
        directory, _ = self.build()
        api = Mock()
        api.kernels_push.return_value = {"error": "Maximum weekly GPU quota reached"}
        with self.assertRaisesRegex(RuntimeError, "quota"):
            launch.push(api, directory)
        self.assertEqual("submission_rejected", read_json(directory / "plan.json")["status"])
        with self.assertRaisesRegex(ValueError, "already attempted"):
            launch.push(api, directory)
        api.kernels_push.assert_called_once()

    def test_network_loss_journals_ambiguous_identity_before_call(self):
        directory, plan = self.build()
        api = Mock()
        def uncertain(*args, **kwargs):
            saved = read_json(directory / "plan.json")
            self.assertEqual("intent_journaled", saved["submission_attempt"]["state"])
            raise TimeoutError("response lost")
        api.kernels_push.side_effect = uncertain
        with self.assertRaises(TimeoutError):
            launch.push(api, directory)
        saved = read_json(directory / "plan.json")
        self.assertEqual("submission_ambiguous", saved["status"])
        self.assertEqual(plan["job"]["job_id"], saved["submission_attempt"]["ref"])
        self.assertEqual("ambiguous", saved["submission_attempt"]["state"])
        self.assertEqual("TimeoutError: response lost", saved["submission_attempt"]["error"])
        self.assertNotIn("http_error", saved["submission_attempt"])
        with self.assertRaisesRegex(ValueError, "already attempted"):
            launch.push(api, directory)
        api.kernels_push.assert_called_once()

    def test_http_failure_keeps_only_structured_diagnostics_and_never_retries(self):
        directory, _ = self.build()
        response = Response()
        response.status_code = 400
        response.headers.update({"X-Request-ID": "request-123", "X-Kaggle-Request-ID": "kaggle-456",
                                 "Set-Cookie": "private-cookie", "Authorization": "private-header"})
        response._content = json.dumps({"code": 3, "status": "INVALID_ARGUMENT",
            "message": "Kernel source is invalid", "request": {"key": "private-request"},
            "details": [{"debug": "private-details"}]}).encode()
        failure = HTTPError("SaveKernel failed", response=response)
        api = Mock()
        api.kernels_push.side_effect = failure
        with self.assertRaises(HTTPError) as raised:
            launch.push(api, directory)
        self.assertIs(failure, raised.exception)
        saved = read_json(directory / "plan.json")
        self.assertEqual("submission_ambiguous", saved["status"])
        self.assertEqual("ambiguous", saved["submission_attempt"]["state"])
        self.assertEqual("HTTPError: SaveKernel failed", saved["submission_attempt"]["error"])
        self.assertEqual({"status_code": 400,
            "request_ids": {"x-request-id": "request-123", "x-kaggle-request-id": "kaggle-456"},
            "response": {"code": 3, "status": "INVALID_ARGUMENT", "message": "Kernel source is invalid"}},
            saved["submission_attempt"]["http_error"])
        self.assertNotIn("private-", json.dumps(saved))
        with self.assertRaisesRegex(ValueError, "already attempted"):
            launch.push(api, directory)
        api.kernels_push.assert_called_once()

    def test_http_diagnostics_bound_redact_and_ignore_non_scalar_fields(self):
        response = Response()
        response.status_code = 400
        response.headers["X-Request-ID"] = "r" * 1000
        response._content = json.dumps({"status": {"request": "do not save"}, "code": [400],
            "error": {"status": "INVALID_ARGUMENT", "code": "c" * 1000,
                      "message": "fixture-secret\n" + "m" * 3000,
                      "request": "do not save"}}).encode()
        with patch.dict(launch.os.environ, {"KAGGLE_KEY": "fixture-secret"}):
            result = launch.submission_http_error(HTTPError("failed", response=response))
        self.assertEqual({"status": "INVALID_ARGUMENT", "code": "c" * 128,
                          "message": "[REDACTED] " + "m" * (1024 - 11)}, result["response"])
        self.assertEqual("r" * 256, result["request_ids"]["x-request-id"])
        self.assertNotIn("fixture-secret", json.dumps(result))

    def test_non_json_http_response_stays_ambiguous_without_saving_raw_body(self):
        directory, _ = self.build()
        response = Response()
        response.status_code = 502
        response.headers["X-Request-ID"] = "proxy-123"
        response._content = b"<html>private proxy error details</html>"
        api = Mock()
        api.kernels_push.side_effect = HTTPError("Bad gateway", response=response)
        with self.assertRaises(HTTPError):
            launch.push(api, directory)
        saved = read_json(directory / "plan.json")
        self.assertEqual("submission_ambiguous", saved["status"])
        self.assertEqual({"status_code": 502, "request_ids": {"x-request-id": "proxy-123"}},
                         saved["submission_attempt"]["http_error"])
        self.assertNotIn("private proxy", json.dumps(saved))
        with self.assertRaisesRegex(ValueError, "already attempted"):
            launch.push(api, directory)
        api.kernels_push.assert_called_once()

    def test_malformed_http_details_do_not_mask_original_failure(self):
        directory, _ = self.build()
        response = Mock(status_code=400)
        response.headers.items.side_effect = TypeError("unusable headers")
        response.json.side_effect = RuntimeError("unusable response")
        failure = HTTPError("original failure", response=response)
        api = Mock()
        api.kernels_push.side_effect = failure
        with self.assertRaises(HTTPError) as raised:
            launch.push(api, directory)
        self.assertIs(failure, raised.exception)
        saved = read_json(directory / "plan.json")
        self.assertEqual("submission_ambiguous", saved["status"])
        self.assertEqual({"status_code": 400}, saved["submission_attempt"]["http_error"])

    def test_wrong_remote_identity_is_ambiguous_never_accepted(self):
        directory, _ = self.build()
        api = Mock()
        api.kernels_push.return_value = {"ref": "/code/fixture/other", "url": "https://kaggle.com/x",
                                         "versionNumber": 1, "kernelId": 1}
        with self.assertRaisesRegex(RuntimeError, "identity"):
            launch.push(api, directory)
        self.assertEqual("submission_ambiguous", read_json(directory / "plan.json")["status"])

    def test_cli_push_and_status_reject_other_credential_owner(self):
        directory, _ = self.build()
        for action in ("push", "status"):
            with self.subTest(action=action), \
                    patch.object(sys, "argv", ["launch_temporal_kaggle.py", action, "--directory", str(directory)]), \
                    patch.object(launch, "authenticated_api", return_value=(Mock(), "other-owner")), \
                    patch.object(launch, action) as operation:
                with self.assertRaisesRegex(ValueError, "credentials owner"):
                    launch.main()
                operation.assert_not_called()

    def test_source_bundle_excludes_files_and_detects_credential_values(self):
        repo = self.root / "source"
        for name in (*launch.SCRIPTS, "src/example.py", "configs/taxonomy.yaml"):
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# fixture\n")
        write_json(repo / "kaggle-private.json", {"username": "private", "key": "secret_fixture_123"})
        (repo / "dataset.bin").write_bytes(b"not source")
        _, manifest = launch.source_bundle(repo)
        self.assertNotIn("kaggle-private.json", manifest["files"])
        self.assertNotIn("dataset.bin", manifest["files"])
        (repo / "src/example.py").write_text("# secret_fixture_123\n")
        with self.assertRaisesRegex(ValueError, "Unsafe source"):
            launch.source_bundle(repo)

    def test_incomplete_or_overlapping_extraction_cannot_build(self):
        for mutation in (lambda p: p["stages"][0].update(complete=False),
                         lambda p: p["stages"][0]["result"].update(completed_videos=43),
                         lambda p: p["stages"][0]["result"].update(partition_index=1)):
            with self.subTest(mutation=mutation):
                plan = baseline_plan()
                mutation(plan)
                with self.assertRaises(ValueError):
                    launch.extraction_sources(plan)


class TemporalRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="temporal-run-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.inputs, self.work = self.root / "input", self.root / "work"
        self.work.mkdir()
        annotations = self.inputs / "behaviorsense-toyota-meta/smarthome_CS_51.json"
        annotations.parent.mkdir(parents=True)
        write_json(annotations, {})
        self.annotations = annotations
        baseline = baseline_plan()
        self.job = {"owner": "fixture", "job_id": "fixture/temporal", "source_sha256": "new" * 20,
                    "extraction_source_sha256": baseline["source_sha256"],
                    "annotation_sha256": sha256(annotations), "session_seconds": 40500,
                    "candidates": copy.deepcopy(stage.CANDIDATES),
                    "feature_sources": launch.extraction_sources(baseline),
                    "baseline": {"validation_macro_ap": .5925371084457453,
                                 "pair_f1": .14488188976377953, "pair_precision": .10697674418604651}}
        for item in baseline["stages"]:
            mount = self.inputs / item["key"]
            (mount / "shards").mkdir(parents=True)
            write_json(mount / "stage_result.json", item["result"])
            for i in range(item["result"]["expected_videos"]):
                (mount / "shards" / f"{item['key']}-{i:04}.npz").write_bytes(b"synthetic feature fixture")
        self.now, self.commands, self.subsets = 0., [], []
        self.incomplete_at, self.timeout_at = None, None

    def artifacts(self, out, loss, seed, complete=True):
        out.mkdir(parents=True, exist_ok=True)
        score = .60 + .003 * (seed + (3 if loss == "asl" else 0))
        convergence = {"converged": complete, "completed_epochs": 45 if complete else 7}
        manifest = {"model": "temporal-fusion", "loss": loss, "seed": seed,
                    "full_eligible_window_coverage_verified": True, "test_loaded_during_training": False,
                    "train": {"subjects": stage.FIT_SUBJECTS, "n_windows": 217090, "n_videos": 274},
                    "validation": {"subjects": stage.VALIDATION_SUBJECTS, "n_windows": 63220, "n_videos": 77},
                    "checkpoint_selection": {"training_seed": seed, "source_split": "validation",
                        "threshold_source_split": "validation", "test_metrics_consulted": False,
                        "validation_comparison_fingerprint": "identical_validation_rows", "best_score": score},
                    "convergence": convergence}
        write_json(out / "manifest.json", manifest)
        write_json(out / "training_status.json", {"status": "complete" if complete else "incomplete_budget",
                   "inference_checkpoint_exported": complete, "convergence": convergence})
        (out / "last_multilabel.pt").write_bytes(b"checkpoint")
        if complete:
            (out / "best_multilabel.pt").write_bytes(f"{loss}-{seed}".encode())
            write_json(out / "thresholds.json", {"source": "validation", "thresholds": [.5]})
            write_json(out / "validation_metrics.json", {"macro_ap": score,
                       "tv_drinking_colabeled_windows": {"f1": .16, "precision": .12}})
            (out / "validation_logits.npz").write_bytes(b"validation predictions")

    def fake_child(self, args, timeout):
        args = list(map(str, args))
        self.commands.append((args, timeout))
        out = Path(args[args.index("--out") + 1])
        if args[2] == "evaluate-test":
            self.assertEqual(6, len([cmd for cmd, _ in self.commands if cmd[2] == "train"]))
            selection = read_json(self.work / "frozen_selection.json")
            checkpoint = Path(args[args.index("--checkpoint") + 1])
            out.mkdir(parents=True)
            write_json(out / "test_metrics.json", {"checkpoint_sha256": sha256(checkpoint),
                "thresholds_refit": False, "checkpoint_selection_on_test": False, "metrics": {"macro_ap": .01},
                "source_provenance": {"retained_starts": [0, 15, 30]}})
            self.assertEqual(selection["checkpoint_sha256"], sha256(checkpoint))
        else:
            loss, seed = args[args.index("--loss") + 1], int(args[args.index("--seed") + 1])
            if (loss, seed) == self.timeout_at:
                raise subprocess.TimeoutExpired(args, timeout)
            self.artifacts(out, loss, seed, complete=(loss, seed) != self.incomplete_at)
        self.now += 100

    def execute(self, preflight_elapsed=0):
        original = stage.feature_sources
        def sources(job, subset):
            if subset == "testing":
                self.assertTrue((self.work / "frozen_selection.json").exists())
                self.assertEqual(6, len(self.commands))
            self.subsets.append(subset)
            return original(job, subset, self.inputs)
        def boot(*args):
            self.now += preflight_elapsed
            return self.fake_child
        with ExitStack() as stack:
            stack.enter_context(patch.object(stage, "bootstrap", side_effect=boot))
            stack.enter_context(patch.object(stage, "dataset_root", return_value=self.annotations.parent))
            stack.enter_context(patch.object(stage, "feature_sources", side_effect=sources))
            stack.enter_context(patch.object(stage.time, "monotonic", side_effect=lambda: self.now))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return stage.execute(self.job, ROOT, self.work)

    def test_one_session_trains_all_six_before_any_test_loading(self):
        result = self.execute()
        self.assertEqual("complete", result["status"])
        self.assertEqual(["training", "testing"], self.subsets)
        self.assertTrue(result["all_six_converged"])
        self.assertTrue(result["improvement_gate_passed"])
        self.assertEqual(("asl", 2), (result["selection"]["loss"], result["selection"]["seed"]))
        self.assertEqual(.615, result["selection"]["validation_macro_ap"])
        self.assertEqual(.01, result["test"]["metrics"]["macro_ap"])
        self.assertNotIn("source_provenance", result["test"])
        self.assertIn("source_provenance", read_json(self.work / result["test_report_relative_path"]))
        self.assertEqual(sha256(self.work / "test/test_metrics.json"), result["test_report_sha256"])
        self.assertFalse(result["automatic_continuation"])
        for args, timeout in self.commands[:6]:
            self.assertNotIn("--resume", args)
            self.assertNotIn("testing", " ".join(args))
            self.assertLessEqual(timeout, 40500)

    def test_time_is_shared_equally_and_returned_to_remaining_candidates(self):
        self.execute()
        allocations = [float(args[args.index("--stop-after-seconds") + 1]) for args, _ in self.commands[:6]]
        self.assertEqual(6600, allocations[0])
        self.assertEqual((40500 - 100 - 900) / 5, allocations[1])
        self.assertEqual(40500 - 500 - 900, allocations[-1])

    def test_incomplete_candidate_prevents_selection_and_test(self):
        self.incomplete_at = ("bce", 1)
        result = self.execute()
        self.assertEqual("incomplete", result["status"])
        self.assertEqual(["training"], self.subsets)
        self.assertEqual(2, len(self.commands))
        self.assertFalse((self.work / "frozen_selection.json").exists())
        self.assertFalse(result["automatic_continuation"])

    def test_timeout_does_not_resubmit_or_read_test(self):
        self.timeout_at = ("bce", 0)
        result = self.execute()
        self.assertEqual("incomplete", result["status"])
        self.assertEqual(1, len(self.commands))
        self.assertEqual(["training"], self.subsets)
        self.assertEqual("process_timeout", result["candidates"][0]["training_status"])

    def test_preflight_can_exhaust_budget_without_starting_training(self):
        result = self.execute(preflight_elapsed=40000)
        self.assertEqual("incomplete", result["status"])
        self.assertEqual([], self.commands)

    def test_existing_training_cannot_be_resumed(self):
        (self.work / "candidates").mkdir()
        with self.assertRaisesRegex(ValueError, "never resumes"):
            self.execute()
        self.assertEqual([], self.commands)

    def test_modified_receipt_missing_shard_or_duplicate_input_fails(self):
        entry = self.job["feature_sources"][0]
        mount = self.inputs / "training-0"
        receipt = read_json(mount / "stage_result.json")
        receipt["source_sha256"] = "wrong"
        write_json(mount / "stage_result.json", receipt)
        with self.assertRaisesRegex(ValueError, "receipt changed"):
            stage.feature_sources(self.job, "training", self.inputs)

    def test_missing_feature_file_is_not_full_coverage(self):
        next((self.inputs / "training-0/shards").glob("*.npz")).unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete shard files"):
            stage.feature_sources(self.job, "training", self.inputs)

    def test_selection_requires_same_validation_and_complete_candidates(self):
        result = self.execute()
        mutated = copy.deepcopy(result["candidates"])
        mutated[0]["status"] = "incomplete"
        with self.assertRaises(ValueError):
            stage.select_candidate(self.job, mutated, {"sha256": "features"})
        mutated = copy.deepcopy(result["candidates"])
        mutated[0]["validation_comparison_fingerprint"] = "different"
        with self.assertRaisesRegex(ValueError, "different validation"):
            stage.select_candidate(self.job, mutated, {"sha256": "features"})

    def test_gate_rejects_map_only_gain_and_preserves_research_candidate(self):
        result = self.execute()
        candidates = result["candidates"]
        candidates[-1]["pair_precision"] = .01
        selected = stage.select_candidate(self.job, candidates, {"sha256": "features"})
        self.assertTrue(selected["gate_passed"])
        self.assertEqual(1, selected["seed"])
        for item in candidates:
            item["pair_f1"] = .01
        selected = stage.select_candidate(self.job, candidates, {"sha256": "features"})
        self.assertFalse(selected["gate_passed"])
        self.assertEqual(("asl", 2), (selected["loss"], selected["seed"]))
        self.assertFalse(selected["serving_model_replaced"])

    def test_changed_annotation_stops_before_loading_features(self):
        self.job["annotation_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "Annotation hash"):
            self.execute()
        self.assertEqual([], self.commands)
        self.assertEqual([], self.subsets)

    def test_runtime_refuses_cpu_fallback_and_wrong_accelerator(self):
        cuda = Mock()
        torch = SimpleNamespace(__version__="2.11.0+cu128", version=SimpleNamespace(cuda="12.8"), cuda=cuda)
        numpy = SimpleNamespace(__version__="2.1.3")
        job = {**self.job, "expected_core_runtime": launch.EXPECTED_CORE_RUNTIME}
        with patch.object(stage.sys, "version_info", (3, 13)), \
                patch.dict(sys.modules, {"torch": torch, "numpy": numpy}):
            cuda.is_available.return_value = False
            with self.assertRaisesRegex(RuntimeError, "CPU fallback"):
                stage.bootstrap(job, ROOT, self.work)
            cuda.is_available.return_value = True
            cuda.get_device_name.return_value = "Tesla T4"
            with self.assertRaisesRegex(RuntimeError, "RTX PRO 6000"):
                stage.bootstrap(job, ROOT, self.work)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    total = result.testsRun - len(result.skipped)
    passed = total - len(result.failures) - len(result.errors)
    print(f"{passed}/{total} passed; {len(result.skipped)} skipped")
    raise SystemExit(0 if result.wasSuccessful() else 1)
