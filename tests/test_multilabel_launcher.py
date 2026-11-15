"""Finite Kaggle orchestration: no network, credentials, or GPU required."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import launch_multilabel_kaggle as launch
from multilabel_kaggle_stage import mounted_sources
import multilabel_kaggle_stage as gpu_stage


class LauncherTests(unittest.TestCase):
    def plan(self):
        return {"owner": "synthetic", "stages": launch.planned_stages(), "selection": None}

    def test_transient_status_failure_retries_without_submitting_another_job(self):
        import requests
        with tempfile.TemporaryDirectory() as folder:
            plan = {"status": "running", "active": "synthetic/already-submitted"}
            calls, delays = [], []
            def transition(api, directory, current):
                calls.append(current["active"])
                if len(calls) == 1:
                    raise requests.exceptions.ConnectionError("temporary disconnect")
                current.update(status="complete", active=None)
                return False
            with patch.object(launch, "advance", side_effect=transition):
                launch.run_coordinator(None, Path(folder), plan, sleep=delays.append)
            self.assertEqual(["synthetic/already-submitted"] * 2, calls)
            self.assertEqual([60], delays)
            self.assertEqual("complete", plan["status"])
            self.assertNotIn("retry_error", plan)

    def test_permanent_failure_is_not_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            delays = []
            with patch.object(launch, "advance", side_effect=ValueError("source mismatch")):
                with self.assertRaisesRegex(ValueError, "source mismatch"):
                    launch.run_coordinator(None, Path(folder), {}, sleep=delays.append)
            self.assertEqual([], delays)

    def test_submission_rejection_cannot_become_a_source_mount(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = {**self.plan(), "active": None, "next_stage": 0, "status": "ready"}
            api = Mock()
            api.kernels_push.return_value = {"error": "submission rejected"}
            with patch.object(launch, "prepare_submission", return_value=("synthetic/preflight", root)):
                with self.assertRaisesRegex(RuntimeError, "submission rejected"):
                    launch.advance(api, root, plan)
            self.assertIsNone(plan["active"])
            self.assertEqual([], plan["stages"][0]["attempts"])
            self.assertEqual(1, len(plan["stages"][0]["rejected_submissions"]))

    def test_uncertain_submission_keeps_its_journaled_identity(self):
        import requests
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = {**self.plan(), "active": None, "next_stage": 0, "status": "ready"}
            api = Mock()
            api.kernels_push.side_effect = requests.exceptions.Timeout("response lost")
            with patch.object(launch, "prepare_submission", return_value=("synthetic/preflight", root)):
                with self.assertRaises(requests.exceptions.Timeout):
                    launch.advance(api, root, plan)
            self.assertEqual("synthetic/preflight", plan["active"])
            self.assertEqual(1, len(plan["stages"][0]["attempts"]))

    def test_schedule_has_full_corpus_and_test_after_selection(self):
        stages = launch.planned_stages()
        self.assertEqual(8, sum(s["stage"] == "extract" and s["subset"] == "training" for s in stages))
        self.assertEqual(4, sum(s["stage"] == "extract" and s["subset"] == "testing" for s in stages))
        pivot = next(i for i, s in enumerate(stages) if s["stage"] == "select")
        self.assertTrue(all(i > pivot for i, s in enumerate(stages) if s.get("subset") == "testing"))
        self.assertEqual(6, sum(s["stage"] == "train" for s in stages))

    def test_training_mounts_all_training_chunks_and_no_test(self):
        plan = self.plan()
        for stage in plan["stages"]:
            if stage["stage"] == "extract":
                stage.update(complete=True, attempts=[{"ref": f"synthetic/{stage['key']}-0", "result": {"status": "resumable"}},
                                                       {"ref": f"synthetic/{stage['key']}-1", "result": {"status": "complete"}}])
        stage = next(s for s in plan["stages"] if s["stage"] == "train")
        stage["attempts"] = [{"ref": "synthetic/old-training", "result": {"status": "resumable"}},
                             {"ref": "synthetic/latest-training", "result": {"status": "resumable"}}]
        sources = launch.stage_sources(plan, stage)
        self.assertEqual(17, len(sources))
        self.assertIn("synthetic/latest-training", sources)
        self.assertNotIn("synthetic/old-training", sources)
        self.assertFalse(any("xte" in ref for ref in sources))

    def test_missing_partition_prevents_training(self):
        plan = self.plan()
        stage = next(s for s in plan["stages"] if s["stage"] == "train")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            launch.stage_sources(plan, stage)

    def test_grouped_submission_preserves_source_and_checks_its_companion(self):
        import hashlib
        bundle, manifest = launch.source_bundle()
        runner = (ROOT / "scripts/grouped_multilabel_extract.py").read_bytes()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "source.b64").write_text(bundle)
            companion = root / "grouped_multilabel_extract.py"
            companion.write_bytes(runner)
            stage = {"key": "xtr04-07", "stage": "extract", "subset": "training",
                     "partition_count": 8, "partition_indices": [4, 5, 6, 7], "attempts": []}
            plan = {"owner": "synthetic", "stages": [stage], "source_manifest": manifest,
                    "source_sha256": manifest["sha256"], "runtime_dataset": "runtime",
                    "runtime_lock": {}, "annotation_sha256": "ann", "stage_seconds": 33000,
                    "group_runner_sha256": hashlib.sha256(runner).hexdigest()}
            ref, out = launch.prepare_submission(root, plan, stage)
            job = launch.read_json(out / "job.json")
            self.assertEqual(manifest["sha256"], job["source_sha256"])
            self.assertEqual([4, 5, 6, 7], job["partition_indices"])
            self.assertEqual(plan["group_runner_sha256"], job["group_runner_sha256"])
            body = (out / "training.py").read_text()
            compile(body, "grouped-training.py", "exec")
            self.assertIn("execute_group(JOB, REPO, WORK)", body)
            self.assertEqual(bundle, (root / "source.b64").read_text())
            companion.write_bytes(runner + b"\n# changed")
            with self.assertRaisesRegex(ValueError, "runner changed"):
                launch.prepare_submission(root, plan, stage)

    def test_grouped_sources_include_every_incremental_attempt_without_cancelled_job(self):
        plan = self.plan()
        completed = []
        for i in range(4):
            completed.append({"key": f"xtr{i:02d}", "stage": "extract", "subset": "training",
                              "complete": True, "attempts": [{"ref": f"synthetic/xtr{i:02d}", "result": {"status": "complete"}}]})
        group = {"key": "xtr04-07", "stage": "extract", "subset": "training", "complete": True,
                 "partition_indices": [4, 5, 6, 7],
                 "attempts": [{"ref": "synthetic/group-a00", "result": {"status": "resumable"}},
                              {"ref": "synthetic/group-a01", "result": {"status": "complete"}}]}
        train = {"key": "sk0", "stage": "train", "attempts": []}
        plan.update(stages=[*completed, group, train],
                    superseded_stages=[{"key": "xtr04", "attempts": [{"ref": "synthetic/cancelled"}]}])
        sources = launch.stage_sources(plan, train)
        self.assertEqual(6, len(sources))
        self.assertEqual(["synthetic/group-a00", "synthetic/group-a01"], sources[-2:])
        self.assertNotIn("synthetic/cancelled", sources)

    def test_mount_failure_is_never_used_as_a_resume_artifact(self):
        extraction = {"key": "data", "stage": "extract", "subset": "training", "complete": True,
                      "attempts": [{"ref": "synthetic/failed-data", "failure": {"kind": "mount"}},
                                   {"ref": "synthetic/good-data", "result": {"status": "complete"}}]}
        train = {"key": "sk0", "stage": "train",
                 "attempts": [{"ref": "synthetic/failed-sk0", "failure": {"kind": "mount"}}]}
        plan = {"stages": [extraction, train]}
        self.assertEqual(["synthetic/good-data"], launch.stage_sources(plan, train))
        train["attempts"].insert(0, {"ref": "synthetic/valid-checkpoint", "result": {"status": "resumable"}})
        self.assertEqual(["synthetic/good-data", "synthetic/valid-checkpoint"], launch.stage_sources(plan, train))
        extraction["attempts"] = [extraction["attempts"][0]]
        with self.assertRaisesRegex(ValueError, "validated complete output"):
            launch.stage_sources(plan, train)

    def test_feature_training_submission_omits_detector_weights_and_wheels(self):
        import hashlib
        bundle, manifest = launch.source_bundle()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "source.b64").write_text(bundle)
            # Source is supplied as a separately checked orchestration companion.
            runner = b"def execute_features(job, repo, work): pass\n"
            (root / "feature_training_stage.py").write_bytes(runner)
            extraction = {"key": "extract", "stage": "extract", "subset": "training", "complete": True,
                          "attempts": [{"ref": "synthetic/extract", "result": {"status": "complete"}}]}
            stage = {"key": "sk0", "stage": "train", "model": "skeleton", "seed": 0,
                     "attempts": [{"ref": "synthetic/failed-sk0", "failure": {"kind": "mount"}}]}
            plan = {"owner": "synthetic", "stages": [extraction, stage], "source_manifest": manifest,
                    "source_sha256": manifest["sha256"], "runtime_dataset": "runtime",
                    "runtime_lock": {}, "annotation_sha256": "ann", "stage_seconds": 33000,
                    "feature_runner_sha256": hashlib.sha256(runner).hexdigest(),
                    "expected_core_runtime": dict(launch.EXPECTED_CORE_RUNTIME)}
            ref, out = launch.prepare_submission(root, plan, stage)
            metadata = launch.read_json(out / "kernel-metadata.json")
            self.assertTrue(ref.endswith("sk0-a01"))
            self.assertEqual(["synthetic/behaviorsense-toyota-meta"], metadata["dataset_sources"])
            self.assertEqual(["synthetic/extract"], metadata["kernel_sources"])
            body = (out / "training.py").read_text()
            compile(body, "feature-training.py", "exec")
            self.assertIn("execute_features(JOB, REPO, WORK)", body)
            self.assertEqual(bundle, (root / "source.b64").read_text())
            (root / "feature_training_stage.py").write_bytes(runner + b"# changed")
            with self.assertRaisesRegex(ValueError, "runner changed"):
                launch.prepare_submission(root, plan, stage)

    def test_grouped_result_cannot_claim_incomplete_inventory_is_complete(self):
        from unittest.mock import Mock
        stage = {"key": "group", "stage": "extract", "subset": "training", "partition_count": 8,
                 "partition_indices": [4, 5, 6, 7], "expected_videos": 175,
                 "attempts": [{"ref": "synthetic/group"}]}
        plan = {"status": "running", "active": "synthetic/group", "next_stage": 0,
                "stages": [stage], "source_sha256": "source", "group_runner_sha256": "runner"}
        result = {"job_id": "synthetic/group", "source_sha256": "source", "status": "complete",
                  "partition_indices": [4, 5, 6, 7], "partition_count": 8, "subset": "training",
                  "orchestration_sha256": "runner", "scope_complete": True,
                  "completed_videos": 174, "expected_videos": 175}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out = root / "group" / "output"
            out.mkdir(parents=True)
            (out / "stage_result.json").write_text(json.dumps(result))
            api = Mock()
            api.kernels_status.return_value = {"status": "COMPLETE"}
            api.kernels_output.return_value = ([], None)
            with self.assertRaisesRegex(ValueError, "every expected video"):
                launch.advance(api, root, plan)
            self.assertFalse(stage.get("complete", False))

    def test_validation_selection_requires_all_converged_and_same_split(self):
        plan = self.plan()
        training = [s for s in plan["stages"] if s["stage"] == "train"]
        for i, stage in enumerate(training):
            stage.update(complete=True, result={"job_id": f"synthetic/{stage['key']}",
                "checkpoint_sha256": str(i), "manifest": {"convergence": {"converged": True},
                "checkpoint_selection": {"source_split": "validation", "test_metrics_consulted": False,
                    "best_score": i / 10, "validation_comparison_fingerprint": "same-targets"}}})
        chosen = launch.select_candidate(plan)
        self.assertEqual("context-fusion", chosen["model"])
        self.assertEqual(2, chosen["seed"])
        training[0]["result"]["manifest"]["convergence"]["converged"] = False
        with self.assertRaisesRegex(ValueError, "convergence"):
            launch.select_candidate(plan)
        training[0]["result"]["manifest"]["convergence"]["converged"] = True
        training[0]["result"]["manifest"]["checkpoint_selection"]["validation_comparison_fingerprint"] = "other"
        with self.assertRaisesRegex(ValueError, "validation targets"):
            launch.select_candidate(plan)

    def test_selection_flattens_two_seed_groups_and_keeps_checkpoint_identity(self):
        groups = []
        for model in ("skeleton", "context-fusion"):
            candidates = []
            for seed in (0, 1, 2):
                candidates.append({"model": model, "seed": seed, "status": "complete",
                    "checkpoint_sha256": f"{seed + (10 if model == 'context-fusion' else 0):064x}",
                    "checkpoint_relative_path": f"candidates/seed{seed}/best_multilabel.pt",
                    "manifest": {"convergence": {"converged": True}, "checkpoint_selection": {
                        "source_split": "validation", "test_metrics_consulted": False,
                        "best_score": .5 + .01 * seed + (.1 if model == "context-fusion" else 0),
                        "validation_comparison_fingerprint": "same-validation"}}})
            groups.append({"key": model, "stage": "train_group", "model": model, "seeds": [0, 1, 2],
                "complete": True, "result": {"job_id": f"synthetic/{model}-all", "candidates": candidates}})
        selected = launch.select_candidate({"stages": groups})
        self.assertEqual("synthetic/context-fusion-all", selected["job_id"])
        self.assertEqual(2, selected["seed"])
        self.assertEqual("candidates/seed2/best_multilabel.pt", selected["checkpoint_relative_path"])
        self.assertEqual(6, len(selected["candidates"]))
        groups[1]["result"]["candidates"][2]["seed"] = 1
        with self.assertRaisesRegex(ValueError, "six prespecified"):
            launch.select_candidate({"stages": groups})

    def test_seed_group_submission_embeds_both_runners_and_mounts_no_peer_model(self):
        import hashlib
        bundle, manifest = launch.source_bundle()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "source.b64").write_text(bundle)
            feature = b"def bootstrap(job, repo, work): pass\n"
            seeds = b"def execute_seed_group(job, repo, work): pass\n"
            (root / "feature_training_stage.py").write_bytes(feature)
            (root / "seed_training_stage.py").write_bytes(seeds)
            data = {"key": "data", "stage": "extract", "subset": "training", "complete": True,
                    "attempts": [{"ref": "synthetic/data", "result": {"status": "complete"}}]}
            skeleton = {"key": "sk-all", "stage": "train_group", "model": "skeleton", "seeds": [0, 1, 2],
                        "stage_seconds": 40500, "attempts": []}
            context = {"key": "ctx-all", "stage": "train_group", "model": "context-fusion", "seeds": [0, 1, 2],
                       "stage_seconds": 40500, "attempts": []}
            plan = {"owner": "synthetic", "stages": [data, skeleton, context], "source_manifest": manifest,
                    "source_sha256": manifest["sha256"], "runtime_dataset": "runtime", "runtime_lock": {},
                    "annotation_sha256": "ann", "stage_seconds": 33000,
                    "feature_runner_sha256": hashlib.sha256(feature).hexdigest(),
                    "seed_runner_sha256": hashlib.sha256(seeds).hexdigest(),
                    "expected_core_runtime": dict(launch.EXPECTED_CORE_RUNTIME)}
            for stage in (skeleton, context):
                ref, out = launch.prepare_submission(root, plan, stage)
                job = launch.read_json(out / "job.json")
                self.assertEqual(40500, job["stage_seconds"])
                self.assertEqual([0, 1, 2], job["seeds"])
                metadata = launch.read_json(out / "kernel-metadata.json")
                self.assertEqual(["synthetic/data"], metadata["kernel_sources"])
                self.assertEqual(["synthetic/behaviorsense-toyota-meta"], metadata["dataset_sources"])
                code = (out / "training.py").read_text()
                compile(code, "seed-group.py", "exec")
                self.assertIn("execute_seed_group(JOB, REPO, WORK)", code)
            (root / "seed_training_stage.py").write_bytes(seeds + b"# changed")
            with self.assertRaisesRegex(ValueError, "runner changed"):
                launch.prepare_submission(root, plan, context)

    def test_mounted_source_rejects_changed_code_and_ambiguous_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job = {"sources": ["synthetic/one"], "source_sha256": "code"}
            result = {"job_id": "synthetic/one", "source_sha256": "other", "status": "complete"}
            (root / "stage_result.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "snapshot"):
                mounted_sources(job, root)
            result["source_sha256"] = "code"
            (root / "stage_result.json").write_text(json.dumps(result))
            self.assertEqual({"synthetic/one"}, set(mounted_sources(job, root)))
            (root / "duplicate").mkdir()
            (root / "duplicate" / "stage_result.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                mounted_sources(job, root)

    def test_frozen_source_and_title_identity(self):
        bundle, manifest = launch.source_bundle()
        again, same = launch.source_bundle()
        self.assertEqual(bundle, again)
        self.assertEqual(manifest, same)
        self.assertFalse(any("kaggle-krishna" in name for name in manifest["files"]))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "source.b64").write_text(bundle)
            plan = {**self.plan(), "source_manifest": manifest, "source_sha256": manifest["sha256"],
                    "runtime_dataset": "runtime", "runtime_lock": {}, "annotation_sha256": "annotations",
                    "stage_seconds": 33000}
            ref, out = launch.prepare_submission(root, plan, plan["stages"][0])
            metadata = launch.read_json(out / "kernel-metadata.json")
            self.assertEqual(ref.split("/")[1], metadata["title"].lower().replace(" ", "-"))
            self.assertTrue(metadata["is_private"])
            self.assertFalse(metadata["enable_internet"])
            self.assertEqual("NvidiaRtxPro6000", metadata["machine_shape"])
            compile((out / "training.py").read_text(), "training.py", "exec")

    def test_production_train_and_evaluate_commands_match_real_cli(self):
        """Exercise the actual stage runner through its GPU/subprocess boundary."""
        import train_adl_multilabel as trainer
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ann = root / "annotations.json"
            ann.write_text("{}")
            work = root / "work"
            work.mkdir()
            sources = {"synthetic/extract": (root / "fit", {"stage": "extract", "subset": "training"})}
            job = {"owner": "synthetic", "job_id": "synthetic/fit", "source_sha256": "source",
                   "stage": "train", "model": "context-fusion", "seed": 2,
                   "annotation_sha256": gpu_stage.sha256(ann), "stage_seconds": 33000}
            commands = []

            def run(arguments):
                args = trainer.parser().parse_args([str(v) for v in arguments[2:]])
                commands.append(args)
                destination = Path(args.out)
                destination.mkdir(exist_ok=True)
                if args.command == "train":
                    self.assertTrue(args.require_full_coverage)
                    self.assertTrue(args.require_convergence)
                    self.assertEqual("context-fusion", args.model)
                    self.assertEqual(["p04", "p15"], args.val_subjects)
                    (destination / "best_multilabel.pt").write_bytes(b"synthetic-checkpoint")
                    (destination / "manifest.json").write_text(json.dumps({"convergence": {"converged": True}}))
                else:
                    self.assertTrue(args.require_full_coverage)
                    (destination / "test_metrics.json").write_text(json.dumps({"metrics": {"macro_ap": .5},
                        "checkpoint_selection_on_test": False, "thresholds_refit": False}))

            with patch.object(gpu_stage, "bootstrap", return_value=run), \
                 patch.object(gpu_stage, "mounted_sources", return_value=sources), \
                 patch.object(gpu_stage, "dataset_root", return_value=root), \
                 patch.object(gpu_stage, "asset", return_value=ann), \
                 patch.object(gpu_stage, "cache_directory", return_value=root / "cache"):
                gpu_stage.execute(job, ROOT, work)
            fit_result = launch.read_json(work / "stage_result.json")
            self.assertEqual("complete", fit_result["status"])
            eval_work = root / "eval"
            eval_work.mkdir()
            chosen = {"job_id": "synthetic/fit", "checkpoint_sha256": fit_result["checkpoint_sha256"]}
            eval_sources = {"synthetic/fit": (work, fit_result),
                            "synthetic/test": (root / "test-shards", {"stage": "extract", "subset": "testing"})}
            with patch.object(gpu_stage, "bootstrap", return_value=run), \
                 patch.object(gpu_stage, "mounted_sources", return_value=eval_sources), \
                 patch.object(gpu_stage, "dataset_root", return_value=root), \
                 patch.object(gpu_stage, "asset", return_value=ann), \
                 patch.object(gpu_stage, "cache_directory", return_value=root / "cache"):
                gpu_stage.execute({**job, "stage": "evaluate", "selection": chosen}, ROOT, eval_work)
            self.assertEqual("complete", launch.read_json(eval_work / "stage_result.json")["status"])
            self.assertEqual(["train", "evaluate-test"], [args.command for args in commands])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LauncherTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(f"{result.testsRun - len(result.errors) - len(result.failures)}/{result.testsRun} passed")
    raise SystemExit(not result.wasSuccessful())
