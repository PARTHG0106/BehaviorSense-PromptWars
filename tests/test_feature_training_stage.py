"""Feature-stage orchestration checks with real mount validation and trainer CLI.

Only GPU bootstrap and the child-process boundary are doubled. Temporary fixtures
exercise provenance, resume, and completion decisions without training a model.
"""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
import copy
from functools import partial
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import feature_training_stage as features  # noqa: E402
import multilabel_kaggle_stage as frozen  # noqa: E402
from train_adl_multilabel import parser as trainer_parser  # noqa: E402


class FeatureStageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="feature-stage-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.inputs = self.root / "input"
        self.work = self.root / "work"
        self.work.mkdir()
        self.cache = self.root / "cache"
        meta = self.inputs / "datasets" / "fixture" / "behaviorsense-toyota-meta"
        meta.mkdir(parents=True)
        self.annotations = meta / "smarthome_CS_51.json"
        frozen.write_json(self.annotations, {})
        self.job = {"owner": "fixture", "job_id": "fixture/sk0-a01", "stage": "train",
                    "model": "skeleton", "seed": 0, "source_sha256": "a" * 64,
                    "feature_runner_sha256": "b" * 64, "stage_seconds": 33000,
                    "annotation_sha256": frozen.sha256(self.annotations), "sources": []}
        self.commands, self.parsed = [], []
        self.child_outcome = "complete"
        self.training_mount, _ = self.mount("extract-training", "extract", subset="training")

    def mount(self, name, stage, *, status="complete", **details):
        root = self.inputs / name
        root.mkdir()
        result = {"job_id": f"fixture/{name}", "source_sha256": self.job["source_sha256"],
                  "stage": stage, "status": status, **details}
        frozen.write_json(root / "stage_result.json", result)
        if stage == "extract":
            (root / "shards").mkdir()
            (root / "shards" / "fixture.npz").write_bytes(b"child-boundary feature fixture")
        elif stage == "train":
            (root / "candidate").mkdir()
            (root / "candidate" / "last_multilabel.pt").write_bytes(b"fixture resume checkpoint")
            (root / "candidate" / "best_multilabel.pt").write_bytes(b"fixture selected checkpoint")
        self.job["sources"].append(result["job_id"])
        return root, result

    def resume_mount(self, name="resume", **details):
        return self.mount(name, "train", status=details.pop("status", "resumable"),
                          model=details.pop("model", "skeleton"), seed=details.pop("seed", 0), **details)

    def evaluation_job(self):
        self.job.update(stage="evaluate", job_id="fixture/test", sources=[])
        self.mount("extract-testing", "extract", subset="testing")
        selected, result = self.mount("selected", "train", model="skeleton", seed=0)
        self.job["selection"] = {"job_id": result["job_id"], "source_split": "validation",
                                 "test_consulted": False,
                                 "checkpoint_sha256": frozen.sha256(selected / "candidate" / "best_multilabel.pt")}
        return selected, result

    def evaluation_group_job(self):
        self.job.update(stage="evaluate", job_id="fixture/test", sources=[])
        self.mount("extract-testing", "extract", subset="testing")
        selected, result = self.mount("selected-group", "train_group", model="context-fusion")
        candidates = []
        for seed in (0, 1, 2):
            relative = f"candidates/seed{seed}/best_multilabel.pt"
            checkpoint = selected / relative
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(f"fixture context-fusion seed {seed} checkpoint".encode())
            candidates.append({"model": "context-fusion", "seed": seed, "status": "complete",
                               "manifest": {"convergence": {"converged": True}},
                               "checkpoint_relative_path": relative,
                               "checkpoint_sha256": frozen.sha256(checkpoint)})
        result["candidates"] = candidates
        frozen.write_json(selected / "stage_result.json", result)
        chosen = candidates[1]
        self.job["selection"] = {"job_id": result["job_id"], "model": chosen["model"], "seed": chosen["seed"],
                                 "source_split": "validation", "test_consulted": False,
                                 "checkpoint_sha256": chosen["checkpoint_sha256"],
                                 "checkpoint_relative_path": chosen["checkpoint_relative_path"]}
        return selected, result

    def fake_child(self, arguments):
        command = list(map(str, arguments))
        self.assertEqual("scripts/train_adl_multilabel.py", command[1])
        args = trainer_parser().parse_args(command[2:])
        self.commands.append(command)
        self.parsed.append(args)
        self.assertEqual("cuda", args.device)
        self.assertTrue(args.require_full_coverage)
        self.assertEqual(str(self.annotations), args.annotations)
        if self.child_outcome == "error":
            raise RuntimeError("synthetic child failure")
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        if args.command == "train":
            if args.resume:
                self.assertEqual(b"fixture resume checkpoint", Path(args.resume).read_bytes())
            exhausted = self.child_outcome == "needs_review"
            manifest = {"model": args.model, "train": {"subjects": ["p03"]},
                        "validation": {"subjects": ["p04", "p15"]},
                        "convergence": {"completed_epochs": 150 if exhausted else 50,
                                        "epoch_budget": 150, "budget_exhausted": exhausted},
                        "test_loaded_during_training": False,
                        "full_eligible_window_coverage_verified": True,
                        "provenance": {"full_per_video_fixture": True}}
            frozen.write_json(out / "manifest.json", manifest)
            if self.child_outcome == "complete":
                (out / "best_multilabel.pt").write_bytes(b"fixture trained checkpoint")
            elif self.child_outcome != "missing_checkpoint":
                (out / "last_multilabel.pt").write_bytes(b"fixture output checkpoint")
        else:
            frozen.write_json(out / "test_metrics.json", {
                "metrics": {"macro_ap": 0.5}, "checkpoint_sha256": self.job["selection"]["checkpoint_sha256"],
                "thresholds_refit": False, "checkpoint_selection_on_test": False, "split": "testing",
                "training_convergence": {"converged": True}, "source_provenance": {"fixture": True}})

    @contextmanager
    def runtime(self, module=features):
        with patch.object(module, "bootstrap", return_value=self.fake_child) as boot, \
             patch.object(module, "dataset_root", side_effect=partial(frozen.dataset_root, inputs=self.inputs)) as datasets, \
             patch.object(module, "mounted_sources", partial(frozen.mounted_sources, inputs=self.inputs)), \
             patch.object(module, "cache_directory", return_value=self.cache), \
             redirect_stdout(io.StringIO()):
            yield boot, datasets

    def execute(self, *, module=features, work=None):
        work = work or self.work
        work.mkdir(parents=True, exist_ok=True)
        with self.runtime(module) as (boot, datasets):
            execute = module.execute_features if module is features else module.execute
            execute(self.job, ROOT, work)
        return frozen.read_json(work / "stage_result.json"), boot, datasets

    def assert_rejected(self, message, error=ValueError):
        with self.runtime(), self.assertRaisesRegex(error, message):
            features.execute_features(self.job, ROOT, self.work)
        self.assertEqual([], self.commands)
        result = frozen.read_json(self.work / "stage_result.json")
        self.assertEqual("failed", result["status"])
        self.assertEqual(self.job["feature_runner_sha256"], result["feature_runner_sha256"])

    def test_training_needs_only_metadata_dataset_and_preserves_result_identity(self):
        result, boot, datasets = self.execute()
        boot.assert_called_once_with(self.job, ROOT, self.work)
        datasets.assert_called_once_with("fixture", "behaviorsense-toyota-meta")
        self.assertEqual("complete", result["status"])
        for key in ("job_id", "stage", "source_sha256", "feature_runner_sha256", "model", "seed"):
            self.assertEqual(self.job[key], result[key])
        self.assertEqual(frozen.sha256(self.work / "candidate" / "best_multilabel.pt"), result["checkpoint_sha256"])
        self.assertNotIn("provenance", result["manifest"])
        args = self.parsed[0]
        self.assertIsNone(args.init)
        self.assertIsNone(args.resume)
        self.assertEqual(["p04", "p15"], args.val_subjects)
        self.assertTrue(args.require_convergence)
        self.assertEqual([str(self.training_mount / "shards" / "*.npz")], args.train_shards)

    def test_both_model_commands_match_original_stage_with_and_without_resume(self):
        # Only the original orchestration needs this unused mount. The companion
        # lookup is asserted separately, so this cannot hide its mount regression.
        (self.inputs / "behavioursense-ww").mkdir()
        resume_root, resume_result = self.resume_mount()
        for model in ("skeleton", "context-fusion"):
            for resume in (False, True):
                with self.subTest(model=model, resume=resume):
                    self.job.update(model=model, sources=["fixture/extract-training"])
                    if resume:
                        resume_result["model"] = model
                        frozen.write_json(resume_root / "stage_result.json", resume_result)
                        self.job["sources"].append(resume_result["job_id"])
                    current = self.root / f"new-{model}-{resume}"
                    baseline = self.root / f"old-{model}-{resume}"
                    self.execute(work=current)
                    actual = self.commands[-1]
                    self.execute(module=frozen, work=baseline)
                    expected = self.commands[-1]
                    self.assertEqual([part.replace(str(baseline), "<WORK>") for part in expected],
                                     [part.replace(str(current), "<WORK>") for part in actual])
                    self.assertEqual(model == "context-fusion", self.parsed[-1].require_context)

    def test_resume_copies_candidate_without_mutating_source(self):
        mount, _ = self.resume_mount()
        before = {path.relative_to(mount): path.read_bytes() for path in mount.rglob("*") if path.is_file()}
        self.execute()
        self.assertEqual(str(self.work / "candidate" / "last_multilabel.pt"), self.parsed[0].resume)
        self.assertEqual(before, {path.relative_to(mount): path.read_bytes()
                                  for path in mount.rglob("*") if path.is_file()})

    def test_training_rejects_complete_resume_instead_of_rerunning_finished_candidate(self):
        self.resume_mount(status="complete")
        self.assert_rejected("safely resumable")

    def test_mount_failure_cannot_supply_a_resume_result(self):
        self.resume_mount(status="failed")
        self.assert_rejected("did not finish safely")

    def test_missing_mount_output_cannot_be_treated_as_a_fresh_resume(self):
        self.job["sources"].append("fixture/mount-failed-before-code")
        self.assert_rejected("Missing input stages", FileNotFoundError)

    def test_resume_requires_last_checkpoint(self):
        mount, _ = self.resume_mount()
        (mount / "candidate" / "last_multilabel.pt").unlink()
        self.assert_rejected("no last_multilabel.pt")

    def test_resume_model_and_seed_are_bound_to_job(self):
        mount, original = self.resume_mount()
        for field, changed in (("model", "context-fusion"), ("seed", 2)):
            with self.subTest(field=field):
                frozen.write_json(mount / "stage_result.json", {**original, field: changed})
                self.assert_rejected("model/seed mismatch")

    def test_multiple_resume_checkpoints_are_rejected(self):
        self.resume_mount("resume-one")
        self.resume_mount("resume-two")
        self.assert_rejected("Only the latest")

    def test_training_forbids_test_mounts_and_requires_training_shards(self):
        self.mount("extract-testing", "extract", subset="testing")
        self.assert_rejected("forbids test mounts")
        self.job["sources"] = []
        self.assert_rejected("requires fitting shards")

    def test_changed_annotations_or_original_source_fail_before_child(self):
        original = self.annotations.read_bytes()
        self.annotations.write_bytes(original + b"\n")
        self.assert_rejected("Annotations changed")
        self.annotations.write_bytes(original)
        result_path = self.training_mount / "stage_result.json"
        result = frozen.read_json(result_path)
        frozen.write_json(result_path, {**result, "source_sha256": "changed"})
        self.assert_rejected("Source snapshot mismatch")

    def test_training_completion_policy_preserves_resumable_and_budget_review(self):
        for outcome in ("resumable", "needs_review"):
            with self.subTest(outcome=outcome):
                self.child_outcome = outcome
                result, _, _ = self.execute(work=self.root / outcome)
                self.assertEqual(outcome, result["status"])
                self.assertNotIn("checkpoint_sha256", result)

    def test_missing_checkpoint_and_child_failure_are_recorded_as_failed(self):
        for outcome, message in (("missing_checkpoint", "did not save"), ("error", "synthetic child failure")):
            with self.subTest(outcome=outcome):
                self.child_outcome = outcome
                work = self.root / outcome
                with self.assertRaisesRegex(RuntimeError, message):
                    self.execute(work=work)
                self.assertEqual("failed", frozen.read_json(work / "stage_result.json")["status"])

    def test_evaluation_command_matches_original_and_preserves_frozen_selection(self):
        self.evaluation_job()
        result, _, datasets = self.execute()
        datasets.assert_called_once_with("fixture", "behaviorsense-toyota-meta")
        actual = self.commands[-1]
        self.assertEqual("complete", result["status"])
        self.assertFalse(result["serving_model_replaced"])
        self.assertFalse(result["test"]["thresholds_refit"])
        self.assertFalse(result["test"]["checkpoint_selection_on_test"])
        self.assertEqual(self.job["selection"], frozen.read_json(self.work / "frozen_selection.json"))
        (self.inputs / "behavioursense-ww").mkdir()
        baseline = self.root / "original-evaluate"
        self.execute(module=frozen, work=baseline)
        self.assertEqual([part.replace(str(baseline), "<WORK>") for part in self.commands[-1]],
                         [part.replace(str(self.work), "<WORK>") for part in actual])

    def test_evaluation_rejects_changed_selected_checkpoint(self):
        selected, _ = self.evaluation_job()
        (selected / "candidate" / "best_multilabel.pt").write_bytes(b"changed checkpoint")
        self.assert_rejected("Selected checkpoint changed")

    def test_evaluation_requires_complete_training_and_official_test_shards(self):
        selected, result = self.evaluation_job()
        frozen.write_json(selected / "stage_result.json", {**result, "status": "resumable"})
        self.assert_rejected("selected completed training stage")
        frozen.write_json(selected / "stage_result.json", result)
        self.job["sources"] = [result["job_id"]]
        self.assert_rejected("No official test shards")

    def test_group_evaluation_loads_exact_selected_seed_checkpoint(self):
        selected, _ = self.evaluation_group_job()
        result, _, datasets = self.execute()
        datasets.assert_called_once_with("fixture", "behaviorsense-toyota-meta")
        checkpoint = selected / "candidates" / "seed1" / "best_multilabel.pt"
        self.assertEqual(str(checkpoint), self.parsed[0].checkpoint)
        self.assertEqual(b"fixture context-fusion seed 1 checkpoint", checkpoint.read_bytes())
        self.assertEqual("complete", result["status"])
        self.assertEqual(self.job["selection"], frozen.read_json(self.work / "frozen_selection.json"))
        self.assertFalse(result["test"]["checkpoint_selection_on_test"])

    def test_group_evaluation_rejects_incomplete_group(self):
        selected, result = self.evaluation_group_job()
        frozen.write_json(selected / "stage_result.json", {**result, "status": "resumable"})
        self.assert_rejected("selected completed training stage")

    def test_group_evaluation_requires_unique_completed_matching_model_and_seed(self):
        selected, original = self.evaluation_group_job()
        variants = []
        for field, value in (("model", "skeleton"), ("seed", 8), ("status", "resumable")):
            result = copy.deepcopy(original)
            result["candidates"][1][field] = value
            variants.append((field, result))
        duplicated = copy.deepcopy(original)
        duplicated["candidates"].append(copy.deepcopy(duplicated["candidates"][1]))
        variants.append(("duplicate", duplicated))
        for name, result in variants:
            with self.subTest(change=name):
                frozen.write_json(selected / "stage_result.json", result)
                self.assert_rejected("one completed model/seed")

    def test_group_evaluation_rejects_noncanonical_and_traversal_paths(self):
        selected, original = self.evaluation_group_job()
        for path in ("../outside.pt", "/tmp/outside.pt", "C:/outside.pt",
                     "candidates/seed1/../../outside.pt", "candidates/seed0/best_multilabel.pt",
                     "candidates\\seed1\\best_multilabel.pt"):
            with self.subTest(path=path):
                self.job["selection"]["checkpoint_relative_path"] = path
                result = copy.deepcopy(original)
                result["candidates"][1]["checkpoint_relative_path"] = path
                frozen.write_json(selected / "stage_result.json", result)
                self.assert_rejected("path is not canonical")

    def test_group_evaluation_rejects_candidate_path_or_hash_that_differs_from_selection(self):
        selected, original = self.evaluation_group_job()
        for field, value in (("checkpoint_relative_path", "candidates/seed0/best_multilabel.pt"),
                             ("checkpoint_sha256", "0" * 64)):
            with self.subTest(field=field):
                result = copy.deepcopy(original)
                result["candidates"][1][field] = value
                frozen.write_json(selected / "stage_result.json", result)
                self.assert_rejected("path/hash differs")

    def test_group_evaluation_rejects_checkpoint_bytes_changed_after_selection(self):
        selected, _ = self.evaluation_group_job()
        (selected / self.job["selection"]["checkpoint_relative_path"]).write_bytes(b"changed checkpoint")
        self.assert_rejected("Selected checkpoint changed")

    def test_group_evaluation_rejects_resolved_checkpoint_outside_mount(self):
        selected, _ = self.evaluation_group_job()
        checkpoint = selected / self.job["selection"]["checkpoint_relative_path"]
        resolve = Path.resolve

        def redirected(path, *args, **kwargs):
            return self.root / "outside.pt" if path == checkpoint else resolve(path, *args, **kwargs)

        # Simulate a mount-provided symlink independently of Windows symlink privileges.
        with patch.object(Path, "resolve", redirected):
            self.assert_rejected("path escapes mounted output")

    def test_extraction_cannot_use_feature_only_bootstrap(self):
        self.job["stage"] = "extract"
        with self.runtime() as (boot, _), self.assertRaisesRegex(ValueError, "only supports"):
            features.execute_features(self.job, ROOT, self.work)
        boot.assert_not_called()


class CoreRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="feature-core-runtime-test-")
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.core = {"python": "3.13", "torch": "2.11.0+cu128", "cuda": "12.8", "numpy": "2.1.3"}
        self.job = {"source_sha256": "a" * 64, "feature_runner_sha256": "b" * 64,
                    "expected_core_runtime": dict(self.core)}
        self.cuda = SimpleNamespace(
            is_available=Mock(return_value=True),
            get_device_name=Mock(return_value="NVIDIA RTX PRO 6000 Blackwell Server Edition"),
            get_device_capability=Mock(return_value=(12, 0)),
            get_arch_list=Mock(return_value=["sm_120"]),
            get_device_properties=Mock(return_value=SimpleNamespace(total_memory=101973819392)))
        self.finite = Mock(return_value=True)
        self.torch = SimpleNamespace(__version__=self.core["torch"], version=SimpleNamespace(cuda="12.8"),
                                     cuda=self.cuda, ones=Mock(return_value=MagicMock()),
                                     isfinite=Mock(return_value=SimpleNamespace(all=self.finite)))
        self.numpy = SimpleNamespace(__version__=self.core["numpy"])

    @contextmanager
    def runtime(self, version=(3, 13, 15)):
        with patch.dict(sys.modules, {"torch": self.torch, "numpy": self.numpy}), \
             patch.object(features.sys, "version_info", version), \
             patch.object(features, "dataset_root", side_effect=AssertionError("No dataset is needed")), \
             patch.object(features.subprocess, "run") as child, \
             redirect_stdout(io.StringIO()):
            yield child

    def test_core_bootstrap_reports_versions_without_installing_packages_or_mounting_datasets(self):
        with self.runtime() as child:
            run = features.bootstrap(self.job, ROOT, self.work)
            child.assert_not_called()
            run(["python", "training.py", Path("features")])
            child.assert_called_once_with(["python", "training.py", "features"], cwd=ROOT, check=True)
        report = frozen.read_json(self.work / "environment.json")
        self.assertEqual(self.core, report["core_runtime"])
        self.assertFalse(report["extraction_runtime_installed"])
        self.assertEqual(self.job["feature_runner_sha256"], report["feature_runner_sha256"])
        self.torch.ones.assert_called_once_with((64, 64), device="cuda")
        self.finite.assert_called_once_with()

    def test_core_version_expectations_are_checked_exactly_when_supplied(self):
        for key in self.core:
            with self.subTest(key=key):
                self.job["expected_core_runtime"] = {**self.core, key: "changed"}
                with self.runtime(), self.assertRaisesRegex(ValueError, "Core runtime differs"):
                    features.bootstrap(self.job, ROOT, self.work)
        self.job.pop("expected_core_runtime")
        with self.runtime():
            features.bootstrap(self.job, ROOT, self.work)

    def test_unverified_python_version_is_rejected(self):
        with self.runtime(version=(3, 14, 0)), self.assertRaisesRegex(RuntimeError, "Python 3.13"):
            features.bootstrap(self.job, ROOT, self.work)

    def test_cuda_device_and_native_blackwell_support_are_required(self):
        checks = ((self.cuda.is_available, False, "CUDA unavailable"),
                  (self.cuda.get_device_name, "Other GPU", "authorized RTX PRO 6000"),
                  (self.cuda.get_device_capability, (9, 0), "authorized RTX PRO 6000"),
                  (self.cuda.get_arch_list, ["sm_90"], "native Blackwell support"),
                  (self.finite, False, "CUDA arithmetic check failed"))
        for check, value, message in checks:
            with self.subTest(message=message):
                original = check.return_value
                check.return_value = value
                with self.runtime(), self.assertRaisesRegex(RuntimeError, message):
                    features.bootstrap(self.job, ROOT, self.work)
                check.return_value = original


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    total = result.testsRun - len(result.skipped)
    passed = total - len(result.failures) - len(result.errors)
    print(f"{passed}/{total} passed; {len(result.skipped)} skipped")
    raise SystemExit(0 if result.wasSuccessful() else 1)
