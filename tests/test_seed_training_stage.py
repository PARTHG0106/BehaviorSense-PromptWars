"""Grouped orchestration tests using real filesystem mounts and trainer parsing."""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
from functools import partial
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import feature_training_stage as features  # noqa: E402
import multilabel_kaggle_stage as frozen  # noqa: E402
import seed_training_stage as group  # noqa: E402
from train_adl_multilabel import parser as trainer_parser  # noqa: E402


class SeedGroupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="seed-group-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.inputs, self.work, self.cache = (self.root / name for name in ("input", "work", "cache"))
        self.work.mkdir()
        meta = self.inputs / "datasets" / "fixture" / "behaviorsense-toyota-meta"
        meta.mkdir(parents=True)
        self.annotations = meta / "smarthome_CS_51.json"
        frozen.write_json(self.annotations, {})
        self.job = {"owner": "fixture", "job_id": "fixture/sk-group", "stage": "train_group",
                    "model": "skeleton", "seeds": [0, 1, 2], "source_sha256": "a" * 64,
                    "feature_runner_sha256": "b" * 64, "seed_runner_sha256": "c" * 64,
                    "expected_core_runtime": {"python": "3.13", "torch": "2.11.0+cu128"},
                    "stage_seconds": 40500, "annotation_sha256": frozen.sha256(self.annotations),
                    "sources": []}
        self.training_mount, _ = self.mount("extract-training", "extract", subset="training")
        self.commands, self.parsed, self.outcomes = [], [], {}
        self.now, self.child_seconds = 0., 100.

    def mount(self, name, stage, status="complete", **details):
        root = self.inputs / name
        root.mkdir(parents=True)
        result = {"job_id": f"fixture/{name}", "source_sha256": self.job["source_sha256"],
                  "stage": stage, "status": status, **details}
        frozen.write_json(root / "stage_result.json", result)
        if stage == "extract":
            (root / "shards").mkdir()
            (root / "shards" / "fixture.npz").write_bytes(b"feature fixture")
        self.job["sources"].append(result["job_id"])
        return root, result

    def artifacts(self, out, seed, outcome="complete"):
        out.mkdir(parents=True, exist_ok=True)
        converged, exhausted = outcome == "complete", outcome == "needs_review"
        convergence = {"converged": converged, "completed_epochs": 150 if exhausted else 50,
                       "epoch_budget": 150, "budget_exhausted": exhausted}
        manifest = {"model": self.job["model"], "checkpoint_selection": {"training_seed": seed},
                    "convergence": convergence, "train": {"subjects": ["p03"]},
                    "validation": {"subjects": ["p04", "p15"]},
                    "test_loaded_during_training": False,
                    "full_eligible_window_coverage_verified": True,
                    "provenance": {"large_fixture": True}}
        frozen.write_json(out / "manifest.json", manifest)
        frozen.write_json(out / "training_status.json", {
            "status": "complete" if converged else "epoch_budget_exhausted_without_convergence"
                      if exhausted else "resumable_session_pause",
            "convergence": convergence, "inference_checkpoint_exported": converged})
        (out / "last_multilabel.pt").write_bytes(f"resume seed {seed}".encode())
        if converged:
            (out / "best_multilabel.pt").write_bytes(f"inference seed {seed}".encode())
            (out / "validation_logits.npz").write_bytes(b"preserved inference artifact")

    def resume_mount(self, states=("complete", "resumable", "pending"), name="resume"):
        details = {key: self.job[key] for key in ("model", "seeds", "seed_runner_sha256",
                                                "feature_runner_sha256", "expected_core_runtime")}
        root, result = self.mount(name, "train_group", status="resumable", **details)
        entries = []
        for seed, state in enumerate(states):
            if state == "pending":
                entries.append({"seed": seed, "model": self.job["model"], "status": state})
            else:
                out = root / "candidates" / f"seed{seed}"
                self.artifacts(out, seed, state)
                entries.append(group.candidate_result(out, self.job["model"], seed))
        result["candidates"] = entries
        frozen.write_json(root / "stage_result.json", result)
        return root, result

    def fake_child(self, arguments):
        command = list(map(str, arguments))
        self.assertEqual("scripts/train_adl_multilabel.py", command[1])
        args = trainer_parser().parse_args(command[2:])
        self.commands.append(command)
        self.parsed.append(args)
        if args.resume:
            self.assertEqual(f"resume seed {args.seed}".encode(), Path(args.resume).read_bytes())
        outcome = self.outcomes.get(args.seed, "complete")
        if outcome == "error":
            raise RuntimeError("synthetic child failure")
        self.artifacts(Path(args.out), args.seed, outcome)
        self.now += self.child_seconds

    @contextmanager
    def runtime(self):
        with patch.object(group, "bootstrap", return_value=self.fake_child) as bootstrap, \
             patch.object(group, "dataset_root", side_effect=partial(frozen.dataset_root, inputs=self.inputs)), \
             patch.object(group, "mounted_sources", partial(frozen.mounted_sources, inputs=self.inputs)), \
             patch.object(group, "cache_directory", return_value=self.cache), \
             patch.object(group.time, "monotonic", side_effect=lambda: self.now), \
             redirect_stdout(io.StringIO()):
            yield bootstrap

    def execute(self):
        with self.runtime():
            group.execute_seed_group(self.job, ROOT, self.work)
        return frozen.read_json(self.work / "stage_result.json")

    def rejected(self, message, error=ValueError):
        with self.assertRaisesRegex(error, message):
            self.execute()
        self.assertEqual("failed", frozen.read_json(self.work / "stage_result.json")["status"])
        self.assertEqual([], self.commands)

    def test_three_seeds_sequential_and_complete_only_when_all_converged(self):
        result = self.execute()
        self.assertEqual([0, 1, 2], [args.seed for args in self.parsed])
        self.assertEqual("complete", result["status"])
        self.assertEqual(3, result["completed_seeds"])
        for key in ("model", "seeds", "stage", "source_sha256", "seed_runner_sha256",
                    "feature_runner_sha256", "expected_core_runtime"):
            self.assertEqual(self.job[key], result[key])

    def test_both_model_commands_preserve_every_original_scientific_argument(self):
        for model in ("skeleton", "context-fusion"):
            with self.subTest(model=model):
                self.job["model"] = model
                self.work = self.root / model
                self.commands.clear()
                self.parsed.clear()
                self.execute()
                actual = list(self.parsed)
                single = {**self.job, "stage": "train", "seed": 0}
                baseline_work = self.root / f"baseline-{model}"
                baseline_work.mkdir()
                with patch.object(features, "bootstrap", return_value=self.fake_child), \
                     patch.object(features, "dataset_root", side_effect=partial(frozen.dataset_root, inputs=self.inputs)), \
                     patch.object(features, "mounted_sources", partial(frozen.mounted_sources, inputs=self.inputs)), \
                     patch.object(features, "cache_directory", return_value=self.cache), \
                     redirect_stdout(io.StringIO()):
                    features.execute_features(single, ROOT, baseline_work)
                baseline = vars(self.parsed[-1])
                for args in actual:
                    changed = {"out", "seed", "cache_dir", "stop_after_seconds"}
                    self.assertEqual({key: value for key, value in baseline.items() if key not in changed},
                                     {key: value for key, value in vars(args).items() if key not in changed})
                    self.assertEqual(model == "context-fusion", args.require_context)

    def test_global_budget_decreases_between_seeds(self):
        self.execute()
        self.assertEqual([40500., 40400., 40300.], [args.stop_after_seconds for args in self.parsed])

    def test_session_budget_stops_with_pending_seeds_and_no_false_completion(self):
        self.child_seconds = 40501
        result = self.execute()
        self.assertEqual("resumable", result["status"])
        self.assertEqual(["complete", "pending", "pending"], [item["status"] for item in result["candidates"]])
        self.assertEqual(1, len(self.commands))

    def test_budget_spent_during_preflight_starts_no_child(self):
        with self.runtime() as boot:
            def preflight(*args):
                self.now = 40501
                return self.fake_child
            boot.side_effect = preflight
            group.execute_seed_group(self.job, ROOT, self.work)
        result = frozen.read_json(self.work / "stage_result.json")
        self.assertEqual("resumable", result["status"])
        self.assertEqual([], self.commands)

    def test_child_session_pause_keeps_exact_resume_and_later_seeds_pending(self):
        self.outcomes[1] = "resumable"
        result = self.execute()
        self.assertEqual("resumable", result["status"])
        self.assertEqual(["complete", "resumable", "pending"], [item["status"] for item in result["candidates"]])
        self.assertEqual([0, 1], [args.seed for args in self.parsed])
        self.assertIn("resume_checkpoint_sha256", result["candidates"][1])

    def test_epoch_budget_exhaustion_needs_review_while_other_seeds_run(self):
        self.outcomes[0] = "needs_review"
        result = self.execute()
        self.assertEqual("needs_review", result["status"])
        self.assertEqual(["needs_review", "complete", "complete"], [item["status"] for item in result["candidates"]])

    def test_complete_resume_skips_finished_seed_and_copies_all_artifacts(self):
        root, _ = self.resume_mount()
        before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
        result = self.execute()
        self.assertEqual([1, 2], [args.seed for args in self.parsed])
        self.assertEqual(str(self.work / "candidates" / "seed1" / "last_multilabel.pt"), self.parsed[0].resume)
        self.assertIsNone(self.parsed[1].resume)
        self.assertEqual("complete", result["status"])
        self.assertEqual(b"preserved inference artifact", (self.work / "candidates" / "seed0" / "validation_logits.npz").read_bytes())
        self.assertEqual(before, {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()})

    def test_all_complete_resume_never_retrains(self):
        self.resume_mount(("complete", "complete", "complete"))
        self.assertEqual("complete", self.execute()["status"])
        self.assertEqual([], self.commands)

    def test_resume_validates_all_identity_fields(self):
        root, value = self.resume_mount()
        for field, wrong in (("model", "context-fusion"), ("seeds", [2, 1, 0]),
                             ("seed_runner_sha256", "wrong"), ("feature_runner_sha256", "wrong"),
                             ("expected_core_runtime", {})):
            with self.subTest(field=field):
                frozen.write_json(root / "stage_result.json", {**value, field: wrong})
                self.rejected(f"Group resume {field} mismatch")

    def test_resume_changed_complete_checkpoint_is_rejected(self):
        root, _ = self.resume_mount()
        (root / "candidates" / "seed0" / "best_multilabel.pt").write_bytes(b"tampered")
        self.rejected("checkpoint mismatch")

    def test_resume_changed_partial_checkpoint_is_rejected(self):
        root, _ = self.resume_mount(("resumable", "pending", "pending"))
        (root / "candidates" / "seed0" / "last_multilabel.pt").write_bytes(b"tampered")
        self.rejected("checkpoint mismatch")

    def test_only_one_latest_group_attempt_is_allowed(self):
        self.resume_mount(name="one")
        self.resume_mount(name="two")
        self.rejected("Only the latest group")

    def test_unrequested_trial_mount_is_ignored(self):
        root, result = self.mount("unmatched-trial", "train", status="failed", model="context-fusion")
        self.job["sources"].remove(result["job_id"])
        self.assertEqual("complete", self.execute()["status"])

    def test_output_has_one_root_summary_compact_candidates_and_distinct_external_caches(self):
        result = self.execute()
        self.assertEqual([self.work / "stage_result.json"], list(self.work.rglob("stage_result.json")))
        self.assertEqual(3, len({args.cache_dir for args in self.parsed}))
        for item, args in zip(result["candidates"], self.parsed):
            self.assertNotIn("provenance", item["manifest"])
            checkpoint = self.work / item["checkpoint_relative_path"]
            self.assertEqual(frozen.sha256(checkpoint), item["checkpoint_sha256"])
            self.assertFalse(Path(args.cache_dir).is_relative_to(self.work))

    def test_nested_old_stage_summary_is_not_copied(self):
        root, _ = self.resume_mount()
        frozen.write_json(root / "candidates" / "seed0" / "stage_result.json", {"job_id": "unrelated"})
        self.execute()
        self.assertEqual([self.work / "stage_result.json"], list(self.work.rglob("stage_result.json")))

    def test_annotations_and_source_provenance_are_verified(self):
        original = self.annotations.read_bytes()
        self.annotations.write_bytes(original + b"\n")
        self.rejected("Annotations changed")
        self.annotations.write_bytes(original)
        path = self.training_mount / "stage_result.json"
        value = frozen.read_json(path)
        frozen.write_json(path, {**value, "source_sha256": "changed"})
        self.rejected("Source snapshot mismatch")

    def test_test_mounts_and_missing_training_shards_are_rejected(self):
        self.mount("testing", "extract", subset="testing")
        self.rejected("forbids test mounts")
        self.job["sources"] = []
        self.rejected("requires fitting shards")

    def test_stale_inference_checkpoint_cannot_fake_convergence(self):
        self.artifacts(self.work, 0, "resumable")
        (self.work / "best_multilabel.pt").write_bytes(b"stale inference")
        item = group.candidate_result(self.work, "skeleton", 0)
        self.assertEqual("resumable", item["status"])
        self.assertNotIn("checkpoint_sha256", item)

    def test_child_failure_is_recorded_and_preserves_known_completed_candidates(self):
        self.outcomes[1] = "error"
        with self.assertRaisesRegex(RuntimeError, "synthetic child failure"):
            self.execute()
        result = frozen.read_json(self.work / "stage_result.json")
        self.assertEqual("failed", result["status"])
        self.assertEqual("complete", result["candidates"][0]["status"])

    def test_invalid_group_and_excessive_budgets_are_rejected_before_bootstrap(self):
        for changes in ({"stage": "train"}, {"seeds": [0, 1]}, {"model": "other"},
                        {"stage_seconds": 43200}, {"stage_seconds": 0}, {"stage_seconds": float("inf")}):
            with self.subTest(changes=changes), self.runtime() as boot:
                with self.assertRaises(ValueError):
                    group.execute_seed_group({**self.job, **changes}, ROOT, self.work)
                boot.assert_not_called()


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    total = result.testsRun - len(result.skipped)
    passed = total - len(result.errors) - len(result.failures)
    print(f"{passed}/{total} passed; {len(result.skipped)} skipped")
    raise SystemExit(0 if result.wasSuccessful() else 1)
