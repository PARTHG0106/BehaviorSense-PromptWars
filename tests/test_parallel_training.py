"""Parallel notebook coordination without credentials, network access, or GPUs."""
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
from parallel_training_control import advance_parallel, validate_group_result


class FakeApi:
    def __init__(self):
        self.events = []
        self.statuses = {}
        self.outputs = {}
        self.push_error = {}
        self.before_push = None

    def kernels_push(self, folder, **kwargs):
        ref = "fixture/" + Path(folder).name
        self.events.append(("push", ref))
        if self.before_push:
            self.before_push(ref)
        if ref in self.push_error:
            value = self.push_error[ref]
            if isinstance(value, Exception):
                raise value
            return value
        assert kwargs == {"timeout": "43200", "acc": "NvidiaRtxPro6000"}
        return {"ref": ref}

    def kernels_status(self, ref):
        self.events.append(("status", ref))
        value = self.statuses.get(ref, {"status": "RUNNING"})
        if isinstance(value, Exception):
            raise value
        return value

    def kernels_output(self, ref, folder, **kwargs):
        self.events.append(("output", ref))
        if ref in self.outputs:
            (Path(folder) / "stage_result.json").write_text(json.dumps(self.outputs[ref]))
        return [], None


class ParallelTrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="parallel-training-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.groups = [
            {"key": "sk012", "stage": "train_group", "model": "skeleton",
             "seeds": [0, 1, 2], "attempts": []},
            {"key": "ctx012", "stage": "train_group", "model": "context-fusion",
             "seeds": [0, 1, 2], "attempts": []},
        ]
        self.plan = {"stages": [{"stage": "preflight", "complete": True}, *self.groups,
                                {"stage": "select"}, {"stage": "evaluate"}],
                     "next_stage": 1, "active": None, "active_jobs": [], "status": "ready",
                     "source_sha256": "a" * 64, "seed_runner_sha256": "b" * 64,
                     "feature_runner_sha256": "c" * 64, "selection": None}
        self.api = FakeApi()
        patcher = patch.object(launch, "prepare_submission", side_effect=self.prepare)
        patcher.start()
        self.addCleanup(patcher.stop)

    def prepare(self, directory, plan, stage):
        slug = f"{stage['key']}-a{len(stage['attempts']):02d}"
        return f"fixture/{slug}", directory / slug

    def advance(self):
        return advance_parallel(self.api, self.directory, self.plan)

    def saved(self):
        return json.loads((self.directory / "plan.json").read_text())

    def result(self, stage, *, status="complete"):
        candidates = [{"model": stage["model"], "seed": seed, "status": "complete",
                       "manifest": {"convergence": {"converged": True},
                                    "checkpoint_selection": {"source_split": "validation",
                                    "test_metrics_consulted": False, "best_score": seed / 10,
                                    "validation_comparison_fingerprint": "same-validation"}},
                       "checkpoint_sha256": str(seed + 1) * 64,
                       "checkpoint_relative_path": f"candidates/seed{seed}/best_multilabel.pt"}
                      for seed in (0, 1, 2)]
        if status in ("resumable", "needs_review"):
            candidates[1] = {"model": stage["model"], "seed": 1, "status": status,
                             "resume_checkpoint_sha256": "d" * 64,
                             "resume_checkpoint_relative_path": "candidates/seed1/last_multilabel.pt"}
            candidates[2] = {"model": stage["model"], "seed": 2, "status": "pending"}
        return {"job_id": stage["active"], "stage": "train_group", "model": stage["model"],
                "seeds": [0, 1, 2], "source_sha256": self.plan["source_sha256"],
                "seed_runner_sha256": self.plan["seed_runner_sha256"],
                "feature_runner_sha256": self.plan["feature_runner_sha256"],
                "status": status, "candidates": candidates}

    def complete(self, stage, *, status="complete"):
        ref = stage["active"]
        self.api.statuses[ref] = {"status": "COMPLETE"}
        self.api.outputs[ref] = self.result(stage, status=status)

    def test_both_submissions_are_durable_before_any_status_poll(self):
        def inspect_journal(ref):
            saved = self.saved()
            group = next(stage for stage in saved["stages"] if stage.get("active") == ref)
            self.assertEqual(ref, group["attempts"][-1]["ref"])
            self.assertFalse(group["attempts"][-1]["submitted"])
            self.assertIn(ref, saved["active_jobs"])
        self.api.before_push = inspect_journal
        self.assertTrue(self.advance())
        self.assertEqual(["push", "push"], [event[0] for event in self.api.events])
        self.assertEqual([stage["active"] for stage in self.groups], self.saved()["active_jobs"])
        self.assertEqual(self.groups[0]["active"], self.plan["active"])
        self.assertFalse(self.advance())
        self.assertEqual(["push", "push", "status", "status"],
                         [event[0] for event in self.api.events])

    def test_uncertain_first_push_still_submits_second_and_never_duplicates(self):
        import requests
        self.api.push_error["fixture/sk012-a00"] = requests.exceptions.Timeout("response lost")
        with self.assertRaises(requests.exceptions.Timeout):
            self.advance()
        self.assertEqual(2, len(self.plan["active_jobs"]))
        self.assertFalse(self.groups[0]["attempts"][0]["submitted"])
        self.assertTrue(self.groups[1]["attempts"][0]["submitted"])
        self.plan = self.saved()
        self.groups = self.plan["stages"][1:3]
        self.advance()
        self.assertEqual(2, sum(event[0] == "push" for event in self.api.events))

    def test_confirmed_rejection_is_not_a_resume_source_and_sibling_runs(self):
        self.api.push_error["fixture/sk012-a00"] = {"error": "capacity unavailable"}
        self.advance()
        self.assertEqual([], self.groups[0]["attempts"])
        self.assertEqual(1, len(self.groups[0]["rejected_submissions"]))
        self.assertIsNone(self.groups[0]["active"])
        self.assertIn("sk012", self.plan["parallel_errors"])
        self.assertEqual("fixture/ctx012-a00", self.groups[1]["active"])
        self.advance()
        self.assertEqual(2, sum(event[0] == "push" for event in self.api.events))

    def test_existing_remote_error_does_not_starve_unsubmitted_sibling(self):
        stage = self.groups[0]
        stage["active"] = "fixture/sk012-a00"
        stage["attempts"].append({"ref": stage["active"], "submitted": True})
        self.api.statuses[stage["active"]] = {"status": "ERROR", "failureMessage": "generic error"}
        self.advance()
        self.assertEqual(["push", "status"], [event[0] for event in self.api.events])
        self.assertIn("generic error", stage["error"])
        self.assertIn("failure", stage["attempts"][0])
        self.assertEqual("running", self.plan["status"])
        self.complete(self.groups[1])
        with self.assertRaisesRegex(RuntimeError, "requires review"):
            self.advance()
        self.assertTrue(self.groups[1]["complete"])
        self.assertEqual("needs_attention", self.saved()["status"])
        self.assertEqual(1, self.plan["next_stage"])
        self.assertIsNone(self.plan["selection"])

    def test_one_completion_waits_for_other_before_selection(self):
        self.advance()
        self.complete(self.groups[0])
        self.advance()
        self.assertEqual(2, self.plan["next_stage"])
        self.assertEqual([self.groups[1]["active"]], self.plan["active_jobs"])
        self.assertIsNone(self.plan["selection"])
        self.complete(self.groups[1])
        self.advance()
        self.assertEqual(3, self.saved()["next_stage"])
        self.assertEqual([], self.plan["active_jobs"])
        self.assertIsNone(self.plan["active"])
        selected = launch.select_candidate(self.plan)
        self.assertEqual(6, len(selected["candidates"]))
        self.assertEqual(6, len({(item["model"], item["seed"]) for item in selected["candidates"]}))
        self.assertEqual("candidates/seed2/best_multilabel.pt", selected["checkpoint_relative_path"])
        self.assertIn(selected["job_id"], [stage["result"]["job_id"] for stage in self.groups])

    def test_transient_status_failure_does_not_block_other_completion(self):
        import requests
        self.advance()
        self.api.statuses[self.groups[0]["active"]] = requests.exceptions.ConnectionError("offline")
        self.complete(self.groups[1])
        with self.assertRaises(requests.exceptions.ConnectionError):
            self.advance()
        self.assertTrue(self.groups[1]["complete"])
        self.assertIsNotNone(self.groups[0]["active"])
        self.api.statuses[self.groups[0]["active"]] = {"status": "RUNNING"}
        self.advance()
        self.assertNotIn("retry_error", self.groups[0])

    def test_delayed_output_does_not_block_other_completion(self):
        self.advance()
        self.api.statuses[self.groups[0]["active"]] = {"status": "COMPLETE"}
        self.complete(self.groups[1])
        with self.assertRaises(launch.OutputPendingError):
            self.advance()
        self.assertTrue(self.groups[1]["complete"])
        self.assertIsNotNone(self.groups[0]["active"])
        self.assertNotIn("result", self.groups[0]["attempts"][0])

    def test_resumable_group_gets_new_attempt_without_rerunning_complete_sibling(self):
        self.advance()
        self.complete(self.groups[0], status="resumable")
        self.complete(self.groups[1])
        self.advance()
        self.assertFalse(self.groups[0].get("complete", False))
        self.advance()
        self.assertEqual("fixture/sk012-a01", self.groups[0]["active"])
        self.assertEqual("resumable", self.groups[0]["attempts"][0]["result"]["status"])
        self.assertEqual(1, len(self.groups[1]["attempts"]))

    def test_resume_limit_stops_only_affected_group(self):
        self.groups[0]["attempts"] = [{"ref": f"fixture/old-{index}"} for index in range(16)]
        self.advance()
        self.assertIn("16 resume", self.groups[0]["error"])
        self.assertIsNotNone(self.groups[1]["active"])
        self.assertEqual(1, sum(event[0] == "push" for event in self.api.events))

    def test_invalid_completed_group_cannot_be_used_for_selection_or_resume(self):
        self.advance()
        self.complete(self.groups[0])
        self.api.outputs[self.groups[0]["active"]]["candidates"][0]["checkpoint_relative_path"] = "../bad.pt"
        self.complete(self.groups[1])
        with self.assertRaisesRegex(RuntimeError, "canonical"):
            self.advance()
        self.assertTrue(self.groups[1]["complete"])
        self.assertNotIn("result", self.groups[0]["attempts"][0])
        self.assertFalse(self.groups[0].get("complete", False))

    def test_result_validation_rejects_wrong_identity_hash_model_seed_or_convergence(self):
        self.advance()
        original = self.result(self.groups[0])
        edits = [
            lambda value: value.update(job_id="fixture/wrong"),
            lambda value: value.update(source_sha256="wrong"),
            lambda value: value.update(feature_runner_sha256="wrong"),
            lambda value: value.update(seed_runner_sha256="wrong"),
            lambda value: value.update(model="context-fusion"),
            lambda value: value["candidates"][1].update(seed=0),
            lambda value: value["candidates"][0].update(checkpoint_sha256="invalid"),
            lambda value: value["candidates"][0].update(status="pending"),
            lambda value: value["candidates"][0]["manifest"]["convergence"].update(converged=False),
        ]
        for index, edit in enumerate(edits):
            with self.subTest(index=index):
                value = copy.deepcopy(original)
                edit(value)
                with self.assertRaises(ValueError):
                    validate_group_result(self.groups[0], value, self.plan)

    def test_only_two_adjacent_prespecified_groups_are_allowed(self):
        self.groups[1]["seeds"] = [0, 1]
        with self.assertRaisesRegex(ValueError, "two adjacent"):
            self.advance()
        self.assertEqual([], self.api.events)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    total = result.testsRun - len(result.skipped)
    passed = total - len(result.errors) - len(result.failures)
    print(f"{passed}/{total} passed; {len(result.skipped)} skipped")
    raise SystemExit(0 if result.wasSuccessful() else 1)
