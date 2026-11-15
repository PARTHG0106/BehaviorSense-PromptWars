"""CPU checks for several frozen extraction partitions sharing one GPU session.

Only the GPU bootstrap/child boundary and clock are doubled. Mount lookup,
annotation loading, partitioning, CLI parsing, and manifest validation stay real.
All data and outputs are temporary test fixtures, never published artifacts.
"""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
import copy
from functools import partial
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import grouped_multilabel_extract as grouped  # noqa: E402
import multilabel_kaggle_stage as stage  # noqa: E402
from extract_multilabel_windows import (  # noqa: E402
    atomic_json, build_parser, completion_status, partition_videos, subset_videos,
)
from behaviorsense.data.toyota import load_tsu_annotations  # noqa: E402


class GroupedExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="grouped-extraction-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "input"
        self.work = self.root / "work"
        self.work.mkdir()
        weights = self.inputs / "behavioursense-ww"
        weights.mkdir(parents=True)
        for name in ("rtmo-l.onnx", "rtdetr-l.onnx"):
            (weights / name).write_bytes(b"test-only model boundary fixture")
        (self.inputs / "rgb-untrimmed").mkdir()
        meta = self.inputs / "datasets" / "fixture" / "behaviorsense-toyota-meta"
        meta.mkdir(parents=True)
        self.annotations = meta / "smarthome_CS_51.json"
        data = {f"P03T{take:02d}C01": {"subset": "training", "duration": 1000,
                "actions": [[19, 10, 900]]} for take in range(1, 17)}
        data["P02T01C01"] = {"subset": "testing", "duration": 1000, "actions": []}
        atomic_json(self.annotations, data)
        self.videos = subset_videos(load_tsu_annotations(self.annotations), "training")
        self.parts = partition_videos(self.videos, 8)
        self.config = {"selection_mode": "full", "windows_per_video": None,
                       "candidate_stride": 15, "decode_mode": "sequential"}
        self.fingerprint = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        self.job = {"job_id": "fixture/xtr04-07", "owner": "fixture", "stage": "extract",
                    "subset": "training", "partition_count": 8, "partition_indices": [4, 5, 6, 7],
                    "source_sha256": "frozen-original-source", "group_runner_sha256": "runner-sha",
                    "annotation_sha256": stage.sha256(self.annotations), "stage_seconds": 100,
                    "expected_videos": 8, "expected_config_fingerprint": self.fingerprint,
                    "sources": []}
        self.expected = {index: {video.vid for video in self.parts[index]}
                         for index in self.job["partition_indices"]}
        self.clock = 1000.0
        self.bootstrap_seconds = 7.0
        self.partition_seconds = {}
        self.completed_limits = {}
        self.commands = []

    def manifest(self, index, *, completed=None, job=None):
        job = job or self.job
        parts = partition_videos(self.videos, job["partition_count"])
        scope = parts[index]
        completed = {video.vid for video in scope} if completed is None else set(completed)
        reports = [{"video_id": video.vid, "retained_windows": 3,
                    "config_fingerprint": self.fingerprint}
                   for video in scope if video.vid in completed]
        return {"subset": job["subset"], "partition": {"index": index, "count": job["partition_count"]},
                "config": copy.deepcopy(self.config), "fingerprint": self.fingerprint,
                **completion_status(self.videos, scope, reports, "full"),
                "windows": sum(report["retained_windows"] for report in reports), "videos": reports}

    def write_manifest(self, out, index, *, completed=None, job=None, value=None):
        job = job or self.job
        out.mkdir(parents=True, exist_ok=True)
        suffix = (f"_part{index:03d}-of{job['partition_count']:03d}"
                  if job["partition_count"] > 1 else "")
        path = out / f"manifest_{job['subset']}{suffix}.json"
        atomic_json(path, value if value is not None else self.manifest(index, completed=completed, job=job))
        return path

    def fake_child(self, arguments):
        self.assertEqual("scripts/extract_multilabel_windows.py", arguments[1])
        args = build_parser().parse_args(list(map(str, arguments[2:])))
        self.commands.append(args)
        self.assertEqual("full", args.selection)
        self.assertIsNone(args.max_videos)
        self.assertEqual(15, args.candidate_stride)
        self.assertEqual("sequential", args.decode_mode)
        self.assertTrue(args.annotations.is_file())
        self.assertTrue(args.rtmo_model.is_file())
        self.assertTrue(args.context_model.is_file())
        part = partition_videos(self.videos, args.partition_count)[args.partition_index]
        count = self.completed_limits.get(args.partition_index, len(part))
        self.write_manifest(args.out, args.partition_index,
                            completed={video.vid for video in part[:count]})
        self.clock += self.partition_seconds.get(args.partition_index, 11.0)

    @contextmanager
    def runtime(self):
        def bootstrap(job, repo, work):
            self.assertEqual(ROOT, repo)
            self.assertEqual(self.work, work)
            self.clock += self.bootstrap_seconds
            return self.fake_child

        # Redirect lookup roots while retaining the production mount discovery and
        # identity checks. No annotations, hashes, partition maps, or summaries are mocked.
        with patch.object(grouped, "bootstrap", side_effect=bootstrap) as boot, \
             patch.object(grouped, "dataset_root", partial(stage.dataset_root, inputs=self.inputs)), \
             patch.object(grouped, "mounted_sources", partial(stage.mounted_sources, inputs=self.inputs)), \
             patch.object(grouped.time, "monotonic", side_effect=lambda: self.clock), \
             redirect_stdout(io.StringIO()):
            yield boot

    def execute(self):
        with self.runtime() as boot:
            grouped.execute_group(self.job, ROOT, self.work)
        return stage.read_json(self.work / "stage_result.json"), boot

    def resume_mount(self):
        ref = "fixture/previous-group-attempt"
        root = self.inputs / "previous-group-attempt"
        root.mkdir()
        previous = {"job_id": ref, "stage": "extract", "subset": "training",
                    "status": "resumable", "source_sha256": self.job["source_sha256"],
                    "orchestration_sha256": self.job["group_runner_sha256"],
                    "partition_count": 8, "partition_indices": [4, 5, 6, 7]}
        atomic_json(root / "stage_result.json", previous)
        self.write_manifest(root / "shards", 4)
        self.job["sources"] = [ref]
        return root, previous

    def test_four_partitions_run_sequentially_after_one_bootstrap_into_one_flat_directory(self):
        result, boot = self.execute()
        boot.assert_called_once_with(self.job, ROOT, self.work)
        self.assertEqual([4, 5, 6, 7], [args.partition_index for args in self.commands])
        self.assertEqual({self.work / "shards"}, {args.out for args in self.commands})
        self.assertEqual([93.0, 82.0, 71.0, 60.0],
                         [args.stop_after_seconds for args in self.commands])
        self.assertEqual("complete", result["status"])
        self.assertTrue(result["scope_complete"])
        self.assertEqual((8, 8, 24), (result["completed_videos"], result["expected_videos"], result["windows"]))
        self.assertEqual(self.fingerprint, result["config_fingerprint"])
        self.assertEqual(4, len(list((self.work / "shards").glob("manifest_*.json"))))
        group = stage.read_json(self.work / "group_manifest.json")
        for key in ("stage", "subset", "source_sha256", "orchestration_sha256",
                    "partition_count", "partition_indices", "scope_complete", "partitions"):
            self.assertEqual(result[key], group[key], key)
        self.assertNotIn("videos", result)
        self.assertNotIn("scope_video_ids", result)

    def test_resume_passes_previous_flat_mount_to_every_partition_without_mutating_it(self):
        mount, _ = self.resume_mount()
        before = {path.relative_to(mount): path.read_bytes() for path in mount.rglob("*") if path.is_file()}
        result, _ = self.execute()
        self.assertEqual("complete", result["status"])
        self.assertEqual([[mount / "shards"]] * 4, [args.resume_from for args in self.commands])
        after = {path.relative_to(mount): path.read_bytes() for path in mount.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_resume_rejects_a_different_group_or_original_source_before_any_child(self):
        mount, previous = self.resume_mount()
        for field, value in (("stage", "train"), ("subset", "testing"), ("partition_count", 4),
                             ("partition_indices", [0, 1, 2, 3]),
                             ("orchestration_sha256", "changed-runner"),
                             ("source_sha256", "changed-extractor")):
            with self.subTest(field=field):
                atomic_json(mount / "stage_result.json", {**previous, field: value})
                with self.runtime(), self.assertRaises(ValueError):
                    grouped.execute_group(self.job, ROOT, self.work)
                self.assertEqual("failed", stage.read_json(self.work / "stage_result.json")["status"])
                self.assertEqual([], self.commands)

    def test_global_budget_includes_bootstrap_and_prevents_later_partition_launches(self):
        self.job["stage_seconds"] = 50
        self.partition_seconds = {4: 20, 5: 22}
        result, _ = self.execute()
        self.assertEqual([4, 5], [args.partition_index for args in self.commands])
        self.assertEqual([43.0, 23.0], [args.stop_after_seconds for args in self.commands])
        self.assertEqual("resumable", result["status"])
        self.assertFalse(result["scope_complete"])
        self.assertEqual((4, 8), (result["completed_videos"], result["expected_videos"]))
        self.assertEqual([True, True, False, False], [p["scope_complete"] for p in result["partitions"]])

    def test_exhausted_bootstrap_budget_launches_no_partition_and_reports_resumable(self):
        self.job["stage_seconds"] = 5
        result, boot = self.execute()
        boot.assert_called_once()
        self.assertEqual([], self.commands)
        self.assertEqual("resumable", result["status"])
        self.assertEqual(0, result["completed_videos"])
        self.assertFalse(result["scope_complete"])

    def test_partial_partition_stops_the_group_and_cannot_claim_completion(self):
        self.completed_limits[4] = 1
        result, _ = self.execute()
        self.assertEqual([4], [args.partition_index for args in self.commands])
        self.assertEqual("resumable", result["status"])
        self.assertEqual("resume_required", result["phase"])
        self.assertEqual((1, 8, 3), (result["completed_videos"], result["expected_videos"], result["windows"]))
        self.assertFalse(result["scope_complete"])
        self.assertFalse(stage.read_json(self.work / "group_manifest.json")["scope_complete"])

    def test_inventory_uses_real_partitioner_and_excludes_testing_videos(self):
        self.assertEqual(self.expected, grouped.expected_partitions(self.annotations, self.job))
        self.assertNotIn("P02T01C01", set().union(*self.expected.values()))
        self.job["expected_videos"] = 9
        with self.runtime(), self.assertRaisesRegex(ValueError, "inventory"):
            grouped.execute_group(self.job, ROOT, self.work)
        self.assertEqual([], self.commands)

    def test_changed_annotation_bytes_fail_before_extraction(self):
        self.annotations.write_text(self.annotations.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.runtime(), self.assertRaisesRegex(ValueError, "Annotations"):
            grouped.execute_group(self.job, ROOT, self.work)
        self.assertEqual([], self.commands)

    def test_missing_partition_manifests_keep_the_group_incomplete(self):
        out = self.work / "shards"
        out.mkdir()
        empty = grouped.summarize_manifests(out, self.job, self.expected)
        self.assertFalse(empty["scope_complete"])
        self.assertEqual((0, 8, 0), (empty["completed_videos"], empty["expected_videos"], empty["windows"]))
        for index in (4, 5, 6):
            self.write_manifest(out, index)
        partial_summary = grouped.summarize_manifests(out, self.job, self.expected)
        self.assertFalse(partial_summary["scope_complete"])
        self.assertEqual(6, partial_summary["completed_videos"])
        self.assertFalse(partial_summary["partitions"][-1]["scope_complete"])

    def test_single_partition_uses_the_frozen_extractors_unsuffixed_manifest_name(self):
        job = {**self.job, "partition_count": 1, "partition_indices": [0], "expected_videos": 16}
        expected = {0: {video.vid for video in self.videos}}
        self.write_manifest(self.work / "shards", 0, job=job)
        summary = grouped.summarize_manifests(self.work / "shards", job, expected)
        self.assertTrue(summary["scope_complete"])
        self.assertEqual(16, summary["completed_videos"])

    def test_partition_indices_must_be_nonempty_sorted_unique_in_range_integers(self):
        self.assertEqual([4, 5, 6, 7], grouped.validate_group(self.job))
        for value in ([], [4, 4], [5, 4], [-1], [8], [True], [4.0], (4, 5), "4,5", None):
            with self.subTest(indices=value), self.assertRaises(ValueError):
                grouped.validate_group({**self.job, "partition_indices": value})
        for changes in ({"partition_count": 0}, {"partition_count": -1}, {"partition_count": True},
                        {"partition_count": 8.0}, {"stage": "train"}, {"subset": "all"},
                        {"stage_seconds": 0}, {"stage_seconds": 33001}, {"stage_seconds": float("nan")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                grouped.validate_group({**self.job, **changes})

    def test_tampered_scope_ids_counts_reports_or_partition_identity_are_rejected(self):
        manifest = self.manifest(4)
        ids = manifest["scope_video_ids"]
        outside = next(iter(self.expected[5]))
        changes = [
            {"scope_video_ids": [ids[0], outside]},
            {"scope_video_ids": [ids[0], ids[0]]},
            {"scope_expected_videos": 99},
            {"completed_video_ids": [ids[0], ids[0]]},
            {"completed_video_ids": [ids[0], outside]},
            {"completed_videos": 99},
            {"videos": [manifest["videos"][0], manifest["videos"][0]]},
            {"videos": manifest["videos"][:1]},
            {"windows": 99},
            {"subset": "testing"},
            {"partition": {"index": 5, "count": 8}},
            {"partition": {"index": 4, "count": 4}},
            {"config": {"selection_mode": "stratified", "windows_per_video": 64}},
            {"config": {"selection_mode": "full", "windows_per_video": 64}},
        ]
        for changed in changes:
            with self.subTest(changed=changed):
                self.write_manifest(self.work / "shards", 4, value={**manifest, **changed})
                with self.assertRaises(ValueError):
                    grouped.summarize_manifests(self.work / "shards", self.job, self.expected)

    def test_falsely_complete_partial_manifest_is_rejected(self):
        manifest = self.manifest(4, completed={next(iter(self.expected[4]))})
        manifest["scope_complete"] = True
        self.write_manifest(self.work / "shards", 4, value=manifest)
        with self.assertRaisesRegex(ValueError, "completion"):
            grouped.summarize_manifests(self.work / "shards", self.job, self.expected)

    def test_changed_fingerprint_is_rejected_against_frozen_experiment_and_other_partitions(self):
        altered = {**self.manifest(4), "fingerprint": "changed-config"}
        self.write_manifest(self.work / "shards", 4, value=altered)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            grouped.summarize_manifests(self.work / "shards", self.job, self.expected)
        self.write_manifest(self.work / "shards", 5)
        no_prior_fingerprint = {**self.job, "expected_config_fingerprint": None}
        with self.assertRaisesRegex(ValueError, "fingerprints differ"):
            grouped.summarize_manifests(self.work / "shards", no_prior_fingerprint, self.expected)

    def test_overlapping_partition_inventory_is_rejected_even_before_extraction(self):
        expected = {**self.expected, 5: self.expected[4]}
        with self.assertRaisesRegex(ValueError, "overlap"):
            grouped.summarize_manifests(self.work / "shards", self.job, expected)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    total = result.testsRun - len(result.skipped)
    passed = total - len(result.failures) - len(result.errors)
    print(f"{passed}/{total} passed; {len(result.skipped)} skipped")
    raise SystemExit(not result.wasSuccessful())
