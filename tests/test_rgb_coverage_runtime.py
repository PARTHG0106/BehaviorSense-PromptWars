"""Synthetic sparse timing, verified-cache accounting and gated execution tests."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import test_rgb_launcher as fixtures
import rgb_coverage_budget as budget
import rgb_coverage_contract as contract

stage = fixtures.stage


def digest(vid):
    return hashlib.sha256(vid.encode()).hexdigest()


def make_job():
    job = fixtures.fixture_job()
    original = list(job["pilot"]["pilot_video_ids"])
    additional = []
    for subject in stage.FIT_SUBJECTS:
        additional.extend(r["video_id"] for r in job["video_inventory"]
                          if r["subject"] == subject and r["video_id"] not in original)
        # Balance the synthetic additions, without using production metadata.
        additional = additional[:3 * (stage.FIT_SUBJECTS.index(subject) + 1)]
    additional = additional[:24]
    pilot = sorted(original + additional)
    probes = job["pilot"]["probe_video_ids"]
    sparse = [additional[0], additional[-1]]
    cache = []
    for r in job["video_inventory"]:
        if r["video_id"] in original or r["subject"] in stage.VALIDATION_SUBJECTS:
            cache.append({"video_id": r["video_id"], "n_windows": r["n_windows"],
                          "feature_sha256": digest(r["video_id"]),
                          "source_shard_sha256": r["shard_sha256"], "source_video_sha256": r["source_sha256"]})
    job.update(rgb_protocol=stage.PREFETCH_PROTOCOL, prefetch_batches=2, opencv_threads=1,
               encoder_preference=["swin3d_t"])
    job["pilot"]["pilot_video_ids"] = pilot
    job["coverage"] = {"schema": "synthetic.coverage", "probe_video_ids": probes + sparse,
                       "core_probe_video_ids": probes, "sparse_probe_video_ids": sparse,
                       "folds": dict(zip(("A", "B", "C"), [stage.FIT_SUBJECTS[i:i+3] for i in (0, 3, 6)])),
                       "cache_reuse": {"kernel_source": "fixture/old/1", "records": cache}}
    return job


def timing(row, work_per_batch=.003, decode_per_frame=.000002):
    frames = row["source_frames"]
    batches = (row["n_windows"] + 7) // 8
    def cpu(seconds, calls=1):
        return {"calls": calls, "failed_calls": 0, "wall_seconds": seconds,
                "cuda_seconds": None, "timing_mode": "cpu_wall_clock"}
    def cuda(seconds):
        return {"calls": batches, "failed_calls": 0, "wall_seconds": seconds,
                "cuda_seconds": seconds, "timing_mode": "cuda_events_and_synchronized_wall_clock"}
    return {"schema": "behaviorsense.rgb_stage_timings.v1", "stages": {
        "decode": cpu(frames * decode_per_frame, frames), "input_validation": cpu(.01),
        "serialize": cpu(.01), "scene_crop": cpu(batches * work_per_batch),
        "forward": cuda(batches * work_per_batch), "transfer": cuda(batches * work_per_batch)}}


def report(row, reused=False):
    value = {"video_id": row["video_id"], "subject": row["subject"], "status": "complete",
             "arch": "swin3d_t", "n_windows": row["n_windows"], "reused": reused,
             "output_sha256": digest(row["video_id"]), "source_shard_sha256": row["shard_sha256"],
             "source_video_sha256": row["source_sha256"], "decoded_frames": row["source_frames"],
             "elapsed_seconds": 1.}
    if reused:
        value["cache_reuse_verified"] = True
    else:
        value["coverage_timings"] = timing(row)
    return value


class CoverageBudgetTests(unittest.TestCase):
    def setUp(self):
        self.job = make_job()
        self.by_id = {r["video_id"]: r for r in self.job["video_inventory"]}
        self.probes = self.job["coverage"]["probe_video_ids"]
        self.fresh = [report(self.by_id[v]) for v in self.probes]
        self.reused = {r["video_id"]: report(self.by_id[r["video_id"]], True)
                       for r in self.job["coverage"]["cache_reuse"]["records"] if r["video_id"] not in self.probes}
        self.pilot = sorted(set(self.job["pilot"]["pilot_video_ids"]) | {
            v for v, r in self.by_id.items() if r["subject"] in stage.VALIDATION_SUBJECTS})

    def forecast(self, elapsed=100.):
        return budget.projected_coverage_budget(self.job, self.fresh, elapsed, self.pilot, self.by_id, self.reused)

    def test_exact_cache_discount_and_every_remaining_video_cost(self):
        value = self.forecast()
        self.assertTrue(value["passed"])
        self.assertEqual(110, value["verified_reused_videos"])
        self.assertEqual(22, value["remaining_pilot_videos"])
        self.assertEqual(421, value["remaining_full_videos"])
        rates = value["rates"]
        def cost(vid):
            r = self.by_id[vid]
            return rates["decode_seconds_per_frame"] * r["source_frames"] + rates["work_seconds_per_batch"] * ((r["n_windows"]+7)//8) + rates["overhead_seconds_per_video"]
        done = set(self.probes) | set(self.reused)
        self.assertAlmostEqual(100 + 600 + sum(cost(v) for v in set(self.pilot)-done), value["projected_pilot_seconds"])
        self.assertAlmostEqual(100 + 600 + 3600 + sum(cost(v) for v in set(self.by_id)-done), value["projected_job_seconds"])
        self.assertAlmostEqual(sum(cost(v) for v, r in self.by_id.items() if r["subset"] == "testing"), value["projected_test_seconds"])

    def test_single_window_clip_cost_uses_one_batch_not_a_per_window_multiplier(self):
        row = deepcopy(self.by_id[self.probes[-1]])
        row["n_windows"], row["source_frames"] = 1, 100000
        sparse = report(row)
        sparse["elapsed_seconds"] = .24
        costs = budget.probe_costs(sparse, row, 8)
        self.assertEqual(1, costs["batches"])
        self.assertAlmostEqual(.000002, costs["decode_seconds_per_frame"])
        self.assertAlmostEqual(.009, costs["work_seconds_per_batch"])
        self.assertGreaterEqual(costs["overhead_seconds_per_video"], .02)

    def test_more_decoded_frames_in_remaining_sparse_video_raise_budget(self):
        before = self.forecast()
        vid = next(v for v in self.pilot if v not in self.probes and v not in self.reused)
        self.by_id[vid]["source_frames"] += 1000000
        after = self.forecast()
        self.assertAlmostEqual(1000000*before["rates"]["decode_seconds_per_frame"],
                               after["projected_pilot_seconds"]-before["projected_pilot_seconds"])

    def test_all_probes_complete_fresh_ordered_and_validation_coverage_required(self):
        for kind in ("missing_probe", "reordered", "reused", "wrong_rows", "missing_validation", "duplicate_pilot"):
            self.setUp()
            if kind == "missing_probe": self.fresh.pop()
            elif kind == "reordered": self.fresh.reverse()
            elif kind == "reused": self.fresh[-1]["reused"] = True
            elif kind == "wrong_rows": self.fresh[-1]["n_windows"] += 1
            elif kind == "missing_validation": self.pilot.pop()
            else: self.pilot.append(self.pilot[0])
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.forecast()

    def test_missing_altered_unverified_or_test_cache_gets_no_credit(self):
        for kind in ("missing", "hash", "flag", "test"):
            self.setUp()
            key = next(iter(self.reused))
            if kind == "missing": self.reused.pop(key)
            elif kind == "hash": self.reused[key]["output_sha256"] = "0"*64
            elif kind == "flag": self.reused[key]["cache_reuse_verified"] = False
            else:
                vid = next(v for v,r in self.by_id.items() if r["subset"] == "testing")
                self.reused[vid] = report(self.by_id[vid], True)
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.forecast()

    def test_bad_timing_never_produces_a_budget_pass(self):
        for kind in ("missing", "failed", "nan", "cuda", "frames", "calls"):
            self.setUp()
            stages = self.fresh[-1]["coverage_timings"]["stages"]
            if kind == "missing": stages.pop("serialize")
            elif kind == "failed": stages["forward"]["failed_calls"] = 1
            elif kind == "nan": stages["decode"]["wall_seconds"] = float("nan")
            elif kind == "cuda": stages["forward"]["cuda_seconds"] = None
            elif kind == "frames": self.fresh[-1]["decoded_frames"] = 0
            else: stages["decode"]["calls"] -= 1
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.forecast()

    def test_exact_pilot_budget_boundary_and_one_second_over(self):
        estimate = self.forecast(elapsed=0.)["projected_pilot_seconds"]
        self.assertTrue(self.forecast(elapsed=4500-estimate)["pilot_fits"])
        self.assertFalse(self.forecast(elapsed=4501-estimate)["pilot_fits"])


class FakeProfile:
    def __init__(self, *args, **kwargs): self.row = None
    def summary(self): return timing(self.row)


class CoverageExecutionTests(unittest.TestCase):
    def setUp(self):
        self.run = fixtures.RGBExecutionTests()
        self.run.setUp()
        self.addCleanup(self.run.doCleanups)
        self.run.job = make_job()
        self.run.job["annotation_sha256"] = fixtures.sha256(self.run.annotations)
        self.run.by_id = {r["video_id"]: r for r in self.run.job["video_inventory"]}
        self.cache_failure = self.sparse_timeout = False
        self.invalid_sparse_timing = False
        original = self.run.fake_extract
        def extract(video, shard, output, encoder, info, batch_size, deadline, **kwargs):
            vid = Path(video).stem
            if self.sparse_timeout and vid in self.run.job["coverage"]["sparse_probe_video_ids"]:
                raise TimeoutError("synthetic sparse decode deadline")
            out = Path(output)
            value = original(video, shard, out, encoder, info, batch_size, deadline)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(vid, encoding="utf-8")
            value.update(output_sha256=digest(vid), output_path=str(out), decoded_frames=self.run.by_id[vid]["source_frames"])
            profile = kwargs.get("profile")
            if profile is not None:
                profile.row = self.run.by_id[vid]
                if self.invalid_sparse_timing and vid in self.run.job["coverage"]["sparse_probe_video_ids"]:
                    profile.summary = lambda: {"schema": "missing", "stages": {}}
            return value
        self.run.fake_extract = extract

    def reuse(self, job, work, shards, encoder_info, deadline):
        if self.cache_failure: raise ValueError("synthetic cache hash mismatch")
        self.run.now += 5
        probes = set(job["coverage"]["probe_video_ids"])
        result, records = {}, []
        for entry in job["coverage"]["cache_reuse"]["records"]:
            vid = entry["video_id"]
            records.append({"video_id": vid, "installed": vid not in probes,
                            "source_sha256_verified": True, "array_validation_verified": True})
            if vid in probes: continue
            row = self.run.by_id[vid]
            out = Path(work)/"rgb_features/swin3d_t"/(vid+".npz")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(vid, encoding="utf-8")
            result[vid] = {**report(row, True), "output_path": str(out),
                           "encoder_sha256": encoder_info["weights_sha256"],
                           "preprocessing_sha256": encoder_info["preprocessing_sha256"]}
        fixtures.write_json(Path(work)/"cache_reuse.json", {"status": "complete", "verified": True,
            "kernel_source": "fixture/old/1", "verified_source_files": 113, "installed_files": 110,
            "test_features_loaded": False, "fresh_probe_video_ids": sorted(probes), "records": records})
        return result

    def profile(self, job, work, encoder_info, run_variant, **kwargs):
        fresh, runs = [], []
        for vid in job["pilot"]["probe_video_ids"]:
            profile = FakeProfile()
            out = Path(work)/"rgb_features/swin3d_t"/(vid+".npz")
            r = run_variant("optimized", vid, out, kwargs["deadline"], profile)
            fresh.append(r)
            runs.append({"variant": "optimized", "video_id": vid, "timings": profile.summary(),
                         "elapsed_wall_seconds": r["elapsed_seconds"]})
        document = {"status": "complete", "all_three_equivalence_checks_passed": True,
                    "optimized_reports": fresh, "runs": runs}
        (Path(work)/"profiling/swin3d_t").mkdir(parents=True, exist_ok=True)
        fixtures.write_json(Path(work)/"profiling/swin3d_t/profile_report.json", document)
        return document

    def execute(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(contract, "validate_coverage_contract", side_effect=lambda job: job["coverage"]))
            stack.enter_context(patch.dict(sys.modules, {
                "rgb_cache_reuse": SimpleNamespace(prepare_reused_features=self.reuse),
                "rgb_profile_stage": SimpleNamespace(profile_probe_variants=self.profile, TimingProfile=FakeProfile),
                "cv2": SimpleNamespace(setNumThreads=lambda _: None)}))
            return self.run.execute()

    def test_success_reuses_110_freshly_times_five_and_preserves_all_gates(self):
        result = self.execute()
        self.assertEqual("complete", result["status"])
        self.assertEqual(3, len(self.run.commands))
        self.assertEqual(["training", "testing"], self.run.binds)
        choice = fixtures.read_json(self.run.work/"encoder_selection.json")
        self.assertEqual(self.run.job["coverage"]["probe_video_ids"], choice["fresh_probe_video_ids"])
        self.assertEqual(110, choice["projection"]["verified_reused_videos"])
        self.assertTrue(stage.coverage_probe_receipts_valid(self.run.job, self.run.work, choice))
        sparse = fixtures.read_json(self.run.work/"sparse_probe_report.json")
        self.assertTrue(all(r["reused"] is False for r in sparse["reports"]))
        self.assertEqual(426, len(self.run.extractions))
        full_calls = [x for x in self.run.extractions if x[1] not in self.run.job["pilot"]["pilot_video_ids"]
                      and self.run.by_id[x[1]]["subject"] not in stage.VALIDATION_SUBJECTS
                      and self.run.by_id[x[1]]["subset"] == "training"]
        expected_deadline = 40500-choice["projection"]["projected_test_seconds"]-3600
        self.assertTrue(all(abs(x[3]-expected_deadline) < 1e-8 for x in full_calls))

    def test_quality_failure_stops_after_137_video_pilot_with_27_fresh_extractions(self):
        self.run.pilot_pass = False
        result = self.execute()
        self.assertEqual("quality_rejected", result["status"])
        self.assertEqual(1, len(self.run.commands))
        self.assertEqual(["training"], self.run.binds)
        self.assertEqual(27, len(self.run.extractions))

    def test_cache_mismatch_stops_before_any_probe_or_head(self):
        self.cache_failure = True
        with self.assertRaisesRegex(ValueError, "cache hash"): self.execute()
        self.assertEqual([], self.run.extractions)
        self.assertEqual([], self.run.commands)

    def test_sparse_timeout_stops_before_heads_or_test(self):
        self.sparse_timeout = True
        result = self.execute()
        self.assertEqual("throughput_rejected", result["status"])
        self.assertEqual([], self.run.commands)
        self.assertEqual(["training"], self.run.binds)

    def test_incomplete_sparse_timing_cannot_pass_forecast(self):
        self.invalid_sparse_timing = True
        with self.assertRaisesRegex(ValueError, "timings"): self.execute()
        self.assertEqual([], self.run.commands)

    def test_parent_will_not_extend_deadline_with_missing_or_changed_receipts(self):
        self.run.pilot_pass = False
        self.execute()
        choice = fixtures.read_json(self.run.work/"encoder_selection.json")
        self.assertTrue(stage.coverage_probe_receipts_valid(self.run.job, self.run.work, choice))
        path = self.run.work/"sparse_probe_report.json"
        original = path.read_bytes()
        changed = json.loads(original)
        changed["completed_elapsed_seconds"] = 901
        fixtures.write_json(path, changed)
        self.assertFalse(stage.coverage_probe_receipts_valid(self.run.job, self.run.work, choice))
        path.write_bytes(original)
        cache_path = self.run.work/"cache_reuse.json"
        changed = fixtures.read_json(cache_path)
        changed["records"][0]["array_validation_verified"] = False
        fixtures.write_json(cache_path, changed)
        self.assertFalse(stage.coverage_probe_receipts_valid(self.run.job, self.run.work, choice))


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    total = result.testsRun-len(result.skipped)
    print(f"{total-len(result.failures)-len(result.errors)}/{total} passed; {len(result.skipped)} skipped")
    raise SystemExit(0 if result.wasSuccessful() else 1)
