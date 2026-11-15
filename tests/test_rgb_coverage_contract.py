"""Synthetic contract checks; no corpus, private artifacts, network or arrays.

The generated C99/C98/C97 records below are artificial. Only the public protocol
cardinalities and taxonomy-mask positions are retained; no experimental video
selection, per-video counts, source hashes or derived label facts are fixtures.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rgb_coverage_contract as contract


def synthetic_job():
    subjects = contract.FIT_SUBJECTS
    supported = [0, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21]
    names = ["class_" + str(c) for c in range(22)]
    names[10], names[13] = "drinking", "watching_tv"
    core = [f"P{s[1:]}T{take:02}C99" for s in subjects for take in range(1, 5)]
    old_extra = [f"P{subjects[i % 9][1:]}T{50 + i // 9:02}C99" for i in range(20)]
    additions = sorted(f"P{s[1:]}T{80 + i:02}C99" for i, s in enumerate((*subjects[6:], subjects[0])))
    selected = sorted(core + old_extra + additions)
    original = sorted(core[:16] + old_extra)
    additional = sorted(set(selected) - set(original))
    pairs = core[:24] + additions
    selected_counts = {vid: 669 + (i < 53) for i, vid in enumerate(selected)}
    sparse_id, frame_id = core[32:34]
    removed = selected_counts[sparse_id] - 1
    selected_counts[sparse_id] = 1
    selected_counts[original[0]] += removed
    remainder = [f"P{subjects[i % 9][1:]}T{20 + i // 9:02}C99" for i in range(214)]
    residual = 217090 - sum(selected_counts.values())
    counts = {**selected_counts, **{vid: residual // 214 + (i < residual % 214)
                                   for i, vid in enumerate(remainder)}}
    facts = []
    for vid, n in sorted(counts.items()):
        both = (44 if vid == pairs[-1] else 22) if vid in pairs else 0
        states = [n - both - 2, 1, 1, both] if n > 1 else [1, 0, 0, 0]
        positive = [int(c in supported) for c in range(22)]
        positive[10], positive[13] = states[1] + both, states[2] + both
        known = [n if c in supported else 0 for c in range(22)]
        frames = 1000000 if vid == sparse_id else 500000 if vid == frame_id else n * 25 + 10
        facts.append({
            "video_id": vid, "subject": "p" + vid[1:3], "take_group": vid[:6], "n_windows": n,
            "positive_counts": positive, "known_counts": known,
            "negative_counts": [k - p for k, p in zip(known, positive)], "four_state_counts": states,
            "source_ref": "fixture/training", "source_sha256": contract.canonical_hash(["source", vid]),
            "source_bytes": 100, "source_frame_count": frames, "source_fps": 25., "duration_seconds": frames / 25,
            "shard_filename": f"toyota_multilabel_20hz_training_{vid}.npz",
            "shard_sha256": contract.canonical_hash(["shard", vid]),
            "retained_starts_sha256": contract.canonical_hash(["starts", vid]),
            "extraction_metadata_sha256": contract.canonical_hash(["metadata", vid]),
        })
    video_inventory = [{**{key: v[key] for key in ("video_id", "subject", "n_windows", "source_ref", "source_sha256",
                       "source_bytes", "source_fps", "duration_seconds", "shard_filename", "shard_sha256")},
                        "source_frames": v["source_frame_count"], "source_filename": v["video_id"] + ".mp4",
                        "subset": "training", "pair_windows": 0} for v in facts]
    for subset, members, size, total, camera in (
        ("training", contract.VALIDATION_SUBJECTS, 77, 63220, 98),
        ("testing", ("p02", "p05", "p08", "p10", "p11", "p14", "p16", "p18", "p20"), 185, 153537, 97),
    ):
        for i in range(size):
            s = members[i % len(members)]
            vid = f"P{s[1:]}T{1 + i // len(members):02}C{camera}"
            video_inventory.append({"video_id": vid, "subject": s, "subset": subset,
                "n_windows": total // size + (i < total % size), "source_ref": "fixture/" + subset,
                "source_sha256": contract.canonical_hash(["source", vid]), "source_filename": vid + ".mp4",
                "source_bytes": 100, "source_frames": 25000, "source_fps": 25., "duration_seconds": 1000.,
                "shard_filename": f"toyota_multilabel_20hz_{subset}_{vid}.npz",
                "shard_sha256": contract.canonical_hash(["shard", vid]), "pair_windows": 0})
    feature_contract = {"schema": "behaviorsense.rgb_action_features.v1", "arch": "swin3d_t", "feature_dim": 768,
                        "views": ["scene", "person"], "encoder_sha256": "a" * 64,
                        "preprocessing_sha256": "b" * 64, "extraction_implementation_sha256": "c" * 64}
    records = []
    for source in sorted(video_inventory, key=lambda v: v["video_id"]):
        vid = source["video_id"]
        if vid not in original and source["subject"] not in contract.VALIDATION_SUBJECTS:
            continue
        records.append({"video_id": vid, "subject": source["subject"], "subset": "training",
            "role": "fitting" if vid in original else "validation", "n_windows": source["n_windows"],
            "source_video_sha256": source["source_sha256"], "source_shard_sha256": source["shard_sha256"],
            "feature_sha256": contract.canonical_hash(["features", vid]), "fingerprint": contract.canonical_hash(["fingerprint", vid]),
            "relative_path": f"rgb_features/swin3d_t/{vid}.npz",
            **{key: feature_contract[key] for key in ("arch", "feature_dim", "encoder_sha256", "preprocessing_sha256",
                                                    "extraction_implementation_sha256")}})
    inventory = [{key: v[key] for key in ("video_id", "n_windows", "feature_sha256", "source_video_sha256",
                                         "source_shard_sha256", "fingerprint")} for v in records]
    core_probes = [core[i] for i in (4, 8, 12)]
    feature_sources = [{"ref": "fixture/training", "subset": "training", "receipt_sha256": "d" * 64}]
    coverage = {
        "schema": contract.SCHEMA, "plan_sha256": contract.PLAN_SHA256, "inventory_sha256": contract.INVENTORY_SHA256,
        "core_selection_sha256": contract.SELECTION_SHA256, "recipe_sha256": contract.RECIPE_SHA256,
        "original_video_ids": original, "core_video_ids": core, "pilot_video_ids": selected,
        "pair_coverage_additions": additions, "additional_rgb_video_ids": additional,
        "core_probe_video_ids": core_probes, "sparse_probe_video_ids": [sparse_id, frame_id],
        "probe_video_ids": core_probes + [sparse_id, frame_id],
        "folds": {label: list(subjects[i::3]) for i, label in enumerate("ABC")},
        "class_names": names, "supervised_class_ids": supported, "fitting_inventory": facts,
        "source_checks": {k: "e" * 64 for k in ("annotation_sha256", "extraction_source_sha256", "assets_manifest_sha256")},
        "forecast_policy": deepcopy(contract.FORECAST_POLICY),
        "video_inventory_sha256": contract.canonical_hash(sorted(video_inventory, key=lambda v: v["video_id"])),
        "feature_sources_sha256": contract.canonical_hash(feature_sources),
        "cache_reuse": {"kernel_source": contract.KERNEL_SOURCE, "source_job_id": contract.KERNEL_SOURCE.rsplit("/", 1)[0],
            "feature_contract": feature_contract, "feature_contract_sha256": contract.canonical_hash(feature_contract),
            "feature_inventory_sha256": contract.canonical_hash(inventory), "records": records,
            "provenance_files": [{"relative_path": path, "sha256": "f" * 64} for path in
                                 ("job.json", "source_manifest.json", "extraction_swin3d_t.json", "encoder_selection.json",
                                  "stage_result.json", "pilot/manifest.json")]},
    }
    coverage["coverage_summary"] = contract._summary(coverage, {v["video_id"]: v for v in facts})
    coverage["pilot"] = {"pilot_video_ids": selected, "probe_video_ids": core_probes,
                         "n_fit_videos": 60, "fit_windows": 40193, "pair_windows": 638}
    coverage["contract_sha256"] = contract.canonical_hash(coverage)
    return {**deepcopy(contract.FROZEN_BUDGET), **coverage["source_checks"], "gate_criteria": deepcopy(contract.GATE_CRITERIA),
            "video_inventory": video_inventory, "feature_sources": feature_sources,
            "pilot": deepcopy(coverage["pilot"]), "coverage": coverage}


class CoverageContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_job = synthetic_job()
        cls.coverage = cls.original_job["coverage"]

    def setUp(self):
        # The production digest anchors private reviewed evidence. Unit tests
        # substitute only that root digest, then exercise the real validators.
        self.enterContext(patch.object(contract, "FROZEN_PAYLOAD_SHA256", self.coverage["contract_sha256"]))

    def job(self):
        job = deepcopy(self.original_job)
        return job

    def assert_semantic_rejection(self, job, message):
        """Exercise recomputation separately from the immutable content anchor."""
        coverage = job["coverage"]
        coverage["contract_sha256"] = contract.canonical_hash({k: v for k, v in coverage.items()
                                                              if k != "contract_sha256"})
        with patch.object(contract, "FROZEN_PAYLOAD_SHA256", coverage["contract_sha256"]):
            with self.assertRaisesRegex(ValueError, message):
                contract.validate_coverage_contract(job)

    def test_synthetic_contract_validates_all_fitting_and_cache_facts(self):
        job = self.job()
        coverage = contract.validate_coverage_contract(job)
        self.assertEqual(274, len(coverage["fitting_inventory"]))
        self.assertEqual(60, len(coverage["pilot_video_ids"]))
        self.assertEqual(40193, coverage["coverage_summary"]["fitting_windows"])
        self.assertEqual(638, coverage["coverage_summary"]["pair_windows"])
        self.assertEqual(28, len(coverage["coverage_summary"]["all_available_pair_positive_take_groups"]))
        self.assertEqual(15, len(coverage["coverage_summary"]["per_class"]))
        cache = coverage["cache_reuse"]
        self.assertEqual(113, len(cache["records"]))
        self.assertEqual(36, sum(r["role"] == "fitting" for r in cache["records"]))
        self.assertEqual(77, sum(r["role"] == "validation" for r in cache["records"]))
        self.assertEqual(contract.KERNEL_SOURCE, cache["kernel_source"])
        self.assertEqual(contract.FROZEN_PAYLOAD_SHA256, coverage["contract_sha256"])

    def test_five_fresh_probes_include_new_sparse_videos(self):
        coverage = self.coverage
        self.assertEqual(coverage["core_probe_video_ids"], coverage["pilot"]["probe_video_ids"])
        self.assertEqual(coverage["core_probe_video_ids"] + coverage["sparse_probe_video_ids"], coverage["probe_video_ids"])
        self.assertTrue(set(coverage["sparse_probe_video_ids"]) <= set(coverage["additional_rgb_video_ids"]))
        facts = {v["video_id"]: v for v in coverage["fitting_inventory"]}
        first, second = coverage["sparse_probe_video_ids"]
        self.assertEqual(1, facts[first]["n_windows"])
        self.assertEqual(max(facts[v]["source_frame_count"] for v in coverage["additional_rgb_video_ids"] if v != first),
                         facts[second]["source_frame_count"])

    def test_sparse_ties_use_video_identity(self):
        facts = {"B": {"source_frame_count": 20, "n_windows": 2},
                 "A": {"source_frame_count": 10, "n_windows": 1},
                 "D": {"source_frame_count": 30, "n_windows": 4},
                 "C": {"source_frame_count": 30, "n_windows": 5}}
        self.assertEqual(["A", "C"], contract._sparse_probes(facts, list(facts)))

    def test_runtime_validator_performs_no_file_reads(self):
        with patch.object(Path, "open", side_effect=AssertionError("runtime must not open arrays or artifacts")):
            contract.validate_coverage_contract(self.job())

    def test_builder_rejects_changed_plan_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / contract.PLAN
            path.parent.mkdir(parents=True)
            path.write_text('{"synthetic": true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance hash changed"):
                contract.build_coverage_contract(path, root)

    def test_builder_rejects_changed_evidence_or_source_file(self):
        # A minimal local lineage reaches each independent receipt loop without
        # requiring the real plan, source tree or coverage inventory.
        for section in ("evidence_sha256", "source_and_report_sha256"):
            with self.subTest(section=section), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                plan = root / contract.PLAN
                plan.parent.mkdir(parents=True)
                plan.write_text('{}', encoding="utf-8")
                source = root / (contract.BASE if section == "evidence_sha256" else "") / "synthetic.json"
                source.write_text('{"changed": true}', encoding="utf-8")
                status = {"recommended_plan": "availability-20261014/expanded_pilot.json", "coverage_work_complete": True,
                          "evidence_sha256": {}, "source_and_report_sha256": {}}
                status[section]["synthetic.json"] = "0" * 64
                (root / contract.BASE / "continuation-status.json").write_text(json.dumps(status), encoding="utf-8")
                with patch.object(contract, "PLAN_SHA256", contract.sha256(plan)):
                    with self.assertRaisesRegex(ValueError, "provenance hash changed"):
                        contract.build_coverage_contract(plan, root)

    def test_contract_rehash_cannot_change_reviewed_facts(self):
        job = self.job()
        coverage = job["coverage"]
        coverage["coverage_summary"]["pair_windows"] += 1
        coverage["contract_sha256"] = contract.canonical_hash({k: v for k, v in coverage.items()
                                                              if k != "contract_sha256"})
        with self.assertRaisesRegex(ValueError, "reviewed frozen evidence"):
            contract.validate_coverage_contract(job)

    def test_unknown_mask_is_not_negative_supervision(self):
        job = self.job()
        video = job["coverage"]["fitting_inventory"][0]
        video["known_counts"][1] = video["n_windows"]
        video["negative_counts"][1] = video["n_windows"]
        self.assert_semantic_rejection(job, "known mask")

    def test_known_positive_and_four_state_counts_are_recomputed(self):
        for key in ("positive_counts", "negative_counts", "four_state_counts"):
            job = self.job()
            job["coverage"]["fitting_inventory"][0][key][0] += 1
            with self.subTest(key=key):
                self.assert_semantic_rejection(job, "counts|labels")

    def test_held_out_identity_cannot_enter_fitting_facts_or_folds(self):
        job = self.job()
        job["coverage"]["fitting_inventory"][0]["subject"] = "p04"
        self.assert_semantic_rejection(job, "held-out")
        job = self.job()
        job["coverage"]["folds"]["A"][0] = "p04"
        self.assert_semantic_rejection(job, "fitting subjects only")

    def test_duplicate_and_missing_selected_id_are_rejected(self):
        for key in ("original_video_ids", "core_video_ids", "pilot_video_ids", "pair_coverage_additions"):
            job = self.job()
            job["coverage"][key][-1] = job["coverage"][key][0]
            with self.subTest(key=key):
                self.assert_semantic_rejection(job, "duplicate")

    def test_summary_class_and_pair_coverage_are_recomputed(self):
        for mutate in (lambda c: c["coverage_summary"].__setitem__("pair_windows", 637),
                       lambda c: c["coverage_summary"]["per_class"]["class_17"].__setitem__("positive_windows", 23),
                       lambda c: c["coverage_summary"]["pair_positive_take_groups_by_fold"]["A"].pop()):
            job = self.job()
            mutate(job["coverage"])
            self.assert_semantic_rejection(job, "coverage count report differs")

    def test_sparse_probe_cannot_be_replaced_by_cached_core_probe(self):
        job = self.job()
        coverage = job["coverage"]
        coverage["sparse_probe_video_ids"][0] = coverage["core_probe_video_ids"][0]
        coverage["probe_video_ids"] = coverage["core_probe_video_ids"] + coverage["sparse_probe_video_ids"]
        self.assert_semantic_rejection(job, "fresh sparse probes")

    def test_plan_pilot_is_bound_to_job(self):
        job = self.job()
        job["pilot"]["pilot_video_ids"].pop()
        with self.assertRaisesRegex(ValueError, "job pilot differs"):
            contract.validate_coverage_contract(job)

    def test_all_job_sources_and_video_metadata_are_bound(self):
        for key in ("source_frames", "source_bytes", "source_sha256", "shard_sha256", "n_windows"):
            job = self.job()
            # This covers testing metadata as well as fitting/validation facts,
            # without opening any held-out data array.
            record = next(v for v in job["video_inventory"] if v["subset"] == "testing")
            record[key] = "0" * 64 if key.endswith("sha256") else record[key] + 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "video inventory binding"):
                contract.validate_coverage_contract(job)
        job = self.job()
        job["feature_sources"][0]["ref"] = "untrusted/replacement"
        with self.assertRaisesRegex(ValueError, "feature source receipts"):
            contract.validate_coverage_contract(job)
        for key in self.coverage["source_checks"]:
            job = self.job()
            job[key] = "0" * 64
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "job source binding"):
                contract.validate_coverage_contract(job)

    def test_original_time_caps_quality_gates_and_runtime_cannot_change(self):
        for key in contract.FROZEN_BUDGET:
            job = self.job()
            job[key] = None
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "frozen budget or runtime"):
                contract.validate_coverage_contract(job)
        job = self.job()
        job["gate_criteria"]["pair_f1_min"] = 0.29
        with self.assertRaisesRegex(ValueError, "quality gates"):
            contract.validate_coverage_contract(job)

    def test_cache_shard_source_and_encoder_provenance_are_bound(self):
        for key in ("source_video_sha256", "source_shard_sha256", "encoder_sha256", "feature_sha256", "fingerprint"):
            job = self.job()
            job["coverage"]["cache_reuse"]["records"][0][key] = "0" * 64
            with self.subTest(key=key):
                self.assert_semantic_rejection(job, "cache")
        job = self.job()
        job["coverage"]["cache_reuse"]["kernel_source"] = contract.KERNEL_SOURCE[:-1] + "2"
        self.assert_semantic_rejection(job, "pinned prior kernel")

    def test_malformed_and_incomplete_contracts_fail_closed(self):
        for value in (None, {}, {"schema": contract.SCHEMA}):
            job = self.job()
            job["coverage"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                contract.validate_coverage_contract(job)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    total = result.testsRun - len(result.skipped)
    passed = total - len(result.failures) - len(result.errors)
    print(f"{passed}/{total} passed; {len(result.skipped)} skipped")
    raise SystemExit(0 if result.wasSuccessful() else 1)
