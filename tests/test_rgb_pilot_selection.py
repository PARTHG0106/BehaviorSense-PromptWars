"""Synthetic CPU-only coverage, optimizer, and provenance regression tests.

No real manifests, label arrays, video files, or held-out data are opened here.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import select_rgb_pilot as selector


SUBJECTS = selector.FIT_SUBJECTS
CLASSES = list(range(15))
FOLDS = {label: list(SUBJECTS[start:start + 3])
         for label, start in zip("ABC", (0, 3, 6))}


def set_states(video, states):
    """Change synthetic labels while retaining a complete supervision mask."""
    video["four_state_counts"] = list(states)
    video["n_windows"] = sum(states)
    video["duration_seconds"] = max(1., 2. * sum(states))
    video["positive_counts"][10] = states[1] + states[3]
    video["positive_counts"][13] = states[2] + states[3]
    video["known_counts"] = [sum(states) if c in CLASSES else 0 for c in range(22)]
    video["negative_counts"] = [known - positive for known, positive in
                                zip(video["known_counts"], video["positive_counts"])]


def set_positive(video, c, count):
    video["positive_counts"][c] = count
    video["negative_counts"][c] = video["known_counts"][c] - count


def make_video(subject, take, camera=1, windows=40, rare=False):
    vid = f"P{subject[1:]}T{take:02d}C{camera:02d}"
    video = {"video_id": vid, "subject": subject, "take_group": vid[:6],
             "camera": camera, "positive_counts": [0] * 22}
    for c in CLASSES:
        video["positive_counts"][c] = int(windows > 0)
    video["positive_counts"][14] = int(rare and windows > 0)
    set_states(video, [windows - 30, 10, 10, 10] if windows else [0, 0, 0, 0])
    return video


def inventory_fixture():
    names = [f"class_{c}" for c in range(22)]
    names[10], names[13] = "drinking", "watching_tv"
    videos = []
    for index, subject in enumerate(SUBJECTS):
        videos.extend([make_video(subject, 1), make_video(subject, 1, camera=2),
                       make_video(subject, 2), make_video(subject, 3),
                       make_video(subject, 4, windows=100 + index, rare=True),
                       make_video(subject, 5, windows=60)])
    return {"schema": selector.SCHEMA, "fitting_subjects": list(SUBJECTS),
            "class_names": names, "supervised_class_ids": list(CLASSES),
            "folds": deepcopy(FOLDS), "videos": videos,
            "provenance": {"frozen_input_hashes_verified": True,
                           "input_files": {"synthetic": {"path": "synthetic.json", "sha256": "a" * 64}},
                           "label_source_files": {"synthetic_labels.py": "b" * 64}}}


def expected_selection(inventory):
    """The three cheapest rare-class/probe videos, one from each fixed fold."""
    rare_subjects = {SUBJECTS[0], SUBJECTS[3], SUBJECTS[6]}
    return sorted(v["video_id"] for v in inventory["videos"]
                  if v["camera"] == 1 and
                  (int(v["video_id"][4:6]) in (1, 2, 3)
                   or int(v["video_id"][4:6]) == (4 if v["subject"] in rare_subjects else 5)))


class InventoryValidationTests(unittest.TestCase):
    def test_valid_synthetic_counts_and_masks_are_accepted(self):
        inventory = inventory_fixture()
        self.assertIs(inventory, selector.validate_inventory(inventory))

    def test_bad_counts_and_supervision_masks_are_rejected(self):
        cases = ("negative", "boolean", "fractional", "mismatched_counts", "missing_mask",
                 "unsupported_mask", "state_sum", "state_pair_disagreement")
        for case in cases:
            with self.subTest(case=case):
                inventory = inventory_fixture()
                video = inventory["videos"][0]
                if case in ("negative", "boolean", "fractional"):
                    video["positive_counts"][0] = {"negative": -1, "boolean": True, "fractional": 1.5}[case]
                elif case == "mismatched_counts":
                    video["negative_counts"][0] -= 1
                elif case == "missing_mask":
                    video["known_counts"][0] -= 1
                    video["negative_counts"][0] -= 1
                elif case == "unsupported_mask":
                    video["known_counts"][21] = 1
                    video["negative_counts"][21] = 1
                elif case == "state_sum":
                    video["four_state_counts"][0] += 1
                else:
                    video["four_state_counts"][1] += 1
                    video["four_state_counts"][0] -= 1
                with self.assertRaises(ValueError):
                    selector.validate_inventory(inventory)

    def test_heldout_subject_cannot_enter_inventory_or_folds(self):
        for location in ("fitting_subjects", "videos", "folds"):
            with self.subTest(location=location):
                inventory = inventory_fixture()
                if location == "fitting_subjects":
                    inventory[location][0] = "p02"
                elif location == "videos":
                    inventory[location][0] = make_video("p02", 1)
                else:
                    inventory[location]["A"][0] = "p02"
                with self.assertRaisesRegex(ValueError, "held-out|fitting subjects"):
                    selector.validate_inventory(inventory)

    def test_duplicate_video_and_changed_take_camera_are_rejected(self):
        for field, value in (("video_id", "P03T01C02"), ("take_group", "P03T09"), ("camera", 2)):
            with self.subTest(field=field):
                inventory = inventory_fixture()
                inventory["videos"][0][field] = value
                with self.assertRaises(ValueError):
                    selector.validate_inventory(inventory)

    def test_unverified_or_malformed_provenance_is_rejected(self):
        for kind in ("unverified", "missing_receipts", "invalid_hash"):
            with self.subTest(kind=kind):
                inventory = inventory_fixture()
                provenance = inventory["provenance"]
                if kind == "unverified":
                    provenance["frozen_input_hashes_verified"] = False
                elif kind == "missing_receipts":
                    provenance["input_files"] = {}
                else:
                    provenance["label_source_files"]["synthetic_labels.py"] = "not-a-hash"
                with self.assertRaises(ValueError):
                    selector.validate_inventory(inventory)


class FoldSelectionTests(unittest.TestCase):
    def rare_class_inventory(self):
        inventory = inventory_fixture()
        inventory["videos"] = [make_video(s, 1, windows=100, rare=s in SUBJECTS[:3]) for s in SUBJECTS]
        return inventory

    def test_new_canonical_folds_repair_original_class_gap(self):
        inventory = self.rare_class_inventory()
        result = selector.choose_folds(inventory)
        self.assertFalse(result["original_fold_support"]["feasible"])
        self.assertEqual("chosen", result["status"])
        self.assertEqual(280, result["partitions_examined"])
        self.assertEqual(90, result["feasible_partitions"])
        self.assertEqual({"A": ["p03", "p09", "p12"], "B": ["p06", "p13", "p17"],
                          "C": ["p07", "p19", "p25"]}, result["folds"])
        self.assertEqual(0, result["row_spread"])
        self.assertEqual(0, result["pair_window_spread"])
        for subjects in result["folds"].values():
            self.assertEqual(1, len(set(subjects) & set(SUBJECTS[:3])))
        self.assertTrue(result["support"]["feasible"])

    def test_canonical_folds_ignore_inventory_order(self):
        inventory = self.rare_class_inventory()
        original = selector.choose_folds(inventory)
        random.Random(91).shuffle(inventory["videos"])
        inventory["fitting_subjects"].reverse()
        inventory["supervised_class_ids"].reverse()
        shuffled = selector.choose_folds(inventory)
        for key in ("folds", "row_spread", "pair_window_spread", "partitions_examined", "feasible_partitions"):
            self.assertEqual(original[key], shuffled[key])

    def test_two_positive_subjects_prove_folds_impossible(self):
        inventory = self.rare_class_inventory()
        set_positive(inventory["videos"][2], 14, 0)
        result = selector.choose_folds(inventory)
        self.assertEqual("proven_infeasible", result["status"])
        self.assertEqual(280, result["partitions_examined"])
        self.assertEqual(0, result["feasible_partitions"])
        self.assertNotIn("folds", result)


class WholeVideoSelectionTests(unittest.TestCase):
    def test_optimal_selection_satisfies_every_constraint_and_minimizes_windows(self):
        inventory = inventory_fixture()
        result = selector.select_videos(inventory, FOLDS)
        self.assertEqual("optimal", result["status"])
        self.assertEqual(expected_selection(inventory), result["selected_video_ids"])
        self.assertEqual(result["selected_video_ids"], result["pilot_video_ids"])
        selected = [v for v in inventory["videos"] if v["video_id"] in result["selected_video_ids"]]
        self.assertEqual(36, len(selected))
        self.assertEqual({s: 4 for s in SUBJECTS}, dict(Counter(v["subject"] for v in selected)))
        self.assertEqual(36, len({v["take_group"] for v in selected}))
        self.assertEqual(1749, sum(v["n_windows"] for v in selected))
        self.assertGreaterEqual(sum(v["four_state_counts"][3] for v in selected), 100)
        for subject in SUBJECTS:
            videos = [v for v in selected if v["subject"] == subject]
            self.assertGreaterEqual(len({v["take_group"] for v in videos if v["four_state_counts"][3]}), 2)
            self.assertTrue(all(sum(v["four_state_counts"][state] for v in videos) > 0 for state in range(4)))
        for c in CLASSES:
            positive_subjects = {v["subject"] for v in selected if v["positive_counts"][c]}
            self.assertGreaterEqual(len(positive_subjects), 3)
            for subjects in FOLDS.values():
                self.assertTrue(positive_subjects & set(subjects))
                self.assertTrue(positive_subjects - set(subjects))
        for subjects in FOLDS.values():
            self.assertGreaterEqual(len({v["take_group"] for v in selected
                                         if v["subject"] in subjects and v["four_state_counts"][3]}), 6)
        self.assertGreaterEqual(len({v["subject"] for v in selected if 100 <= v["n_windows"] <= 1000}), 3)
        self.assertTrue(result["audit"]["passed"])
        self.assertFalse(result["constraint_relaxation"])
        self.assertLess(result["objective"]["stable_id_tie_cost"], .25)

    def test_shuffled_inputs_select_identical_whole_videos_and_probes(self):
        inventory = inventory_fixture()
        first = selector.select_videos(inventory, FOLDS)
        random.Random(23).shuffle(inventory["videos"])
        inventory["supervised_class_ids"].reverse()
        second = selector.select_videos(inventory, dict(reversed(list(FOLDS.items()))))
        self.assertEqual(first["selected_video_ids"], second["selected_video_ids"])
        self.assertEqual(first["probe_video_ids"], second["probe_video_ids"])
        self.assertEqual(first["objective"], second["objective"])

    def test_insufficient_distinct_takes_cannot_be_filled_with_second_camera(self):
        inventory = inventory_fixture()
        inventory["videos"] = [v for v in inventory["videos"] if v["subject"] != "p03"
                               or int(v["video_id"][4:6]) in (1, 2, 4)]
        result = selector.select_videos(inventory, FOLDS)
        self.assertEqual("proven_infeasible", result["status"])
        self.assertEqual([], result["selected_video_ids"])

    def test_one_available_pair_take_is_accepted_and_flagged_when_fold_has_six(self):
        inventory = inventory_fixture()
        available = {"p03": {"P03T01"}, "p06": {"P06T01", "P06T02"},
                     "p07": {"P07T01", "P07T02", "P07T03"}}
        for v in inventory["videos"]:
            if v["subject"] in available and v["take_group"] not in available[v["subject"]]:
                states = v["four_state_counts"][:]
                states[0] += states[3]
                states[3] = 0
                set_states(v, states)
        result = selector.select_videos(inventory, FOLDS)
        self.assertEqual("optimal", result["status"])
        subject = result["audit"]["per_subject"]["p03"]
        # The two cameras of T01 are one available take, and one selected take.
        self.assertEqual(1, subject["available_pair_positive_takes"])
        self.assertEqual(1, subject["required_pair_positive_takes"])
        self.assertEqual(["P03T01"], subject["pair_positive_takes"])
        self.assertTrue(subject["pair_take_availability_shortfall"])
        self.assertEqual(6, len(result["audit"]["pair_positive_takes_by_fold"]["A"]))
        self.assertFalse(result["audit"]["per_subject"]["p06"]["pair_take_availability_shortfall"])
        self.assertTrue(result["audit"]["passed"])
        self.assertFalse(result["constraint_relaxation"])

    def test_zero_available_pair_takes_still_requires_one_and_is_infeasible(self):
        inventory = inventory_fixture()
        for v in inventory["videos"]:
            if v["subject"] == "p03":
                states = v["four_state_counts"][:]
                states[0] += states[3]
                states[3] = 0
                set_states(v, states)
        result = selector.select_videos(inventory, FOLDS)
        self.assertEqual("proven_infeasible", result["status"])
        self.assertEqual([], result["selected_video_ids"])
        audited = selector.audit_selection(inventory, expected_selection(inventory), FOLDS)
        subject = audited["per_subject"]["p03"]
        self.assertEqual(0, subject["available_pair_positive_takes"])
        self.assertEqual(1, subject["required_pair_positive_takes"])
        self.assertTrue(subject["pair_take_availability_shortfall"])
        self.assertIn("p03: insufficient pair-positive takes", audited["violations"])

    def test_five_pair_positive_takes_per_fold_remains_infeasible(self):
        inventory = inventory_fixture()
        available = {"p03": {"P03T01"}, "p06": {"P06T01", "P06T02"},
                     "p07": {"P07T01", "P07T02"}}
        for v in inventory["videos"]:
            if v["subject"] in available and v["take_group"] not in available[v["subject"]]:
                states = v["four_state_counts"][:]
                states[0] += states[3]
                states[3] = 0
                set_states(v, states)
        self.assertTrue(selector.fold_support(inventory, FOLDS)["feasible"])
        self.assertEqual("proven_infeasible", selector.select_videos(inventory, FOLDS)["status"])
        audited = selector.audit_selection(inventory, expected_selection(inventory), FOLDS)
        self.assertEqual(5, len(audited["pair_positive_takes_by_fold"]["A"]))
        self.assertIn("A: fewer than six pair-positive takes", audited["violations"])
        self.assertFalse(any("insufficient pair-positive takes" in v for v in audited["violations"]))

    def test_two_available_pair_takes_cannot_be_reduced_to_one_selected_take(self):
        inventory = inventory_fixture()
        for v in inventory["videos"]:
            if v["subject"] == "p06" and v["take_group"] not in {"P06T01", "P06T04"}:
                states = v["four_state_counts"][:]
                states[0] += states[3]
                states[3] = 0
                set_states(v, states)
        audited = selector.audit_selection(inventory, expected_selection(inventory), FOLDS)
        self.assertIn("p06: insufficient pair-positive takes", audited["violations"])
        self.assertEqual(2, audited["per_subject"]["p06"]["required_pair_positive_takes"])
        result = selector.select_videos(inventory, FOLDS)
        self.assertEqual("optimal", result["status"])
        self.assertEqual(["P06T01", "P06T04"], result["audit"]["per_subject"]["p06"]["pair_positive_takes"])

    def test_pair_window_minimum_is_not_replaced_by_positive_video_count(self):
        inventory = inventory_fixture()
        for v in inventory["videos"]:
            states = v["four_state_counts"][:]
            states[0] += states[3] - 1
            states[3] = 1
            set_states(v, states)
        self.assertTrue(selector.choose_folds(inventory)["support"]["feasible"])
        self.assertEqual("proven_infeasible", selector.select_videos(inventory, FOLDS)["status"])

    def test_each_subject_needs_all_four_states_even_when_folds_have_them(self):
        inventory = inventory_fixture()
        for v in inventory["videos"]:
            if v["subject"] == "p03":
                states = v["four_state_counts"][:]
                states[0] += states[1]
                states[1] = 0
                set_states(v, states)
        self.assertTrue(selector.fold_support(inventory, FOLDS)["feasible"])
        self.assertEqual("proven_infeasible", selector.select_videos(inventory, FOLDS)["status"])

    def test_three_class_positive_subjects_cannot_all_come_from_one_fold(self):
        inventory = inventory_fixture()
        for v in inventory["videos"]:
            if v["subject"] not in FOLDS["A"]:
                set_positive(v, 14, 0)
        self.assertEqual("proven_infeasible", selector.select_videos(inventory, FOLDS)["status"])

    def test_probes_require_three_subjects_with_eligible_selected_video(self):
        inventory = inventory_fixture()
        for v in inventory["videos"]:
            if v["subject"] not in SUBJECTS[:2] and v["n_windows"] >= 100:
                set_states(v, [50, 10, 10, 10])
        self.assertEqual("proven_infeasible", selector.select_videos(inventory, FOLDS)["status"])

    def test_solver_limit_never_accepts_even_a_feasible_incumbent(self):
        inventory = inventory_fixture()
        chosen = set(expected_selection(inventory))

        def limited(objective, **kwargs):
            vector = np.zeros(len(objective))
            for i, v in enumerate(sorted(inventory["videos"], key=lambda v: v["video_id"])):
                vector[i] = int(v["video_id"] in chosen)
            return SimpleNamespace(status=1, message="time limit reached", x=vector, mip_gap=.01)

        with patch("scipy.optimize.milp", side_effect=limited):
            result = selector.select_videos(inventory, FOLDS)
        self.assertEqual("solver_limit", result["status"])
        self.assertEqual([], result["selected_video_ids"])
        self.assertNotIn("audit", result)
        self.assertNotIn("pilot_video_ids", result)
        self.assertIn("unresolved", result["reason"])
        self.assertFalse(result["constraint_relaxation"])

    def test_solver_infeasible_and_error_are_distinct_from_limit(self):
        for status, expected in ((2, "proven_infeasible"), (4, "solver_error")):
            with self.subTest(status=status), patch("scipy.optimize.milp", return_value=SimpleNamespace(
                    status=status, message="synthetic solver status", x=None)):
                result = selector.select_videos(inventory_fixture(), FOLDS)
                self.assertEqual(expected, result["status"])
                self.assertEqual([], result["selected_video_ids"])
                self.assertFalse(result["constraint_relaxation"])

    def test_forged_optimal_integer_selection_fails_independent_audit(self):
        def forged(objective, **kwargs):
            return SimpleNamespace(status=0, message="claimed optimal", x=np.zeros(len(objective)))

        with patch("scipy.optimize.milp", side_effect=forged):
            with self.assertRaisesRegex(ValueError, "failed independent audit"):
                selector.select_videos(inventory_fixture(), FOLDS)

    def test_noninteger_nonfinite_out_of_bounds_and_wrong_size_solutions_are_rejected(self):
        for kind in ("fractional", "nan", "out_of_bounds", "wrong_size", "missing"):
            with self.subTest(kind=kind):
                def forged(objective, **kwargs):
                    vector = np.zeros(len(objective))
                    if kind == "wrong_size":
                        vector = vector[:-1]
                    elif kind == "missing":
                        vector = None
                    else:
                        vector[0] = {"fractional": .5, "nan": np.nan, "out_of_bounds": 2.}[kind]
                    return SimpleNamespace(status=0, message="claimed optimal", x=vector)

                with patch("scipy.optimize.milp", side_effect=forged):
                    with self.assertRaisesRegex(ValueError, "invalid optimal integer solution"):
                        selector.select_videos(inventory_fixture(), FOLDS)

    def test_audit_rejects_unknown_duplicate_and_same_take_video_ids(self):
        inventory = inventory_fixture()
        selected = expected_selection(inventory)
        with self.assertRaisesRegex(ValueError, "unknown selected"):
            selector.audit_selection(inventory, selected[:-1] + ["P02T01C01"], FOLDS)
        with self.assertRaisesRegex(ValueError, "duplicate selected"):
            selector.audit_selection(inventory, selected + selected[:1], FOLDS)
        selected[selected.index("P03T03C01")] = "P03T01C02"
        audited = selector.audit_selection(inventory, selected, FOLDS)
        self.assertFalse(audited["passed"])
        self.assertIn("multiple cameras selected from the same take", audited["violations"])


class ProbeSelectionTests(unittest.TestCase):
    def test_probes_are_selected_eligible_deterministic_and_from_distinct_subjects(self):
        inventory = inventory_fixture()
        selected = expected_selection(inventory)
        expected = ["P03T04C01", "P09T04C01", "P17T04C01"]
        self.assertEqual(expected, selector.select_probes(inventory, selected))
        # Add a faster eligible unselected video; it cannot become a probe.
        extra = make_video("p03", 9, windows=100)
        extra["duration_seconds"] = 1.
        inventory["videos"].append(extra)
        random.Random(42).shuffle(inventory["videos"])
        selected.reverse()
        actual = selector.select_probes(inventory, selected)
        self.assertEqual(expected, actual)
        by_id = {v["video_id"]: v for v in inventory["videos"]}
        self.assertEqual(3, len({by_id[vid]["subject"] for vid in actual}))
        self.assertTrue(set(actual) <= set(selected))
        self.assertTrue(all(100 <= by_id[vid]["n_windows"] <= 1000 for vid in actual))

    def test_multiple_eligible_cameras_from_two_subjects_do_not_make_three_probes(self):
        inventory = inventory_fixture()
        inventory["videos"] = [make_video(s, take, windows=100, rare=True)
                               for s in SUBJECTS[:2] for take in (1, 2, 3)]
        with self.assertRaisesRegex(ValueError, "three eligible probe subjects"):
            selector.select_probes(inventory, [v["video_id"] for v in inventory["videos"]])


class ArtifactBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rgb-pilot-selection-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "scripts").mkdir()
        builder = self.root / "scripts/build_rgb_coverage_inventory.py"
        builder.write_text("# synthetic builder source\n", encoding="utf-8")
        source = self.root / "synthetic_source.txt"
        source.write_text("synthetic provenance only\n", encoding="utf-8")
        self.inventory = inventory_fixture()
        # Match the production completeness contract using synthetic zero-window
        # placeholders and one expensive video. No real data is consulted.
        fillers = [make_video(s, take, windows=0) for s in SUBJECTS for take in range(6, 40)]
        self.inventory["videos"].extend(fillers[:274 - len(self.inventory["videos"])])
        remaining = 217090 - sum(v["n_windows"] for v in self.inventory["videos"])
        set_states(self.inventory["videos"][-1], [remaining - 30, 10, 10, 10])
        receipt = {"path": str(source), "sha256": selector.sha256(source)}
        self.inventory["provenance"]["input_files"] = {"synthetic": receipt}
        self.inventory["provenance"]["label_source_files"] = {str(source): receipt["sha256"]}
        self.inventory_path = self.root / "inventory.json"
        self.inventory_path.write_text(json.dumps(self.inventory), encoding="utf-8")
        self.audit = {"schema": "behaviorsense.rgb_fitting_coverage_inventory_audit.v1",
                      "status": "verified_complete", "inventory_file_sha256": selector.sha256(self.inventory_path),
                      "inventory_canonical_sha256": selector.canonical_hash(self.inventory),
                      "builder_sha256": selector.sha256(builder),
                      "input_files": deepcopy(self.inventory["provenance"]["input_files"]),
                      "n_videos": 274, "n_windows": 217090}
        for key in ("all_274_extraction_metadata_audits_replayed", "frozen_input_hashes_verified",
                    "frozen_label_helper_hashes_verified", "full_fitting_manifest_aggregates_match",
                    "pilot_fitting_manifest_aggregates_match"):
            self.audit[key] = True
        self.audit_path = self.root / "inventory_audit.json"
        self.save_audit()

    def save_audit(self):
        self.audit_path.write_text(json.dumps(self.audit), encoding="utf-8")

    def test_matching_synthetic_audit_and_source_hashes_are_accepted(self):
        selector.validate_inventory(self.inventory)
        result = selector.verify_inventory_artifact(self.inventory_path, self.inventory, root=self.root)
        self.assertEqual(selector.sha256(self.audit_path), result["sha256"])
        self.assertEqual(1, len(selector.verify_input_files(self.inventory)))

    def test_changed_metadata_and_missing_audit_are_rejected_before_optimization(self):
        for kind in ("changed_inventory", "missing_audit"):
            with self.subTest(kind=kind):
                self.save_audit()
                if kind == "changed_inventory":
                    changed = deepcopy(self.inventory)
                    changed["videos"][0]["duration_seconds"] += 1
                    self.inventory_path.write_text(json.dumps(changed), encoding="utf-8")
                else:
                    self.inventory_path.write_text(json.dumps(self.inventory), encoding="utf-8")
                    self.audit_path.unlink()
                with patch.object(selector, "select_videos") as optimize:
                    with self.assertRaisesRegex(ValueError, "audited artifact|missing verified inventory audit"):
                        selector.run(self.inventory_path, self.root / "output")
                optimize.assert_not_called()
                self.assertFalse((self.root / "output/recipe.json").exists())

    def test_changed_builder_incomplete_audit_and_missing_checks_are_rejected(self):
        original = deepcopy(self.audit)
        for kind in ("builder", "incomplete", "missing_check", "count", "provenance"):
            with self.subTest(kind=kind):
                self.audit = deepcopy(original)
                if kind == "builder":
                    self.audit["builder_sha256"] = "0" * 64
                elif kind == "incomplete":
                    self.audit["status"] = "incomplete"
                elif kind == "missing_check":
                    self.audit["frozen_label_helper_hashes_verified"] = False
                elif kind == "count":
                    self.audit["n_windows"] -= 1
                else:
                    self.audit["input_files"] = {}
                self.save_audit()
                with self.assertRaises(ValueError):
                    selector.verify_inventory_artifact(self.inventory_path, self.inventory, root=self.root)

    def test_changed_provenance_source_is_detected(self):
        source = Path(self.inventory["provenance"]["input_files"]["synthetic"]["path"])
        source.write_text("changed synthetic provenance\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "provenance hash changed"):
            selector.verify_input_files(self.inventory)

    def test_run_freezes_recipe_before_optimization_and_refuses_overwrite(self):
        output = self.root / "output"
        real_verify = selector.verify_inventory_artifact
        real_select = selector.select_videos

        def verify(path, inventory):
            return real_verify(path, inventory, root=self.root)

        def select(inventory, folds):
            recipe = json.loads((output / "recipe.json").read_text(encoding="utf-8"))
            self.assertEqual(selector.sha256(self.inventory_path), recipe["inventory_sha256"])
            self.assertEqual(selector.canonical_hash(selector.POLICY), recipe["policy_sha256"])
            self.assertEqual(Path(selector.__file__).read_bytes(), (output / "selector_source.py").read_bytes())
            self.assertEqual(recipe["source_sha256"], selector.sha256(output / "selector_source.py"))
            self.assertFalse((output / "selection.json").exists())
            return real_select(inventory, folds)

        with patch.object(selector, "verify_inventory_artifact", side_effect=verify), \
                patch.object(selector, "select_videos", side_effect=select):
            result = selector.run(self.inventory_path, output)
            self.assertEqual("optimal", result["status"])
            self.assertFalse(result["model_candidate_created"])
            self.assertEqual(selector.sha256(output / "recipe.json"), result["recipe_sha256"])
            frozen = {path.name: path.read_bytes() for path in output.iterdir()}
            with self.assertRaisesRegex(ValueError, "immutable selection output"):
                selector.run(self.inventory_path, output)
        self.assertEqual(frozen, {path.name: path.read_bytes() for path in output.iterdir()})


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    total = result.testsRun - len(result.skipped)
    passed = total - len(result.failures) - len(result.errors)
    print(f"{passed}/{total} passed; {len(result.skipped)} skipped")
    raise SystemExit(0 if result.wasSuccessful() else 1)
