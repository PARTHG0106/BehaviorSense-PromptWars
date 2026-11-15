"""Offline provenance and label-coverage regression tests on synthetic records only."""

from contextlib import ExitStack, contextmanager
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import build_rgb_coverage_inventory as builder
from behaviorsense.data.toyota import COARSE_V11, TSU_CLASSES, TsuVideo

DRINK = COARSE_V11.index("drinking")
TV = COARSE_V11.index("watching_tv")
READ = COARSE_V11.index("reading")
LABEL_FILES = ("src/behaviorsense/data/multilabel_skeleton.py", "src/behaviorsense/data/toyota.py")


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class ArrayArchive:
    """The sole permitted np.load result; never backed by a real array file."""

    def __init__(self, arrays):
        self.arrays = arrays
        self.files = list(arrays)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __getitem__(self, key):
        return self.arrays[key]


class SyntheticSnapshot:
    """Match fixed inventory cardinalities without using any project data.

    The 36 cached videos provide four states for every fitting subject. A further
    238 one-window videos keep full-inventory replay inexpensive. One has two
    adjacent half-window actions: co-label support without simultaneous overlap.
    Expected counts are built from these assigned states, independently of the
    production annotation replay and full-coverage audit, which remain unmocked.
    """

    def __init__(self):
        self.root = Path(tempfile.gettempdir()).resolve() / "synthetic-rgb-coverage"
        self.hashes = {relative: digest(name) for name, relative in builder.INPUT_PATHS.items()}
        self.hashes.update({relative: digest(relative) for relative in LABEL_FILES})
        self.pins = {name: self.hashes[builder.INPUT_PATHS[name]] for name in builder.FROZEN_INPUT_HASHES}
        self.supported = list(builder.supervised_coarse(builder.TSU_TO_COARSE))
        self.annotations, shards, planned, cached_shards = {}, [], [], []
        cached_parts = {name: [] for name in ("video_ids", "subjects", "starts", "targets", "target_mask", "tv_drinking_overlap_frames")}
        total_positive = np.zeros(len(COARSE_V11), np.int64)
        pilot_positive = np.zeros_like(total_positive)
        n_total = n_overlap = n_both = 0
        identities = [(s, 1, camera) for s in builder.FIT_SUBJECTS for camera in range(1, 5)]
        identities += [(builder.FIT_SUBJECTS[i % 9], 2 + i // 9, 1) for i in range(238)]
        for index, (subject, take, camera) in enumerate(identities):
            video_id = f"P{subject[1:]}T{take:02d}C{camera:02d}"
            n = 884 if index == 0 else 885 if index < 36 else 1
            starts = np.arange(n, dtype=np.int64) * 15
            frames = int(np.ceil((int(starts[-1]) + 30) * 25 / 20))
            state = (camera - 1) if index < 36 else index % 4
            split_halves = index == 273
            if split_halves:
                state = 3
                actions = ((TSU_CLASSES.index("Drink.From_cup"), 0, 18),
                           (TSU_CLASSES.index("Watch_TV"), 19, 37))
            else:
                fine = (["Read"] if state == 0 else
                        (["Drink.From_cup"] if state == 1 else
                         ["Watch_TV"] if state == 2 else ["Drink.From_cup", "Watch_TV"]))
                actions = tuple((TSU_CLASSES.index(name), 0, frames) for name in fine)
            self.annotations[video_id] = TsuVideo(video_id, int(subject[1:]), take, camera,
                                                   "training", frames, actions)
            report = {
                "schema_version": 2, "selection_mode": "full", "window_cap": None,
                "full_dense_coverage": True, "video_id": video_id, "subject": subject,
                "subset": "training", "source_sha256": digest(video_id),
                "config_fingerprint": digest("fixture-config"), "source_fps": 25.,
                "source_frame_count": frames, "annotation_duration_frames": frames,
                "candidate_stride": 15, "candidate_windows": n, "selected_windows": n,
                "excluded_annotation_coverage": 0, "selected_starts": starts.tolist(),
                "retained_windows": n, "retained_starts": starts.tolist(),
                "dropped_visibility": 0, "source_bytes": frames * 3,
            }
            identity = {key: report[key] for key in ("config_fingerprint", "video_id", "subset", "source_sha256")}
            report["fingerprint"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            shard = {"extraction": report, "sha256": digest(video_id + "-shard"),
                     "n_windows": n, "path": video_id + ".npz"}
            shards.append(shard)
            overlap = 30 if state == 3 and not split_halves else 0
            planned.append({"video_id": video_id, "subject": subject, "subset": "training",
                            "shard_sha256": shard["sha256"], "source_sha256": report["source_sha256"],
                            "source_fps": 25., "source_frames": frames, "n_windows": n,
                            "pair_windows": n if overlap else 0,
                            "source_ref": "synthetic/" + video_id, "shard_filename": video_id + ".npz"})
            active = [READ] if state == 0 else [DRINK] if state == 1 else [TV] if state == 2 else [DRINK, TV]
            positive = np.zeros(len(COARSE_V11), np.int64)
            positive[active] = n
            total_positive += positive
            n_total += n
            n_both += n if state == 3 else 0
            n_overlap += n if overlap else 0
            if index < 36:
                cached_shards.append(copy.deepcopy(shard))
                pilot_positive += positive
                targets = np.zeros((n, len(COARSE_V11)), np.float32)
                targets[:, active] = 1
                known = np.zeros_like(targets, dtype=bool)
                known[:, self.supported] = True
                values = {"video_ids": np.full(n, video_id), "subjects": np.full(n, subject),
                          "starts": starts, "targets": targets, "target_mask": known,
                          "tv_drinking_overlap_frames": np.full(n, overlap, np.int16)}
                for name, array in values.items():
                    cached_parts[name].append(array)
        self.cached = {name: np.concatenate(parts) for name, parts in cached_parts.items()}
        self.split_id = video_id
        self.first_shard, self.first_plan = shards[0], planned[0]
        self.pilot_shards = cached_shards
        negative = np.zeros_like(total_positive)
        negative[self.supported] = n_total - total_positive[self.supported]
        pilot_negative = np.zeros_like(total_positive)
        pilot_negative[self.supported] = 31859 - pilot_positive[self.supported]
        annotation_hash = self.hashes[builder.INPUT_PATHS["annotations"]]
        label_sources = [{"path": path, "sha256": self.hashes[path]} for path in LABEL_FILES]
        # Poisoned records would fail immediately if broader partitions were processed.
        shards.append({"extraction": {"subject": "p02", "video_id": "P02T01C01"}})
        planned.append({"subject": "p02", "video_id": "P02T01C01"})
        self.full = {
            "class_names": list(COARSE_V11), "annotation_sha256": annotation_hash,
            "source_files": label_sources,
            "train": {"subjects": list(builder.FIT_SUBJECTS), "n_videos": 274, "n_windows": n_total,
                      "positive": total_positive.tolist(), "negative": negative.tolist(),
                      "tv_drinking_actual_overlap_windows": n_overlap,
                      "tv_drinking_actual_overlap_sample_positions": n_overlap * 30,
                      "tv_drinking_colabeled_windows": n_both},
            "provenance": {"sample_fps": 20., "annotation_fps": 25., "window_frames": 30,
                           "min_overlap": .5, "supervised_class_ids": self.supported,
                           "source_shards": shards},
        }
        self.plan = {"source_manifest": {"files": {path: self.hashes[path] for path in LABEL_FILES}},
                     "job": {"annotation_sha256": annotation_hash, "video_inventory": planned}}
        self.pilot = {"class_names": list(COARSE_V11), "annotation_sha256": annotation_hash,
                      "source_provenance": {"source_shards": cached_shards},
                      "train": {"positive": pilot_positive.tolist(), "negative": pilot_negative.tolist()}}
        self.protocol = {"folds": copy.deepcopy(builder.ORIGINAL_FOLDS),
                         "source_plan_sha256": self.hashes[builder.INPUT_PATHS["rgb_plan"]],
                         "fitting_video_ids": [s["extraction"]["video_id"] for s in cached_shards]}
        self.provenance = {"frozen_manifest_sha256": self.hashes[builder.INPUT_PATHS["pilot_manifest"]],
                           "files": {"fit_rows.npz": {"sha256": self.hashes[builder.INPUT_PATHS["pilot_rows"]]}},
                           "status": "verified_complete", "annotation_sha256": annotation_hash,
                           "forbidden_inputs_loaded": {"validation": False, "test": False}}
        self.documents = {"training_manifest": self.full, "rgb_plan": self.plan,
                          "pilot_manifest": self.pilot, "pilot_protocol": self.protocol,
                          "pilot_provenance": self.provenance}

    @contextmanager
    def readers(self):
        paths = {self.root / builder.INPUT_PATHS[name]: value for name, value in self.documents.items()}

        def file_hash(path):
            if Path(path) == Path(builder.__file__):
                return digest("synthetic-builder")
            return self.hashes[Path(path).relative_to(self.root).as_posix()]

        def array_load(path, *, allow_pickle):
            if Path(path) != self.root / builder.INPUT_PATHS["pilot_rows"] or allow_pickle is not False:
                raise AssertionError("Only the synthetic fitting-row archive may be loaded")
            return ArrayArchive(self.cached)

        with ExitStack() as stack:
            stack.enter_context(patch.object(builder, "ROOT", self.root))
            stack.enter_context(patch.object(builder, "FROZEN_INPUT_HASHES", self.pins))
            hashes = stack.enter_context(patch.object(builder, "sha256", side_effect=file_hash))
            metadata = stack.enter_context(patch.object(builder, "read_json", side_effect=lambda path: paths[Path(path)]))
            annotations = stack.enter_context(patch.object(builder, "load_tsu_annotations", return_value=self.annotations))
            arrays = stack.enter_context(patch.object(builder.np, "load", side_effect=array_load))
            yield SimpleNamespace(hashes=hashes, metadata=metadata, annotations=annotations, arrays=arrays)


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = SyntheticSnapshot()

    def rejected(self, fragment, *, before_arrays=False, error=ValueError):
        with self.fixture.readers() as readers:
            with self.assertRaisesRegex(error, fragment):
                builder.build_inventory(self.fixture.root)
            if before_arrays:
                readers.arrays.assert_not_called()
        return readers

    def test_full_replay_preserves_unknown_taxonomy_and_distinct_pair_definitions(self):
        with self.fixture.readers() as readers:
            inventory, audit = builder.build_inventory(self.fixture.root)
        readers.arrays.assert_called_once_with(self.fixture.root / builder.INPUT_PATHS["pilot_rows"], allow_pickle=False)
        summary = inventory["summary"]
        self.assertEqual((summary["n_videos"], summary["n_subjects"]), (274, 9))
        self.assertEqual(inventory["original_pilot_summary"]["n_windows"], 31859)
        self.assertEqual(len(audit["cached_fitting_checks"]), 36)
        unsupported = set(range(len(COARSE_V11))) - set(self.fixture.supported)
        self.assertEqual(len(unsupported), 7)
        for index in unsupported:
            self.assertEqual([summary[key][index] for key in ("positive_counts", "known_counts", "negative_counts")], [0, 0, 0])
        for index in self.fixture.supported:
            self.assertEqual(summary["known_counts"][index], summary["n_windows"])
            self.assertEqual(summary["negative_counts"][index], summary["n_windows"] - summary["positive_counts"][index])
        absent_index = next(index for index in self.fixture.supported if index not in (READ, DRINK, TV))
        absent = inventory["availability"]["per_class"][COARSE_V11[absent_index]]
        self.assertEqual((absent["positive_windows"], absent["positive_subject_count"]), (0, 0))
        self.assertFalse(absent["at_least_three_positive_subjects_possible"])
        drinking = inventory["availability"]["per_class"]["drinking"]
        self.assertEqual(drinking["positive_subjects"], list(builder.FIT_SUBJECTS))
        self.assertTrue(drinking["positive_support_in_every_original_held_fold_possible"])
        split = next(v for v in inventory["videos"] if v["video_id"] == self.fixture.split_id)
        self.assertEqual(split["four_state_counts"], [0, 0, 0, 1])
        self.assertEqual(split["colabeled_without_actual_overlap_windows"], 1)
        self.assertEqual(split["actual_overlap_windows"], 0)
        self.assertEqual(split["legacy_extraction_pair_stratum_count"], 0)
        self.assertEqual(summary["four_state_counts"][3], summary["actual_overlap_windows"] + 1)
        self.assertTrue(inventory["availability"]["all_four_states_available_in_every_subject"])
        self.assertEqual(audit["scope"]["array_files_loaded"], [builder.INPUT_PATHS["pilot_rows"]])
        self.assertFalse(any(value for key, value in audit["scope"].items() if key != "array_files_loaded"))

    def test_pinned_input_mismatches_fail_before_any_metadata_or_array_load(self):
        for name in self.fixture.pins:
            with self.subTest(name=name):
                original = self.fixture.hashes[builder.INPUT_PATHS[name]]
                self.fixture.hashes[builder.INPUT_PATHS[name]] = digest("tampered")
                readers = self.rejected("Frozen " + name + " hash differs", before_arrays=True)
                readers.metadata.assert_not_called()
                readers.annotations.assert_not_called()
                self.fixture.hashes[builder.INPUT_PATHS[name]] = original

    def test_alternate_root_is_rejected_before_any_input_is_read(self):
        with self.fixture.readers() as readers:
            with self.assertRaisesRegex(ValueError, "[Rr]oot"):
                builder.build_inventory(self.fixture.root / "other-workspace")
            readers.hashes.assert_not_called()
            readers.metadata.assert_not_called()
            readers.annotations.assert_not_called()
            readers.arrays.assert_not_called()

    def test_label_helper_source_mismatch_fails_before_label_or_array_load(self):
        self.fixture.plan["source_manifest"]["files"][LABEL_FILES[0]] = digest("new-code")
        readers = self.rejected("Frozen label helper differs", before_arrays=True)
        readers.annotations.assert_not_called()

    def test_unverified_or_forbidden_cache_cannot_be_reused(self):
        self.fixture.provenance["status"] = "incomplete"
        self.rejected("not verified", before_arrays=True)
        self.fixture.provenance["status"] = "verified_complete"
        self.fixture.provenance["forbidden_inputs_loaded"]["validation"] = True
        self.rejected("forbidden inputs", before_arrays=True)

    def test_annotation_hash_and_label_semantics_are_bound_before_arrays(self):
        self.fixture.pilot["annotation_sha256"] = digest("different-annotations")
        self.rejected("Annotation provenance differs", before_arrays=True)
        self.fixture.pilot["annotation_sha256"] = self.fixture.full["annotation_sha256"]
        self.fixture.full["provenance"]["sample_fps"] = 25.
        self.rejected("Label semantics changed", before_arrays=True)

    def test_protocol_partition_and_fold_changes_are_rejected(self):
        self.fixture.full["train"]["subjects"][0] = "p02"
        self.rejected("Fitting subject partition changed", before_arrays=True)
        self.fixture.full["train"]["subjects"] = list(builder.FIT_SUBJECTS)
        self.fixture.protocol["folds"]["A"][0] = "p03"
        self.rejected("Original frozen folds differ", before_arrays=True)

    def test_cached_row_subject_and_count_guards(self):
        original = self.fixture.cached["subjects"][0]
        self.fixture.cached["subjects"][0] = "p02"
        self.rejected("non-fitting subject")
        self.fixture.cached["subjects"][0] = original
        self.fixture.cached["starts"] = self.fixture.cached["starts"][:-1]
        self.rejected("Expected 31,859 cached fitting rows")

    def test_every_cached_target_mask_start_and_overlap_is_replayed(self):
        for name in ("starts", "targets", "target_mask", "tv_drinking_overlap_frames"):
            with self.subTest(array=name):
                values = self.fixture.cached[name]
                original = values.flat[0].copy()
                values.flat[0] = not original if values.dtype.kind == "b" else original + 1
                self.rejected("P03T01C01 " + name, error=AssertionError)
                values.flat[0] = original

    def test_per_video_cached_subject_must_match_even_if_global_subject_set_matches(self):
        self.fixture.cached["subjects"][0] = "p06"
        self.rejected("Pilot subject differs")

    def test_source_clock_and_source_hash_must_match_the_plan(self):
        self.fixture.first_plan["source_fps"] = 30.
        self.rejected("Source clock differs")
        self.fixture.first_plan["source_fps"] = 25.
        self.fixture.first_plan["source_sha256"] = digest("different-source")
        self.rejected("Video provenance differs")

    def test_non_training_annotations_are_rejected(self):
        video = self.fixture.annotations["P03T01C01"]
        self.fixture.annotations[video.vid] = replace(video, subset="testing")
        self.rejected("Non-fitting annotation")

    def test_duplicate_or_non_grid_retained_starts_fail_the_real_full_audit(self):
        starts = self.fixture.first_shard["extraction"]["retained_starts"]
        for invalid in (0, 16):
            with self.subTest(start=invalid):
                starts[1] = invalid
                self.rejected("retained windows disagree")

    def test_incomplete_eligible_grid_cannot_claim_full_coverage(self):
        report = self.fixture.first_shard["extraction"]
        report["selected_starts"] = report["selected_starts"][:-1]
        report["selected_windows"] -= 1
        self.rejected("omitted eligible full-coverage windows")

    def test_legacy_pair_stratum_cannot_be_replaced_with_colabel_counts(self):
        # The synthetic final video has co-label support, but no intersection.
        self.fixture.plan["job"]["video_inventory"][273]["pair_windows"] = 1
        self.rejected("Legacy pair stratum differs")

    def test_aggregate_positive_drift_is_rejected_after_reconstruction(self):
        self.fixture.full["train"]["positive"][DRINK] += 1
        self.rejected("Full fitting aggregate differs: positive_counts")


class ArtifactPreservationTests(unittest.TestCase):
    def test_identical_repeat_is_a_noop_and_divergent_snapshot_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "inventory.json"
            value = {"schema": "fixture", "n_windows": 42}
            builder.write_preserving(path, value)
            original, timestamp = path.read_bytes(), path.stat().st_mtime_ns
            builder.write_preserving(path, value)
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), (original, timestamp))
            with self.assertRaisesRegex(ValueError, "Existing artifact differs; preserve it"):
                builder.write_preserving(path, {**value, "n_windows": 43})
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), (original, timestamp))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    total = result.testsRun - len(result.skipped)
    passed = total - len(result.failures) - len(result.errors)
    print(f"{passed}/{total} passed; {len(result.skipped)} skipped")
    raise SystemExit(0 if result.wasSuccessful() else 1)
