"""Timestamp, overlap and leakage guards for the untrimmed multilabel data path."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behaviorsense.data.multilabel_skeleton import (  # noqa: E402
    load_multilabel_shards,
    targets_for_windows,
)
from behaviorsense.data import multilabel_skeleton as multilabel_data
from behaviorsense.data.toyota import (  # noqa: E402
    COARSE_V11,
    TSU_CLASSES,
    TSU_TO_COARSE,
    TsuVideo,
    supervised_coarse,
)

TV = COARSE_V11.index("watching_tv")
DRINK = COARSE_V11.index("drinking")
TV_FINE = TSU_CLASSES.index("Watch_TV")
CUP = TSU_CLASSES.index("Drink.From_cup")
BOTTLE = TSU_CLASSES.index("Drink.From_bottle")
READ_FINE = TSU_CLASSES.index("Read")


def _video(actions, subject=3, duration=1000):
    return TsuVideo(vid=f"P{subject:02d}T01C01", subject=subject, take=1, camera=1,
                    subset="testing" if subject in (2, 10) else "training",
                    duration=duration, actions=tuple(actions))


def _annotations(path, videos):
    path.write_text(json.dumps({video.vid: {"subset": video.subset, "duration": video.duration,
                                           "actions": video.actions} for video in videos}),
                    encoding="utf-8")
    return path


def _shard(path, subjects=(3,), starts=(0,), **overrides):
    starts = np.asarray(starts, dtype=np.int64)
    n = len(starts)
    data = {
        "skeletons": np.zeros((n, 30, 2, 17, 3), dtype=np.float16),
        "subjects": np.asarray([f"p{subject:02d}" for subject in subjects]),
        "clips": np.asarray([f"P{subject:02d}T01C01.mp4" for subject in subjects]),
        "starts": starts,
        "datasets": np.asarray(["toyota_untrimmed"] * n),
        "source_frames": np.rint((starts[:, None] + np.arange(30)) * 25 / 20).astype(np.int64),
        "source_fps": np.full(n, 25, dtype=np.float32),
        # Deliberately invalid legacy labels: successful loading proves they cannot erase
        # simultaneous ground truth or leak into the new targets.
        "labels": np.full(n, 999, dtype=np.int64),
        "fine_labels": np.full(n, 999, dtype=np.int64),
    }
    data.update(overrides)
    data = {key: value for key, value in data.items() if value is not None}
    np.savez_compressed(path, **data)
    return path


def _reject(call, fragment):
    try:
        call()
    except ValueError as error:
        assert fragment in str(error), str(error)
    else:
        raise AssertionError(f"invalid data accepted; expected {fragment!r}")


def test_md1_original_overlap_overrides_argmax_and_preserves_both_official_sides():
    with tempfile.TemporaryDirectory() as temporary:
        tmp = Path(temporary)
        videos = [_video([(TV_FINE, 0, 300), (CUP, 25, 150)], subject) for subject in (3, 2)]
        annotation = _annotations(tmp / "annotations.json", videos)
        shard = _shard(tmp / "toyota_untrimmed_20hz_0000.npz", (3, 2), (20, 20))
        data = load_multilabel_shards([shard], annotation)
        assert data["targets"].shape == data["target_mask"].shape == (2, len(COARSE_V11))
        assert np.all(data["targets"][:, [TV, DRINK]] == 1)
        assert data["tv_drinking_overlap_frames"].tolist() == [30, 30]
        assert data["subjects"].tolist() == ["p03", "p02"]
        assert data["video_ids"].tolist() == ["P03T01C01", "P02T01C01"]
        assert data["starts"].tolist() == [20, 20]
        assert np.flatnonzero(data["target_mask"][0]).tolist() == list(supervised_coarse(TSU_TO_COARSE))
        assert not data["target_mask"][:, COARSE_V11.index("sitting")].any()
        assert data["skeletons"].shape == (2, 30, 2, 17, 3)
        provenance = data["provenance"]
        assert provenance["class_names"] == list(COARSE_V11)
        assert provenance["annotations_sha256"] == hashlib.sha256(annotation.read_bytes()).hexdigest()
        assert provenance["source_shards"][0]["sha256"] == hashlib.sha256(shard.read_bytes()).hexdigest()
        assert provenance["source_shards"][0]["clock"] == "source_frames"
        assert provenance["legacy_single_label_targets_used"] is False


def test_md2_union_does_not_double_count_and_colabel_is_not_simultaneity():
    starts = np.asarray([0], dtype=np.int64)
    frames = np.rint(np.arange(30)[None] * 25 / 20).astype(np.int64)
    # Each action covers exactly half the sampled window; no frame has both actions.
    adjacent = _video([(TV_FINE, 0, 18), (CUP, 19, 37)])
    target, overlap = targets_for_windows(adjacent, starts, source_frames=frames, source_fps=25)
    assert target[0, TV] == target[0, DRINK] == 1
    assert overlap[0] == 0

    duplicate = _video([(TV_FINE, 0, 100), (CUP, 0, 9), (BOTTLE, 0, 9)])
    target, overlap = targets_for_windows(duplicate, starts, source_frames=frames, source_fps=25)
    assert target[0, DRINK] == 0 and overlap[0] == 8
    consecutive = _video([(TV_FINE, 0, 100), (CUP, 0, 9), (BOTTLE, 10, 18)])
    target, overlap = targets_for_windows(consecutive, starts, source_frames=frames, source_fps=25)
    assert target[0, DRINK] == 1 and overlap[0] == 15


def test_md3_gaps_are_masked_and_filtered_rows_keep_their_true_times():
    with tempfile.TemporaryDirectory() as temporary:
        tmp = Path(temporary)
        annotation = _annotations(tmp / "annotations.json", [_video([(TV_FINE, 0, 100),
                                                                       (READ_FINE, 225, 300)])])
        # Starts are deliberately nonconsecutive and out of time order. A row-index-derived
        # clock would call all three TV, and a blanket mask would train on the final gap.
        shard = _shard(tmp / "toyota_untrimmed_20hz_0000.npz", (3, 3, 3), (180, 0, 400))
        data = load_multilabel_shards([shard], annotation)
        assert data["targets"][:, TV].tolist() == [0, 1, 0]
        assert data["targets"][:, COARSE_V11.index("reading")].tolist() == [1, 0, 0]
        assert data["annotation_coverage"].tolist() == [1, 1, 0]
        assert not data["target_mask"][2].any()
        assert data["provenance"]["n_masked_gap_windows"] == 1


def test_md4_exact_decoder_rounding_is_used_and_rate_drift_is_refused():
    starts = np.asarray([0], dtype=np.int64)
    frames = np.rint(np.arange(30)[None] * 25 / 20).astype(np.int64)
    video = _video([(TV_FINE, 8, 25)])
    exact, _ = targets_for_windows(video, starts, source_frames=frames, source_fps=25)
    nominal, _ = targets_for_windows(video, starts)
    assert exact[0, TV] == 1 and nominal[0, TV] == 0
    shifted = frames + 1
    _reject(lambda: targets_for_windows(video, starts, source_frames=shifted, source_fps=25),
            "disagree with starts")
    _reject(lambda: targets_for_windows(video, starts, source_frames=frames), "supplied together")
    _reject(lambda: targets_for_windows(video, starts, sample_fps=float("nan")), "sample_fps")
    _reject(lambda: targets_for_windows(video, starts, min_overlap=0), "min_overlap")


def test_md5_missing_timestamps_trimmed_inputs_duplicates_and_identity_leaks_fail():
    with tempfile.TemporaryDirectory() as temporary:
        tmp = Path(temporary)
        annotation = _annotations(tmp / "annotations.json", [_video([(TV_FINE, 0, 500)])])
        missing = _shard(tmp / "missing.npz")
        with np.load(missing) as source:
            old_values = {name: source[name] for name in source.files if name != "starts"}
        np.savez_compressed(missing, **old_values)
        _reject(lambda: load_multilabel_shards([missing], annotation), "missing ['starts']")
        trimmed = _shard(tmp / "trimmed.npz", datasets=np.asarray(["toyota_trimmed"]))
        _reject(lambda: load_multilabel_shards([trimmed], annotation), "only toyota_untrimmed")
        wrong_subject = _shard(tmp / "wrong_subject.npz", subjects=(2,),
                               clips=np.asarray(["P03T01C01.mp4"]))
        _reject(lambda: load_multilabel_shards([wrong_subject], annotation), "subject disagrees")
        first = _shard(tmp / "first.npz")
        second = _shard(tmp / "second.npz")
        _reject(lambda: load_multilabel_shards([first, second], annotation), "duplicate window")
        _reject(lambda: load_multilabel_shards([first, first], annotation), "more than once")
        wrong_rate = _shard(tmp / "toyota_untrimmed_10hz_0000.npz")
        _reject(lambda: load_multilabel_shards([wrong_rate], annotation), "filename rate")
        bad_shape = _shard(tmp / "bad_shape.npz", skeletons=np.zeros((1, 30, 1, 17, 3), np.float32))
        _reject(lambda: load_multilabel_shards([bad_shape], annotation), "expected skeletons")
        bad_start = _shard(tmp / "bad_start.npz")
        with np.load(bad_start) as source:
            values = {name: source[name] for name in source.files}
        values["starts"] = np.asarray([0.5])
        np.savez_compressed(bad_start, **values)
        _reject(lambda: load_multilabel_shards([bad_start], annotation), "nonnegative integers")
        wrong_subset = tmp / "wrong_subset.json"
        wrong_subset.write_text(json.dumps({"P03T01C01": {"subset": "testing", "duration": 1000,
                                                          "actions": [[TV_FINE, 0, 500]]}}))
        _reject(lambda: load_multilabel_shards([first], wrong_subset), "cross-subject split")


def test_md6_audited_empty_extractions_are_counted_but_never_become_training_rows():
    with tempfile.TemporaryDirectory() as temporary:
        tmp = Path(temporary)
        annotation = _annotations(tmp / "annotations.json", [_video([(TV_FINE, 0, 500)])])
        good = _shard(tmp / "good.npz")
        report = np.asarray(json.dumps({"schema_version": 1, "retained_windows": 0}))
        empty = _shard(tmp / "empty.npz", subjects=(), starts=(), extraction_metadata=report)
        data = load_multilabel_shards([empty, good], annotation)
        assert len(data["skeletons"]) == 1
        assert data["provenance"]["n_empty_shards_skipped"] == 1
        assert data["provenance"]["source_shards"][0]["excluded"] == "empty_extraction"
        _reject(lambda: load_multilabel_shards([empty], annotation), "all supplied shards are empty")
        unaudited = _shard(tmp / "unaudited.npz", subjects=(), starts=())
        _reject(lambda: load_multilabel_shards([unaudited, good], annotation), "no extraction audit")
        wrong_clock = _shard(tmp / "wrong_clock.npz", annotation_fps=np.asarray(30.0))
        _reject(lambda: load_multilabel_shards([wrong_clock], annotation), "stored annotation_fps")


def _full_fixture(directory):
    video = _video([(TV_FINE, 0, 100)], duration=100)
    annotation = _annotations(directory / "annotations.json", [video])
    starts = [0, 15, 30, 45]
    identity = {"config_fingerprint": "a" * 64, "video_id": video.vid,
                "subset": "training", "source_sha256": "b" * 64}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    report = dict(identity, fingerprint=fingerprint, schema_version=2, subject="p03",
        selection_mode="full", window_cap=None, full_dense_coverage=True, candidate_stride=15,
        source_fps=25., source_frame_count=100, candidate_windows=4, selected_windows=4,
        excluded_annotation_coverage=0, retained_windows=4, dropped_visibility=0,
        selected_starts=starts, retained_starts=starts)
    skeletons = np.zeros((4, 30, 2, 17, 3), np.float16)
    skeletons[:, :15, 0, :8, 2] = .9
    path = _shard(directory / "full.npz", subjects=(3,) * 4, starts=starts,
                  skeletons=skeletons, extraction_metadata=np.asarray(json.dumps(report)),
                  fingerprint=np.asarray(fingerprint))
    return path, annotation


def _load_full_fixture(path, annotation, **kwargs):
    inventory = hashlib.sha256(b"P03T01C01").hexdigest()
    with patch.dict(multilabel_data.OFFICIAL_VIDEO_COUNTS, training=1), patch.dict(
            multilabel_data.OFFICIAL_INVENTORY_SHA256, training=inventory):
        return load_multilabel_shards(path, annotation, require_full_coverage=True,
                                     expected_subset="training", **kwargs)


def _mutate_shard(path, mutation):
    with np.load(path, allow_pickle=False) as shard:
        payload = {key: shard[key] for key in shard.files}
    mutation(payload)
    np.savez_compressed(path, **payload)


def test_md7_full_audit_binds_source_hash_and_both_fingerprints():
    with tempfile.TemporaryDirectory() as temporary:
        tmp = Path(temporary)
        path, annotation = _full_fixture(tmp)
        data = _load_full_fixture(path, annotation)
        assert data["provenance"]["full_coverage_verified"] is True
        assert len(data["skeletons"]) == 4

        def changed_source(payload):
            report = json.loads(str(payload["extraction_metadata"]))
            report["source_sha256"] = "c" * 64
            payload["extraction_metadata"] = np.asarray(json.dumps(report))
        _mutate_shard(path, changed_source)
        _reject(lambda: _load_full_fixture(path, annotation), "fingerprint disagrees with source")
        path, annotation = _full_fixture(tmp)
        _mutate_shard(path, lambda payload: payload.update(fingerprint=np.asarray("d" * 64)))
        _reject(lambda: _load_full_fixture(path, annotation), "stored shard fingerprint")


def test_md8_full_audit_binds_row_clock_to_the_audited_video():
    with tempfile.TemporaryDirectory() as temporary:
        path, annotation = _full_fixture(Path(temporary))
        def changed_rate(payload):
            payload["source_fps"] = np.full(4, 30.)
            payload["source_frames"] = np.rint(
                (payload["starts"][:, None] + np.arange(30)) * 30 / 20).astype(np.int64)
        _mutate_shard(path, changed_rate)
        _reject(lambda: _load_full_fixture(path, annotation), "row source clock disagrees")


def test_md9_full_audit_rechecks_retained_primary_person_visibility():
    with tempfile.TemporaryDirectory() as temporary:
        tmp = Path(temporary)
        path, annotation = _full_fixture(tmp)
        _mutate_shard(path, lambda payload: payload["skeletons"].fill(0))
        _reject(lambda: _load_full_fixture(path, annotation), "visibility gate")
        path, annotation = _full_fixture(tmp)
        def just_below_frame_floor(payload):
            payload["skeletons"][0, 14, 0, :, 2] = 0
        _mutate_shard(path, just_below_frame_floor)
        _reject(lambda: _load_full_fixture(path, annotation), "visibility gate")
        path, annotation = _full_fixture(tmp)
        # The stored float16 representation of .3 is > .3 in float32. The
        # extraction path can retain it, so the recheck must not round its cutoff.
        _mutate_shard(path, lambda payload: payload["skeletons"].__setitem__((slice(None), slice(0, 15), 0, slice(0, 8), 2), .3))
        assert _load_full_fixture(path, annotation)["provenance"]["full_coverage_verified"]


def test_md10_full_data_cache_stays_disk_backed_and_preserves_labels():
    with tempfile.TemporaryDirectory() as temporary:
        tmp = Path(temporary)
        path, annotation = _full_fixture(tmp)
        data = _load_full_fixture(path, annotation, cache_dir=tmp / "cache")
        assert isinstance(data["skeletons"], np.memmap)
        assert data["targets"][:, TV].tolist() == [1., 1., 1., 1.]
        assert data["provenance"]["storage"].startswith("disk_backed")
        data["skeletons"]._mmap.close()  # Permit TemporaryDirectory cleanup on Windows.


if __name__ == "__main__":
    tests = [value for key, value in sorted(globals().items()) if key.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as error:
            failures += 1
            print(f"FAIL {test.__name__}: {type(error).__name__}: {error}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    sys.exit(bool(failures))
