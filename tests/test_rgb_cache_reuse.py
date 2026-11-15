"""Training-only cache reuse checks with synthetic bytes and a mocked array loader."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rgb_cache_reuse as reuse


def digest(value):
    return hashlib.sha256(value).hexdigest()


class CacheFixture:
    def __init__(self, directory, monkeypatch):
        self.mount, self.work = directory / "mount", directory / "work"
        self.output = self.mount / "frozen-output"
        self.output.mkdir(parents=True)
        self.shards, self.metadata, self.paths = {}, {}, {}
        self.loaded = []
        self.clock = 1.0
        implementation_files = {"extractor": "scripts/extract_rgb_action_features.py",
                                "rgb_action": "src/behaviorsense/models/rgb_action.py"}
        implementation = {key: digest((ROOT / path).read_bytes()) for key, path in implementation_files.items()}
        self.info = {"arch": "swin3d_t", "weights_sha256": "e" * 64, "feature_dim": 768,
                     "preprocessing": {"fixture": True}, "preprocessing_sha256": "f" * 64}
        self.feature_contract = {"schema": "behaviorsense.rgb_action_features.v1", "arch": "swin3d_t",
            "feature_dim": 768, "encoder_sha256": self.info["weights_sha256"], "encoder_info": self.info,
            "preprocessing_sha256": self.info["preprocessing_sha256"], "views": ["scene", "person"],
            "extraction_implementation_sha256": reuse.canonical_sha256(implementation)}
        fitting = [f"P{subject[1:]}T{index:02}C01" for subject in reuse.FIT_SUBJECTS for index in range(1, 5)]
        validation = [f"P{reuse.VALIDATION_SUBJECTS[index % 2][1:]}T{index // 2 + 1:02}C01"
                      for index in range(77)]
        self.probes = [fitting[0], fitting[4], fitting[8], "P03T99C01", "P06T99C01"]
        records, inventory, source_reports = [], [], []
        for vid in fitting + validation + self.probes[-2:] + ["P02T01C01"]:
            shard = directory / "pose" / f"{vid}.npz"
            shard.parent.mkdir(exist_ok=True)
            shard.write_bytes(f"synthetic pose {vid}".encode())
            source = {"video_id": vid, "subject": "p" + vid[1:3], "n_windows": 2,
                "subset": "testing" if vid == "P02T01C01" else "training",
                "source_sha256": digest(vid.encode()), "shard_sha256": digest(shard.read_bytes())}
            inventory.append(source)
            # Deliberately include a test mapping: selection must reject it before loading.
            self.shards[vid] = shard
            if vid not in fitting + validation:
                continue
            path = self.output / "rgb_features" / "swin3d_t" / f"{vid}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"synthetic RGB {vid}".encode())
            self.paths[vid] = path
            record = {"video_id": vid, "subject": source["subject"], "subset": "training",
                "role": "fitting" if vid in fitting else "validation", "n_windows": 2,
                "relative_path": f"rgb_features/swin3d_t/{vid}.npz", "feature_sha256": digest(path.read_bytes()),
                "source_video_sha256": source["source_sha256"], "source_shard_sha256": source["shard_sha256"],
                "fingerprint": digest((vid + "fingerprint").encode()),
                **{key: self.feature_contract[key] for key in ("encoder_sha256", "preprocessing_sha256",
                   "extraction_implementation_sha256", "feature_dim", "arch")}}
            metadata = {**record, "status": "complete", "decoded_frames": 90, "retrieved_frames": 80,
                "encoder_views": 4, "peak_cached_frames": 53, "prefetch_batches": 2, "spatial_backend": "native"}
            self.metadata[vid] = metadata
            source_reports.append({key: metadata[key] for key in reuse.REPORT_FIELDS} |
                {"output_sha256": record["feature_sha256"], "reused": False})
            records.append(record)
        source_id = reuse.SOURCE_KERNEL.rsplit("/", 1)[0]
        source_identity = {"job_id": source_id, "source_sha256": "a" * 64, "extraction_source_sha256": "b" * 64}
        feature_inventory = [{key: row[key] for key in reuse.RECEIPT_FIELDS} for row in records]
        inventory_sha = reuse.canonical_sha256(feature_inventory)
        self.documents = {
            "job.json": {**source_identity, "pilot": {"pilot_video_ids": fitting}, "video_inventory": inventory},
            "source_manifest.json": {"sha256": source_identity["source_sha256"],
                "files": {path: implementation[key] for key, path in implementation_files.items()}},
            "extraction_swin3d_t.json": source_reports,
            "encoder_selection.json": {"job_id": source_id, "source_sha256": source_identity["source_sha256"],
                                       "arch": "swin3d_t", "encoder_info": self.info},
            "stage_result.json": {**source_identity, "status": "quality_rejected", "test_features_loaded": False,
                                  "completed_pilot_videos": 113, "total_pilot_videos": 113},
            "pilot/manifest.json": {"feature_contract": self.feature_contract, "feature_inventory": feature_inventory,
                                    "feature_inventory_sha256": inventory_sha}}
        self.cache = {"kernel_source": reuse.SOURCE_KERNEL, "source_job_id": source_id,
            "source_sha256": source_identity["source_sha256"], "extraction_source_sha256": source_identity["extraction_source_sha256"],
            "feature_contract": self.feature_contract, "feature_contract_sha256": reuse.canonical_sha256(self.feature_contract),
            "feature_inventory_sha256": inventory_sha, "records": records, "provenance_files": []}
        for name in self.documents:
            self.save_document(name)
        self.job = {"extraction_source_sha256": source_identity["extraction_source_sha256"],
                    "video_inventory": copy.deepcopy(inventory),
                    "coverage": {"cache_reuse": self.cache, "probe_video_ids": self.probes}}
        monkeypatch.setattr(reuse, "dataset_root", lambda owner, slug: self.mount)
        monkeypatch.setattr(reuse, "validate_rgb_cache", self.validate)
        monkeypatch.setattr(reuse.time, "monotonic", lambda: self.clock)

    def save_document(self, name):
        path = self.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.documents[name]), encoding="utf-8")
        specification = {"relative_path": name, "sha256": digest(path.read_bytes())}
        self.cache["provenance_files"] = [entry for entry in self.cache["provenance_files"]
                                          if entry["relative_path"] != name] + [specification]

    def validate(self, source, shard, deadline):
        vid = source.stem
        assert source == self.paths[vid]
        assert shard == self.shards[vid]
        assert deadline == 100
        self.loaded.append(vid)
        return object(), copy.deepcopy(self.feature_contract), copy.deepcopy(self.metadata[vid])

    def run(self, deadline=100):
        return reuse.prepare_reused_features(self.job, self.work, self.shards, self.info, deadline)

    def assert_no_publication(self):
        assert not (self.work / "cache_reuse.json").exists()
        assert not list(self.work.glob(".rgb-cache-reuse-*"))
        assert not list(self.work.glob("rgb_features/**/*.npz"))


@pytest.fixture
def cache(tmp_path, monkeypatch):
    return CacheFixture(tmp_path, monkeypatch)


def test_verifies_all_113_and_installs_only_110_with_exact_reports(cache):
    source_before = {path: path.read_bytes() for path in cache.output.rglob("*") if path.is_file()}
    reports = cache.run()
    assert len(cache.loaded) == len(set(cache.loaded)) == 113
    assert len(reports) == 110
    assert not set(reports) & set(cache.probes)
    assert "P02T01C01" not in cache.loaded
    for vid, report in reports.items():
        assert set(report) == set(reuse.REPORT_FIELDS) | {
            "reused", "cache_reuse_verified", "elapsed_seconds", "output_path", "output_sha256"}
        assert report["reused"] is True and report["cache_reuse_verified"] is True
        assert Path(report["output_path"]).read_bytes() == cache.paths[vid].read_bytes()
        assert report["output_sha256"] == digest(cache.paths[vid].read_bytes())
    receipt = json.loads((cache.work / "cache_reuse.json").read_text())
    assert receipt["status"] == "complete" and receipt["verified"] is True
    assert (receipt["verified_source_files"], receipt["verified_fitting_files"], receipt["verified_validation_files"]) == (113, 36, 77)
    assert (receipt["installed_files"], receipt["installed_fitting_files"], receipt["installed_validation_files"]) == (110, 33, 77)
    assert receipt["test_features_loaded"] is False
    assert set(receipt["fresh_probe_video_ids"]) == set(cache.probes)
    assert set(receipt["skipped_cached_probe_video_ids"]) == set(cache.probes[:3])
    assert all(row["source_sha256_verified"] and row["array_validation_verified"] for row in receipt["records"])
    assert {path: path.read_bytes() for path in source_before} == source_before
    assert not list(cache.work.glob(".rgb-cache-reuse-*"))


@pytest.mark.parametrize("index", [0, 35, 112])
def test_bad_feature_bytes_rejected_including_skipped_probe(cache, index):
    record = cache.cache["records"][index]
    cache.paths[record["video_id"]].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="cache checksum mismatch"):
        cache.run()
    assert record["video_id"] not in cache.loaded
    cache.assert_no_publication()


def test_missing_file_is_not_a_partial_reuse_success(cache):
    cache.paths[cache.probes[0]].rename(cache.paths[cache.probes[0]].with_suffix(".npz.partial"))
    with pytest.raises(ValueError, match="Missing or ambiguous"):
        cache.run()
    assert not cache.loaded
    cache.assert_no_publication()


def test_duplicate_same_bytes_mount_rejected(cache):
    path = cache.paths[cache.probes[0]]
    duplicate = cache.mount / "duplicate" / "rgb_features" / "swin3d_t" / path.name
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="ambiguous mounted"):
        cache.run()
    assert not cache.loaded


def test_duplicate_provenance_mount_rejected_even_with_identical_bytes(cache):
    duplicate = cache.mount / "duplicate" / "job.json"
    duplicate.parent.mkdir()
    duplicate.write_bytes((cache.output / "job.json").read_bytes())
    with pytest.raises(ValueError, match="ambiguous mounted"):
        cache.run()
    assert not cache.loaded


def test_profiling_cache_outside_reuse_directory_is_not_selected(cache):
    path = cache.output / "profiling" / "swin3d_t" / "baseline" / f"{cache.probes[0]}.npz"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"unrelated baseline probe")
    assert len(cache.run()) == 110


def test_unexpected_test_cache_detected_without_loading(cache):
    test_path = cache.output / "rgb_features" / "swin3d_t" / "P02T01C01.npz"
    test_path.write_bytes(b"must never be loaded")
    with pytest.raises(ValueError, match="forbidden test selection"):
        cache.run()
    assert not cache.loaded
    cache.assert_no_publication()


def test_test_record_cannot_replace_training_record(cache):
    row = cache.cache["records"][0]
    row.update(video_id="P02T01C01", subset="testing", subject="p02")
    cache.job["coverage"]["probe_video_ids"] = [cache.cache["records"][index]["video_id"] for index in (1, 4, 8)] + cache.probes[-2:]
    with pytest.raises(ValueError, match="forbidden training source"):
        cache.run()
    assert not cache.loaded


@pytest.mark.parametrize("field", ["encoder_sha256", "preprocessing_sha256", "source_video_sha256",
                                    "source_shard_sha256", "fingerprint", "n_windows", "extraction_implementation_sha256"])
def test_validated_metadata_still_must_match_frozen_record(cache, field):
    cache.metadata[cache.probes[0]][field] = 99 if field == "n_windows" else "0" * 64
    with pytest.raises(ValueError, match="rows/source/provenance mismatch"):
        cache.run()
    cache.assert_no_publication()


@pytest.mark.parametrize("field,value", [("status", "partial"), ("prefetch_batches", 0),
                                         ("spatial_backend", "reference"), ("decoded_frames", -1)])
def test_partial_or_changed_extraction_configuration_rejected(cache, field, value):
    cache.metadata[cache.probes[0]][field] = value
    with pytest.raises(ValueError, match="Incomplete or changed"):
        cache.run()
    cache.assert_no_publication()


def test_array_validator_failure_rejects_nonfinite_or_wrong_row_cache(cache, monkeypatch):
    def reject(source, shard, deadline):
        assert source == cache.paths[cache.probes[0]] and shard == cache.shards[cache.probes[0]]
        raise ValueError("RGB features must be finite float16 [N,2,D]")
    monkeypatch.setattr(reuse, "validate_rgb_cache", reject)
    with pytest.raises(ValueError, match="finite float16"):
        cache.run()
    cache.assert_no_publication()


def test_loaded_encoder_identity_must_match_cache(cache):
    info = copy.deepcopy(cache.info)
    info["weights_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="loaded encoder"):
        reuse.prepare_reused_features(cache.job, cache.work, cache.shards, info, 100)
    assert not cache.loaded


def test_validator_contract_must_match_frozen_cache(cache, monkeypatch):
    def different(source, shard, deadline):
        values, contract, metadata = cache.validate(source, shard, deadline)
        contract["encoder_info"]["precision"] = "changed"
        return values, contract, metadata
    monkeypatch.setattr(reuse, "validate_rgb_cache", different)
    with pytest.raises(ValueError, match="encoder/preprocessing/implementation mismatch"):
        cache.run()
    cache.assert_no_publication()


def test_source_implementation_must_be_unchanged(cache):
    cache.documents["source_manifest.json"]["files"]["scripts/extract_rgb_action_features.py"] = "0" * 64
    cache.save_document("source_manifest.json")
    with pytest.raises(ValueError, match="implementation changed"):
        cache.run()
    assert not cache.loaded


@pytest.mark.parametrize("name", sorted(reuse.PROVENANCE_FILES))
def test_each_source_provenance_file_is_byte_verified(cache, name):
    with (cache.output / name).open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="provenance checksum mismatch"):
        cache.run()
    assert not cache.loaded


def test_pose_hash_rechecked_before_array_validation(cache):
    cache.shards[cache.probes[0]].write_bytes(b"different source rows")
    with pytest.raises(ValueError, match="pose shard checksum mismatch"):
        cache.run()
    assert not cache.loaded
    cache.assert_no_publication()


@pytest.mark.parametrize("deadline", [None, float("inf"), float("nan"), True])
def test_deadline_must_be_finite(cache, deadline):
    with pytest.raises(ValueError, match="finite absolute"):
        cache.run(deadline)
    assert not cache.loaded


def test_expired_deadline_reads_no_arrays(cache):
    cache.clock = 100
    with pytest.raises(TimeoutError, match="probe deadline"):
        cache.run()
    assert not cache.loaded


def test_deadline_around_array_load_cleans_staged_files(cache, monkeypatch):
    def expire(source, shard, deadline):
        value = cache.validate(source, shard, deadline)
        if len(cache.loaded) == 20:
            cache.clock = 100
        return value
    monkeypatch.setattr(reuse, "validate_rgb_cache", expire)
    with pytest.raises(TimeoutError, match="probe deadline"):
        cache.run()
    assert len(cache.loaded) == 20
    cache.assert_no_publication()


def test_hashing_checks_deadline_between_chunks(cache, monkeypatch):
    calls = []
    monkeypatch.setattr(reuse, "CHUNK_BYTES", 1)
    def check(deadline):
        calls.append(deadline)
        if len(calls) == 10:
            raise TimeoutError("chunk deadline")
    monkeypatch.setattr(reuse, "_deadline", check)
    with pytest.raises(TimeoutError, match="chunk deadline"):
        reuse._hash_file(cache.paths[cache.probes[0]], 100)
    assert len(calls) == 10


def test_copy_detects_source_change_after_array_validation(cache, monkeypatch):
    copy_verified = reuse._copy_verified
    def changed(source, target, expected, deadline):
        source.write_bytes(b"changed between validation and copy")
        copy_verified(source, target, expected, deadline)
    monkeypatch.setattr(reuse, "_copy_verified", changed)
    with pytest.raises(ValueError, match="changed during its verified copy"):
        cache.run()
    cache.assert_no_publication()


def test_skipped_probe_source_change_after_array_validation_is_rejected(cache, monkeypatch):
    def changed(source, shard, deadline):
        result = cache.validate(source, shard, deadline)
        source.write_bytes(b"changed during array validation")
        return result
    monkeypatch.setattr(reuse, "validate_rgb_cache", changed)
    with pytest.raises(ValueError, match="changed during probe verification"):
        cache.run()
    cache.assert_no_publication()


def test_deadline_during_publication_rolls_back_installed_files(cache, monkeypatch):
    replace = Path.replace
    published = []
    def expire(self, target):
        result = replace(self, target)
        if self.suffix == ".npz":
            published.append(target)
            if len(published) == 4:
                cache.clock = 100
        return result
    monkeypatch.setattr(Path, "replace", expire)
    with pytest.raises(TimeoutError, match="probe deadline"):
        cache.run()
    assert len(published) == 4
    cache.assert_no_publication()


def test_receipt_write_failure_rolls_back_all_installed_files(cache, monkeypatch):
    def fail(path, value):
        assert value["verified_source_files"] == 113 and value["installed_files"] == 110
        path.with_suffix(".json.partial").write_text("partial")
        raise OSError("disk full")
    monkeypatch.setattr(reuse, "write_json", fail)
    with pytest.raises(OSError, match="disk full"):
        cache.run()
    cache.assert_no_publication()
    assert not (cache.work / "cache_reuse.json.partial").exists()


def test_cannot_write_inside_source_mount(cache):
    with pytest.raises(ValueError, match="inside its mounted source"):
        reuse.prepare_reused_features(cache.job, cache.output / "work", cache.shards, cache.info, 100)
    assert not cache.loaded


@pytest.mark.parametrize("index", [0, 3, 4])
def test_existing_destination_is_preserved_and_rejected(cache, index):
    output = cache.work / "rgb_features" / "swin3d_t" / f"{cache.probes[index]}.npz"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"must be fresh")
    with pytest.raises(ValueError, match="clean destination"):
        cache.run()
    assert output.read_bytes() == b"must be fresh"
    assert not cache.loaded


def test_destination_symlink_escape_is_rejected(cache, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    cache.work.mkdir()
    try:
        (cache.work / "rgb_features").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("This Windows account cannot create directory symlinks")
    with pytest.raises(ValueError, match="escapes the work directory"):
        cache.run()
    assert not cache.loaded
    assert not list(outside.iterdir())


if __name__ == "__main__":
    class CountSummary:
        def pytest_terminal_summary(self, terminalreporter):
            stats = terminalreporter.stats
            passed = len(stats.get("passed", []))
            failed = len(stats.get("failed", [])) + len(stats.get("error", []))
            skipped = len(stats.get("skipped", []))
            terminalreporter.write_line(f"{passed}/{passed + failed} passed; {skipped} skipped")

    raise SystemExit(pytest.main([str(Path(__file__).resolve()), "-q", "--color=no",
                                 "-p", "no:cacheprovider"], plugins=[CountSummary()]))
