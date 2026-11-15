"""Verify a frozen training-only RGB cache before publishing reusable features.

The mounted kernel output is read-only.  Every expected source file, including
the old copies of fresh probes, is checked before any reusable file is installed.
No cache hit can supply a fresh throughput measurement.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import tempfile
import time

from multilabel_kaggle_stage import dataset_root, write_json
from temporal_kaggle_stage import FIT_SUBJECTS, VALIDATION_SUBJECTS, canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
SOURCE_KERNEL = "krishnagupta02468/bs-rgb-prefetch-20261013/1"
PROVENANCE_FILES = {"job.json", "source_manifest.json", "extraction_swin3d_t.json",
                    "encoder_selection.json", "stage_result.json", "pilot/manifest.json"}
REPORT_FIELDS = ("status", "n_windows", "video_id", "subject", "arch", "feature_dim",
    "source_video_sha256", "source_shard_sha256", "encoder_sha256", "preprocessing_sha256",
    "decoded_frames", "retrieved_frames", "encoder_views", "peak_cached_frames")
RECEIPT_FIELDS = ("video_id", "n_windows", "feature_sha256", "source_video_sha256",
                  "source_shard_sha256", "fingerprint")
CHUNK_BYTES = 8 * 1024 * 1024


def _deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("RGB cache verification reached its finite probe deadline")


def _sha_value(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("RGB cache provenance requires a SHA-256 digest")


def _hash_file(path, deadline):
    _deadline(deadline)
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            _deadline(deadline)
            block = handle.read(CHUNK_BYTES)
            _deadline(deadline)
            if not block:
                break
            digest.update(block)
    _deadline(deadline)
    return digest.hexdigest()


def _mounted_file(root, relative_path, deadline):
    """Use the asset helper's recursive convention, but reject even equal copies."""
    relative = PurePosixPath(relative_path)
    if (relative.is_absolute() or ".." in relative.parts or "\\" in relative_path
            or str(relative) != relative_path or ":" in relative_path):
        raise ValueError("Invalid relative RGB cache provenance path")
    _deadline(deadline)
    found = []
    for path in root.rglob(relative_path):
        _deadline(deadline)
        if path.is_file():
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Mounted RGB cache file escapes its source root")
            found.append(path)
    if len(found) != 1:
        raise ValueError(f"Missing or ambiguous mounted RGB cache file: {relative_path}")
    return found[0]


def _verified_json(path, expected, deadline):
    """Hash the exact bytes that are decoded, with deadline checks around I/O."""
    _sha_value(expected)
    digest, chunks = hashlib.sha256(), []
    with path.open("rb") as handle:
        while True:
            _deadline(deadline)
            block = handle.read(CHUNK_BYTES)
            _deadline(deadline)
            if not block:
                break
            digest.update(block)
            chunks.append(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"Mounted RGB cache provenance checksum mismatch: {path.name}")
    _deadline(deadline)
    result = json.loads(b"".join(chunks).decode("utf-8"))
    _deadline(deadline)
    return result


def validate_rgb_cache(path, shard_path, deadline):
    """Lazy seam for the established array validator; never loads model weights."""
    from train_rgb_multilabel import validate_rgb_cache as validate
    return validate(path, shard_path, deadline)


def _contract(job, train_shards, encoder_info):
    coverage = job["coverage"]
    cache = coverage["cache_reuse"]
    if (cache.get("kernel_source") != SOURCE_KERNEL
            or cache.get("source_job_id") != SOURCE_KERNEL.rsplit("/", 1)[0]
            or cache.get("extraction_source_sha256") != job["extraction_source_sha256"]):
        raise ValueError("RGB cache source differs from the frozen kernel provenance")
    for key in ("source_sha256", "extraction_source_sha256", "feature_contract_sha256",
                "feature_inventory_sha256"):
        _sha_value(cache.get(key))
    feature_contract = cache["feature_contract"]
    if (canonical_sha256(feature_contract) != cache["feature_contract_sha256"]
            or feature_contract.get("arch") != "swin3d_t"
            or feature_contract.get("feature_dim") != 768
            or feature_contract.get("encoder_info") != encoder_info
            or feature_contract.get("encoder_sha256") != encoder_info.get("weights_sha256")
            or feature_contract.get("preprocessing_sha256") != encoder_info.get("preprocessing_sha256")):
        raise ValueError("RGB cache encoder/preprocessing contract differs from the loaded encoder")
    inventory = job["video_inventory"]
    by_id = {row["video_id"]: row for row in inventory}
    if len(by_id) != len(inventory):
        raise ValueError("Duplicate video in RGB cache source inventory")
    records = cache["records"]
    ids = [row["video_id"] for row in records]
    if len(ids) != 113 or len(set(ids)) != 113:
        raise ValueError("RGB cache requires exactly 113 distinct complete source files")
    probes = coverage["probe_video_ids"]
    if (len(probes) != 5 or len(set(probes)) != 5 or len(set(probes) & set(ids)) != 3
            or any(vid not in by_id or by_id[vid]["subset"] != "training"
                   or by_id[vid]["subject"] not in FIT_SUBJECTS for vid in probes)):
        raise ValueError("RGB cache requires all five fresh fitting probes and three cached exclusions")
    roles = {"fitting": set(), "validation": set()}
    for row in records:
        vid = row["video_id"]
        source = by_id.get(vid, {})
        role = "validation" if source.get("subject") in VALIDATION_SUBJECTS else "fitting"
        if (source.get("subset") != "training" or row.get("subset") != "training"
                or source.get("subject") not in FIT_SUBJECTS + VALIDATION_SUBJECTS
                or row.get("subject") != source["subject"] or row.get("role") != role
                or vid not in train_shards):
            raise ValueError(f"RGB cache selects a missing or forbidden training source: {vid}")
        if (row.get("relative_path") != f"rgb_features/swin3d_t/{vid}.npz"
                or row.get("n_windows") != source["n_windows"]
                or type(row.get("n_windows")) is not int or row["n_windows"] < 1
                or row.get("source_video_sha256") != source["source_sha256"]
                or row.get("source_shard_sha256") != source["shard_sha256"]
                or any(row.get(key) != feature_contract.get(key) for key in
                       ("arch", "feature_dim", "encoder_sha256", "preprocessing_sha256",
                        "extraction_implementation_sha256"))):
            raise ValueError(f"RGB cache record disagrees with the frozen video contract: {vid}")
        for key in ("feature_sha256", "source_video_sha256", "source_shard_sha256",
                    "fingerprint", "encoder_sha256", "preprocessing_sha256",
                    "extraction_implementation_sha256"):
            _sha_value(row.get(key))
        roles[role].add(vid)
    validation = {vid for vid, row in by_id.items() if row["subject"] in VALIDATION_SUBJECTS}
    if len(roles["fitting"]) != 36 or len(roles["validation"]) != 77 or roles["validation"] != validation:
        raise ValueError("RGB cache must cover 36 old fitting videos and all 77 validation videos")
    return cache, by_id, roles, set(probes)


def _provenance(cache, root, by_id, roles, deadline):
    specifications = cache["provenance_files"]
    names = [entry["relative_path"] for entry in specifications]
    if len(names) != len(set(names)) or set(names) != PROVENANCE_FILES:
        raise ValueError("RGB cache needs the exact six distinct source provenance files")
    documents = {}
    for entry in specifications:
        path = _mounted_file(root, entry["relative_path"], deadline)
        documents[entry["relative_path"]] = _verified_json(path, entry["sha256"], deadline)
    source_job, stage = documents["job.json"], documents["stage_result.json"]
    for document in (source_job, stage):
        if any(document.get(key) != cache[key] for key in ("source_sha256", "extraction_source_sha256")):
            raise ValueError("RGB cache source job/snapshot identity changed")
        if document.get("job_id") != cache["source_job_id"]:
            raise ValueError("RGB cache source job identity changed")
    if (stage.get("status") != "quality_rejected" or stage.get("test_features_loaded") is not False
            or stage.get("completed_pilot_videos") != 113
            or stage.get("total_pilot_videos") != 113
            or set(source_job["pilot"]["pilot_video_ids"]) != roles["fitting"]):
        raise ValueError("RGB cache source is not the complete frozen training-only pilot")
    old_inventory = {row["video_id"]: row for row in source_job["video_inventory"]}
    for row in cache["records"]:
        source = old_inventory.get(row["video_id"], {})
        current = by_id[row["video_id"]]
        if any(source.get(key) != current.get(key) for key in
               ("subject", "subset", "n_windows", "source_sha256", "shard_sha256")):
            raise ValueError("RGB cache old/current source video inventories disagree")
    manifest = documents["source_manifest.json"]
    if manifest.get("sha256") != cache["source_sha256"]:
        raise ValueError("RGB cache source manifest snapshot identity changed")
    implementation_files = {"extractor": "scripts/extract_rgb_action_features.py",
                            "rgb_action": "src/behaviorsense/models/rgb_action.py"}
    hashes = {}
    for key, relative in implementation_files.items():
        hashes[key] = _hash_file(ROOT / relative, deadline)
        if hashes[key] != manifest["files"].get(relative):
            raise ValueError("RGB cache extractor implementation changed since source extraction")
    if canonical_sha256(hashes) != cache["feature_contract"]["extraction_implementation_sha256"]:
        raise ValueError("RGB cache extraction implementation fingerprint changed")
    selection = documents["encoder_selection.json"]
    if (selection.get("arch") != "swin3d_t"
            or selection.get("encoder_info") != cache["feature_contract"]["encoder_info"]
            or selection.get("job_id") != cache["source_job_id"]
            or selection.get("source_sha256") != cache["source_sha256"]):
        raise ValueError("RGB cache source encoder selection changed")
    pilot = documents["pilot/manifest.json"]
    inventory = pilot.get("feature_inventory", [])
    expected = [{key: row[key] for key in RECEIPT_FIELDS} for row in cache["records"]]
    if (pilot.get("feature_contract") != cache["feature_contract"]
            or canonical_sha256(inventory) != cache["feature_inventory_sha256"]
            or pilot.get("feature_inventory_sha256") != cache["feature_inventory_sha256"]
            or len(inventory) != 113
            or sorted(inventory, key=lambda row: row["video_id"]) != sorted(expected, key=lambda row: row["video_id"])):
        raise ValueError("RGB cache source head provenance disagrees with frozen cache records")
    reports = documents["extraction_swin3d_t.json"]
    by_report = {row["video_id"]: row for row in reports}
    if len(by_report) != 113 or len(reports) != 113 or set(by_report) != {row["video_id"] for row in cache["records"]}:
        raise ValueError("RGB cache extraction reports do not cover exactly 113 source files")
    return by_report


def _copy_verified(source, target, expected, deadline):
    digest = hashlib.sha256()
    with source.open("rb") as incoming, target.open("xb") as outgoing:
        while True:
            _deadline(deadline)
            block = incoming.read(CHUNK_BYTES)
            _deadline(deadline)
            if not block:
                break
            outgoing.write(block)
            digest.update(block)
    _deadline(deadline)
    if digest.hexdigest() != expected or _hash_file(target, deadline) != expected:
        raise ValueError("RGB cache changed during its verified copy")


def prepare_reused_features(job, work, train_shards, encoder_info, deadline):
    """Return 110 verified cache-hit reports, leaving all five probes fresh.

    All 113 mounted caches pass the established pose/RGB array validator.  The
    three old core-probe files are verified and deliberately not installed; the
    two sparse probes have no old cache.  Failures publish no partial receipt or
    reusable cache files.  Only the supplied fitting/validation shards are read.
    """
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
        raise ValueError("RGB cache reuse requires a finite absolute monotonic deadline")
    began = time.monotonic()
    _deadline(deadline)
    cache, by_id, roles, probes = _contract(job, train_shards, encoder_info)
    owner, slug, _ = cache["kernel_source"].split("/")
    root = Path(dataset_root(owner, slug)).resolve()
    _deadline(deadline)
    work = Path(work).resolve()
    destination = work / "rgb_features" / "swin3d_t"
    if work.is_relative_to(root):
        raise ValueError("RGB cache destination cannot write inside its mounted source")
    if not destination.resolve().is_relative_to(work):
        raise ValueError("RGB cache destination escapes the work directory")
    records = cache["records"]
    outputs = {row["video_id"]: destination / f"{row['video_id']}.npz" for row in records}
    receipt_path = work / "cache_reuse.json"
    if (receipt_path.exists() or receipt_path.with_suffix(".json.partial").exists()
            or any(destination.glob("*.npz"))):
        raise ValueError("RGB cache reuse requires a clean destination without stale cache files")
    source_reports = _provenance(cache, root, by_id, roles, deadline)
    paths = {row["video_id"]: _mounted_file(root, row["relative_path"], deadline) for row in records}
    directories = {path.parent.resolve() for path in paths.values()}
    if len(directories) != 1:
        raise ValueError("Ambiguous mounted RGB cache feature directories")
    feature_directory = directories.pop()
    mounted = set()
    for path in feature_directory.rglob("*.npz"):
        _deadline(deadline)
        mounted.add(path.resolve())
    _deadline(deadline)
    if mounted != {path.resolve() for path in paths.values()}:
        raise ValueError("Unexpected or duplicate RGB cache files, including forbidden test selection")
    work.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=".rgb-cache-reuse-", dir=work)).resolve()
    if staged.parent != work:
        raise ValueError("RGB cache staging directory must stay within the work directory")
    staged_paths, installed, reports, verification = [], [], {}, []
    try:
        for row in records:
            _deadline(deadline)
            started = time.monotonic()
            vid, source = row["video_id"], paths[row["video_id"]]
            if _hash_file(source, deadline) != row["feature_sha256"]:
                raise ValueError(f"Mounted RGB cache checksum mismatch: {vid}")
            if "bytes" in row and source.stat().st_size != row["bytes"]:
                raise ValueError(f"Mounted RGB cache byte count mismatch: {vid}")
            if _hash_file(train_shards[vid], deadline) != row["source_shard_sha256"]:
                raise ValueError(f"Mounted RGB cache pose shard checksum mismatch: {vid}")
            _deadline(deadline)
            values, actual_contract, metadata = validate_rgb_cache(source, train_shards[vid], deadline)
            del values
            _deadline(deadline)
            if actual_contract != cache["feature_contract"]:
                raise ValueError(f"RGB cache encoder/preprocessing/implementation mismatch: {vid}")
            if any(metadata.get(key) != row[key] for key in
                   ("video_id", "subject", "n_windows", "source_video_sha256", "source_shard_sha256",
                    "fingerprint", "arch", "feature_dim", "encoder_sha256", "preprocessing_sha256",
                    "extraction_implementation_sha256")):
                raise ValueError(f"RGB cache rows/source/provenance mismatch: {vid}")
            previous = source_reports[vid]
            if (metadata.get("status") != "complete" or any(key not in metadata for key in REPORT_FIELDS)
                    or any(previous.get(key) != metadata[key] for key in REPORT_FIELDS)
                    or previous.get("output_sha256") != row["feature_sha256"]
                    or metadata.get("spatial_backend", "native") != "native"
                    or metadata.get("prefetch_batches", 0) != 2):
                raise ValueError(f"Incomplete or changed RGB cache extraction report: {vid}")
            entry = dict(row, source_path=str(source), source_sha256_verified=True,
                         array_validation_verified=True, installed=vid not in probes)
            if vid not in probes:
                temporary = staged / f"{vid}.npz"
                staged_paths.append(temporary)
                _copy_verified(source, temporary, row["feature_sha256"], deadline)
                reports[vid] = {key: metadata[key] for key in REPORT_FIELDS} | {
                    "reused": True, "cache_reuse_verified": True,
                    "elapsed_seconds": round(time.monotonic() - started, 6),
                    "output_path": str(outputs[vid]), "output_sha256": row["feature_sha256"]}
                entry.update(output_path=str(outputs[vid]), output_sha256=row["feature_sha256"])
            else:
                # Skipped probes still need byte verification bound to the
                # validated source, even though no copy will recheck them.
                if _hash_file(source, deadline) != row["feature_sha256"]:
                    raise ValueError(f"Mounted RGB cache changed during probe verification: {vid}")
                entry["reason"] = "fresh_probe_required"
            verification.append(entry)
        _deadline(deadline)
        if len(reports) != 110 or set(reports) & probes:
            raise ValueError("RGB cache verified installation does not contain exactly 110 non-probes")
        destination.mkdir(parents=True, exist_ok=True)
        for temporary in staged_paths:
            _deadline(deadline)
            output = outputs[temporary.stem]
            if output.exists():
                raise ValueError("RGB cache destination appeared during verification")
            temporary.replace(output)
            installed.append(output)
        receipt = {
            "schema": "behaviorsense.rgb_cache_reuse.v1", "status": "complete", "verified": True,
            "kernel_source": cache["kernel_source"], "source_job_id": cache["source_job_id"],
            "source_sha256": cache["source_sha256"], "extraction_source_sha256": cache["extraction_source_sha256"],
            "feature_contract_sha256": cache["feature_contract_sha256"],
            "feature_inventory_sha256": cache["feature_inventory_sha256"],
            "provenance_files": cache["provenance_files"], "verified_source_files": len(verification),
            "verified_fitting_files": len(roles["fitting"]), "verified_validation_files": len(roles["validation"]),
            "installed_files": len(reports), "installed_fitting_files": len(set(reports) & roles["fitting"]),
            "installed_validation_files": len(set(reports) & roles["validation"]),
            "fresh_probe_video_ids": sorted(probes), "skipped_cached_probe_video_ids": sorted(probes & set(paths)),
            "test_features_loaded": False, "elapsed_seconds": round(time.monotonic() - began, 6),
            "records": verification}
        _deadline(deadline)
        write_json(receipt_path, receipt)
        _deadline(deadline)
        return reports
    except BaseException:
        for path in installed:
            path.unlink(missing_ok=True)
        receipt_path.unlink(missing_ok=True)
        receipt_path.with_suffix(".json.partial").unlink(missing_ok=True)
        raise
    finally:
        for path in staged_paths:
            path.unlink(missing_ok=True)
        staged.rmdir()
