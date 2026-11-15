"""Stage reviewed RGB assets and freeze one finite private Kaggle experiment."""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import re
import zipfile

from launch_multilabel_kaggle import ROOT, TRAINING_IMAGE, EXPECTED_CORE_RUNTIME, authenticated_api, api_document, coordinator_lock
from launch_temporal_kaggle import extraction_sources, push as push_frozen, status as remote_status, verify_submission
from multilabel_kaggle_stage import read_json, write_json, sha256
from temporal_kaggle_stage import canonical_sha256, FIT_SUBJECTS, VALIDATION_SUBJECTS
from rgb_kaggle_stage import GATE_CRITERIA

BASE = ROOT / "artifacts/rgb-throughput-20261013"
DEFAULT_ASSETS = ROOT / "artifacts/pipeline-rebuild-20261012/assets"
DEFAULT_RUN = BASE / "prefetch-run"
ASSET_SLUG = "bs-rgb-assets-20261012"
RUN_SLUG = "bs-rgb-prefetch-20261013"
WHEEL_NAME = "torchvision-0.26.0+cu128-cp313-cp313-manylinux_2_28_x86_64.whl"
WHEEL_SHA256 = "cb1f6184a7ba30fba40580e1a01a6604a86c55e79fdda187f40116ee680441ec"
EXTRACTION_SOURCE_SHA256 = "407c5aa9e4178a989a452731c21fc46d02f9844eba0003e7216f1b4e1ecacc0f"
WEIGHTS = {
    "swin3d_b": {"filename": "swin3d_b_22k-7c6ae6fa.pth",
                 "url": "https://download.pytorch.org/models/swin3d_b_22k-7c6ae6fa.pth",
                 "sha256_prefix": "7c6ae6fa", "sha256": "7c6ae6fa165f481a9c71156644a7c0e61bb393e470ca3671b8d24a30d365ffc6",
                 "bytes": 381822446, "pretraining": "ImageNet22K then Kinetics400", "feature_dim": 1024},
    "swin3d_t": {"filename": "swin3d_t-7615ae03.pth",
                 "url": "https://download.pytorch.org/models/swin3d_t-7615ae03.pth",
                 "sha256_prefix": "7615ae03", "sha256": "7615ae035996b65eb38dad437ae533d2dfcd36f9f89d28c0f0fa7bfb8e6b3130",
                 "bytes": 127447114, "pretraining": "ImageNet1K then Kinetics400", "feature_dim": 768},
}
UPSTREAM_REFERENCE_PREPROCESSING = {"frames": 32, "frame_rate": 15, "crop_size": [224, 224], "resize_short_side": 256,
                 "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
                 "views": ["scene", "person"], "source_clock": "verified original frame indices; no duration rescaling",
                 "reference": "https://github.com/pytorch/vision/blob/v0.26.0/torchvision/models/video/swin_transformer.py"}


def actual_preprocessing(root=ROOT):
    """Read the encoder's constant transform facts without importing NumPy/Torch."""
    constants = {}
    class ResolveConstants(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id not in constants:
                raise ValueError(f"Nonliteral preprocessing dependency: {node.id}")
            return ast.copy_location(ast.Constant(constants[node.id]), node)
    source = Path(root) / "src/behaviorsense/models/rgb_action.py"
    for statement in ast.parse(source.read_text(encoding="utf-8")).body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            name = statement.targets[0].id
            if name in ("RGB_FRAMES", "RGB_FPS", "PREPROCESSING"):
                constants[name] = ast.literal_eval(ResolveConstants().visit(statement.value))
    return constants["PREPROCESSING"]


def download_file(url, target, expected=None, prefix=None):
    """Verified TLS, atomic final files, and no network work in a GPU session."""
    import requests
    target = Path(target)
    if target.exists():
        actual = sha256(target)
        if (expected and actual != expected) or (prefix and not actual.startswith(prefix)):
            raise ValueError(f"Existing asset checksum mismatch: {target.name}")
        return {"sha256": actual, "bytes": target.stat().st_size, "url": url}
    partial = target.with_suffix(target.suffix + ".partial")
    with requests.get(url, stream=True, timeout=(30, 90)) as response:
        response.raise_for_status()
        digest, count = hashlib.sha256(), 0
        with partial.open("wb") as output:
            for block in response.iter_content(8 << 20):
                if block:
                    output.write(block)
                    digest.update(block)
                    count += len(block)
        actual = digest.hexdigest()
        if ((expected and actual != expected) or (prefix and not actual.startswith(prefix))
                or (not response.headers.get("Content-Encoding") and response.headers.get("Content-Length")
                    and count != int(response.headers["Content-Length"]))):
            raise ValueError(f"Downloaded asset checksum/length mismatch: {target.name}")
    partial.replace(target)
    return {"sha256": actual, "bytes": count, "url": url}


def stage_assets(directory=DEFAULT_ASSETS, owner=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files, encoders = {}, {}
    for arch, specification in WEIGHTS.items():
        value = download_file(specification["url"], directory / specification["filename"],
                              expected=specification["sha256"])
        if value["bytes"] != specification["bytes"]:
            raise ValueError(f"Unexpected pretrained asset length: {arch}")
        files[specification["filename"]] = value
        encoders[arch] = {**specification, **value}
        print(f"Verified {arch}: {value['bytes']} bytes {value['sha256']}", flush=True)
    wheel_url = "https://download-r2.pytorch.org/whl/cu128/" + WHEEL_NAME.replace("+", "%2B")
    files[WHEEL_NAME] = download_file(wheel_url, directory / WHEEL_NAME, expected=WHEEL_SHA256)
    licenses = {
        "torchvision-LICENSE.txt": "https://raw.githubusercontent.com/pytorch/vision/v0.26.0/LICENSE",
        "Video-Swin-Transformer-LICENSE.txt": "https://raw.githubusercontent.com/SwinTransformer/Video-Swin-Transformer/master/LICENSE",
    }
    for filename, url in licenses.items():
        files[filename] = download_file(url, directory / filename)
    preprocessing = actual_preprocessing()
    manifest = {"schema_version": 1, "encoders": encoders, "files": files,
                "preprocessing": preprocessing, "preprocessing_sha256": canonical_sha256(preprocessing),
                "upstream_reference_preprocessing": UPSTREAM_REFERENCE_PREPROCESSING,
                "torchvision": "0.26.0+cu128",
                "torchvision_wheel": WHEEL_NAME, "core_runtime": EXPECTED_CORE_RUNTIME,
                "upstream_license_files": list(licenses), "contains_credentials": False,
                "weights_source": UPSTREAM_REFERENCE_PREPROCESSING["reference"], "dataset_visibility_required": "private"}
    write_json(directory / "asset-manifest.json", manifest)
    if owner:
        write_json(directory / "dataset-metadata.json", {"id": f"{owner}/{ASSET_SLUG}",
                   "title": "BehaviorSense RGB encoder assets 20261012", "licenses": [{"name": "other"}],
                   "description": "Private upstream video encoder weights and matching runtime wheel. Source URLs, upstream code licenses and checksums are in asset-manifest.json."})
    return manifest


def verify_assets(directory):
    directory = Path(directory)
    manifest = read_json(directory / "asset-manifest.json")
    if (set(manifest.get("encoders", {})) != set(WEIGHTS)
            or manifest.get("torchvision") != "0.26.0+cu128"
            or manifest.get("torchvision_wheel") != WHEEL_NAME
            or manifest.get("core_runtime") != EXPECTED_CORE_RUNTIME
            or manifest.get("dataset_visibility_required") != "private"):
        raise ValueError("Unexpected private RGB asset/runtime contract")
    for arch, specification in WEIGHTS.items():
        actual = manifest["encoders"][arch]
        if any(actual.get(key) != specification[key] for key in specification):
            raise ValueError(f"Unapproved pretrained encoder contract: {arch}")
        if manifest["files"][specification["filename"]]["sha256"] != specification["sha256"]:
            raise ValueError(f"Pretrained encoder manifest hash mismatch: {arch}")
    if manifest["files"][WHEEL_NAME]["sha256"] != WHEEL_SHA256:
        raise ValueError("Matched torchvision wheel hash mismatch")
    preprocessing = actual_preprocessing()
    if (manifest.get("preprocessing") != preprocessing
            or manifest.get("preprocessing_sha256") != canonical_sha256(preprocessing)):
        raise ValueError("Asset preprocessing differs from the actual frozen RGB transform; restage local assets")
    expected = set(manifest["files"]) | {"asset-manifest.json", "dataset-metadata.json"}
    unexpected = [p.name for p in directory.iterdir() if p.name not in expected]
    if unexpected:
        raise ValueError(f"Unexpected files in private asset folder: {unexpected}")
    for name, value in manifest["files"].items():
        path = directory / name
        if (path.name != name or path.is_symlink() or path.parent != directory
                or sha256(path) != value["sha256"] or path.stat().st_size != value["bytes"]):
            raise ValueError(f"RGB asset changed: {name}")
    return manifest


def upload_assets(api, owner, directory):
    """An explicit CLI action; never invoked by staging, building or GPU execution."""
    directory = Path(directory)
    verify_assets(directory)
    metadata = read_json(directory / "dataset-metadata.json")
    if metadata["id"] != f"{owner}/{ASSET_SLUG}":
        raise ValueError("Asset metadata owner differs from authenticated owner")
    receipt_path = directory.parent / "asset-upload-receipt.json"
    intent_path = directory.parent / "asset-upload-intent.json"
    with coordinator_lock(directory.parent):
        if intent_path.exists() or receipt_path.exists():
            raise ValueError("An asset upload was already attempted; reconcile it before retrying")
        intent = {"dataset": metadata["id"], "private": True, "state": "intent_journaled",
                  "manifest_sha256": sha256(directory / "asset-manifest.json")}
        write_json(intent_path, intent)
        try:
            response = api_document(api.dataset_create_new(str(directory), public=False, quiet=True,
                                                          convert_to_csv=False, dir_mode="skip"))
        except Exception as exc:
            write_json(intent_path, {**intent, "state": "ambiguous", "error": f"{type(exc).__name__}: {exc}"})
            raise
        write_json(receipt_path, response)
        if response.get("error"):
            write_json(intent_path, {**intent, "state": "rejected"})
            raise RuntimeError(f"Private asset upload rejected: {response['error']}")
        # Creation only records the API receipt. Availability/privacy are verified
        # from remote metadata before a GPU submission is permitted.
        write_json(intent_path, {**intent, "state": "receipt_recorded"})
        return response


def video_inventory(training_manifest, test_report, sources):
    inventory = []
    for subset, provenance in (("training", training_manifest["provenance"]),
                               ("testing", test_report["source_provenance"])):
        for shard in provenance["source_shards"]:
            extraction = shard["extraction"]
            matches = [item["ref"] for item in sources if item["subset"] == subset
                       and "/" + item["ref"].split("/")[-1] + "/" in shard["path"]]
            if len(matches) != 1 or extraction["subset"] != subset:
                raise ValueError("Cannot bind video to its frozen extraction source")
            inventory.append({"video_id": extraction["video_id"], "subject": extraction["subject"],
                "subset": subset, "n_windows": shard["n_windows"], "shard_sha256": shard["sha256"],
                "shard_filename": Path(shard["path"]).name, "source_ref": matches[0],
                "source_filename": extraction["source_filename"], "source_sha256": extraction["source_sha256"],
                "source_bytes": extraction["source_bytes"], "source_frames": extraction["source_frame_count"],
                "source_fps": extraction["source_fps"],
                "duration_seconds": extraction["source_frame_count"] / extraction["source_fps"],
                "pair_windows": extraction.get("retained_strata", {}).get("tv_drinking", 0)})
    if len(inventory) != 536 or len({item["video_id"] for item in inventory}) != 536:
        raise ValueError("RGB plan requires the full 536-video inventory")
    for subset, videos, windows in (("training", 351, 280310), ("testing", 185, 153537)):
        selected = [x for x in inventory if x["subset"] == subset]
        if len(selected) != videos or sum(x["n_windows"] for x in selected) != windows:
            raise ValueError(f"Unexpected frozen {subset} coverage")
    return sorted(inventory, key=lambda value: value["video_id"])


def choose_pilot(inventory, annotations, taxonomy_path=ROOT / "src/behaviorsense/data/toyota.py"):
    """Fit-only annotation coverage; exactly four whole videos per fitting subject."""
    facts = {}
    for statement in ast.parse(Path(taxonomy_path).read_text(encoding="utf-8")).body:
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            if statement.target.id in ("TSU_CLASSES", "TSU_TO_COARSE"):
                facts[statement.target.id] = ast.literal_eval(statement.value)
    fine, mapping = facts["TSU_CLASSES"], facts["TSU_TO_COARSE"]
    fit = [item for item in inventory if item["subject"] in FIT_SUBJECTS]
    classes = {item["video_id"]: {mapping.get(fine[int(action[0])]) for action in annotations[item["video_id"]]["actions"]}
               - {None} for item in fit}
    supported = set(mapping.values())
    frequency = {label: sum(label in labels for labels in classes.values()) for label in supported}
    covered, selected = set(), []
    for _ in range(4):
        for subject in FIT_SUBJECTS:
            options = [item for item in fit if item["subject"] == subject and item not in selected]
            if not options:
                raise ValueError(f"Insufficient fitting videos for {subject}")
            def score(item):
                labels = classes[item["video_id"]]
                novelty = sum(1 / max(1, frequency[label]) for label in labels - covered)
                return (novelty, min(item["pair_windows"], 100), len(labels), -item["n_windows"], item["video_id"])
            chosen = max(options, key=score)
            selected.append(chosen)
            covered |= classes[chosen["video_id"]]
    if covered != supported or sum(item["pair_windows"] for item in selected) < 100:
        raise ValueError("Pilot does not cover all supported activities and fitting co-label support")
    eligible = [item for item in selected if 100 <= item["n_windows"] <= 1000]
    eligible.sort(key=lambda item: (item["duration_seconds"] / item["n_windows"], item["video_id"]))
    probes = []
    for quantile in (0.15, 0.5, 0.85):
        target = round(quantile * max(0, len(eligible) - 1))
        options = [(index, item) for index, item in enumerate(eligible)
                   if item not in probes and item["subject"] not in {x["subject"] for x in probes}]
        if not options:
            raise ValueError("Need three complete short probe videos from different fitting subjects")
        probes.append(min(options, key=lambda pair: (abs(pair[0] - target), pair[1]["video_id"]))[1])
    return {"pilot_video_ids": sorted(x["video_id"] for x in selected),
            "probe_video_ids": [x["video_id"] for x in probes], "n_fit_videos": len(selected),
            "fit_windows": sum(x["n_windows"] for x in selected),
            "pair_windows": sum(x["pair_windows"] for x in selected), "covered_activities": sorted(covered),
            "selection_uses": "fitting subjects, fitting video annotations and extraction metadata only",
            "validation": "all 77 videos belonging to p04 and p15, without video selection"}


def source_bundle(root=ROOT):
    paths = sorted((root / "src").rglob("*.py")) + sorted((root / "scripts").rglob("*.py"))
    paths.append(root / "configs/taxonomy.yaml")
    for required in ("train_rgb_multilabel.py", "extract_rgb_action_features.py", "rgb_kaggle_stage.py", "rgb_profile_stage.py"):
        if not (root / "scripts" / required).is_file():
            raise FileNotFoundError(f"RGB source is not ready: {required}")
    secrets = [value.encode() for name in ("KAGGLE_KEY", "KAGGLE_API_TOKEN") if (value := os.environ.get(name))]
    for path in root.glob("kaggle*.json"):
        value = json.loads(path.read_text(encoding="utf-8-sig")).get("key")
        if value:
            secrets.append(str(value).encode())
    stream, files = io.BytesIO(), {}
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Source path escapes repository")
            relative, data = path.relative_to(root).as_posix(), path.read_bytes()
            if any(secret in data for secret in secrets):
                raise ValueError(f"Credential value found in {relative}")
            if path.suffix == ".py":
                compile(data, relative, "exec")
            info = zipfile.ZipInfo(relative, date_time=(2026, 10, 13, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
            files[relative] = hashlib.sha256(data).hexdigest()
    raw = stream.getvalue()
    return base64.b64encode(raw).decode("ascii"), {"sha256": hashlib.sha256(raw).hexdigest(),
        "files": files, "contains_credentials": False, "contains_data": False}


MAX_KERNEL_SOURCE_BYTES = 1_000_000


def validate_source_size(body):
    if len(body) >= MAX_KERNEL_SOURCE_BYTES:
        raise ValueError(f"Kaggle kernel source must be less than {MAX_KERNEL_SOURCE_BYTES:,} bytes; "
                         f"generated {len(body):,} bytes")
    return body


def runtime(bundle, manifest, job):
    import zlib
    # Coverage provenance can exceed Kaggle's source limit when embedded as repr.
    # Encoding the JSON preserves the job and the separately hashed source ZIP.
    encoded_job = base64.b64encode(zlib.compress(
        json.dumps(job, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8"), 9
    )).decode("ascii")
    return f'''# One private RGB experiment. Assets are staged; no internet or credentials.
import base64, hashlib, io, json, os, sys, time, zipfile, zlib
from pathlib import Path
STARTED = time.monotonic()
WORK = Path("/kaggle/working")
REPO = WORK / "rgb_source"
REPO.mkdir(parents=True, exist_ok=True)
JOB = json.loads(zlib.decompress(base64.b64decode({encoded_job!r}, validate=True)))
MANIFEST = {manifest!r}
raw = base64.b64decode({bundle!r}, validate=True)
assert hashlib.sha256(raw).hexdigest() == JOB["source_sha256"] == MANIFEST["sha256"]
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    assert set(archive.namelist()) == set(MANIFEST["files"])
    for name in archive.namelist():
        destination = REPO / name
        assert destination.resolve().is_relative_to(REPO.resolve())
        body = archive.read(name)
        assert hashlib.sha256(body).hexdigest() == MANIFEST["files"][name]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(body)
(WORK / "job.json").write_text(json.dumps(JOB, indent=2))
(WORK / "source_manifest.json").write_text(json.dumps(MANIFEST, indent=2))
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]
os.environ["PYTHONPATH"] = str(REPO / "src")
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "4"
from rgb_kaggle_stage import supervise
supervise(JOB, REPO, WORK, started_at=STARTED)
'''


def build(directory=DEFAULT_RUN, assets=DEFAULT_ASSETS, owner=None, root=ROOT,
          coverage_plan=None, run_slug=None):
    directory, assets, root = Path(directory), Path(assets), Path(root)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("A frozen RGB experiment already exists; do not rebuild it")
    if coverage_plan is not None and (not run_slug or run_slug == RUN_SLUG):
        raise ValueError("Coverage mode requires an explicit new run slug; the original kernel is frozen")
    slug = RUN_SLUG if run_slug is None else run_slug
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug) or len(slug) > 50:
        raise ValueError("Run slug must contain lowercase letters, digits and hyphens, at most 50 characters")
    coverage = None
    if coverage_plan is not None:
        from rgb_coverage_contract import build_coverage_contract
        coverage = build_coverage_contract(coverage_plan, root)
    asset_manifest = verify_assets(assets)
    if not owner or read_json(assets / "dataset-metadata.json")["id"] != f"{owner}/{ASSET_SLUG}":
        raise ValueError("Asset owner must match the frozen experiment owner")
    old = read_json(root / "artifacts/training-tv-20261007/full-run/plan.json")
    if old["source_sha256"] != EXTRACTION_SOURCE_SHA256:
        raise ValueError("The original extraction source hash differs from reviewed provenance")
    sources = extraction_sources(old, include_test=True)
    previous = root / "artifacts/temporal-multilabel-20261011/output"
    inventory = video_inventory(read_json(previous / "candidates/asl-seed1/manifest.json"),
                                read_json(previous / "test/test_metrics.json"), sources)
    annotations = read_json(root / "artifacts/training-tv-20261007/data/smarthome_CS_51.json")
    if sha256(root / "artifacts/training-tv-20261007/data/smarthome_CS_51.json") != old["annotation_sha256"]:
        raise ValueError("Fitting annotation file differs from reviewed provenance")
    pilot = (coverage["pilot"] if coverage is not None else
             choose_pilot(inventory, annotations, root / "src/behaviorsense/data/toyota.py"))
    if coverage is not None:
        for required in ("rgb_coverage_contract.py", "rgb_cache_reuse.py", "rgb_coverage_budget.py"):
            if not (root / "scripts" / required).is_file():
                raise FileNotFoundError(f"RGB coverage source is not ready: {required}")
    bundle, manifest = source_bundle(root)
    job = {"job_id": f"{owner}/{slug}", "owner": owner, "stage": "rgb_single_session",
        "rgb_protocol": "native-prefetch-profile.v3", "prefetch_batches": 2,
        "opencv_threads": 1,
        "session_seconds": 40500, "pilot_seconds": 4500, "probe_seconds": 900,
        "throughput_safety_factor": 1.25, "full_head_and_export_reserve": 3600,
        "pilot_head_reserve": 600, "source_sha256": manifest["sha256"],
        "extraction_source_sha256": old["source_sha256"], "annotation_sha256": old["annotation_sha256"],
        "expected_core_runtime": EXPECTED_CORE_RUNTIME, "feature_sources": sources,
        "video_inventory": inventory, "pilot": pilot, "gate_criteria": GATE_CRITERIA,
        "asset_dataset": f"{owner}/{ASSET_SLUG}/1", "asset_dataset_slug": ASSET_SLUG,
        "assets_manifest_sha256": sha256(assets / "asset-manifest.json"), "assets": asset_manifest,
        "encoder_preference": ["swin3d_t"], "batch_size": 8,
        "test_benchmark_previously_inspected": True, "automatic_continuation": False,
        "serving_model_replaced": False}
    if coverage is not None:
        job["coverage"] = coverage
    from rgb_kaggle_stage import validate_job
    validate_job(job)
    metadata = {"id": job["job_id"], "title": slug.replace("-", " ").title(),
        "code_file": "training.py", "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "machine_shape": "NvidiaRtxPro6000", "docker_image": TRAINING_IMAGE,
        "docker_image_pinning_type": "original", "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "dataset_sources": [job["asset_dataset"], f"{owner}/rgb-untrimmed/1", f"{owner}/behaviorsense-toyota-meta"],
        "kernel_sources": kernel_mounts(job), "model_sources": []}
    body = validate_source_size(runtime(bundle, manifest, job).encode("utf-8"))
    compile(body, "training.py", "exec")
    submission = directory / "submission"
    submission.mkdir(parents=True, exist_ok=True)
    (submission / "training.py").write_bytes(body)
    write_json(submission / "kernel-metadata.json", metadata)
    write_json(submission / "job.json", job)
    write_json(directory / "source_manifest.json", manifest)
    (directory / "source.b64").write_text(bundle, encoding="ascii")
    plan = {"schema_version": 1, "status": "prepared", "job": job, "submission": "submission",
        "source_manifest": manifest, "timeout_seconds": 43200, "single_session": True,
        "automatic_continuation": False, "submission_hashes": {name: sha256(submission / name)
            for name in ("training.py", "kernel-metadata.json", "job.json")}}
    write_json(directory / "plan.json", plan)
    return plan


def kernel_mounts(job):
    """Preserve extraction mounts and append only the pinned reviewed cache run."""
    refs = [entry["ref"] for entry in job["feature_sources"]]
    if "coverage" in job:
        cache_ref = job["coverage"]["cache_reuse"]["kernel_source"]
        if cache_ref != "krishnagupta02468/bs-rgb-prefetch-20261013/1":
            raise ValueError("Coverage cache mount must use the reviewed original kernel version 1")
        if cache_ref in refs:
            raise ValueError("Coverage cache kernel must be distinct from extraction mounts")
        refs.append(cache_ref)
    return refs


def push(api, directory):
    """Validate the RGB protocol before the existing one-attempt submit journal."""
    from rgb_kaggle_stage import validate_job
    directory = Path(directory)
    plan = read_json(directory / "plan.json")
    if plan.get("status") != "prepared" or plan.get("submission_attempt"):
        raise ValueError("Submission already attempted; automatic resubmission is forbidden")
    validate_job(plan["job"])
    verify_submission(directory, plan)
    validate_source_size((directory / "submission/training.py").read_bytes())
    metadata = read_json(directory / "submission/kernel-metadata.json")
    if (metadata.get("id") != plan["job"]["job_id"]
            or metadata.get("docker_image") != TRAINING_IMAGE
            or metadata.get("competition_sources") != ["arc-prize-2026-arc-agi-3"]
            or metadata.get("kernel_sources") != kernel_mounts(plan["job"])
            or metadata.get("dataset_sources") != [plan["job"]["asset_dataset"],
                f"{plan['job']['owner']}/rgb-untrimmed/1", f"{plan['job']['owner']}/behaviorsense-toyota-meta"]):
        raise ValueError("Frozen RGB mounts or runtime differ from the reviewed experiment")
    evidence = verify_remote_inputs(api, plan["job"])
    write_json(directory / "remote_input_verification.json", evidence)
    return push_frozen(api, directory)


def verify_remote_inputs(api, job):
    """Read-only readiness/privacy/version checks before the single GPU commit."""
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest, ApiGetDatasetStatusRequest
    datasets = {}
    with api.build_kaggle_client() as client:
        for ref in (job["asset_dataset"], f"{job['owner']}/rgb-untrimmed/1"):
            owner, slug, version = ref.split("/")
            request, status_request = ApiGetDatasetRequest(), ApiGetDatasetStatusRequest()
            for value in (request, status_request):
                value.owner_slug, value.dataset_slug = owner, slug
            data = client.datasets.dataset_api_client.get_dataset(request)
            state = client.datasets.dataset_api_client.get_dataset_status(status_request)
            evidence = {"ref": data.ref, "is_private": data.is_private,
                        "current_version_number": data.current_version_number, "status": state.status.name.lower()}
            if (evidence["ref"] != owner + "/" + slug or evidence["is_private"] is not True
                    or evidence["current_version_number"] != int(version) or evidence["status"] != "ready"):
                raise ValueError(f"Required private dataset version is not verified ready: {ref}: {evidence}")
            datasets[ref] = evidence
    listed = api.dataset_list_files(job["asset_dataset"], page_size=100)
    if listed.error_message or listed.next_page_token:
        raise ValueError("Private RGB asset file listing is incomplete")
    files = {item.name: item.total_bytes for item in listed.dataset_files}
    expected = {name: info["bytes"] for name, info in job["assets"]["files"].items()}
    names = {name: (name.replace("+", "") if name == WHEEL_NAME and name not in files else name)
             for name in expected}
    # Kaggle consumes dataset-metadata.json during upload, while asset-manifest
    # remains a mounted file and gets its exact hash checked in GPU preflight.
    if (len(files) != len(listed.dataset_files) or set(files) != set(names.values()) | {"asset-manifest.json"}
            or any(files[names[name]] != size for name, size in expected.items())):
        raise ValueError("Remote private RGB assets differ from the frozen file inventory")
    result = {"datasets": datasets, "asset_files": files, "ready_private_versions_verified": True,
              "mounted_asset_filenames": names,
              "asset_manifest_sha256_expected_at_runtime": job["assets_manifest_sha256"]}
    if "coverage" in job:
        result["cache_kernel"] = verify_remote_cache_kernel(api, job)
    return result


def verify_remote_cache_kernel(api, job):
    """Bind current status to verified version-one source before and after the read."""
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest, ApiGetKernelSessionStatusRequest
    reference = kernel_mounts(job)[-1]
    owner, slug, version = reference.split("/")
    expected = job["coverage"]["cache_reuse"]["training_source_text_sha256"]

    def source_snapshot(client):
        # Versioned pull/status routes are unsupported for this private source.
        # A bare lookup is accepted only while it is still exactly frozen v1.
        request = ApiGetKernelRequest()
        request.user_name, request.kernel_slug = owner, slug
        data = client.get_kernel(request)
        metadata = data.metadata
        snapshot = {"ref": getattr(metadata, "ref", None),
                    "is_private": getattr(metadata, "is_private", None),
                    "current_version_number": getattr(metadata, "current_version_number", None)}
        if (snapshot["ref"] != owner + "/" + slug or snapshot["is_private"] is not True
                or snapshot["current_version_number"] != int(version)):
            raise ValueError(f"Required private cache kernel version is not verified complete: {reference}: {snapshot}")
        source = data.blob.source if data.blob is not None else None
        if not isinstance(source, str) or hashlib.sha256(source.encode("utf-8")).hexdigest() != expected:
            raise ValueError("Remote cache kernel source differs from the frozen version 1 submission")
        snapshot["training_source_text_sha256"] = expected
        return snapshot

    status_request = ApiGetKernelSessionStatusRequest()
    status_request.user_name, status_request.kernel_slug = owner, slug
    with api.build_kaggle_client() as client:
        kernels = client.kernels.kernels_api_client
        before = source_snapshot(kernels)
        state = kernels.get_kernel_session_status(status_request)
        after = source_snapshot(kernels)
    evidence = {"kernel_source": reference, **after, "requested_version": int(version),
                "status": state.status.name.lower(),
                "lookup_scope": "current version bound before and after status",
                "source_snapshots": {"before_status": before, "after_status": after}}
    if evidence["status"] != "complete":
        raise ValueError(f"Required private cache kernel version is not verified complete: {reference}: {evidence}")
    evidence["source_verified"] = True
    return evidence


def outputs(api, directory, logs_only=False):
    directory = Path(directory)
    plan = read_json(directory / "plan.json")
    pattern = r"(?!)" if logs_only else r"(^|/)(stage_result|environment|frozen_selection|encoder_selection|probe_report|profile_report|sparse_probe_report|cache_reuse|coverage_execution|supervision_started|execution_started|pilot_video_ids|pilot_passed|gate|gate_config|training_status|validation_metrics|validation_by_subject|test_metrics|manifest|history|thresholds|job|source_manifest|training_pose_feature_inventory|testing_pose_feature_inventory|extraction_swin3d_[bt])\.json$|(^|/)(best_rgb_multilabel\.pt|validation_logits\.npz|test_logits\.npz)$"
    paths, _ = api.kernels_output(plan["job"]["job_id"], str(directory / "output"),
                                  file_pattern=pattern, force=True, quiet=True)
    return {"downloaded": paths, "scientific_completion_requires_output_audit": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage-assets", "upload-assets", "build", "push", "status", "logs", "output"))
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--directory", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--coverage-plan", type=Path,
                        help="Verified expanded fitting-only coverage plan for a new experiment")
    parser.add_argument("--run-slug", help="New Kaggle kernel slug (required with --coverage-plan)")
    parser.add_argument("--credentials", type=Path, default=ROOT / "kaggle-krishna.json")
    args = parser.parse_args()
    if args.action != "build" and (args.coverage_plan is not None or args.run_slug is not None):
        parser.error("--coverage-plan and --run-slug apply only to build")
    if args.action == "stage-assets":
        owner = json.loads(args.credentials.read_text(encoding="utf-8-sig"))["username"]
        result = stage_assets(args.assets, owner)
        print(json.dumps({"status": "staged_locally", "encoders": result["encoders"], "upload_performed": False}, indent=2))
        return
    if args.action == "build":
        owner = json.loads(args.credentials.read_text(encoding="utf-8-sig"))["username"]
        result = build(args.directory, args.assets, owner,
                       coverage_plan=args.coverage_plan, run_slug=args.run_slug)
        print(json.dumps({"status": result["status"], "job_id": result["job"]["job_id"],
                          "pilot": result["job"]["pilot"], "source_sha256": result["job"]["source_sha256"]}, indent=2))
        return
    api, owner = authenticated_api(args.credentials)
    if args.action == "upload-assets":
        result = upload_assets(api, owner, args.assets)
    else:
        plan = read_json(args.directory / "plan.json")
        if owner != plan["job"]["owner"]:
            raise ValueError("Authenticated credentials owner differs from frozen job owner")
        if args.action == "push":
            result = push(api, args.directory)
        elif args.action == "status":
            result = remote_status(api, args.directory)
        else:
            result = outputs(api, args.directory, logs_only=args.action == "logs")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
