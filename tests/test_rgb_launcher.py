"""Offline single-session orchestration tests; no network, weights, Torch or GPU."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import base64
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import launch_rgb_kaggle as launch
import rgb_kaggle_stage as stage
from multilabel_kaggle_stage import read_json, sha256, write_json


def inventory():
    result = []
    for subset, subjects, count, total in (
        ("training", stage.FIT_SUBJECTS, 274, 217090),
        ("training", stage.VALIDATION_SUBJECTS, 77, 63220),
        ("testing", ["p02", "p05", "p08", "p10", "p11", "p14", "p16", "p18", "p20"], 185, 153537),
    ):
        for index in range(count):
            subject = subjects[index % len(subjects)]
            vid = "P" + subject[1:] + f"T{index // len(subjects) + 1:02}C01"
            result.append({"video_id": vid, "subject": subject, "subset": subset,
                "n_windows": total // count + (index < total % count),
                "shard_filename": vid + ".npz", "shard_sha256": "c" * 64,
                "source_filename": vid + ".mp4", "source_sha256": "d" * 64,
                "source_ref": "fixture/source", "source_bytes": 100, "source_frames": 3000,
                "source_fps": 30, "duration_seconds": 100., "pair_windows": 10})
    return result


def fixture_job():
    records = inventory()
    pilot = sorted(x["video_id"] for subject in stage.FIT_SUBJECTS
                   for x in [r for r in records if r["subject"] == subject][:4])
    probes = [next(x for x in pilot if x.startswith("P" + subject[1:])) for subject in stage.FIT_SUBJECTS[:3]]
    return {"job_id": "fixture/" + launch.RUN_SLUG, "owner": "fixture", "stage": "rgb_single_session",
        "session_seconds": 40500, "pilot_seconds": 4500, "probe_seconds": 900,
        "throughput_safety_factor": 1.25, "full_head_and_export_reserve": 3600,
        "pilot_head_reserve": 600, "batch_size": 8, "automatic_continuation": False,
        "encoder_preference": ["swin3d_b", "swin3d_t"], "video_inventory": records,
        "pilot": {"pilot_video_ids": pilot, "probe_video_ids": probes},
        "source_sha256": "e" * 64, "extraction_source_sha256": launch.EXTRACTION_SOURCE_SHA256,
        "annotation_sha256": "a" * 64, "assets_manifest_sha256": "f" * 64,
        "assets": {"encoders": copy.deepcopy(launch.WEIGHTS), "preprocessing": {"fixture": "test"},
                   "preprocessing_sha256": "1" * 64},
        "asset_dataset": "fixture/" + launch.ASSET_SLUG + "/1", "asset_dataset_slug": launch.ASSET_SLUG,
        "gate_criteria": copy.deepcopy(launch.GATE_CRITERIA), "feature_sources": [],
        "expected_core_runtime": launch.EXPECTED_CORE_RUNTIME}


class RGBProtocolTests(unittest.TestCase):
    def test_normalized_wheel_mount_is_unambiguous_and_restorable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            name = launch.WHEEL_NAME
            normalized = root / name.replace("+", "")
            normalized.write_bytes(b"verified wheel fixture")
            manifest = {"torchvision_wheel": name}
            self.assertEqual(normalized, stage.mounted_asset(root, name, manifest))
            (root / name).write_bytes(b"unexpected duplicate")
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                stage.mounted_asset(root, name, manifest)

    def test_transform_is_read_from_actual_encoder_without_torch_import(self):
        value = launch.actual_preprocessing(ROOT)
        self.assertEqual(32, value["frames"])
        self.assertEqual(15., value["sample_fps"])
        self.assertEqual(["scene", "person"], value["views"])
        self.assertIn("pad", value["scene"])
        self.assertEqual([256, 256], value["resize_size"])

    def test_protocol_rejects_expanded_budget_resume_or_provenance_impersonation(self):
        stage.validate_job(fixture_job())
        for key, value in (("session_seconds", 43200), ("pilot_seconds", 5400),
                           ("probe_seconds", 1800), ("automatic_continuation", True),
                           ("source_sha256", launch.EXTRACTION_SOURCE_SHA256)):
            job = fixture_job()
            job[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                stage.validate_job(job)

    def test_pilot_cannot_select_validation_test_or_concentrate_subjects(self):
        for subject in ("p04", "p02", "p03"):
            job = fixture_job()
            replacement = next(x["video_id"] for x in job["video_inventory"]
                               if x["subject"] == subject and x["video_id"] not in job["pilot"]["pilot_video_ids"])
            job["pilot"]["pilot_video_ids"][-1] = replacement
            with self.subTest(subject=subject), self.assertRaises(ValueError):
                stage.validate_job(job)

    def test_selector_is_fit_only_deterministic_and_exactly_four_per_person(self):
        with tempfile.TemporaryDirectory() as temporary:
            taxonomy = Path(temporary) / "taxonomy.py"
            taxonomy.write_text("TSU_CLASSES: list = ['watch', 'drink']\nTSU_TO_COARSE: dict = {'watch':'watching_tv','drink':'drinking'}\n")
            records = inventory()
            annotations = {x["video_id"]: {"actions": [[0, 0, 1], [1, 0, 1]]}
                           for x in records if x["subject"] in stage.FIT_SUBJECTS}
            expected = launch.choose_pilot(records, annotations, taxonomy)
            # No validation/test annotations exist, and their extraction statistics
            # are adversarial. Neither can affect the fitting-only selector.
            mutated = copy.deepcopy(records)
            for row in mutated:
                if row["subject"] not in stage.FIT_SUBJECTS:
                    row["pair_windows"] = 10 ** 9
                    row["n_windows"] = 1
            self.assertEqual(expected, launch.choose_pilot(list(reversed(mutated)), annotations, taxonomy))
            self.assertEqual(36, expected["n_fit_videos"])
            self.assertEqual(3, len(set(expected["probe_video_ids"])))

    def test_projection_uses_slowest_whole_video_and_all_remaining_test_work(self):
        job = fixture_job()
        by_id = stage.validate_job(job)
        reports = [{"video_id": vid, "status": "complete", "reused": False,
                    "n_windows": by_id[vid]["n_windows"],
                    "elapsed_seconds": by_id[vid]["n_windows"] * rate}
                   for vid, rate in zip(job["pilot"]["probe_video_ids"], (.01, .03, .02))]
        pilot = job["pilot"]["pilot_video_ids"] + [x["video_id"] for x in by_id.values()
                                                  if x["subject"] in stage.VALIDATION_SUBJECTS]
        value = stage.projected_budget(job, reports, 100, pilot, by_id)
        remaining = sum(x["n_windows"] for x in by_id.values()) - sum(x["n_windows"] for x in reports)
        self.assertAlmostEqual(.0375, value["seconds_per_window_conservative"])
        self.assertAlmostEqual(100 + .0375 * remaining + 3600 + 600, value["projected_job_seconds"])
        reports[0]["reused"] = True
        with self.assertRaisesRegex(ValueError, "fresh complete"):
            stage.projected_budget(job, reports, 100, pilot, by_id)


class RGBExecutionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rgb-execution-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.work = self.root / "work"
        self.work.mkdir()
        self.annotations = self.root / "smarthome_CS_51.json"
        write_json(self.annotations, {})
        self.job = fixture_job()
        self.job["annotation_sha256"] = sha256(self.annotations)
        self.by_id = stage.validate_job(self.job)
        self.now = 0.
        self.commands, self.binds, self.loads, self.extractions = [], [], [], []
        self.cached = set()
        self.base_rate, self.tiny_rate = .005, .002
        self.pilot_pass = self.full_pass = True
        self.preflight_seconds = 0.
        self.child_timeout = None
        self.late_probe_report = False

    def fake_load(self, checkpoint, expected, arch, device):
        self.loads.append(arch)
        self.now += 2
        return object(), {"arch": arch, "weights_sha256": expected,
                          "preprocessing": self.job["assets"]["preprocessing"], "preprocessing_sha256": "1" * 64}

    def fake_extract(self, video, shard, output, encoder, info, batch_size, deadline):
        vid = Path(video).stem
        row = self.by_id[vid]
        reused = output in self.cached
        duration = .001 if reused else row["n_windows"] * (self.base_rate if info["arch"] == "swin3d_b" else self.tiny_rate)
        if self.now + duration > deadline:
            self.now = max(self.now, deadline)
            raise TimeoutError("synthetic whole-video deadline")
        self.now += duration
        if self.late_probe_report and len(self.extractions) == 2:
            # Emulate a slow final report hash after the extractor's last
            # cooperative deadline check, returning a complete cache too late.
            self.now = self.job["probe_seconds"] + 1
        self.extractions.append((info["arch"], vid, reused, deadline))
        if row["subset"] == "testing":
            self.assertTrue((self.work / "frozen_selection.json").is_file())
            self.assertTrue(self.pilot_pass and self.full_pass)
        self.cached.add(output)
        return {"status": "complete", "video_id": vid, "subject": row["subject"],
            "n_windows": row["n_windows"], "source_video_sha256": row["source_sha256"],
            "source_shard_sha256": row["shard_sha256"], "encoder_sha256": info["weights_sha256"],
            "preprocessing_sha256": info["preprocessing_sha256"], "elapsed_seconds": duration, "reused": reused}

    def fake_bind(self, job, subset, work):
        self.binds.append(subset)
        if subset == "testing":
            self.assertTrue((self.work / "frozen_selection.json").is_file())
        return [self.root / subset / "*.npz"], {x["video_id"]: self.root / x["shard_filename"]
                                               for x in self.by_id.values() if x["subset"] == subset}

    def fake_child(self, arguments, timeout):
        args = list(map(str, arguments))
        self.commands.append(args)
        command = args[2] if args[2] == "evaluate-test" else args[args.index("--scope") + 1]
        if command == self.child_timeout:
            raise subprocess.TimeoutExpired(args, timeout)
        out = Path(args[args.index("--out") + 1])
        out.mkdir(parents=True)
        if command == "pilot":
            self.assertNotIn("testing", " ".join(args))
            write_json(out / "gate.json", {"passed": self.pilot_pass, "reason": "fixture pilot gate",
                "criteria": self.job["gate_criteria"], "metrics": {}})
        elif command == "full":
            self.assertNotIn("testing", " ".join(args))
            (out / "best_rgb_multilabel.pt").write_bytes(b"frozen fixture checkpoint")
            write_json(out / "thresholds.json", {"source": "validation", "thresholds": [.5]})
            write_json(out / "frozen_selection.json", {"gate_passed": self.full_pass,
                "checkpoint_relative_path": "best_rgb_multilabel.pt",
                "checkpoint_sha256": sha256(out / "best_rgb_multilabel.pt"),
                "thresholds_sha256": sha256(out / "thresholds.json")})
        else:
            checkpoint = Path(args[args.index("--checkpoint") + 1])
            write_json(out / "test_metrics.json", {"checkpoint_sha256": sha256(checkpoint),
                "thresholds_refit": False, "checkpoint_selection_on_test": False,
                "full_eligible_window_coverage_verified": True, "metrics": {"macro_ap": .01}})
        self.now += 100

    def execute(self):
        def boot(*args):
            self.now += self.preflight_seconds
            return self.root, self.fake_child
        def videos(job, subset):
            if subset == "testing":
                self.assertTrue((self.work / "frozen_selection.json").is_file())
            return {x["video_id"]: self.root / x["source_filename"]
                    for x in self.by_id.values() if x["subset"] == subset}
        def write_receipt(path, value):
            if Path(path).name == "stage_result.json" or Path(path).name.startswith("extraction_"):
                return
            write_json(path, value)
        with ExitStack() as stack:
            stack.enter_context(patch.object(stage, "write_json", side_effect=write_receipt))
            stack.enter_context(patch.object(stage, "prepare_assets", side_effect=boot))
            stack.enter_context(patch.object(stage, "dataset_root", return_value=self.root))
            stack.enter_context(patch.object(stage, "asset", side_effect=lambda root, name: root / name))
            stack.enter_context(patch.object(stage, "bind_shards", side_effect=self.fake_bind))
            stack.enter_context(patch.object(stage, "bind_videos", side_effect=videos))
            stack.enter_context(patch.object(stage.time, "monotonic", side_effect=lambda: self.now))
            stack.enter_context(patch.dict(sys.modules, {
                "behaviorsense.models.rgb_action": SimpleNamespace(load_rgb_encoder=self.fake_load),
                "extract_rgb_action_features": SimpleNamespace(extract_video_features=self.fake_extract),
                "torch": SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None,
                    OutOfMemoryError=type("OutOfMemoryError", (RuntimeError,), {})))}))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return stage.execute(self.job, ROOT, self.work)

    def test_one_full_head_after_matched_pilot_then_all_test_and_probe_caches_reused(self):
        result = self.execute()
        self.assertEqual("complete", result["status"])
        self.assertEqual(["training", "testing"], self.binds)
        self.assertEqual(["swin3d_b"], self.loads)
        self.assertEqual(3, len(self.commands))
        self.assertEqual("full", self.commands[1][self.commands[1].index("--scope") + 1])
        self.assertEqual("evaluate-test", self.commands[2][2])
        fresh = [(arch, vid) for arch, vid, reused, _ in self.extractions if not reused]
        self.assertEqual(536, len(fresh))
        self.assertEqual(536, len(set(fresh)))
        self.assertEqual(.01, result["test"]["metrics"]["macro_ap"])
        self.assertFalse(result["automatic_continuation"])

    def test_tiny_fallback_after_base_projection_rejection(self):
        self.base_rate = .05
        result = self.execute()
        self.assertEqual("complete", result["status"])
        self.assertEqual(["swin3d_b", "swin3d_t"], self.loads)
        probes = read_json(self.work / "probe_report.json")
        self.assertEqual(["rejected_projection", "passed"], [x["status"] for x in probes])
        self.assertEqual(3, len(probes[0]["videos"]))

    def test_tiny_fallback_retains_time_after_base_whole_video_timeout(self):
        self.base_rate = 10.
        result = self.execute()
        self.assertEqual("complete", result["status"])
        probes = read_json(self.work / "probe_report.json")
        self.assertEqual(["probe_deadline", "passed"], [x["status"] for x in probes])
        self.assertLessEqual(probes[1]["projection"]["projected_pilot_seconds"], 4500)

    def test_both_throughput_failures_stop_without_heads_or_test(self):
        self.base_rate = self.tiny_rate = .10
        result = self.execute()
        self.assertEqual("throughput_rejected", result["status"])
        self.assertEqual([], self.commands)
        self.assertEqual(["training"], self.binds)

    def test_pilot_failure_stops_before_remaining_training_and_test(self):
        self.pilot_pass = False
        result = self.execute()
        self.assertEqual("quality_rejected", result["status"])
        self.assertEqual(1, len(self.commands))
        self.assertEqual(["training"], self.binds)
        self.assertEqual(113, len({vid for _, vid, _, _ in self.extractions}))
        self.assertFalse((self.work / "frozen_selection.json").exists())

    def test_full_failure_stops_before_any_test_binding_or_test_pixels(self):
        self.full_pass = False
        result = self.execute()
        self.assertEqual("quality_rejected", result["status"])
        self.assertEqual(2, len(self.commands))
        self.assertEqual(["training"], self.binds)
        self.assertFalse(result["test_features_loaded"])

    def test_bootstrap_time_is_charged_to_shared_probe_and_pilot_budget(self):
        self.preflight_seconds = 890
        result = self.execute()
        self.assertEqual("throughput_rejected", result["status"])
        self.assertEqual([], self.commands)
        self.assertEqual([], self.loads)

    def test_late_complete_probe_report_cannot_expand_into_pilot(self):
        self.late_probe_report = True
        result = self.execute()
        self.assertEqual("throughput_rejected", result["status"])
        self.assertEqual([], self.commands)
        self.assertFalse((self.work / "encoder_selection.json").exists())

    def test_timeout_cannot_train_another_head_or_load_test(self):
        self.child_timeout = "pilot"
        result = self.execute()
        self.assertEqual("pilot_budget_exhausted", result["status"])
        self.assertEqual(1, len(self.commands))
        self.assertEqual(["training"], self.binds)

    def test_existing_execution_cannot_resume_even_after_early_rejection(self):
        self.base_rate = self.tiny_rate = .1
        self.execute()
        with self.assertRaisesRegex(ValueError, "cannot resume"):
            self.execute()


class RGBLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rgb-launch-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        remote = patch.object(launch, "verify_remote_inputs", return_value={"fixture": True})
        self.remote = remote.start()
        self.addCleanup(remote.stop)

    def build(self):
        repo = self.root / "source"
        assets = self.root / "assets"
        assets.mkdir()
        write_json(assets / "dataset-metadata.json", {"id": "fixture/" + launch.ASSET_SLUG})
        write_json(assets / "asset-manifest.json", fixture_job()["assets"])
        for name in ("scripts/train_rgb_multilabel.py", "scripts/extract_rgb_action_features.py",
                     "scripts/rgb_kaggle_stage.py", "scripts/rgb_profile_stage.py", "src/example.py", "configs/taxonomy.yaml"):
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# source fixture\n")
        ann = repo / "artifacts/training-tv-20261007/data/smarthome_CS_51.json"
        ann.parent.mkdir(parents=True)
        write_json(ann, {})
        old = repo / "artifacts/training-tv-20261007/full-run/plan.json"
        old.parent.mkdir(parents=True)
        write_json(old, {"source_sha256": launch.EXTRACTION_SOURCE_SHA256, "annotation_sha256": sha256(ann)})
        previous = repo / "artifacts/temporal-multilabel-20261011/output"
        (previous / "candidates/asl-seed1").mkdir(parents=True)
        (previous / "test").mkdir()
        write_json(previous / "candidates/asl-seed1/manifest.json", {})
        write_json(previous / "test/test_metrics.json", {})
        directory = self.root / "run"
        with patch.object(launch, "verify_assets", return_value=fixture_job()["assets"]), \
                patch.object(launch, "extraction_sources", return_value=[]), \
                patch.object(launch, "video_inventory", return_value=inventory()), \
                patch.object(launch, "choose_pilot", return_value=fixture_job()["pilot"]):
            plan = launch.build(directory, assets, "fixture", repo)
        return directory, plan

    def execute_runtime_fixture(self, body, supervise, work_name="runtime"):
        """Execute the real extraction envelope in a temp directory, with training mocked."""
        compile(body, "training.py", "exec")
        work = self.root / work_name
        marker = 'WORK = Path("/kaggle/working")'
        self.assertEqual(1, body.count(marker))
        body = body.replace(marker, f"WORK = Path({str(work)!r})")
        namespace = {}
        with patch.dict(sys.modules, {"rgb_kaggle_stage": SimpleNamespace(supervise=supervise)}), \
                patch.dict(launch.os.environ), patch.object(sys, "path", sys.path.copy()), \
                patch("time.monotonic", return_value=123.5):
            exec(compile(body, "training.py", "exec"), namespace)
        return namespace

    def test_build_is_private_finite_offline_with_separate_source_hash(self):
        directory, plan = self.build()
        meta = read_json(directory / "submission/kernel-metadata.json")
        self.assertEqual("NvidiaRtxPro6000", meta["machine_shape"])
        self.assertEqual(launch.TRAINING_IMAGE, meta["docker_image"])
        self.assertEqual(3, len(meta["dataset_sources"]))
        self.assertTrue(meta["is_private"])
        self.assertFalse(meta["enable_internet"])
        self.assertEqual(43200, plan["timeout_seconds"])
        self.assertNotEqual(plan["job"]["source_sha256"], plan["job"]["extraction_source_sha256"])
        compile((directory / "submission/training.py").read_text(), "training.py", "exec")
        with self.assertRaises(FileExistsError):
            launch.build(directory)

    def test_compressed_runtime_preserves_large_nested_job_and_verified_source(self):
        directory, plan = self.build()
        bundle = (directory / "source.b64").read_text(encoding="ascii")
        manifest = plan["source_manifest"]
        job = copy.deepcopy(plan["job"])
        job["coverage"] = {"rows": [{"video_id": f"fixture-{index}",
            "provenance": {"nested": [None, True, False, -1, .125, "caf\u00e9\n'quoted'"],
                           "sha256": "a" * 64}, "windows": list(range(20))}
            for index in range(6000)]}
        self.assertGreater(len(repr(job).encode("utf-8")), launch.MAX_KERNEL_SOURCE_BYTES)
        body = launch.runtime(bundle, manifest, job)
        self.assertLess(len(body.encode("utf-8")), launch.MAX_KERNEL_SOURCE_BYTES)
        supervise = Mock()
        namespace = self.execute_runtime_fixture(body, supervise)
        work = self.root / "runtime"
        supervise.assert_called_once_with(job, work / "rgb_source", work, started_at=123.5)
        self.assertEqual(job, namespace["JOB"])
        self.assertEqual(job, read_json(work / "job.json"))
        self.assertEqual(manifest, namespace["MANIFEST"])
        self.assertEqual(manifest, read_json(work / "source_manifest.json"))
        self.assertEqual(base64.b64decode(bundle, validate=True), namespace["raw"])
        for name, expected in manifest["files"].items():
            self.assertEqual(expected, sha256(work / "rgb_source" / name))
        self.assertEqual(bundle, (directory / "source.b64").read_text(encoding="ascii"))

    def test_compressed_runtime_still_rejects_changed_archive_and_file_manifest(self):
        directory, plan = self.build()
        bundle = (directory / "source.b64").read_text(encoding="ascii")
        manifest, job = plan["source_manifest"], plan["job"]
        raw = base64.b64decode(bundle, validate=True)
        changed_bundle = base64.b64encode(raw[:-1] + bytes([raw[-1] ^ 1])).decode("ascii")
        changed_manifest = copy.deepcopy(manifest)
        changed_manifest["files"][next(iter(manifest["files"]))] = "0" * 64
        for name, candidate_bundle, candidate_manifest in (
                ("archive", changed_bundle, manifest), ("manifest", bundle, changed_manifest)):
            with self.subTest(change=name):
                supervise = Mock()
                body = launch.runtime(candidate_bundle, candidate_manifest, job)
                with self.assertRaises(AssertionError):
                    self.execute_runtime_fixture(body, supervise, work_name=name)
                supervise.assert_not_called()

    def test_build_rejects_source_at_limit_before_writing_a_prepared_plan(self):
        with patch.object(launch, "runtime", return_value="#" * launch.MAX_KERNEL_SOURCE_BYTES):
            with self.assertRaisesRegex(ValueError, "less than 1,000,000 bytes"):
                self.build()
        self.assertFalse((self.root / "run").exists())
        self.remote.assert_not_called()

    def test_build_measures_utf8_bytes_instead_of_characters(self):
        body = "#" + "\u00e9" * (launch.MAX_KERNEL_SOURCE_BYTES // 2)
        self.assertLess(len(body), launch.MAX_KERNEL_SOURCE_BYTES)
        with patch.object(launch, "runtime", return_value=body):
            with self.assertRaisesRegex(ValueError, "less than 1,000,000 bytes"):
                self.build()
        self.assertFalse((self.root / "run").exists())

    def test_build_preserves_exact_utf8_bytes_just_below_limit(self):
        body = "#\u00e9\n" + "#" * (launch.MAX_KERNEL_SOURCE_BYTES - 5)
        self.assertEqual(launch.MAX_KERNEL_SOURCE_BYTES - 1, len(body.encode("utf-8")))
        with patch.object(launch, "runtime", return_value=body):
            directory, plan = self.build()
        self.assertEqual("prepared", plan["status"])
        self.assertEqual(body.encode("utf-8"), (directory / "submission/training.py").read_bytes())

    def test_push_rejects_frozen_oversized_source_before_any_network(self):
        directory, plan = self.build()
        source = directory / "submission/training.py"
        source.write_bytes(b"#" * launch.MAX_KERNEL_SOURCE_BYTES)
        plan["submission_hashes"]["training.py"] = sha256(source)
        write_json(directory / "plan.json", plan)
        api = Mock()
        with self.assertRaisesRegex(ValueError, "less than 1,000,000 bytes"):
            launch.push(api, directory)
        self.remote.assert_not_called()
        self.assertEqual([], api.mock_calls)
        self.assertEqual(plan, read_json(directory / "plan.json"))
        self.assertEqual(launch.MAX_KERNEL_SOURCE_BYTES, source.stat().st_size)
        self.assertFalse((directory / "remote_input_verification.json").exists())

    def test_modified_frozen_source_refuses_submission_before_network(self):
        directory, _ = self.build()
        with (directory / "submission/training.py").open("a") as out:
            out.write("# changed\n")
        api = Mock()
        with self.assertRaisesRegex(ValueError, "changed"):
            launch.push(api, directory)
        api.kernels_push.assert_not_called()

    def test_one_accepted_commit_journals_identity_and_refuses_duplicate(self):
        directory, plan = self.build()
        api = Mock()
        api.kernels_push.return_value = {"ref": "/code/" + plan["job"]["job_id"],
            "url": "https://www.kaggle.com/code/" + plan["job"]["job_id"], "versionNumber": 1, "kernelId": 42}
        launch.push(api, directory)
        self.assertEqual("accepted", read_json(directory / "plan.json")["submission_attempt"]["state"])
        with self.assertRaisesRegex(ValueError, "already attempted"):
            launch.push(api, directory)
        api.kernels_push.assert_called_once()

    def test_ambiguous_push_never_retries(self):
        directory, _ = self.build()
        api = Mock()
        api.kernels_push.side_effect = TimeoutError("response lost")
        with self.assertRaises(TimeoutError):
            launch.push(api, directory)
        self.assertEqual("submission_ambiguous", read_json(directory / "plan.json")["status"])
        with self.assertRaisesRegex(ValueError, "already attempted"):
            launch.push(api, directory)
        api.kernels_push.assert_called_once()

    def test_unready_or_public_assets_stop_before_consuming_one_submission(self):
        directory, _ = self.build()
        self.remote.side_effect = ValueError("Required private dataset version is not verified ready")
        api = Mock()
        with self.assertRaisesRegex(ValueError, "not verified ready"):
            launch.push(api, directory)
        api.kernels_push.assert_not_called()
        self.assertEqual("prepared", read_json(directory / "plan.json")["status"])

    def test_private_asset_upload_journals_before_call_and_refuses_ambiguous_retry(self):
        assets = self.root / "assets"
        assets.mkdir()
        write_json(assets / "dataset-metadata.json", {"id": "fixture/" + launch.ASSET_SLUG})
        write_json(assets / "asset-manifest.json", {})
        api = Mock()
        def uncertain(*args, **kwargs):
            self.assertFalse(kwargs["public"])
            self.assertEqual("intent_journaled", read_json(self.root / "asset-upload-intent.json")["state"])
            raise TimeoutError("response lost")
        api.dataset_create_new.side_effect = uncertain
        with patch.object(launch, "verify_assets"):
            with self.assertRaises(TimeoutError):
                launch.upload_assets(api, "fixture", assets)
            self.assertEqual("ambiguous", read_json(self.root / "asset-upload-intent.json")["state"])
            with self.assertRaisesRegex(ValueError, "already attempted"):
                launch.upload_assets(api, "fixture", assets)
        api.dataset_create_new.assert_called_once()


class RGBRemoteInputTests(unittest.TestCase):
    def setUp(self):
        self.job = fixture_job()
        self.job["assets"]["files"] = {"fixture.pth": {"bytes": 123}}
        self.private, self.version, self.state = True, 1, "READY"
        self.api = MagicMock()
        client = self.api.build_kaggle_client.return_value.__enter__.return_value.datasets.dataset_api_client
        client.get_dataset.side_effect = lambda request: SimpleNamespace(
            ref=request.owner_slug + "/" + request.dataset_slug,
            is_private=self.private, current_version_number=self.version)
        client.get_dataset_status.side_effect = lambda request: SimpleNamespace(status=SimpleNamespace(name=self.state))
        self.api.dataset_list_files.return_value = SimpleNamespace(error_message="", next_page_token="",
            dataset_files=[SimpleNamespace(name="fixture.pth", total_bytes=123),
                           SimpleNamespace(name="asset-manifest.json", total_bytes=456)])

    def verify(self):
        sdk = SimpleNamespace(ApiGetDatasetRequest=SimpleNamespace, ApiGetDatasetStatusRequest=SimpleNamespace)
        with patch.dict(sys.modules, {"kagglesdk.datasets.types.dataset_api_service": sdk}):
            return launch.verify_remote_inputs(self.api, self.job)

    def test_remote_ready_private_exact_version_and_file_lengths_are_verified(self):
        result = self.verify()
        self.assertTrue(result["ready_private_versions_verified"])
        self.assertEqual(2, len(result["datasets"]))
        self.api.dataset_list_files.assert_called_once_with(self.job["asset_dataset"], page_size=100)

    def test_public_wrong_version_or_processing_dataset_is_rejected(self):
        for field, value in (("private", False), ("version", 2), ("state", "PROCESSING")):
            setattr(self, field, value)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "not verified ready"):
                self.verify()
            setattr(self, field, {"private": True, "version": 1, "state": "READY"}[field])

    def test_truncated_or_incomplete_asset_listing_is_rejected(self):
        self.api.dataset_list_files.return_value.dataset_files[0].total_bytes = 12
        with self.assertRaisesRegex(ValueError, "file inventory"):
            self.verify()
        self.api.dataset_list_files.return_value.next_page_token = "more"
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.verify()

    def test_remote_wheel_filename_normalization_is_recorded_with_exact_bytes(self):
        self.job["assets"]["files"][launch.WHEEL_NAME] = {"bytes": 8057045}
        actual = launch.WHEEL_NAME.replace("+", "")
        self.api.dataset_list_files.return_value.dataset_files.append(SimpleNamespace(name=actual, total_bytes=8057045))
        result = self.verify()
        self.assertEqual(actual, result["mounted_asset_filenames"][launch.WHEEL_NAME])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    total = result.testsRun - len(result.skipped)
    passed = total - len(result.failures) - len(result.errors)
    print(f"{passed}/{total} passed; {len(result.skipped)} skipped")
    raise SystemExit(0 if result.wasSuccessful() else 1)
