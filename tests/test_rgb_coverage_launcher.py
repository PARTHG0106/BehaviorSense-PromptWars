"""Synthetic offline launcher wiring and pinned cache-source readiness checks."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import hashlib
import io
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import launch_rgb_kaggle as launch
import rgb_coverage_contract as contract
import rgb_kaggle_stage as stage
import test_rgb_launcher as fixtures
from multilabel_kaggle_stage import read_json, sha256, write_json


CACHE_REF = "krishnagupta02468/bs-rgb-prefetch-20261013/1"
NEW_SLUG = "bs-rgb-coverage-20261014"


class RGBCoverageLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rgb-coverage-launch-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo, self.assets = self.root / "repo", self.root / "assets"
        self.assets.mkdir()
        self.assets_manifest = fixtures.fixture_job()["assets"]
        write_json(self.assets / "dataset-metadata.json", {"id": "fixture/" + launch.ASSET_SLUG})
        write_json(self.assets / "asset-manifest.json", self.assets_manifest)
        for name in ("scripts/train_rgb_multilabel.py", "scripts/extract_rgb_action_features.py",
                     "scripts/rgb_kaggle_stage.py", "scripts/rgb_profile_stage.py",
                     "scripts/rgb_coverage_contract.py", "scripts/rgb_cache_reuse.py",
                     "scripts/rgb_coverage_budget.py",
                     "src/example.py", "configs/taxonomy.yaml"):
            target = self.repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("# Synthetic source fixture\n", encoding="utf-8")
        annotations = self.repo / "artifacts/training-tv-20261007/data/smarthome_CS_51.json"
        annotations.parent.mkdir(parents=True)
        write_json(annotations, {})
        original = self.repo / "artifacts/training-tv-20261007/full-run/plan.json"
        original.parent.mkdir(parents=True)
        write_json(original, {"source_sha256": launch.EXTRACTION_SOURCE_SHA256,
                              "annotation_sha256": sha256(annotations)})
        previous = self.repo / "artifacts/temporal-multilabel-20261011/output"
        for relative in ("candidates/asl-seed1/manifest.json", "test/test_metrics.json"):
            target = previous / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            write_json(target, {})
        self.records = fixtures.inventory()
        fit = [row for row in self.records if row["subject"] in stage.FIT_SUBJECTS]
        ids = sorted(row["video_id"] for row in fit[:60])
        core = fixtures.fixture_job()["pilot"]["probe_video_ids"]
        sparse = [vid for vid in ids if vid not in core][-2:]
        self.coverage = {
            "schema": "synthetic.coverage.v1", "pilot": {"pilot_video_ids": ids,
                "probe_video_ids": core, "n_fit_videos": 60,
                "fit_windows": sum(row["n_windows"] for row in fit[:60])},
            "core_probe_video_ids": core, "sparse_probe_video_ids": sparse,
            "probe_video_ids": core + sparse,
            "cache_reuse": {"kernel_source": CACHE_REF,
                "source_job_id": CACHE_REF.rsplit("/", 1)[0],
                "source_sha256": "a" * 64,
                "training_source_text_sha256": hashlib.sha256(b"# fixture\n").hexdigest()},
        }
        self.plan_path = self.repo / "synthetic-coverage-plan.json"
        write_json(self.plan_path, {"synthetic": True})
        self.sources = [{"ref": "fixture/original-extraction/1", "subset": "training"}]
        stack = self.enterContext(ExitStack())
        stack.enter_context(patch.object(launch, "verify_assets", return_value=self.assets_manifest))
        stack.enter_context(patch.object(launch, "extraction_sources", return_value=self.sources))
        stack.enter_context(patch.object(launch, "video_inventory", return_value=self.records))
        self.choose = stack.enter_context(patch.object(launch, "choose_pilot",
                                                     return_value=fixtures.fixture_job()["pilot"]))
        self.freeze_coverage = stack.enter_context(patch.object(contract, "build_coverage_contract",
                                                               return_value=self.coverage))

    def build(self, coverage=True, run_slug=NEW_SLUG):
        directory = self.root / "run"
        kwargs = {"coverage_plan": self.plan_path, "run_slug": run_slug} if coverage else {}
        if coverage:
            # Contract and runtime semantics have separate synthetic suites. This
            # test isolates how the launcher freezes and passes the verified value.
            with patch.object(stage, "validate_job", return_value={row["video_id"]: row for row in self.records}) as check:
                plan = launch.build(directory, self.assets, "fixture", self.repo, **kwargs)
            check.assert_called_once_with(plan["job"])
        else:
            plan = launch.build(directory, self.assets, "fixture", self.repo, **kwargs)
        return directory, plan

    def test_build_freezes_verified_coverage_and_adds_only_the_pinned_cache_mount(self):
        directory, plan = self.build()
        job = plan["job"]
        self.freeze_coverage.assert_called_once_with(self.plan_path, self.repo)
        self.choose.assert_not_called()
        self.assertEqual(self.coverage, job["coverage"])
        self.assertEqual(self.coverage["pilot"], job["pilot"])
        self.assertEqual(60, len(job["pilot"]["pilot_video_ids"]))
        self.assertEqual(3, len(job["pilot"]["probe_video_ids"]))
        self.assertEqual(2, len(job["coverage"]["sparse_probe_video_ids"]))
        self.assertEqual("fixture/" + NEW_SLUG, job["job_id"])
        self.assertNotEqual(job["source_sha256"], job["coverage"]["cache_reuse"]["source_sha256"])
        self.assertEqual("native-prefetch-profile.v3", job["rgb_protocol"])
        self.assertEqual(["swin3d_t"], job["encoder_preference"])
        self.assertEqual([900, 4500, 40500], [job[key] for key in ("probe_seconds", "pilot_seconds", "session_seconds")])
        self.assertEqual(launch.GATE_CRITERIA, job["gate_criteria"])
        metadata = read_json(directory / "submission/kernel-metadata.json")
        self.assertEqual([self.sources[0]["ref"], CACHE_REF], metadata["kernel_sources"])
        self.assertTrue(metadata["is_private"])
        self.assertFalse(metadata["enable_internet"])
        self.assertFalse(job["automatic_continuation"])
        for name in ("rgb_coverage_contract.py", "rgb_cache_reuse.py", "rgb_coverage_budget.py"):
            self.assertIn("scripts/" + name, plan["source_manifest"]["files"])
        self.assertEqual(job, read_json(directory / "submission/job.json"))
        compile((directory / "submission/training.py").read_text(), "training.py", "exec")

    def test_legacy_default_build_retains_original_selection_slug_and_mounts(self):
        directory, plan = self.build(coverage=False)
        self.freeze_coverage.assert_not_called()
        self.choose.assert_called_once()
        self.assertNotIn("coverage", plan["job"])
        self.assertEqual("fixture/" + launch.RUN_SLUG, plan["job"]["job_id"])
        self.assertEqual(36, len(plan["job"]["pilot"]["pilot_video_ids"]))
        self.assertEqual([self.sources[0]["ref"]],
                         read_json(directory / "submission/kernel-metadata.json")["kernel_sources"])

    def test_coverage_requires_new_valid_slug_before_reading_plan_or_assets(self):
        for slug in (None, "", launch.RUN_SLUG, "other/slug", "UPPERCASE", "../escape", "a" * 51):
            with self.subTest(slug=slug), self.assertRaises(ValueError):
                self.build(run_slug=slug)
        self.freeze_coverage.assert_not_called()
        self.assertFalse((self.root / "run").exists())

    def test_rejected_coverage_lineage_never_freezes_a_submission(self):
        self.freeze_coverage.side_effect = ValueError("coverage provenance lineage differs")
        with self.assertRaisesRegex(ValueError, "lineage"):
            self.build()
        self.assertFalse((self.root / "run").exists())

    def test_coverage_source_dependency_must_exist_in_bundle(self):
        for name in ("rgb_coverage_contract.py", "rgb_cache_reuse.py", "rgb_coverage_budget.py"):
            target = self.repo / "scripts" / name
            original = target.read_bytes()
            target.unlink()
            with self.subTest(name=name), self.assertRaisesRegex(FileNotFoundError, "coverage source"):
                self.build()
            self.assertFalse((self.root / "run").exists())
            target.write_bytes(original)

    def test_unpinned_or_duplicate_cache_mount_is_rejected(self):
        job = {"coverage": copy.deepcopy(self.coverage), "feature_sources": self.sources}
        for ref in (CACHE_REF.rsplit("/", 1)[0], CACHE_REF[:-1] + "2", "fixture/different/1"):
            job["coverage"]["cache_reuse"]["kernel_source"] = ref
            with self.subTest(ref=ref), self.assertRaisesRegex(ValueError, "version 1"):
                launch.kernel_mounts(job)
        job["coverage"]["cache_reuse"]["kernel_source"] = CACHE_REF
        job["feature_sources"] = [{"ref": CACHE_REF}]
        with self.assertRaisesRegex(ValueError, "distinct"):
            launch.kernel_mounts(job)

    def test_push_requires_exact_coverage_mount_even_with_updated_metadata_hash(self):
        directory, plan = self.build()
        metadata_path = directory / "submission/kernel-metadata.json"
        metadata = read_json(metadata_path)
        metadata["kernel_sources"].remove(CACHE_REF)
        write_json(metadata_path, metadata)
        plan["submission_hashes"]["kernel-metadata.json"] = sha256(metadata_path)
        write_json(directory / "plan.json", plan)
        with patch.object(stage, "validate_job"), patch.object(launch, "verify_remote_inputs") as remote, \
                patch.object(launch, "push_frozen") as push:
            with self.assertRaisesRegex(ValueError, "mounts"):
                launch.push(Mock(), directory)
        remote.assert_not_called()
        push.assert_not_called()

    def test_push_accepts_coverage_mount_and_preserves_remote_evidence(self):
        directory, plan = self.build()
        api = Mock()
        evidence = {"cache_kernel": {"source_verified": True, "status": "complete"}}
        with patch.object(stage, "validate_job"), \
                patch.object(launch, "verify_remote_inputs", return_value=evidence) as verify, \
                patch.object(launch, "push_frozen", return_value={"synthetic": "receipt"}) as push:
            result = launch.push(api, directory)
        verify.assert_called_once_with(api, plan["job"])
        push.assert_called_once_with(api, directory)
        self.assertEqual({"synthetic": "receipt"}, result)
        self.assertEqual(evidence, read_json(directory / "remote_input_verification.json"))

    def test_output_filter_includes_coverage_receipts_and_excludes_feature_arrays(self):
        directory, _ = self.build()
        api = Mock()
        api.kernels_output.return_value = ([], None)
        launch.outputs(api, directory)
        pattern = api.kernels_output.call_args.kwargs["file_pattern"]
        for name in ("coverage_execution.json", "cache_reuse.json", "sparse_probe_report.json",
                     "profiling/swin3d_t/profile_report.json"):
            with self.subTest(name=name):
                self.assertIsNotNone(re.search(pattern, name))
        self.assertIsNone(re.search(pattern, "rgb_features/swin3d_t/P03T01C01.npz"))
        launch.outputs(api, directory, logs_only=True)
        self.assertEqual(r"(?!)", api.kernels_output.call_args.kwargs["file_pattern"])

    def test_cli_passes_coverage_plan_and_slug_to_local_build(self):
        identity = self.root / "synthetic-identity.json"
        write_json(identity, {"username": "fixture"})
        argv = ["launch_rgb_kaggle.py", "build", "--assets", str(self.assets),
                "--directory", str(self.root / "run"), "--credentials", str(identity),
                "--coverage-plan", str(self.plan_path), "--run-slug", NEW_SLUG]
        result = {"status": "prepared", "job": {"job_id": "fixture/" + NEW_SLUG,
                  "pilot": self.coverage["pilot"], "source_sha256": "a" * 64}}
        with patch.object(sys, "argv", argv), patch.object(launch, "build", return_value=result) as build, \
                patch.object(launch, "authenticated_api") as authenticate, redirect_stdout(io.StringIO()):
            launch.main()
        build.assert_called_once_with(self.root / "run", self.assets, "fixture",
                                      coverage_plan=self.plan_path, run_slug=NEW_SLUG)
        authenticate.assert_not_called()

    def test_cli_rejects_coverage_flags_on_network_actions_before_authentication(self):
        with patch.object(sys, "argv", ["launch_rgb_kaggle.py", "push", "--run-slug", NEW_SLUG]), \
                patch.object(launch, "authenticated_api") as authenticate, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                launch.main()
        authenticate.assert_not_called()


class RGBCoverageRemoteSourceTests(unittest.TestCase):
    def setUp(self):
        self.source = "# frozen version one\n"
        self.job = {"feature_sources": [], "coverage": {"cache_reuse": {
            "kernel_source": CACHE_REF,
            "training_source_text_sha256": hashlib.sha256(self.source.encode()).hexdigest()}}}
        self.api = MagicMock()
        self.client = self.api.build_kaggle_client.return_value.__enter__.return_value.kernels.kernels_api_client
        self.metadata = SimpleNamespace(ref=CACHE_REF.rsplit("/", 1)[0], is_private=True,
                                        current_version_number=1)
        self.client.get_kernel.return_value = SimpleNamespace(metadata=self.metadata,
                                                               blob=SimpleNamespace(source=self.source))
        self.client.get_kernel_session_status.return_value = SimpleNamespace(status=SimpleNamespace(name="COMPLETE"))
        self.sdk = SimpleNamespace(ApiGetKernelRequest=SimpleNamespace,
                                   ApiGetKernelSessionStatusRequest=SimpleNamespace)

    def verify(self):
        with patch.dict(sys.modules, {"kagglesdk.kernels.types.kernels_api_service": self.sdk}):
            return launch.verify_remote_cache_kernel(self.api, self.job)

    def test_read_only_requests_bind_status_to_version_one_before_and_after(self):
        value = self.verify()
        self.assertTrue(value["source_verified"])
        self.assertEqual("complete", value["status"])
        self.assertEqual(1, value["requested_version"])
        self.assertEqual("current version bound before and after status", value["lookup_scope"])
        self.assertEqual(["get_kernel", "get_kernel_session_status", "get_kernel"],
                         [call[0] for call in self.client.mock_calls])
        self.assertEqual(value["source_snapshots"]["before_status"],
                         value["source_snapshots"]["after_status"])
        for call in self.client.mock_calls:
            request = call.args[0]
            self.assertEqual("krishnagupta02468", request.user_name)
            self.assertEqual(launch.RUN_SLUG, request.kernel_slug)
            self.assertFalse(getattr(request, "version_label", None))
        self.api.kernels_status.assert_not_called()
        self.api.kernels_push.assert_not_called()
        self.api.kernels_output.assert_not_called()

    def test_unready_public_wrong_identity_or_newer_source_version_is_rejected(self):
        for key, value in (("ref", "fixture/different"), ("is_private", False), ("current_version_number", 2)):
            old = getattr(self.metadata, key)
            setattr(self.metadata, key, value)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "not verified complete"):
                self.verify()
            setattr(self.metadata, key, old)
        self.client.get_kernel_session_status.return_value.status.name = "RUNNING"
        with self.assertRaisesRegex(ValueError, "not verified complete"):
            self.verify()

    def test_missing_or_changed_source_cannot_pass_identity_check(self):
        for source in (None, "# changed source\n", self.source.replace("\n", "\r\n")):
            self.client.get_kernel.return_value.blob.source = source
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "source differs"):
                self.verify()

    def test_version_change_around_status_is_rejected_even_when_status_is_complete(self):
        before = copy.deepcopy(self.client.get_kernel.return_value)
        after = copy.deepcopy(before)
        after.metadata.current_version_number = 2
        self.client.get_kernel.side_effect = [before, after]
        with self.assertRaisesRegex(ValueError, "not verified complete"):
            self.verify()
        self.client.get_kernel_session_status.assert_called_once()
        self.assertEqual(2, self.client.get_kernel.call_count)

    def test_source_or_identity_change_after_complete_status_is_rejected(self):
        before = copy.deepcopy(self.client.get_kernel.return_value)
        for field in ("source", "is_private", "ref"):
            after = copy.deepcopy(before)
            if field == "source":
                after.blob.source = "# unexpected replacement\n"
            else:
                setattr(after.metadata, field, False if field == "is_private" else "fixture/different")
            self.client.get_kernel.side_effect = [before, after]
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.verify()

    def test_remote_input_verification_includes_pinned_cache_source_evidence(self):
        self.job.update(owner="fixture", asset_dataset="fixture/assets/1",
            assets={"files": {"fixture.pth": {"bytes": 123}}}, assets_manifest_sha256="a" * 64)
        datasets = self.api.build_kaggle_client.return_value.__enter__.return_value.datasets.dataset_api_client
        datasets.get_dataset.side_effect = lambda request: SimpleNamespace(
            ref=request.owner_slug + "/" + request.dataset_slug, is_private=True, current_version_number=1)
        datasets.get_dataset_status.return_value = SimpleNamespace(status=SimpleNamespace(name="READY"))
        self.api.dataset_list_files.return_value = SimpleNamespace(error_message="", next_page_token="",
            dataset_files=[SimpleNamespace(name="fixture.pth", total_bytes=123),
                           SimpleNamespace(name="asset-manifest.json", total_bytes=456)])
        dataset_sdk = SimpleNamespace(ApiGetDatasetRequest=SimpleNamespace, ApiGetDatasetStatusRequest=SimpleNamespace)
        with patch.dict(sys.modules, {"kagglesdk.kernels.types.kernels_api_service": self.sdk,
                                      "kagglesdk.datasets.types.dataset_api_service": dataset_sdk}):
            result = launch.verify_remote_inputs(self.api, self.job)
        self.assertEqual(CACHE_REF, result["cache_kernel"]["kernel_source"])
        self.assertTrue(result["cache_kernel"]["source_verified"])
        self.assertTrue(result["ready_private_versions_verified"])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    total = result.testsRun - len(result.skipped)
    passed = total - len(result.failures) - len(result.errors)
    print(f"{passed}/{total} passed; {len(result.skipped)} skipped")
    raise SystemExit(0 if result.wasSuccessful() else 1)
