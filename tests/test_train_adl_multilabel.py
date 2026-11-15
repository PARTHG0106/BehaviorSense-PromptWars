"""Regression checks for genuine multi-label gradients, holdouts, and resumable fitting."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_adl_multilabel as trainer
from behaviorsense.eval.multilabel import average_precision, fit_thresholds, masked_metrics
from behaviorsense.models.stgcnpp import STGCNpp
from behaviorsense.models.visual_context import (
    CONTEXT_DIM, CONTEXT_SCHEMA, FEATURE_NAMES, OUTPUT_CONTRACT, PREPROCESSING,
    VALIDATED_MODEL_SHA256, context_features,
)
from behaviorsense.data.toyota import TSU_CLASSES


def test_two_positives_receive_independent_gradients_and_unknown_is_ignored():
    logits = torch.zeros(1, 4, requires_grad=True)
    loss = trainer.masked_bce(logits, torch.tensor([[1., 1., 0., 0.]]),
                              torch.tensor([[True, True, True, False]]))
    loss.backward()
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] < 0
    assert logits.grad[0, 2] > 0 and logits.grad[0, 3] == 0
    assert torch.sigmoid(torch.tensor([4., 4.])).sum() > 1.9


def test_ap_ties_do_not_depend_on_window_order():
    assert average_precision(np.array([.8, .8]), np.array([True, False])) == .5
    assert average_precision(np.array([.8, .8]), np.array([False, True])) == .5


def test_thresholds_are_independent_and_disable_missing_evidence():
    scores = np.array([[.25, .85, .9], [.2, .8, .8], [.1, .65, .7], [.05, .6, .6]])
    targets = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0], [0, 0, 0]])
    mask = np.ones_like(targets, dtype=bool)
    fit = fit_thresholds(scores, targets, mask, min_positive=1, min_negative=1)
    assert fit["thresholds"][0] < fit["thresholds"][1]
    assert fit["thresholds"][2] == 1 and not fit["enabled"][2]
    metrics = masked_metrics(scores, targets, mask, fit["thresholds"], pair=None)
    assert metrics["macro_f1"] == 1
    assert metrics["per_class_ap"][2] is None
    assert fit["source_split"] == "validation"


def test_official_test_subjects_cannot_enter_train_or_validation():
    with pytest.raises(ValueError, match="non-training subjects"):
        trainer.subject_split(np.array(["p03", "p04", "p02"]))
    subjects = np.array(["p03", "p04", "p06", "p07"] * 2)
    train, val = trainer.subject_split(subjects, val_subjects=["p04"])
    assert not set(subjects[train]) & set(subjects[val])
    assert set(subjects[val]) == {"p04"}


def test_shard_order_is_stable_when_resume_mount_roots_change(tmp_path):
    first_mount = tmp_path / "z_partition"
    second_mount = tmp_path / "a_partition"
    first_mount.mkdir()
    second_mount.mkdir()
    alpha, beta = first_mount / "A.npz", second_mount / "B.npz"
    alpha.write_bytes(b"alpha")
    beta.write_bytes(b"beta")
    ordered = trainer.expand_shards([str(second_mount / "*.npz"), str(first_mount / "*.npz"), str(alpha)])
    assert ordered == [str(alpha.resolve()), str(beta.resolve())]


def test_initial_checkpoint_without_provenance_is_never_loaded(tmp_path):
    model = STGCNpp(n_classes=22, base_channels=4)
    before = copy.deepcopy(model.state_dict())
    path = tmp_path / "old.pt"
    torch.save({"model": {"fc.bias": torch.ones(22)},
                "args": {"stream": "bone", "n_classes": 22, "sample_fps": 20.}}, path)
    args = trainer.parser().parse_args(["train", "--shards", "unused", "--annotations", "unused",
                                        "--init", str(path)])
    result = trainer.initialise(model, args, ["p04"])
    assert result["mode"] == "scratch"
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({"checkpoint_sha256": trainer.file_sha256(path),
                                      "fit_subjects": ["p03", "p04"]}), encoding="utf-8")
    args.init_provenance = str(provenance)
    result = trainer.initialise(model, args, ["p04"])
    assert result["mode"] == "scratch" and "exposed" in result["reason"]


def synthetic_data():
    subjects = np.repeat(np.array(["p03", "p04", "p06", "p07"]), 4)
    n = len(subjects)
    skeletons = np.zeros((n, 30, 2, 17, 3), dtype=np.float16)
    rng = np.random.default_rng(19)
    skeletons[:, :, 0, :, :2] = rng.normal(0, .1, (n, 30, 17, 2))
    skeletons[:, :, 0, [5, 6], 1] = .5
    skeletons[:, :, 0, [11, 12], 1] = -.5
    skeletons[:, :, 0, :, 2] = .9
    targets = np.zeros((n, 22), dtype=np.float32)
    targets[:, 13] = np.tile([1, 0, 1, 0], 4)
    targets[:, 10] = np.tile([0, 1, 1, 0], 4)
    mask = np.zeros_like(targets, dtype=bool); mask[:, [10, 13]] = True
    return {"skeletons": skeletons, "subjects": subjects, "targets": targets,
            "target_mask": mask, "starts": np.tile(np.arange(4) * 30, 4),
            "video_ids": np.repeat(["P03T01C01", "P04T01C01", "P06T01C01", "P07T01C01"], 4),
            "sample_fps": 20., "tv_drinking_overlap_frames": np.tile([0, 0, 30, 0], 4),
            "provenance": {"fixture": True}}


def test_training_resume_matches_uninterrupted_and_test_never_refits(tmp_path, monkeypatch):
    data = synthetic_data()
    # Include an unknown gap to ensure it cannot form an empty-mask training batch.
    data["target_mask"][0] = False
    source = tmp_path / "source.npz"; source.write_bytes(b"fixture source")
    annotations = tmp_path / "annotations.json"; annotations.write_text("{}")
    monkeypatch.setattr(trainer, "load_multilabel_shards", lambda *a, **k: copy.deepcopy(data))
    base = ["train", "--shards", str(source), "--annotations", str(annotations),
            "--epochs", "2", "--min-epochs", "1", "--warmup-epochs", "0", "--base-channels", "4",
            "--batch-size", "4", "--workers", "0", "--cpu-threads", "1", "--device", "cpu",
            "--threshold-min-support", "1", "--val-subjects", "p04"]
    full_args = trainer.parser().parse_args(base + ["--out", str(tmp_path / "full")])
    full = trainer.train(full_args)
    partial_args = trainer.parser().parse_args(base + ["--out", str(tmp_path / "resumed"), "--stop-after", "1"])
    trainer.train(partial_args)
    resume = tmp_path / "resumed" / "last_multilabel.pt"
    resumed_args = trainer.parser().parse_args(base + ["--out", str(tmp_path / "resumed"), "--resume", str(resume)])
    resumed = trainer.train(resumed_args)
    for key in full["state_dict"]:
        torch.testing.assert_close(full["state_dict"][key], resumed["state_dict"][key], rtol=0, atol=0)
    assert full["threshold_config"] == resumed["threshold_config"]
    assert full["manifest"]["excluded_unknown_windows"] == 1
    assert not full["manifest"]["test_loaded_during_training"]
    assert "model" not in full and "ema" not in full
    assert full["manifest"]["train"]["tv_drinking_actual_overlap_windows"] > 0
    assert not full["manifest"]["convergence"]["converged"]
    assert full["manifest"]["convergence"]["warning"]
    assert full["manifest"]["checkpoint_selection"]["source_split"] == "validation"
    assert full["manifest"]["checkpoint_selection"]["test_metrics_consulted"] is False
    # Re-exporting an already completed resume must retain selection provenance too.
    resumed_again = trainer.train(resumed_args)
    assert resumed_again["manifest"]["checkpoint_selection"] == resumed["manifest"]["checkpoint_selection"]
    checkpoint = tmp_path / "full" / "best_multilabel.pt"
    before_hash = trainer.file_sha256(checkpoint)
    data["subjects"][:] = "p02"
    monkeypatch.setattr(trainer, "fit_thresholds", lambda *a, **k: pytest.fail("test threshold tuning"))
    test_args = trainer.parser().parse_args(["evaluate-test", "--checkpoint", str(checkpoint),
        "--test-shards", str(source), "--annotations", str(annotations), "--out", str(tmp_path / "test"),
        "--workers", "0", "--batch-size", "4", "--device", "cpu"])
    report = trainer.evaluate_test(test_args)
    assert report["thresholds_refit"] is False
    assert report["checkpoint_selection_on_test"] is False
    assert report["training_convergence"] == full["manifest"]["convergence"]
    assert trainer.file_sha256(checkpoint) == before_hash
    data["subjects"][:] = "p04"
    with pytest.raises(ValueError, match="official CS test"):
        trainer.evaluate_test(test_args)


def test_quality_defaults_and_plateau_report_distinguish_epoch_budget():
    args = trainer.parser().parse_args(["train", "--shards", "unused", "--annotations", "unused"])
    assert (args.epochs, args.min_epochs, args.patience, args.split_seed) == (100, 30, 12, 0)
    before_min = trainer.convergence_report(29, 0, min_epochs=30, epochs=100, patience=12)
    assert not before_min["converged"]
    plateau = trainer.convergence_report(30, 0, min_epochs=30, epochs=100, patience=12)
    assert plateau["converged"] and plateau["status"] == "validation_plateau"
    improving = trainer.convergence_report(100, 99, min_epochs=30, epochs=100, patience=12)
    assert not improving["converged"] and improving["warning"]
    assert improving["status"] == "epoch_budget_exhausted_without_validation_plateau"
    disabled = trainer.convergence_report(100, 0, min_epochs=30, epochs=100, patience=0)
    assert not disabled["converged"]


def test_training_cannot_early_stop_before_minimum_epochs(tmp_path, monkeypatch):
    data = synthetic_data()
    source = tmp_path / "source.npz"; source.write_bytes(b"fixture source")
    annotations = tmp_path / "annotations.json"; annotations.write_text("{}")
    monkeypatch.setattr(trainer, "load_multilabel_shards", lambda *a, **k: copy.deepcopy(data))
    # Constant validation AP exhausts patience at epoch 2, before the required floor.
    monkeypatch.setattr(trainer, "assess", lambda *a, **k: {"macro_ap": .5})
    args = trainer.parser().parse_args(["train", "--shards", str(source),
        "--annotations", str(annotations), "--out", str(tmp_path / "run"),
        "--epochs", "10", "--min-epochs", "3", "--patience", "1",
        "--base-channels", "4", "--batch-size", "4", "--workers", "0",
        "--cpu-threads", "1", "--device", "cpu", "--val-subjects", "p04"])
    result = trainer.train(args)
    convergence = result["manifest"]["convergence"]
    assert convergence["completed_epochs"] == 3
    assert convergence["converged"] and convergence["warning"] is None
    history = json.loads((tmp_path / "run" / "history.json").read_text())
    assert len(history) == 3
    assert result["manifest"]["checkpoint_selection"]["best_epoch_zero_based"] == 0


def test_actual_overlap_diagnostic_excludes_unknown_labels():
    data = synthetic_data()
    data["target_mask"][:] = False
    logits = np.full_like(data["targets"], 8.)
    result = trainer.assess(data, np.arange(len(logits)), logits)
    pair = result["tv_drinking_actual_overlap_windows"]
    assert pair["known_windows"] == pair["support"] == pair["predicted"] == 0


def context_training_fixture(directory):
    """Small real shard/annotation files for plumbing; never an accuracy benchmark."""
    videos = {}
    tv = TSU_CLASSES.index("Watch_TV")
    cup = TSU_CLASSES.index("Drink.From_cup")
    reading = TSU_CLASSES.index("Read")
    for subject in (2, 3, 4, 6, 7):
        videos[f"P{subject:02d}T01C01"] = {
            "subset": "testing" if subject == 2 else "training", "duration": 200,
            "actions": [[tv, 0, 36], [cup, 50, 86], [tv, 100, 136],
                        [cup, 100, 136], [reading, 150, 186]],
        }
    annotations = directory / "annotations.json"
    annotations.write_text(json.dumps(videos), encoding="utf-8")
    metadata = {"schema": CONTEXT_SCHEMA, "feature_dim": CONTEXT_DIM,
                "feature_names": list(FEATURE_NAMES), "model_sha256": VALIDATED_MODEL_SHA256,
                "score_threshold": .25, "keypoint_threshold": .30,
                "preprocessing": PREPROCESSING, "output_contract": OUTPUT_CONTRACT,
                "geometry_missing_value": -1, "geometry_units": "image_diagonal",
                "sampling": {"sample_offsets": [15]}, "synthetic_fixture": True}
    paths = {}
    for subset, subjects in (("training", (3, 4, 6, 7)), ("testing", (2,))):
        n = len(subjects) * 4
        skeletons = synthetic_data()["skeletons"][:n].astype(np.float32)
        skeletons[:, :, 0, :, :2] = skeletons[:, :, 0, :, :2] * 40 + 60
        starts = np.tile(np.array([0, 40, 80, 120]), len(subjects))
        frames = np.rint((starts[:, None] + np.arange(30)) * 25 / 20).astype(np.int64)
        context = []
        for index in range(n):
            objects = []
            if index % 4 in (0, 2):
                objects.append([80, 15, 115, 45, .85, 62])
            if index % 4 in (1, 2):
                objects.append([55, 52, 65, 64, .75, 41])
            if index % 4 == 3:
                objects.append([45, 50, 70, 70, .8, 73])
            context.append(context_features(np.asarray(objects, np.float32), skeletons[index, 15], (120, 160)))
        path = directory / f"context_{subset}_20hz.npz"
        np.savez_compressed(path, skeletons=skeletons.astype(np.float16),
            subjects=np.repeat([f"p{s:02d}" for s in subjects], 4),
            clips=np.repeat([f"P{s:02d}T01C01.mp4" for s in subjects], 4),
            datasets=np.full(n, "toyota_untrimmed"), starts=starts,
            source_frames=frames, source_fps=np.full(n, 25, np.float32),
            sample_fps=np.asarray(20.), annotation_fps=np.asarray(25.),
            visual_context=np.stack(context), context_source_frames=frames[:, [15]],
            context_feature_names=np.asarray(FEATURE_NAMES),
            context_metadata=np.asarray(json.dumps(metadata)))
        paths[subset] = path
    return paths, annotations, metadata


def context_train_args(paths, annotations, out, *extra):
    return trainer.parser().parse_args([
        "train", "--shards", str(paths["training"]), "--annotations", str(annotations),
        "--out", str(out), "--model", "context-fusion", "--require-context",
        "--context-hidden", "8", "--base-channels", "8", "--epochs", "4",
        "--min-epochs", "1", "--patience", "20", "--warmup-epochs", "0",
        "--scheduler", "plateau", "--lr-patience", "1", "--convergence-lr-reductions", "1",
        "--batch-size", "4", "--workers", "0", "--cpu-threads", "1", "--device", "cpu",
        "--precision", "float32", "--threshold-min-support", "1", "--val-subjects", "p04", *extra,
    ])


def test_context_shards_train_resume_and_frozen_test_evaluation(tmp_path, monkeypatch):
    paths, annotations, metadata = context_training_fixture(tmp_path)
    # Freeze only the selection metric to exercise an LR reduction before resume.
    # Data loading, augmentation, gradients, checkpoint serialization and test metrics
    # all execute their real paths; recognition accuracy is not asserted on this fixture.
    with monkeypatch.context() as training_metric:
        training_metric.setattr(trainer, "assess", lambda *a, **k: {"macro_ap": .5})
        full = trainer.train(context_train_args(paths, annotations, tmp_path / "full"))
        partial = trainer.train(context_train_args(paths, annotations, tmp_path / "resumed", "--stop-after", "3"))
        assert partial["status"] == "resumable_session_pause"
        assert not (tmp_path / "resumed" / "best_multilabel.pt").exists()
        checkpoint = tmp_path / "resumed" / "last_multilabel.pt"
        at_pause = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert at_pause["lr_reductions"] >= 1
        resumed = trainer.train(context_train_args(paths, annotations, tmp_path / "resumed", "--resume", str(checkpoint)))
    assert full["format"] == resumed["format"] == trainer.FUSION_FORMAT
    assert full["manifest"]["visual_context"] == metadata
    assert full["manifest"]["provenance"]["storage"].startswith("disk_backed")
    assert full["manifest"]["train"]["tv_drinking_actual_overlap_windows"] > 0
    assert full["manifest"]["checkpoint_selection"]["test_metrics_consulted"] is False
    for key, value in full["state_dict"].items():
        torch.testing.assert_close(value, resumed["state_dict"][key], rtol=0, atol=0)
    assert full["threshold_config"] == resumed["threshold_config"]
    last_full = torch.load(tmp_path / "full" / "last_multilabel.pt", map_location="cpu", weights_only=False)
    last_resumed = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for branch in ("train_state", "ema_state", "best_state"):
        for key, value in last_full[branch].items():
            torch.testing.assert_close(value, last_resumed[branch][key], rtol=0, atol=0)
    assert last_full["scheduler"] == last_resumed["scheduler"]
    assert last_full["lr_reductions"] == last_resumed["lr_reductions"]
    assert last_full["step"] == last_resumed["step"]
    assert last_full["train_state"]["fusion_head.3.weight"].abs().sum() > 0
    inference = tmp_path / "full" / "best_multilabel.pt"
    original_hash = trainer.file_sha256(inference)
    monkeypatch.setattr(trainer, "fit_thresholds", lambda *a, **k: pytest.fail("test threshold tuning"))
    args = trainer.parser().parse_args([
        "evaluate-test", "--checkpoint", str(inference), "--test-shards", str(paths["testing"]),
        "--annotations", str(annotations), "--out", str(tmp_path / "test"),
        "--workers", "0", "--batch-size", "4", "--device", "cpu",
    ])
    report = trainer.evaluate_test(args)
    assert report["thresholds_refit"] is False
    assert report["checkpoint_selection_on_test"] is False
    assert report["split"]["subjects"] == ["p02"]
    assert report["source_provenance"]["visual_context"] == metadata
    assert trainer.file_sha256(inference) == original_hash
    with np.load(tmp_path / "test" / "test_logits.npz") as result:
        assert result["logits"].shape == result["targets"].shape == (4, 22)
        assert np.isfinite(result["logits"]).all()
    # The same tensor dimension cannot hide a changed frozen detector at evaluation.
    with np.load(paths["testing"]) as source:
        changed = {name: source[name] for name in source.files}
    tampered = dict(metadata, model_sha256="1" * 64)
    changed["context_metadata"] = np.asarray(json.dumps(tampered))
    np.savez_compressed(paths["testing"], **changed)
    with pytest.raises(ValueError, match="context"):
        trainer.evaluate_test(args)


def test_required_convergence_never_exports_or_calibrates_an_unfinished_candidate(tmp_path, monkeypatch):
    paths, annotations, _ = context_training_fixture(tmp_path)
    monkeypatch.setattr(trainer, "fit_thresholds", lambda *a, **k: pytest.fail("premature threshold fit"))
    result = trainer.train(context_train_args(paths, annotations, tmp_path / "gated", "--require-convergence"))
    assert result["status"] == "epoch_budget_exhausted_without_convergence"
    assert result["convergence"]["converged"] is False
    assert result["inference_checkpoint_exported"] is False
    assert (tmp_path / "gated" / "last_multilabel.pt").is_file()
    assert not (tmp_path / "gated" / "best_multilabel.pt").exists()
    assert not (tmp_path / "gated" / "thresholds.json").exists()
    saved = json.loads((tmp_path / "gated" / "training_status.json").read_text())
    assert saved["inference_checkpoint_exported"] is False


if __name__ == "__main__":
    class RunnerSummary:
        def pytest_terminal_summary(self, terminalreporter):
            stats = terminalreporter.stats
            passed = len(stats.get("passed", []))
            total = passed + len(stats.get("failed", [])) + len(stats.get("error", []))
            print(f"{passed}/{total} passed; {len(stats.get('skipped', []))} skipped")

    sys.exit(pytest.main([__file__, "-q"], plugins=[RunnerSummary()]))
