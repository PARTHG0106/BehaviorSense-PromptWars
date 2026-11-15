"""Exercise temporal fitting/export with real tiny source-clock/context shards."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT / "tests")]
import train_multilabel_temporal as trainer
from test_train_adl_multilabel import context_training_fixture, synthetic_data


def temporal_data():
    data = synthetic_data()
    data["starts"] = np.tile([0, 15, 45, 60], 4)
    context = np.full((len(data["starts"]), 401), -1, dtype=np.float32)
    context[:, 0] = 1
    features = context[:, 1:].reshape(-1, 80, 5)
    features[:, :, 0] = 0
    features[:, 41] = np.array([.8, .2, -.1, .2, .1])
    features[:, 62] = np.array([.9, -.3, .2, .5, .7])
    data["visual_context"] = context
    return data


def arguments(tmp_path, paths, annotations, loss="bce", **overrides):
    flags = ["train", "--train-shards", str(paths["training"]),
        "--annotations", str(annotations), "--out", str(tmp_path / "candidate"),
        "--loss", loss, "--epochs", "3", "--min-epochs", "1", "--patience", "1",
        "--warmup-epochs", "0", "--lr-patience", "1", "--convergence-lr-reductions", "0",
        "--base-channels", "4", "--context-hidden", "8", "--context-radius", "1",
        "--batch-size", "4", "--workers", "0", "--cpu-threads", "1", "--device", "cpu",
        "--threshold-min-support", "1", "--val-subjects", "p04"]
    args = trainer.parser().parse_args(flags)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_context_keeps_gaps_and_center_labels_without_reading_neighbor_labels():
    data = temporal_data()
    dataset = trainer.TemporalWindows(data, [1], context_radius=1, neighbor_dropout=0)
    item = dataset[0]
    np.testing.assert_array_equal(item[1].numpy(), data["targets"][1])
    np.testing.assert_array_equal(item[2].numpy(), data["target_mask"][1])
    np.testing.assert_array_equal(item[3].numpy()[0], data["visual_context"][0])
    np.testing.assert_array_equal(item[3].numpy()[1], data["visual_context"][1])
    assert not item[3].numpy()[2].any()  # Start 30 is absent, not the next row at 45.
    data["targets"][0] = 1 - data["targets"][0]
    data["target_mask"][0] = False
    again = dataset[0]
    torch.testing.assert_close(again[1], item[1])
    torch.testing.assert_close(again[2], item[2])
    torch.testing.assert_close(again[3], item[3])


def test_temporal_and_center_augmentations_use_the_same_affine_rng():
    data = temporal_data()
    dataset = trainer.TemporalWindows(data, [1], training=True, seed=17,
                                      context_radius=1, neighbor_dropout=0)
    original = trainer.baseline.MultilabelWindows(data, [1], training=True, seed=17)
    dataset.epoch = original.epoch = 3
    actual, center = dataset[0], original[0]
    torch.testing.assert_close(actual[0], center[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[3][1], center[3], rtol=0, atol=0)
    gathered = np.zeros((3, 401), np.float32)
    gathered[:2] = data["visual_context"][:2]
    expected = trainer.baseline.augment_context(gathered, dataset.cfg,
                                                np.random.default_rng((17, 3, 1)))
    np.testing.assert_array_equal(actual[3].numpy(), expected)


def test_neighbor_dropout_never_drops_center_and_does_not_change_targets():
    data = temporal_data()
    data["starts"][:4] = [0, 15, 30, 45]
    dataset = trainer.TemporalWindows(data, [1], training=True, seed=12,
                                      context_radius=1, neighbor_dropout=.999999)
    item = dataset[0]
    assert item[3][1, 0] == 1
    assert not item[3][[0, 2]].any()
    np.testing.assert_array_equal(item[1].numpy(), data["targets"][1])


def test_forward_batch_passes_joints_without_a_second_bone_conversion():
    joints = torch.randn(2, 3, 30, 17, 2)
    context = torch.randn(2, 3, 401)
    received = []

    def model(x, c):
        received.extend((x, c))
        return torch.zeros(2, 22)

    trainer.forward_batch(model, (joints, torch.zeros(2, 22), torch.ones(2, 22), context), "cpu")
    torch.testing.assert_close(received[0], joints, rtol=0, atol=0)
    torch.testing.assert_close(received[1], context, rtol=0, atol=0)


def test_validation_identity_is_order_independent_and_detects_target_changes():
    data = temporal_data()
    rows = np.arange(len(data["starts"]))
    before = trainer.validation_fingerprint(data, rows)
    assert before == trainer.validation_fingerprint(data, rows[::-1])
    data["targets"][0, 10] = 1 - data["targets"][0, 10]
    assert before != trainer.validation_fingerprint(data, rows)


def test_pair_ap_uses_known_colabels_and_records_its_ranking_definition():
    data = temporal_data()
    indices = np.arange(4)
    data["targets"][indices, 10] = [0, 1, 1, 0]
    data["targets"][indices, 13] = [0, 1, 1, 0]
    scores = np.full((4, 22), .1, np.float32)
    scores[:, [10, 13]] = np.array([.1, .9, .7, .8])[:, None]
    logits = np.log(scores / (1 - scores))
    result = trainer.assess(data, indices, logits)
    pair = result["tv_drinking_colabeled_windows"]
    assert pair["average_precision"] == pytest.approx((1 + 2 / 3) / 2)
    assert pair["ranking_score"] == "minimum_of_two_marginal_sigmoid_scores"


@pytest.mark.parametrize("loss", ["bce", "asl"])
def test_real_shards_train_export_and_frozen_test_evaluation(tmp_path, monkeypatch, loss):
    paths, annotations, _ = context_training_fixture(tmp_path)
    args = arguments(tmp_path, paths, annotations, loss)
    actual_predict = trainer.predict
    first_predictions = []

    def stable_first_epoch(model, loader, device, deadline=None):
        # Execute real inference, then hold the first epoch's actual predictions
        # fixed to exercise early stopping deterministically without inventing scores.
        predictions = actual_predict(model, loader, device, deadline)
        if not first_predictions:
            first_predictions.append(predictions.copy())
        return first_predictions[0].copy()

    with monkeypatch.context() as scoped:
        scoped.setattr(trainer, "predict", stable_first_epoch)
        artifact = trainer.train(args)
    folder = Path(args.out)
    state = json.loads((folder / "training_status.json").read_text())
    assert state["status"] == "complete" and state["inference_checkpoint_exported"]
    assert state["completed_epochs"] == 2
    assert not state["automatic_continuation"] and not state["resume_supported"]
    assert artifact["manifest"]["initialization"]["mode"] == "scratch"
    assert artifact["manifest"]["test_loaded_during_training"] is False
    assert artifact["manifest"]["temporal_context"]["lookahead_seconds"] == .75
    if loss == "asl":
        assert artifact["manifest"]["pos_weight"] is None
        assert artifact["manifest"]["loss_configuration"]["gamma_pos"] == 1
    history = json.loads((folder / "history.json").read_text())
    assert all(row["fit_windows_seen"] == 12 for row in history)
    saved = torch.load(folder / "best_multilabel.pt", map_location="cpu", weights_only=False)
    assert saved["format"] == trainer.TEMPORAL_FORMAT
    assert saved["threshold_config"]["source_split"] == "validation"
    assert set(json.loads((folder / "validation_by_subject.json").read_text())) == {"p04"}
    reload_args = copy.copy(args)
    reload_args.cache_dir = str(tmp_path / "reload_cache")
    data = trainer.load_data([paths["training"]], reload_args)
    val = np.flatnonzero(data["subjects"] == "p04")
    model = trainer.build_model(saved["args"])
    model.load_state_dict(saved["state_dict"], strict=True)
    loader = torch.utils.data.DataLoader(trainer.TemporalWindows(data, val, context_radius=1), batch_size=4)
    reloaded = trainer.predict(model, loader, "cpu")
    np.testing.assert_allclose(reloaded, first_predictions[0], rtol=0, atol=1e-6)

    evaluation = trainer.parser().parse_args(["evaluate-test", "--checkpoint", str(folder / "best_multilabel.pt"),
        "--test-shards", str(paths["testing"]), "--annotations", str(annotations),
        "--out", str(tmp_path / "test"), "--device", "cpu", "--workers", "0", "--cpu-threads", "1"])
    with monkeypatch.context() as scoped:
        def forbid_refit(*args, **kwargs):
            raise AssertionError("test evaluation must never refit thresholds")
        scoped.setattr(trainer.baseline, "fit_thresholds", forbid_refit)
        report = trainer.evaluate_test(evaluation)
    assert report["split"]["subjects"] == ["p02"]
    assert set(report["per_subject"]) == {"p02"}
    assert report["split"]["n_windows"] == 4
    assert report["thresholds_refit"] is False and report["checkpoint_selection_on_test"] is False
    assert report["exploratory_test_benchmark"] is True
    assert report["checkpoint_sha256"] == trainer.baseline.file_sha256(folder / "best_multilabel.pt")
    with pytest.raises(ValueError, match="never resumes"):
        trainer.train(args)


def test_budget_stop_never_exports_or_counts_an_unfinished_epoch(tmp_path):
    paths, annotations, _ = context_training_fixture(tmp_path)
    args = arguments(tmp_path, paths, annotations, stop_after_seconds=.000001)
    state = trainer.train(args)
    assert state["status"] == "incomplete_budget"
    assert state["completed_epochs"] == 0
    assert state["interrupted_epoch_counted"] is False
    assert not (Path(args.out) / "best_multilabel.pt").exists()


def test_epoch_limit_without_convergence_is_explicit(tmp_path):
    paths, annotations, _ = context_training_fixture(tmp_path)
    args = arguments(tmp_path, paths, annotations, epochs=1)
    state = trainer.train(args)
    assert state["status"] == "incomplete_epoch_limit"
    assert state["completed_epochs"] == 1 and not state["inference_checkpoint_exported"]
    assert (Path(args.out) / "last_multilabel.pt").is_file()
    assert not (Path(args.out) / "best_multilabel.pt").exists()


if __name__ == "__main__":
    class RunnerSummary:
        def pytest_terminal_summary(self, terminalreporter):
            stats = terminalreporter.stats
            passed = len(stats.get("passed", []))
            total = passed + len(stats.get("failed", [])) + len(stats.get("error", []))
            print(f"{passed}/{total} passed; {len(stats.get('skipped', []))} skipped")

    raise SystemExit(pytest.main([__file__, "-q"], plugins=[RunnerSummary()]))
