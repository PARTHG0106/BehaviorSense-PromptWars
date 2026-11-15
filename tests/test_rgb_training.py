"""Real audited toy shards/cache alignment and finite head-training plumbing."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import train_rgb_multilabel as trainer
import behaviorsense.data.multilabel_skeleton as multilabel_data
from behaviorsense.data.toyota import COARSE_V11, TSU_CLASSES, TSU_TO_COARSE, load_tsu_annotations
from behaviorsense.models import rgb_action
from behaviorsense.models.visual_context import (CONTEXT_DIM, CONTEXT_SCHEMA, FEATURE_NAMES,
    VALIDATED_MODEL_SHA256, PREPROCESSING, OUTPUT_CONTRACT)
from extract_multilabel_windows import select_windows
from extract_rgb_action_features import _array_sha256, source_frame_indices


def criteria():
    return {"macro_ap_gain_over_matched_control": .03, "macro_ap_min": .6379390777,
        "pair_f1_min": .30, "pair_precision_min": .30, "pair_recall_min": .30,
        "subjects": {"p04": {"pair_f1_min": .21808510638, "macro_ap_min": .59126160643},
                     "p15": {"pair_f1_min": .03861003861, "macro_ap_min": .64502818523}}}


def permissive_test_criteria():
    result = criteria()
    for key in result:
        if key != "subjects":
            result[key] = 0.
    for subject in result["subjects"].values():
        subject.update(pair_f1_min=0., macro_ap_min=0.)
    return result


def write_cache(path, shard, payload, report):
    dim, n = 768, len(payload["starts"])
    info = {"arch": "swin3d_t", "weights_sha256": "b"*64, "feature_dim": dim,
            "preprocessing": rgb_action.PREPROCESSING,
            "preprocessing_sha256": rgb_action.PREPROCESSING_SHA256, "device": "fixture"}
    frames, boundary = source_frame_indices(payload["source_frames"][:, 15], 25., report["source_frame_count"])
    rois, observed = zip(*(rgb_action.person_roi(pose, 160, 120) for pose in payload["skeletons"]))
    arrays = {"starts": payload["starts"].astype(np.int64), "video_id": np.asarray(report["video_id"]),
        "subject": np.asarray(report["subject"]), "video_ids": np.full(n, report["video_id"]),
        "subjects": np.full(n, report["subject"]), "source_frames_rgb": frames,
        "center_source_frames": payload["source_frames"][:, 15].astype(np.int64),
        "boundary_duplicate_flags": boundary, "person_rois": np.asarray(rois, np.int64),
        "roi_observed": np.asarray(observed, bool), "source_fps": np.asarray(25., np.float64),
        "features": np.random.default_rng(int(report["video_id"][1:3])).normal(size=(n, 2, dim)).astype(np.float16)}
    identity = {"schema": trainer.SCHEMA, "video_id": report["video_id"], "subject": report["subject"],
        "n_windows": n, "arch": "swin3d_t", "feature_dim": dim, "source_video_sha256": report["source_sha256"],
        "source_shard_sha256": trainer._hash(shard), "encoder_sha256": info["weights_sha256"],
        "encoder_info": info, "preprocessing_sha256": rgb_action.PREPROCESSING_SHA256,
        "extraction_implementation_sha256": "c"*64, "source_fps": 25.,
        "source_frame_count": report["source_frame_count"], "source_width": 160, "source_height": 120,
        "views": ["scene", "person"]}
    metadata = dict(identity, status="complete", fingerprint=trainer.canonical_sha256(identity),
        array_sha256={key: _array_sha256(value) for key, value in arrays.items()})
    np.savez_compressed(path, **arrays, metadata=np.asarray(json.dumps(metadata)))


@pytest.fixture
def corpus(tmp_path):
    annotations = {}
    representatives = {}
    for fine, coarse in TSU_TO_COARSE.items():
        representatives.setdefault(coarse, TSU_CLASSES.index(fine))
    classes = [name for name in COARSE_V11 if name in representatives]
    duration = 150 * len(classes)
    actions = [[representatives[name], index*150, (index+1)*150-1] for index, name in enumerate(classes)]
    drink_span = next((start, end) for fine, start, end in actions if TSU_CLASSES[fine].startswith("Drink."))
    actions.append([TSU_CLASSES.index("Watch_TV"), *drink_span])
    for subject in (2, 3, 4, 6, 15):
        annotations[f"P{subject:02d}T01C01"] = {"subset": "testing" if subject == 2 else "training",
                                                "duration": duration, "actions": actions}
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps(annotations))
    videos = load_tsu_annotations(annotation_path)
    features = tmp_path / "features"
    features.mkdir()
    paths, counts = {}, {}
    context_meta = {"schema": CONTEXT_SCHEMA, "feature_dim": CONTEXT_DIM, "feature_names": list(FEATURE_NAMES),
        "model_sha256": VALIDATED_MODEL_SHA256, "score_threshold": .25, "keypoint_threshold": .3,
        "preprocessing": PREPROCESSING, "output_contract": OUTPUT_CONTRACT,
        "geometry_missing_value": -1, "geometry_units": "image_diagonal", "sampling": {"sample_offsets": [15]}}
    for vid, video in videos.items():
        selection = select_windows(video, duration, 25., selection="full")
        starts, frames, n = selection.starts, selection.source_frames, len(selection.starts)
        pose = np.zeros((n, 30, 2, 17, 3), dtype=np.float16)
        pose[:, :, 0, :, 0] = np.linspace(40, 80, 17)
        pose[:, :, 0, :, 1] = np.linspace(20, 100, 17)
        pose[:, :, 0, :, 2] = .9
        pose[:, :, 0, :, 0] += (np.arange(n) % 5)[:, None, None]
        context = np.zeros((n, CONTEXT_DIM), np.float32)
        context[:, 0] = 1
        identity = {"config_fingerprint": "a"*64, "video_id": vid, "subset": video.subset,
                    "source_sha256": hashlib.sha256(vid.encode()).hexdigest()}
        fingerprint = trainer.baseline.digest(identity)
        report = dict(identity, fingerprint=fingerprint, schema_version=2, subject=f"p{video.subject:02d}",
            selection_mode="full", window_cap=None, full_dense_coverage=True, candidate_stride=15,
            source_fps=25., source_frame_count=duration, candidate_windows=selection.candidate_windows,
            selected_windows=n, excluded_annotation_coverage=selection.candidate_windows-n,
            selected_starts=starts.tolist(), retained_starts=starts.tolist(), retained_windows=n, dropped_visibility=0)
        payload = {"skeletons": pose, "starts": starts, "source_frames": frames,
            "source_fps": np.full(n, 25., np.float64), "subjects": np.full(n, report["subject"]),
            "clips": np.full(n, vid+".mp4"), "datasets": np.full(n, "toyota_untrimmed"),
            "fingerprint": np.asarray(fingerprint), "extraction_metadata": np.asarray(json.dumps(report)),
            "visual_context": context, "context_source_frames": frames[:, [15]],
            "context_feature_names": np.asarray(FEATURE_NAMES), "context_metadata": np.asarray(json.dumps(context_meta))}
        path = tmp_path / (vid+".npz")
        np.savez_compressed(path, **payload)
        write_cache(features/(vid+".npz"), path, payload, report)
        paths[vid], counts[vid] = path, n
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps(["P03T01C01"]))
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps(permissive_test_criteria()))
    return dict(root=tmp_path, annotations=annotation_path, features=features, paths=paths, counts=counts,
                pilot=pilot, gate=gate, videos=videos)


def args_for(corpus, scope="pilot", out="fit", **options):
    paths = [str(path) for vid, path in corpus["paths"].items() if corpus["videos"][vid].subset == "training"]
    args = trainer.parser().parse_args(["train", "--features-dir", str(corpus["features"]), "--shards", *paths,
        "--annotations", str(corpus["annotations"]), "--out", str(corpus["root"]/out), "--scope", scope,
        "--pilot-video-ids", str(corpus["pilot"]), "--gate-config", str(corpus["gate"]), "--device", "cpu",
        "--epochs", "3", "--min-epochs", "1", "--patience", "1", "--lr-patience", "1",
        "--hidden", "8", "--batch-size", "256", "--chunk-size", "64", "--cpu-threads", "1"])
    for key, value in options.items():
        setattr(args, key, value)
    return args


def patch_official_toy_inventory(monkeypatch, corpus):
    for subset in ("training", "testing"):
        vids = sorted(vid for vid, video in corpus["videos"].items() if video.subset == subset)
        monkeypatch.setitem(multilabel_data.OFFICIAL_VIDEO_COUNTS, subset, len(vids))
        monkeypatch.setitem(multilabel_data.OFFICIAL_INVENTORY_SHA256, subset, hashlib.sha256("\n".join(vids).encode()).hexdigest())
    monkeypatch.setattr(trainer, "FULL_COUNTS", {"training_videos": 4,
        "fit_windows": corpus["counts"]["P03T01C01"] + corpus["counts"]["P06T01C01"],
        "validation_windows": corpus["counts"]["P04T01C01"] + corpus["counts"]["P15T01C01"],
        "test_videos": 1, "test_windows": corpus["counts"]["P02T01C01"]})


def rewrite_cache(path, mutate):
    with np.load(path) as source:
        values = {key: source[key] for key in source.files}
    meta = json.loads(str(values.pop("metadata")))
    mutate(values, meta)
    meta["fingerprint"] = trainer.canonical_sha256({key: meta[key] for key in trainer.IDENTITY_KEYS})
    meta["array_sha256"] = {key: _array_sha256(value) for key, value in values.items()}
    np.savez_compressed(path, **values, metadata=np.asarray(json.dumps(meta)))


def test_real_pilot_loading_preserves_all_selected_rows_and_validation_subjects(corpus):
    data = trainer.load_data(args_for(corpus))
    assert set(data["selected_video_ids"]) == {"P03T01C01", "P04T01C01", "P15T01C01"}
    assert len(data["starts"]) == sum(corpus["counts"][vid] for vid in data["selected_video_ids"])
    assert isinstance(data["features"], np.memmap)
    assert data["features"].shape[1] == trainer.CONTROL_DIM + 1536
    assert np.isfinite(data["features"]).all()
    train, val = trainer.baseline.subject_split(data["subjects"], val_subjects=trainer.VAL_SUBJECTS)
    trainer.baseline.require_class_support({"train": trainer.baseline.split_report(data, train),
                                           "validation": trainer.baseline.split_report(data, val)}, 5)
    assert not (data["subjects"] == "p02").any()


def test_missing_validation_video_and_test_input_are_rejected_before_loading(corpus):
    args = args_for(corpus)
    args.shards = [path for path in args.shards if "P15T01C01" not in path]
    with pytest.raises(ValueError, match="selected source coverage is missing"):
        trainer.select_shards(args, corpus["videos"])
    args = args_for(corpus)
    args.shards.append(str(corpus["paths"]["P02T01C01"]))
    with pytest.raises(ValueError, match="forbidden subject/subset"):
        trainer.select_shards(args, corpus["videos"])


def test_full_training_refuses_an_incomplete_fitting_inventory(corpus):
    args = args_for(corpus, scope="full")
    args.shards = [path for path in args.shards if "P06T01C01" not in path]
    with pytest.raises(ValueError, match="exactly every official source video"):
        trainer.select_shards(args, corpus["videos"])


@pytest.mark.parametrize("field", ["source_video_sha256", "source_shard_sha256", "preprocessing_sha256"])
def test_wrong_pixel_shard_or_preprocessing_contract_is_rejected(corpus, field):
    path = corpus["features"]/"P03T01C01.npz"
    rewrite_cache(path, lambda values, meta: meta.update({field: "f"*64}))
    with pytest.raises(ValueError):
        trainer.validate_rgb_cache(path, corpus["paths"]["P03T01C01"])


def test_row_reordering_and_uniform_encoder_mismatch_are_rejected(corpus):
    path = corpus["features"]/"P03T01C01.npz"
    rewrite_cache(path, lambda values, meta: values.update(starts=values["starts"][::-1]))
    with pytest.raises(ValueError, match="order/clock/crop"):
        trainer.validate_rgb_cache(path, corpus["paths"]["P03T01C01"])
    # Restore row order, then forge a different internally consistent encoder ID.
    def change(values, meta):
        values["starts"] = values["starts"][::-1]
        meta["encoder_sha256"] = "d"*64
        meta["encoder_info"]["weights_sha256"] = "d"*64
    rewrite_cache(path, change)
    with pytest.raises(ValueError, match="changed between videos"):
        trainer.load_data(args_for(corpus))


def test_normalization_uses_only_fitting_rows_and_clips_heldout_values():
    values = np.array([[0., 5.], [2., 5.], [100000., -100000.]], np.float32)
    mean, std = trainer.fit_normalization(values, np.array([0, 1]), chunk_size=1)
    np.testing.assert_allclose(mean, [1., 5.])
    np.testing.assert_allclose(std, [1., 1e-4])
    standardized = trainer.standardized_tensor(values, mean, std, "cpu", chunk_size=1).numpy()
    np.testing.assert_array_equal(standardized[2], [8., -8.])
    np.testing.assert_allclose(standardized[:2, 0], [-1., 1.])


def summary(ap=.70, pair=.40):
    metric = {"macro_ap": ap, "tv_drinking_colabeled_windows": {"f1": pair, "precision": pair, "recall": pair}}
    return {"metrics": metric, "per_subject": {subject: copy.deepcopy(metric) for subject in trainer.VAL_SUBJECTS}}


def test_gate_requires_control_gain_and_both_subjects_without_test_scores():
    gate = trainer.gate_decision(criteria(), summary(), summary(.66), "pilot")
    assert gate["passed"]
    assert not trainer.gate_decision(criteria(), summary(), summary(.69), "pilot")["passed"]
    assert trainer.gate_decision(criteria(), summary(), None, "full")["passed"]
    for subject in trainer.VAL_SUBJECTS:
        for key in ("macro_ap", "pair"):
            value = summary()
            if key == "macro_ap": value["per_subject"][subject]["macro_ap"] = .1
            else: value["per_subject"][subject]["tv_drinking_colabeled_windows"]["f1"] = 0.
            assert not trainer.gate_decision(criteria(), value, summary(.6), "pilot")["passed"]


def test_zero_deadline_does_not_load_features_or_export_final(corpus, monkeypatch):
    monkeypatch.setattr(trainer, "load_data", lambda *a, **k: pytest.fail("deadline expired before data load"))
    args = args_for(corpus, stop_after_seconds=0.)
    result = trainer.train(args)
    assert not result["passed"] and result["reason"] == "incomplete_budget"
    assert not (Path(args.out)/"best_rgb_multilabel.pt").exists()
    assert trainer.read(Path(args.out)/"frozen_selection.json")["gate_passed"] is False


def cached_first_predictions(monkeypatch):
    original, cache = trainer.predict, {}
    def held(model, *args, **kwargs):
        if model not in cache:
            cache[model] = original(model, *args, **kwargs)
        return cache[model].copy()
    monkeypatch.setattr(trainer, "predict", held)


def test_pilot_fits_exactly_matched_control_and_rgb_heads(corpus, monkeypatch):
    args = args_for(corpus)
    seen = []
    original = trainer._train_head
    def observe(*values, **keywords):
        seen.append((values[6]["head"], values[2].shape[1], values[6]["seed"]))
        return original(*values, **keywords)
    monkeypatch.setattr(trainer, "_train_head", observe)
    cached_first_predictions(monkeypatch)
    result = trainer.train(args)
    assert [row[0] for row in seen] == ["control", "rgb"]
    assert seen[0][1] == trainer.CONTROL_DIM and seen[1][1] == trainer.CONTROL_DIM + 1536
    assert [row[2] for row in seen] == [0, 0]
    assert result["gate"]["metrics"]["control"] is not None
    assert (Path(args.out)/"control/best_rgb_multilabel.pt").exists()


def test_convergence_reports_lr_reductions_without_delaying_plateau(corpus, monkeypatch):
    args = args_for(corpus, epochs=8, patience=4)
    cached_first_predictions(monkeypatch)
    trainer.train(args)
    for directory in (Path(args.out), Path(args.out)/"control"):
        history = trainer.read(directory/"history.json")
        np.testing.assert_allclose([epoch["learning_rate"] for epoch in history],
                                   [.001, .001, .0005, .0005, .00025])
        manifest = trainer.read(directory/"manifest.json")
        convergence = manifest["convergence"]
        assert convergence["converged"] and convergence["completed_epochs"] == 5
        assert convergence["lr_reductions"] == 2
        assert convergence["last_lr_reduction_epoch_zero_based"] == 4
        assert convergence["required_lr_reductions"] == convergence["lr_settle_epochs"] == 0
        assert manifest["checkpoint_selection"]["best_epoch_zero_based"] == 0
        assert manifest["checkpoint_selection"]["best_score"] == history[0]["validation_macro_ap"]
        assert trainer.read(directory/"training_status.json")["convergence"] == convergence


def test_full_train_export_and_frozen_real_toy_test_without_refitting(corpus, monkeypatch):
    patch_official_toy_inventory(monkeypatch, corpus)
    args = args_for(corpus, scope="full")
    # Freeze each epoch's validation predictions to the true first-epoch output.
    # This exercises plateau completion without inventing metrics or claiming accuracy.
    with monkeypatch.context() as plateau:
        cached_first_predictions(plateau)
        result = trainer.train(args)
    assert result["gate_passed"]
    checkpoint = Path(args.out)/"best_rgb_multilabel.pt"
    original_hash = trainer._hash(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["format"] == trainer.FORMAT and saved["manifest"]["convergence"]["converged"]
    assert saved["normalization"]["source_split"] == "fit"
    # Reload exactly the exported model and normalization, independently of the
    # in-memory model. Its selected validation logits must be reproducible.
    data = trainer.load_data(args_for(corpus, scope="full", out="replay"))
    _, validation = trainer.baseline.subject_split(data["subjects"], val_subjects=trainer.VAL_SUBJECTS)
    model = trainer.RGBHead(saved["config"]["input_dim"], saved["config"]["hidden"], saved["config"]["dropout"])
    model.load_state_dict(saved["state_dict"], strict=True)
    features = trainer.standardized_tensor(data["features"], saved["normalization"]["mean"],
                                           saved["normalization"]["std"], "cpu")
    logits = trainer.predict(model, features, validation, args.batch_size)
    with np.load(Path(args.out)/"validation_logits.npz", allow_pickle=False) as predictions:
        np.testing.assert_array_equal(logits, predictions["logits"])
    assert trainer.temporal.assess(data, validation, logits, saved["threshold_config"]["thresholds"]) == saved["validation_metrics"]
    history = trainer.read(Path(args.out)/"history.json")
    assert all(epoch["fit_windows_seen"] == len(data["starts"]) - len(validation) for epoch in history)
    monkeypatch.setattr(trainer.baseline, "fit_thresholds", lambda *a, **k: pytest.fail("test threshold refit"))
    test_args = trainer.parser().parse_args(["evaluate-test", "--features-dir", str(corpus["features"]),
        "--shards", str(corpus["paths"]["P02T01C01"]), "--annotations", str(corpus["annotations"]),
        "--checkpoint", str(checkpoint), "--out", str(corpus["root"]/"test"), "--device", "cpu", "--cpu-threads", "1"])
    report = trainer.evaluate_test(test_args)
    assert report["thresholds_refit"] is False and report["checkpoint_selection_on_test"] is False
    assert report["split"]["subjects"] == ["p02"]
    assert trainer._hash(checkpoint) == original_hash
    assert report["feature_contract"] == saved["manifest"]["feature_contract"]
    with pytest.raises(ValueError, match="overwrite"):
        trainer.evaluate_test(test_args)

    # A broken export or edited selection must fail before opening any test data.
    monkeypatch.setattr(trainer, "load_data", lambda *a, **k: pytest.fail("invalid selection loaded test labels"))
    for mutate in (
        lambda value: value.update(gate_passed=False),
        lambda value: value.update(checkpoint_sha256="f"*64),
        lambda value: value.update(thresholds_sha256="f"*64),
        lambda value: value["gate"].update(passed=False),
    ):
        changed = copy.deepcopy(result)
        mutate(changed)
        trainer.baseline.write_json(Path(args.out)/"frozen_selection.json", changed)
        test_args.out = str(corpus["root"]/"refused_test")
        with pytest.raises(ValueError, match="frozen selection"):
            trainer.evaluate_test(test_args)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.])
def test_invalid_test_deadlines_are_refused_before_reading_checkpoint(tmp_path, monkeypatch, value):
    args = trainer.parser().parse_args(["evaluate-test", "--features-dir", "missing", "--shards", "missing",
        "--annotations", "missing", "--checkpoint", "missing", "--out", str(tmp_path/"test")])
    args.stop_after_seconds = value
    monkeypatch.setattr(trainer, "_hash", lambda *a, **k: pytest.fail("invalid deadline reached input reads"))
    with pytest.raises(ValueError, match="finite RGB runtime"):
        trainer.evaluate_test(args)


@pytest.mark.parametrize("mean,std", [([0.], [0.]), ([0.], [-1.]), ([float("inf")], [1.]),
                                    ([0.], [float("inf")]), ([0., 1.], [1., 1.])])
def test_invalid_normalization_cannot_be_hidden_by_clipping(mean, std):
    with pytest.raises(ValueError, match="normalization requires"):
        trainer.standardized_tensor(np.array([[1.]], np.float32), mean, std, "cpu")


def test_nonfinite_input_features_cannot_be_hidden_by_clipping():
    with pytest.raises(ValueError, match="non-finite input"):
        trainer.standardized_tensor(np.array([[float("inf")]], np.float32), np.array([0.]), np.array([1.]), "cpu")


def test_bad_gate_configuration_is_rejected_before_data_loading(corpus, monkeypatch):
    gate = permissive_test_criteria()
    # Even the relative gain unused by the full gate is a frozen finite criterion.
    gate["macro_ap_gain_over_matched_control"] = float("nan")
    corpus["gate"].write_text(json.dumps(gate))
    monkeypatch.setattr(trainer, "load_data", lambda *a, **k: pytest.fail("invalid gate reached training data"))
    with pytest.raises(ValueError, match="gate thresholds"):
        trainer.train(args_for(corpus, scope="full"))


def test_epoch_limit_without_plateau_exports_no_rgb_checkpoint(corpus, monkeypatch):
    args = args_for(corpus, epochs=1, min_epochs=1)
    result = trainer.train(args)
    assert result["passed"] is False and result["reason"] == "incomplete_epoch_limit"
    assert not list(Path(args.out).rglob("best_rgb_multilabel.pt"))
    assert trainer.read(Path(args.out)/"control/training_status.json")["convergence"]["converged"] is False


def test_budget_stop_after_optimizer_step_does_not_count_unfinished_epoch(corpus, monkeypatch):
    args = args_for(corpus)
    expired = False
    original = torch.optim.AdamW.step
    def expire_after_step(optimizer, *values, **keywords):
        nonlocal expired
        result = original(optimizer, *values, **keywords)
        expired = True
        return result
    def deadline(_):
        if expired:
            raise TimeoutError("fixture expires during first epoch")
    monkeypatch.setattr(torch.optim.AdamW, "step", expire_after_step)
    monkeypatch.setattr(trainer, "deadline_check", deadline)
    result = trainer.train(args)
    assert result["passed"] is False and result["reason"] == "incomplete_budget"
    status = trainer.read(Path(args.out)/"control/training_status.json")
    assert status["completed_epochs"] == 0 and status["interrupted_epoch_counted"] is False
    assert not list(Path(args.out).rglob("best_rgb_multilabel.pt"))


def test_head_learns_a_separable_masked_problem_with_real_optimizer_steps(tmp_path):
    # Synthetic separability verifies the learning path, not real RGB accuracy.
    sign = np.tile(np.array([-1., 1.], np.float32), 64)
    features = torch.from_numpy(np.stack([sign, -sign], axis=1))
    targets = np.zeros((len(sign), len(COARSE_V11)), np.float32)
    mask = np.zeros_like(targets, dtype=bool)
    targets[:, list(trainer.baseline.PAIR)] = (sign > 0)[:, None]
    mask[:, list(trainer.baseline.PAIR)] = True
    subjects = np.array(["p03"]*96 + ["p04"]*16 + ["p15"]*16)
    data = {"targets": targets, "target_mask": mask, "subjects": subjects,
            "video_ids": subjects.copy(), "starts": np.arange(len(sign))}
    train, val = np.arange(96), np.arange(96, 128)
    args = trainer.parser().parse_args(["train", "--scope", "full", "--gate-config", "unused",
        "--features-dir", "unused", "--shards", "unused", "--annotations", "unused", "--out", str(tmp_path),
        "--device", "cpu", "--hidden", "8", "--dropout", "0", "--lr", ".05", "--weight-decay", "0",
        "--batch-size", "24", "--epochs", "20", "--min-epochs", "6", "--patience", "3"])
    trainer.baseline.set_seed(0)
    initial = trainer.RGBHead(2, args.hidden, args.dropout)
    initial_state = {name: value.clone() for name, value in initial.state_dict().items()}
    manifest = {"head": "rgb", "train": trainer.baseline.split_report(data, train),
                "validation_fingerprint": trainer.temporal.validation_fingerprint(data, val)}
    artifact = trainer._train_head(args, data, features, train, val,
        (np.zeros(2, np.float32), np.ones(2, np.float32)), manifest, tmp_path/"learned", None)
    assert artifact is not None and artifact["manifest"]["convergence"]["converged"]
    assert any(not torch.equal(initial_state[name], value) for name, value in artifact["state_dict"].items())
    history = trainer.read(tmp_path/"learned/history.json")
    assert history[-1]["train_loss"] < history[0]["train_loss"] * .5
    assert all(epoch["fit_windows_seen"] == len(train) for epoch in history)
    assert artifact["validation_metrics"]["macro_ap"] == 1.


if __name__ == "__main__":
    class RunnerSummary:
        def pytest_terminal_summary(self, terminalreporter):
            stats = terminalreporter.stats
            passed = len(stats.get("passed", []))
            total = passed + len(stats.get("failed", [])) + len(stats.get("error", []))
            print(f"{passed}/{total} passed; {len(stats.get('skipped', []))} skipped")

    raise SystemExit(pytest.main([__file__, "-q"], plugins=[RunnerSummary()]))
