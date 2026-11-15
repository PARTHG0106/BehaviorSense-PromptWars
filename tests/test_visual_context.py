"""CPU checks for the inspected RT-DETR contract and learned context fusion.

Detector doubles validate the adapter seam only. A real CUDA forward is required
by the hosted preflight; these tests do not make a recognition-accuracy claim.
"""

from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import cv2  # Load the extension before patch.dict restores sys.modules in adapter tests.
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behaviorsense.models.context_fusion import STGCNContextFusion  # noqa: E402
from behaviorsense.models.stgcnpp import STGCNpp  # noqa: E402
from behaviorsense.models.visual_context import (  # noqa: E402
    COCO_NAMES, CONTEXT_DIM, FEATURE_NAMES, RTDETRContextDetector, VALIDATED_MODEL_SHA256,
    context_features, postprocess_rtdetr, preprocess_rtdetr,
)


def detections():
    raw = np.zeros((1, 300, 6), dtype=np.float32)
    raw[0, 0] = [0.5, 0.5, 0.5, 0.5, 0.8, 62]
    raw[0, 1] = [0.25, 0.25, 0.1, 0.1, 0.1, 41]
    return raw


def pose():
    # Visible box [20,10,60,90], centre [40,50].
    result = np.zeros((1, 17, 3), dtype=np.float32)
    for index, xy in {0: (40, 10), 5: (20, 30), 6: (60, 30), 9: (60, 45),
                      10: (20, 45), 11: (30, 60), 12: (50, 60), 15: (30, 90)}.items():
        result[0, index] = [*xy, 0.9]
    return result


class VisualContextTests(unittest.TestCase):
    def test_preprocess_rgb_unit_range_and_stretch(self):
        frame = np.empty((2, 4, 3), dtype=np.uint8)
        frame[:] = [10, 20, 30]
        frame[1] = [100, 150, 200]
        tensor = preprocess_rtdetr(frame)
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        np.testing.assert_allclose(tensor[0, :, 0, 0], np.array([30, 20, 10]) / 255)
        np.testing.assert_allclose(tensor[0, :, -1, -1], np.array([200, 150, 100]) / 255)
        with self.assertRaisesRegex(ValueError, "uint8"):
            preprocess_rtdetr(frame.astype(np.float32))

    def test_normalised_cxcywh_decodes_non_square_source_without_second_sigmoid(self):
        decoded = postprocess_rtdetr(detections(), (400, 1600, 3))
        self.assertEqual(decoded.shape, (1, 6))
        np.testing.assert_allclose(decoded[0], [400, 100, 1200, 300, 0.8, 62])
        # A raw score of 0.1 must stay below threshold (sigmoid(0.1) would exceed it).
        self.assertNotIn(41, decoded[:, 5])
        raw = detections()
        raw[0, 0, :4] = [0, 0, 0.5, 0.5]
        np.testing.assert_allclose(postprocess_rtdetr(raw, (100, 200))[0, :4], [0, 0, 50, 25])

    def test_unknown_or_corrupted_outputs_fail(self):
        for bad in (np.zeros((1, 84, 300)), np.zeros((1, 300, 6), dtype=np.int32)):
            with self.assertRaises(ValueError):
                postprocess_rtdetr(bad, (100, 200))
        for column, value in ((0, 640), (4, np.nan), (4, -0.3), (5, 62.5), (5, 80)):
            bad = detections()
            bad[0, 0, column] = value
            with self.assertRaises(ValueError):
                postprocess_rtdetr(bad, (100, 200))

    def test_object_and_person_geometry_use_actual_pixels(self):
        boxes = np.array([[60, 40, 80, 60, 0.8, 62], [0, 0, 10, 10, 0.6, 62],
                          [55, 40, 65, 50, 0.9, 41]], dtype=np.float32)
        features = context_features(boxes, pose(), (100, 100, 3))
        self.assertEqual(features.shape, (401,))
        self.assertEqual(len(set(FEATURE_NAMES)), CONTEXT_DIM)
        self.assertEqual(features[0], 1)
        rows = features[1:].reshape(80, 5)
        diagonal = np.sqrt(20000)
        np.testing.assert_allclose(rows[62], [0.8, 30 / diagonal, 0,
                                               np.sqrt(20**2 + 30**2) / diagonal, 0], atol=1e-7)
        self.assertEqual(rows[41, 4], 0)  # Real wrist-box intersection, not missing data.
        np.testing.assert_array_equal(rows[73], [0, -1, -1, -1, -1])
        # Horizontal reflection changes signed dx only; geometric distance is invariant.
        mirrored_pose = pose()
        mirrored_pose[..., 0] = 100 - mirrored_pose[..., 0]
        mirrored_boxes = boxes.copy()
        mirrored_boxes[:, 0] = 100 - boxes[:, 2]
        mirrored_boxes[:, 2] = 100 - boxes[:, 0]
        reflected = context_features(mirrored_boxes, mirrored_pose, (100, 100))[1:].reshape(80, 5)
        np.testing.assert_allclose(reflected[62, [0, 2, 3, 4]], rows[62, [0, 2, 3, 4]])
        self.assertAlmostEqual(float(reflected[62, 1]), -float(rows[62, 1]))

    def test_missing_joints_have_missing_geometry_and_association_is_per_object(self):
        boxes = np.array([[60, 40, 80, 60, 0.8, 62]], dtype=np.float32)
        observed = context_features(boxes, None, (100, 100))[1:].reshape(80, 5)
        np.testing.assert_allclose(observed[62], [0.8, -1, -1, -1, -1])
        occluded = pose()
        occluded[:, [0, 9, 10], :] = 0
        row = context_features(boxes, occluded, (100, 100))[1:].reshape(80, 5)[62]
        self.assertGreaterEqual(row[1], 0)
        np.testing.assert_array_equal(row[3:], [-1, -1])
        distant = pose()
        distant[..., 0] += 300
        grouped = context_features(boxes, np.concatenate((distant, pose())), (100, 500))
        self.assertAlmostEqual(float(grouped[1:].reshape(80, 5)[62, 1]), 30 / np.hypot(100, 500))
        empty = context_features(np.empty((0, 6)), pose(), (100, 100))
        self.assertEqual(empty[0], 1)
        self.assertTrue(np.all(empty[1:].reshape(80, 5)[:, 0] == 0))

    def test_loader_rejects_cuda_fallback_and_unvalidated_weights(self):
        fake = types.SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
        with patch("behaviorsense.models.visual_context.file_sha256", return_value=VALIDATED_MODEL_SHA256):
            with patch.dict(sys.modules, {"onnxruntime": fake}):
                with self.assertRaisesRegex(RuntimeError, "CUDAExecutionProvider"):
                    RTDETRContextDetector("test.onnx")
        with patch("behaviorsense.models.visual_context.file_sha256", return_value="wrong"):
            with self.assertRaisesRegex(ValueError, "unvalidated"):
                RTDETRContextDetector("test.onnx")
        fake.get_available_providers = lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
        fake.InferenceSession = lambda *a, **kw: types.SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
        with patch("behaviorsense.models.visual_context.file_sha256", return_value=VALIDATED_MODEL_SHA256):
            with patch.dict(sys.modules, {"onnxruntime": fake}):
                with self.assertRaisesRegex(RuntimeError, "failed to initialize CUDA"):
                    RTDETRContextDetector("test.onnx")

    def test_adapter_seam_and_metadata_on_validated_session(self):
        calls = []
        session = types.SimpleNamespace(
            get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
            disable_fallback=lambda: calls.append("disable_fallback"),
            get_inputs=lambda: [types.SimpleNamespace(name="images", type="tensor(float)", shape=[1, 3, 640, 640])],
            get_outputs=lambda: [types.SimpleNamespace(name="output0", type="tensor(float)", shape=[1, 300, 6])],
            get_modelmeta=lambda: types.SimpleNamespace(custom_metadata_map={
                "head": "RTDETRDecoder", "version": "8.4.115", "names": repr(dict(enumerate(COCO_NAMES)))}),
            run=lambda names, inputs: (calls.append((names, inputs["images"].shape)) or [detections()]),
        )
        fake = types.SimpleNamespace(get_available_providers=session.get_providers,
                                     InferenceSession=lambda *a, **kw: session)
        with patch("behaviorsense.models.visual_context.file_sha256", return_value=VALIDATED_MODEL_SHA256):
            with patch.dict(sys.modules, {"onnxruntime": fake}):
                detector = RTDETRContextDetector("test.onnx")
                result = detector.features(np.zeros((100, 200, 3), dtype=np.uint8), pose())
        self.assertEqual(result.shape, (401,))
        self.assertEqual(calls, ["disable_fallback", (["output0"], (1, 3, 640, 640))])
        self.assertEqual(detector.metadata["model_sha256"], VALIDATED_MODEL_SHA256)
        self.assertTrue(detector.metadata["frozen"])


class ContextFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        torch.manual_seed(5)
        self.x = torch.randn(2, 3, 12, 17, 2)
        self.context = torch.from_numpy(np.stack([
            context_features(np.array([[60, 40, 80, 60, .8, 62]], np.float32), pose(), (100, 100)),
            context_features(np.empty((0, 6)), pose(), (100, 100)),
        ]))

    def test_baseline_weights_and_logits_are_preserved_exactly(self):
        baseline = STGCNpp(n_classes=22, base_channels=4, dropout=0).eval()
        fusion = STGCNContextFusion(n_classes=22, base_channels=4, context_hidden=8, dropout=0).eval()
        loaded = fusion.load_state_dict(baseline.state_dict(), strict=False)
        self.assertFalse(loaded.unexpected_keys)
        self.assertTrue(all(key.startswith(("context_encoder.", "fusion_head.")) for key in loaded.missing_keys))
        with torch.no_grad():
            expected = baseline(self.x)
            self.assertTrue(torch.equal(fusion(self.x, self.context), expected))
            self.assertTrue(torch.equal(fusion.forward_skeleton(self.x), expected))

    def test_padding_missing_context_and_learned_fusion_gradients(self):
        fusion = STGCNContextFusion(n_classes=22, base_channels=4, context_hidden=8, dropout=0).eval()
        with torch.no_grad():
            fusion.fusion_head[-1].weight.normal_(0, 0.2)
        single = fusion(self.x, self.context)
        padded = torch.stack((self.context, torch.zeros_like(self.context)), dim=1)
        self.assertTrue(torch.allclose(fusion(self.x, padded), single, atol=1e-7))
        self.assertTrue(torch.equal(fusion(self.x, torch.zeros_like(self.context)), fusion.forward_skeleton(self.x)))
        changed = self.context.clone()
        changed[:, 1 + 62 * 5] = 0.01
        self.assertFalse(torch.allclose(single, fusion(self.x, changed), atol=1e-7))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(single, torch.ones_like(single))
        loss.backward()
        gradient = fusion.context_encoder[1].weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0)
        self.assertGreater(float(fusion.blocks[0].gcn.conv.weight.grad.abs().sum()), 0)
        with self.assertRaisesRegex(ValueError, "context must"):
            fusion(self.x, torch.zeros(2, 400))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    total = result.testsRun - len(result.skipped)
    passed = total - len(result.failures) - len(result.errors)
    print(f"{passed}/{total} passed; {len(result.skipped)} skipped")
    sys.exit(0 if result.wasSuccessful() else 1)
