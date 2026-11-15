"""CPU checks for per-upload GPU assignment and the sampled-frame limit.

These exercise subprocess construction and decoding with model doubles; a hosted
two-GPU run is still needed to measure throughput and real CUDA model loading.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behaviorsense import video  # noqa: E402


class VideoDeviceTests(unittest.TestCase):
    def test_selected_gpu_respects_parent_ordinals_and_uuid_masks(self):
        for mask, wanted in [(None, "1"), ("3,1", "1"),
                             ("GPU-first,GPU-second", "GPU-second"),
                             ("MIG-GPU-first/1/0,MIG-GPU-second/2/0", "MIG-GPU-second/2/0")]:
            with self.subTest(mask=mask):
                parent = {"PYTHONPATH": "source", "LD_LIBRARY_PATH": "cuda-libs"}
                if mask is not None:
                    parent["CUDA_VISIBLE_DEVICES"] = mask
                original = dict(parent)
                child, pose, reid = video._child_device_config(parent, "cuda:1", None)
                self.assertEqual(child["CUDA_VISIBLE_DEVICES"], wanted)
                self.assertEqual((pose, reid), ("cuda", "cuda:0"))
                self.assertEqual(child["LD_LIBRARY_PATH"], "cuda-libs")
                self.assertEqual(parent, original)

    def test_cpu_fallbacks_and_explicit_second_gpu_are_remapped(self):
        parent = {"CUDA_VISIBLE_DEVICES": "3,1"}
        for pose, reid, mask, expected in [
            ("cuda:1", "cpu", "1", ("cuda", "cpu")),
            ("cpu", "cuda:1", "1", ("cpu", "cuda:0")),
            ("cpu", None, "3,1", ("cpu", "cpu")),
            ("cuda:1", "cuda", "1", ("cuda", "cuda:0")),
            ("cuda:1", "cuda:0", "1,3", ("cuda", "cuda:1")),
        ]:
            with self.subTest(pose=pose, reid=reid):
                child, actual_pose, actual_reid = video._child_device_config(parent, pose, reid)
                self.assertEqual(child["CUDA_VISIBLE_DEVICES"], mask)
                self.assertEqual((actual_pose, actual_reid), expected)
        self.assertEqual(parent, {"CUDA_VISIBLE_DEVICES": "3,1"})

    def test_disabled_or_out_of_range_gpus_cannot_escape_parent_mask(self):
        for mask, device in [("", "cuda"), ("-1", "cuda"), ("0", "cuda:1"),
                             ("0,-1,1", "cuda:1"), ("0,,1", "cuda:1")]:
            with self.subTest(mask=mask):
                with self.assertRaisesRegex(ValueError, "CUDA_VISIBLE_DEVICES"):
                    video._child_device_config({"CUDA_VISIBLE_DEVICES": mask}, device, None)
        for device in ("cuda:-1", "cuda:one", "mps"):
            with self.assertRaisesRegex(ValueError, "unsupported device"):
                video._child_device_config({}, device, None)

    def test_overlapping_uploads_get_distinct_child_masks_without_global_changes(self):
        barrier = threading.Barrier(2)

        def complete(cmd, *, env, **kwargs):
            barrier.wait(timeout=5)
            result = {
                "mask": env["CUDA_VISIBLE_DEVICES"],
                "pose": cmd[cmd.index("--device") + 1],
                "reid": cmd[cmd.index("--osnet-device") + 1],
                "limit": int(cmd[cmd.index("--max-frames") + 1]),
            }
            Path(cmd[cmd.index("--out") + 1]).write_text(json.dumps(result), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-first,GPU-second"}):
            before = dict(os.environ)
            with patch.object(video, "_run_watching_progress", side_effect=complete):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(video.extract_isolated, "clip.mp4", rtmo="pose.onnx",
                                           osnet="reid.pth", device=f"cuda:{index}",
                                           osnet_device="cuda", max_frames=7)
                               for index in range(2)]
                    results = [future.result(timeout=10) for future in futures]
            self.assertEqual(os.environ, before)
        self.assertEqual([item["mask"] for item in results], ["GPU-first", "GPU-second"])
        for item in results:
            self.assertEqual((item["pose"], item["reid"], item["limit"]), ("cuda", "cuda:0", 7))

    def test_rtmo_cpu_fallback_is_explicit_and_cuda_provider_failure_is_visible(self):
        model = types.SimpleNamespace(session=types.SimpleNamespace(
            get_providers=lambda: ["CPUExecutionProvider"]))
        module = types.ModuleType("rtmlib")
        module.RTMO = lambda **kwargs: model
        with patch.dict(sys.modules, {"rtmlib": module}):
            self.assertEqual(video.RTMODetector("pose.onnx", device="cpu").providers,
                             ["CPUExecutionProvider"])
            with self.assertRaisesRegex(RuntimeError, "not CUDA"):
                video.RTMODetector("pose.onnx", device="cuda")
            with self.assertRaisesRegex(ValueError, "extract_isolated"):
                video.RTMODetector("pose.onnx", device="cuda:1")


class FakeCapture:
    def __init__(self, count, fps):
        self.count = count
        self.fps = fps
        self.reads = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, key):
        return {cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_WIDTH: 16,
                cv2.CAP_PROP_FRAME_HEIGHT: 8}.get(key, 0)

    def read(self):
        if self.reads == self.count:
            return False, None
        self.reads += 1
        return True, np.zeros((8, 16, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class VideoFrameLimitTests(unittest.TestCase):
    def test_exact_limit_and_unsampled_tail_are_accepted_but_next_sample_is_overflow(self):
        # At 30 -> 15 Hz, the retained source frames are 0, 2, 4. Source frame 5
        # is an unsampled tail; only decoding frame 6 proves a fourth sample exists.
        for count, kept, truncated in [(4, 2, False), (5, 3, False),
                                        (6, 3, False), (7, 3, True), (10, 3, True)]:
            with self.subTest(source_frames=count):
                capture = FakeCapture(count, 30)
                detector = types.SimpleNamespace(providers=["CPUExecutionProvider"])
                calls = []
                detector.detect = lambda frame: calls.append(frame) or []
                with patch.object(cv2, "VideoCapture", return_value=capture), \
                        patch.object(video, "RTMODetector", return_value=detector):
                    result = video.extract_observations("clip.mp4", rtmo="pose.onnx",
                                                        device="cpu", max_frames=3)
                self.assertEqual(result["frames_kept"], kept)
                self.assertEqual(result["truncated"], truncated)
                self.assertEqual(len(calls), kept, "overflow detection must not run another pose")
                self.assertEqual(result["frames_read"], min(count, 7))
                self.assertTrue(capture.released)

    def test_invalid_limits_and_rates_fail_before_opening_video(self):
        with patch.object(cv2, "VideoCapture") as capture:
            for limit in (0, -1, 1.5, True):
                with self.assertRaisesRegex(ValueError, "max_frames"):
                    video.extract_observations("clip.mp4", rtmo="pose.onnx", max_frames=limit)
            for rate in (0, -1, float("nan"), float("inf")):
                with self.assertRaisesRegex(ValueError, "target_fps"):
                    video.extract_observations("clip.mp4", rtmo="pose.onnx", target_fps=rate)
            capture.assert_not_called()

    def test_operator_frame_limit_is_validated_and_defaults_to_forty_minutes(self):
        with patch.dict(os.environ):
            os.environ.pop("BS_MAX_FRAMES", None)
            self.assertEqual(video._max_frames_from_env(), 36000)
            os.environ["BS_MAX_FRAMES"] = "12000"
            self.assertEqual(video._max_frames_from_env(), 12000)
            for raw in ("", "0", "-3", "two", "1.5"):
                os.environ["BS_MAX_FRAMES"] = raw
                with self.assertRaisesRegex(ValueError, "BS_MAX_FRAMES"):
                    video._max_frames_from_env()


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    total = result.testsRun - len(result.skipped)
    passed = total - len(result.failures) - len(result.errors)
    print(f"{passed}/{total} passed; {len(result.skipped)} skipped")
    sys.exit(0 if result.wasSuccessful() else 1)
