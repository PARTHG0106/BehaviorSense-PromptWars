"""One upload per subprocess; the parent selects its physical T4 before imports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))

    def progress(**event):
        print(json.dumps(event), file=sys.stderr, flush=True)

    progress(step="loading_models")
    import torch
    import onnxruntime as ort
    from behaviorsense.models.rgb_hybrid import FrozenRGBHybrid, T4RGBEncoder
    from behaviorsense.models.visual_context import load_context_detector
    from behaviorsense.service.rgb_hybrid_video import process_video
    from behaviorsense.video import RTMODetector, _gallery_from_json, _osnet_embedder

    torch.set_num_threads(2)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each RGB worker requires exactly one assigned CUDA GPU.")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    # Resolve the CUDA/cuDNN libraries installed with the pinned Torch wheel
    # before ONNX Runtime constructs either detector session.
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    bundle = Path(config["bundle"])
    model = FrozenRGBHybrid(bundle, "cuda:0")
    progress(step="loading_swin_fp32")
    encoder = T4RGBEncoder(bundle / "swin3d_t-7615ae03.pth", "cuda:0")
    # Both ONNX sessions see only the GPU assigned by CUDA_VISIBLE_DEVICES.
    detector = RTMODetector(config["rtmo"], device="cuda")
    objects = load_context_detector(config["rtdetr"], require_cuda=True)
    if config.get("smoke_only"):
        import numpy as np
        progress(step="testing_model_forwards")
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        detector.detect(frame)
        context = objects.features(frame, np.zeros((2, 17, 3), dtype=np.float32))[None]
        features = encoder(np.zeros((2, 32, 224, 224, 3), dtype=np.uint8))[None]
        prediction = model.predict(np.zeros((1, 30, 2, 17, 3), dtype=np.float16), context,
            features, np.asarray([0], dtype=np.int64), np.asarray(["smoke"]))
        if prediction["scores"].shape != (1, 22) or not np.isfinite(prediction["scores"]).all():
            raise RuntimeError("Hybrid inference smoke check failed.")
        result = {"stages": [], "smoke_passed": True, "gpu": torch.cuda.get_device_name(0),
                  "pose_providers": detector.providers, "object_providers": objects.metadata["providers"],
                  "model_id": model.manifest["model_id"], "rgb_precision": "float32"}
        Path(config["result"]).write_text(json.dumps(result), encoding="utf-8")
        progress(step="complete")
        return
    embedder, gallery = None, None
    if config.get("use_osnet"):
        osnet = bundle / "osnet_ain_x1_0_msmt17.pth"
        if not osnet.is_file():
            raise ValueError("OSNet is required for enrollment or a supplied gallery.")
        from behaviorsense.models.rgb_hybrid import file_sha256
        expected = model.manifest["files"][osnet.name]["sha256"]
        if file_sha256(osnet) != expected:
            raise ValueError("OSNet checkpoint checksum differs.")
        embedder = _osnet_embedder(str(osnet), "cuda:0")
        if config.get("gallery_file"):
            gallery = _gallery_from_json(Path(config["gallery_file"]).read_text(encoding="utf-8"))
    progress(step="decoding")
    result = process_video(config["video"], model, encoder, detector, objects,
        embedder=embedder, gallery=gallery, max_frames=config["max_frames"], progress=progress)
    Path(config["result"]).write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    progress(step="complete")


if __name__ == "__main__":
    main()
