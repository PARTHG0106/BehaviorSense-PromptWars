"""Frozen RT-DETR evidence for learned multi-label context fusion.

This adapter deliberately supports the inspected Ultralytics export, rather than
guessing a detector's box format from the last dimension. That export returns
normalised ``cx, cy, width, height, score, class_id`` after top-k selection.

The features describe visible objects and 2-D proximity. A TV detection, a nearby
head, or a nearby wrist does not establish gaze, attention, use, or an activity.
No activity labels or hand-written class priors are used here.

Only NumPy is imported eagerly; CPU dataset readers do not need CUDA or ORT.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np


COCO_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)
CONTEXT_SCHEMA = "rtdetr-coco80-relative-v1"
FEATURE_FIELDS = ("score", "center_dx", "center_dy", "head_distance", "wrist_distance")
FEATURE_NAMES = ("detector_observed",) + tuple(
    f"{name.replace(' ', '_')}.{field}" for name in COCO_NAMES for field in FEATURE_FIELDS
)
CONTEXT_DIM = len(FEATURE_NAMES)
VALIDATED_MODEL_SHA256 = "614c5379aaa7743753e74ffa313a4d94950fc76b10809c35f41c2be9b101b83c"
PREPROCESSING = "uint8-BGR -> bilinear-stretch-640x640 -> RGB-float32/255 -> NCHW"
OUTPUT_CONTRACT = "float32[1,300,6]: normalized-cxcywh, sigmoid-score, zero-based-COCO80-id"
DEFAULT_SCORE_THRESHOLD = 0.25
DEFAULT_KEYPOINT_THRESHOLD = 0.30


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _threshold(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0 < value <= 1:
        raise ValueError(f"{name} must be in (0,1]")
    return value


def preprocess_rtdetr(frame_bgr: np.ndarray) -> np.ndarray:
    """Use RT-DETR's stretch resize, not YOLO's aspect-preserving letterbox.

    Boxes are normalised to the stretched input. Multiplying each coordinate by
    the original width/height in postprocessing exactly reverses that resize.
    """
    import cv2

    frame = np.asarray(frame_bgr)
    if frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1:
        raise ValueError("RT-DETR needs a non-empty HWC image with three BGR channels")
    if frame.dtype != np.uint8:
        raise ValueError("RT-DETR input must be uint8 BGR in [0,255]")
    resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1)[None],
                                dtype=np.float32) / np.float32(255.0)


def postprocess_rtdetr(output: np.ndarray, image_shape: tuple[int, ...], *,
                       score_threshold: float = DEFAULT_SCORE_THRESHOLD) -> np.ndarray:
    """Return ``[K,6]`` pixel xyxy/score/class from the validated top-k output.

    No second sigmoid or NMS is applied. RT-DETR's set predictions already have
    sigmoid scores; generic YOLO postprocessing would silently corrupt this model.
    """
    threshold = _threshold(score_threshold, "score_threshold")
    if len(image_shape) < 2 or min(image_shape[:2]) <= 0:
        raise ValueError("image_shape needs a positive height and width")
    raw = np.asarray(output)
    if raw.shape != (1, 300, 6) or not np.issubdtype(raw.dtype, np.floating):
        raise ValueError(f"unsupported RT-DETR output {raw.shape}/{raw.dtype}; expected [1,300,6]")
    if not np.isfinite(raw).all():
        raise ValueError("non-finite RT-DETR output")
    rows = raw[0].astype(np.float32, copy=True)
    if np.any(rows[:, :5] < -1e-6) or np.any(rows[:, :5] > 1 + 1e-6):
        raise ValueError("RT-DETR boxes and sigmoid scores must be normalised to [0,1]")
    class_ids = rows[:, 5]
    if (np.any(np.abs(class_ids - np.rint(class_ids)) > 1e-6) or
            np.any(class_ids < 0) or np.any(class_ids >= len(COCO_NAMES))):
        raise ValueError("RT-DETR returned invalid COCO class IDs")
    rows = rows[rows[:, 4] >= threshold]
    xywh = rows[:, :4].copy()
    height, width = image_shape[:2]
    rows[:, :2] = (xywh[:, :2] - xywh[:, 2:] / 2) * (width, height)
    rows[:, 2:4] = (xywh[:, :2] + xywh[:, 2:] / 2) * (width, height)
    rows[:, [0, 2]] = np.clip(rows[:, [0, 2]], 0, width)
    rows[:, [1, 3]] = np.clip(rows[:, [1, 3]], 0, height)
    return rows[(rows[:, 2] > rows[:, 0]) & (rows[:, 3] > rows[:, 1])]


def _point_box_distance(points: np.ndarray, box: np.ndarray) -> float:
    """Euclidean distance from the nearest observed point to the closed box."""
    delta = np.maximum(np.maximum(box[:2] - points, points - box[2:]), 0)
    return float(np.linalg.norm(delta, axis=1).min())


def context_features(detections: np.ndarray, poses_pixels: np.ndarray | None,
                     image_shape: tuple[int, ...], *,
                     keypoint_threshold: float = DEFAULT_KEYPOINT_THRESHOLD) -> np.ndarray:
    """Build [401] features from pixel detections and the simultaneous pixel pose.

    Element 0 records a successfully observed detector frame, including a frame
    with no objects. For each COCO class, select its highest-scoring detection and
    store score, object-minus-person centre dx/dy, head-to-box distance, and nearest
    wrist-to-box distance. All distances use the image diagonal as the unit.

    The associated person has the closest visible-keypoint bounding-box centre.
    It requires four confident joints, including two shoulders/hips. A head uses
    the nose when visible, otherwise the mean of at least two visible eyes/ears.
    Missing geometry is -1; a missing object has score 0 and all four geometry
    values -1. In particular, zero distance denotes actual 2-D box overlap.
    Person association is proximity only, not a claim of object ownership/use.
    """
    threshold = _threshold(keypoint_threshold, "keypoint_threshold")
    if len(image_shape) < 2 or min(image_shape[:2]) <= 0:
        raise ValueError("image_shape needs a positive height and width")
    det = np.asarray(detections, dtype=np.float32)
    if det.ndim != 2 or det.shape[1] != 6 or not np.isfinite(det).all():
        raise ValueError("detections must be finite [K,6] pixel xyxy/score/class")
    if len(det) and (np.any(det[:, 2:4] <= det[:, :2]) or np.any(det[:, 4] <= 0) or
                     np.any(det[:, 4] > 1) or np.any(det[:, 5] < 0) or
                     np.any(det[:, 5] >= 80) or np.any(det[:, 5] != np.rint(det[:, 5]))):
        raise ValueError("invalid detection box, score, or class")
    if poses_pixels is None:
        poses = np.empty((0, 17, 3), dtype=np.float32)
    else:
        poses = np.asarray(poses_pixels, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (17, 3) or not np.isfinite(poses).all():
        raise ValueError("poses_pixels must be finite [M,17,3] pixel xy/confidence")
    if np.any(poses[..., 2] < 0) or np.any(poses[..., 2] > 1):
        raise ValueError("pose confidences must be in [0,1]")
    people = []
    for pose in poses:
        visible = pose[:, 2] >= threshold
        if visible.sum() < 4 or visible[[5, 6, 11, 12]].sum() < 2:
            continue
        xy = pose[visible, :2]
        centre = (xy.min(axis=0) + xy.max(axis=0)) / 2
        if visible[0]:
            head = pose[[0], :2]
        elif visible[1:5].sum() >= 2:
            head = pose[1:5, :2][visible[1:5]].mean(axis=0, keepdims=True)
        else:
            head = np.empty((0, 2), dtype=np.float32)
        wrists = pose[[9, 10], :2][visible[[9, 10]]]
        people.append((centre, head, wrists))
    result = np.empty(CONTEXT_DIM, dtype=np.float32)
    result[0] = 1.0
    rows = result[1:].reshape(len(COCO_NAMES), len(FEATURE_FIELDS))
    rows[:] = -1.0
    rows[:, 0] = 0.0
    diagonal = float(np.hypot(*image_shape[:2]))
    for class_id in np.unique(det[:, 5]).astype(int):
        candidates = det[det[:, 5] == class_id]
        obj = candidates[int(np.argmax(candidates[:, 4]))]
        rows[class_id, 0] = obj[4]
        if not people:
            continue
        centre = (obj[:2] + obj[2:4]) / 2
        person = min(people, key=lambda person: float(np.linalg.norm(centre - person[0])))
        rows[class_id, 1:3] = (centre - person[0]) / diagonal
        if len(person[1]):
            rows[class_id, 3] = _point_box_distance(person[1], obj[:4]) / diagonal
        if len(person[2]):
            rows[class_id, 4] = _point_box_distance(person[2], obj[:4]) / diagonal
    return result


class RTDETRContextDetector:
    """Frozen ONNX inference with a pinned, inspected model/contract."""

    def __init__(self, model_path: str | Path, *, require_cuda: bool = True,
                 score_threshold: float = DEFAULT_SCORE_THRESHOLD,
                 keypoint_threshold: float = DEFAULT_KEYPOINT_THRESHOLD) -> None:
        path = Path(model_path)
        digest = file_sha256(path)
        if digest != VALIDATED_MODEL_SHA256:
            raise ValueError("unvalidated RT-DETR weights; inspect output/preprocessing before adding their SHA256")
        self.score_threshold = _threshold(score_threshold, "score_threshold")
        self.keypoint_threshold = _threshold(keypoint_threshold, "keypoint_threshold")
        import onnxruntime as ort

        if require_cuda and "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("RT-DETR context extraction requires ONNX Runtime CUDAExecutionProvider")
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if require_cuda
                     else ["CPUExecutionProvider"])
        self.session = ort.InferenceSession(str(path), providers=providers)
        active = self.session.get_providers()
        if require_cuda and (not active or active[0] != "CUDAExecutionProvider"):
            raise RuntimeError(f"RT-DETR failed to initialize CUDA; active providers: {active}")
        # Stop ORT rerunning the entire graph on CPU following a CUDA runtime error.
        self.session.disable_fallback()
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or inputs[0].name != "images" or inputs[0].type != "tensor(float)" or
                list(inputs[0].shape) != [1, 3, 640, 640] or len(outputs) != 1 or
                outputs[0].name != "output0" or outputs[0].type != "tensor(float)" or
                list(outputs[0].shape) != [1, 300, 6]):
            raise ValueError("RT-DETR ONNX input/output does not match the validated contract")
        meta = self.session.get_modelmeta().custom_metadata_map
        try:
            names = ast.literal_eval(meta.get("names", ""))
        except (SyntaxError, ValueError) as exc:
            raise ValueError("RT-DETR is missing its COCO class metadata") from exc
        if (names != dict(enumerate(COCO_NAMES)) or meta.get("head") != "RTDETRDecoder" or
                meta.get("version") != "8.4.115"):
            raise ValueError("RT-DETR metadata does not match the inspected Ultralytics COCO export")
        self.metadata = {
            "schema": CONTEXT_SCHEMA, "feature_dim": CONTEXT_DIM,
            "feature_names": list(FEATURE_NAMES), "model_sha256": digest,
            "score_threshold": self.score_threshold,
            "keypoint_threshold": self.keypoint_threshold,
            "preprocessing": PREPROCESSING, "output_contract": OUTPUT_CONTRACT,
            "provider": active[0], "providers": active, "frozen": True,
            "geometry_missing_value": -1, "geometry_units": "image_diagonal",
            "interpretation": "object visibility and 2-D proximity; does not measure gaze or attention",
        }

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        outputs = self.session.run(["output0"], {"images": preprocess_rtdetr(frame_bgr)})
        return postprocess_rtdetr(outputs[0], frame_bgr.shape,
                                  score_threshold=self.score_threshold)

    def features(self, frame_bgr: np.ndarray, poses_pixels: np.ndarray | None = None) -> np.ndarray:
        return context_features(self.detect(frame_bgr), poses_pixels, frame_bgr.shape,
                                keypoint_threshold=self.keypoint_threshold)


def load_context_detector(model_path: str | Path, *, require_cuda: bool = True,
                          score_threshold: float = DEFAULT_SCORE_THRESHOLD,
                          keypoint_threshold: float = DEFAULT_KEYPOINT_THRESHOLD) -> RTDETRContextDetector:
    return RTDETRContextDetector(model_path, require_cuda=require_cuda,
                                score_threshold=score_threshold,
                                keypoint_threshold=keypoint_threshold)
