"""Native head pixels, optional mask-limited SR, and calibrated crop cameras.

The full source image stays at its original resolution. Only a head bbox is
super-resolved; inference context is removed before returning the 4x crop.
The existing 2D parser uses resized RGB only for labels, never as SR input.
"""

import builtins
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
from PIL import Image

from .io import sha256

DETECTOR_COMMIT = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
DETECTOR_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
DETECTOR_BYTES = 232589
PARSER_COMMIT = "8a4729d95118d0e97c44185f9bdef3d6bfeaaf99"
PARSER_WEIGHT_SHA256 = (
    "18c44e06b77e47da86fe2b48fac78e0ccbd0b5241c829f9487ebf7fd4a6389e6"
)
PARSER_WEIGHT_BYTES = 55361505
PARSER_SOURCE_FILES = {
    "models/bisenet.py": (
        7568,
        "d73e5e81ebee1e8e9754af559d7d38e5494f5ba475595bad004bea3e4fc7dd12",
    ),
    "models/resnet.py": (
        7099,
        "2001df461431fc8735645345b549bf86bf216dd5c47879cbf9f5077f9b9c740d",
    ),
}
HEAD_LABELS = (*range(1, 14), 17, 18)
HEAD_EXPANSION = dict(
    horizontal_face_height_fraction=0.7,
    top_face_height_fraction=0.6,
    bottom_face_height_fraction=0.15,
)


def _rgb(rgb):
    if (
        not isinstance(rgb, np.ndarray)
        or rgb.dtype != np.uint8
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or min(rgb.shape[:2]) < 1
    ):
        raise ValueError("Expected a nonempty original uint8 HWC RGB image")
    return rgb


def _array_sha(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _integer_box(bbox):
    box = np.asarray(bbox)
    if box.shape != (4,) or not np.issubdtype(box.dtype, np.integer):
        raise ValueError("Head bbox must have four integer exclusive pixel edges")
    x0, y0, x1, y1 = map(int, box)
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise ValueError("Head bbox must be nonempty with nonnegative origin")
    return x0, y0, x1, y1


def head_bbox(face_bbox, wh):
    """Expand a current YuNet face bbox using its height; clip to image WH."""
    face = np.asarray(face_bbox, dtype=np.float64)
    size = np.asarray(wh)
    if (
        face.shape != (4,)
        or not np.isfinite(face).all()
        or face[2] <= face[0]
        or face[3] <= face[1]
        or size.shape != (2,)
        or not np.issubdtype(size.dtype, np.integer)
        or np.any(size <= 0)
    ):
        raise ValueError("Invalid face bbox or image dimensions")
    width, height = map(int, size)
    x0, y0, x1, y1 = face
    h = y1 - y0
    box = [
        min(width, max(0, math.floor(x0 - 0.7 * h))),
        min(height, max(0, math.floor(y0 - 0.6 * h))),
        min(width, max(0, math.ceil(x1 + 0.7 * h))),
        min(height, max(0, math.ceil(y1 + 0.15 * h))),
    ]
    _integer_box(box)
    return box


def verify_head_detector(detector_weight):
    path = Path(detector_weight).resolve()
    if path.stat().st_size != DETECTOR_BYTES or sha256(path) != DETECTOR_SHA256:
        raise RuntimeError("Head detector differs from pinned YuNet2023mar")
    return dict(
        repository="opencv/opencv_zoo",
        source_commit=DETECTOR_COMMIT,
        detector_weight=str(path),
        detector_weight_sha256=DETECTOR_SHA256,
        detector_weight_bytes=DETECTOR_BYTES,
    )


def verify_head_parser(parser_source, parser_weight):
    source, weight = Path(parser_source).resolve(), Path(parser_weight).resolve()
    files = {}
    for relative, (size, digest) in PARSER_SOURCE_FILES.items():
        path = source / relative
        raw = path.read_bytes()
        canonical = raw.replace(b"\r\n", b"\n")
        if len(canonical) != size or hashlib.sha256(canonical).hexdigest() != digest:
            raise RuntimeError(
                "Head parser source differs from pinned yakhyo: " + relative
            )
        files[relative] = dict(
            path=str(path),
            canonical_lf_bytes=size,
            canonical_lf_sha256=digest,
            raw_sha256=hashlib.sha256(raw).hexdigest(),
        )
    if (
        weight.stat().st_size != PARSER_WEIGHT_BYTES
        or sha256(weight) != PARSER_WEIGHT_SHA256
    ):
        raise RuntimeError("Head parser weight differs from pinned resnet18.pt")
    return dict(
        repository="yakhyo/face-parsing",
        source_commit=PARSER_COMMIT,
        source_directory=str(source),
        source_files=files,
        weight=str(weight),
        weight_sha256=PARSER_WEIGHT_SHA256,
        weight_bytes=PARSER_WEIGHT_BYTES,
    )


def _private_parser(source):
    """Load unchanged source with only its absolute resnet import redirected.

    SwinIR also imports a top-level `models`. Its namespace and Python's global
    importer are never changed. Compiling the verified bytes avoids writing a
    __pycache__ into the vendor source directory.
    """
    source = Path(source).resolve()
    namespace = (
        "_msv2_headparser_" + hashlib.sha256(str(source).encode()).hexdigest()[:16]
    )
    if namespace + ".bisenet" in sys.modules:
        return sys.modules[namespace + ".bisenet"]
    package = ModuleType(namespace)
    package.__path__ = [str(source / "models")]
    sys.modules[namespace] = package

    def load(name, filename, importer=builtins.__import__):
        path = source / "models" / filename
        spec = importlib.util.spec_from_file_location(namespace + "." + name, path)
        module = importlib.util.module_from_spec(spec)
        module.__dict__["__builtins__"] = {**vars(builtins), "__import__": importer}
        sys.modules[module.__name__] = module
        try:
            exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
        except BaseException:
            sys.modules.pop(module.__name__, None)
            raise
        return module

    resnet = load("resnet", "resnet.py")

    def scoped_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name == "models.resnet" and fromlist:
            return resnet
        return builtins.__import__(name, globals, locals, fromlist, level)

    return load("bisenet", "bisenet.py", scoped_import)


class HeadDetector:
    """Pinned YuNet detection on original RGB, CPU only and two threads."""

    def __init__(self, detector_weight):
        self.assets = verify_head_detector(detector_weight)
        import cv2

        cv2.setNumThreads(2)
        self.cv2 = cv2
        self.score_threshold = 0.8
        self.detector = cv2.FaceDetectorYN.create(
            str(Path(detector_weight).resolve()),
            "",
            (320, 320),
            self.score_threshold,
            0.3,
            5000,
            cv2.dnn.DNN_BACKEND_OPENCV,
            cv2.dnn.DNN_TARGET_CPU,
        )

    def boxes(self, rgb):
        rgb = _rgb(rgb)
        height, width = rgb.shape[:2]
        boxes, faces = [], []
        metadata = dict(
            state="no_head",
            reason="no_face_detection",
            source_wh=[width, height],
            source_rgb_sha256=_array_sha(rgb),
            device="cpu",
            cpu_threads=2,
            detector_score_threshold=self.score_threshold,
            detector_count=0,
            head_count=0,
            expansion=dict(HEAD_EXPANSION),
            bbox_convention="Native integer pixel-corner edges, upper exclusive",
            includes_all_long_hair=False,
            faces=faces,
        )
        if min(width, height) < 8:
            metadata["reason"] = "source_too_small_for_detection"
            return boxes, metadata
        self.detector.setInputSize((width, height))
        _, detections = self.detector.detect(
            self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR)
        )
        if detections is None:
            return boxes, metadata
        if detections.ndim != 2 or detections.shape[1] != 15:
            raise RuntimeError("Unexpected YuNet detection layout")
        if not np.isfinite(detections).all():
            raise RuntimeError("Nonfinite YuNet detection")
        metadata["detector_count"] = len(detections)
        # Stable confidence/position order also stabilizes per-source head IDs.
        detections = sorted(
            detections, key=lambda row: (-float(row[-1]), float(row[0]), float(row[1]))
        )
        for detection in detections:
            x, y, w, h = map(float, detection[:4])
            score = float(detection[-1])
            if score < self.score_threshold or w <= 0 or h <= 0:
                continue
            face = [x, y, x + w, y + h]
            try:
                bbox = head_bbox(face, [width, height])
            except ValueError:
                # A box fully beyond the captured image supplies no head view.
                continue
            boxes.append(bbox)
            faces.append(
                dict(face_bbox_native=face, bbox_native=bbox, confidence=score)
            )
        if boxes:
            metadata.update(state="heads_found", reason="current_yunet_faces")
        metadata["head_count"] = len(boxes)
        return boxes, metadata


class HeadPixelExtractor:
    """Existing yakhyo 19-label head parsing on CPU, localized by 2D boxes."""

    def __init__(self, parser_source, parser_weight, detector_weight):
        # Validate every asset before importing optional inference packages.
        parser_assets = verify_head_parser(parser_source, parser_weight)
        detector_assets = verify_head_detector(detector_weight)
        import torch

        torch.set_num_threads(2)
        network = _private_parser(parser_source)
        original = network.resnet18
        try:
            # The complete parser checkpoint supplies this backbone. Avoid the
            # upstream constructor's otherwise unused ImageNet download.
            network.resnet18 = lambda: original(weights=None)
            self.model = network.BiSeNet(19, "resnet18").cpu().eval()
        finally:
            network.resnet18 = original
        state = torch.load(Path(parser_weight), map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.torch = torch
        self.mean = torch.tensor([0.485, 0.456, 0.406], device="cpu").view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device="cpu").view(1, 3, 1, 1)
        self.detector = HeadDetector(detector_weight)
        self.assets = dict(
            parser=parser_assets,
            detector=detector_assets,
            device="cpu",
            cpu_threads=2,
            head_labels=list(HEAD_LABELS),
            excluded_labels=[0, 14, 15, 16],
            label_scope="face, facial features, ears, hair and hat; no neck or clothes",
            checkpoint_load="strict complete state_dict, weights_only=True; backbone constructor weights=None",
        )

    def _parse(self, native_rgb):
        small = Image.fromarray(native_rgb).resize(
            (512, 512), Image.Resampling.BILINEAR
        )
        x = (
            self.torch.from_numpy(np.array(small, copy=True))
            .permute(2, 0, 1)[None]
            .float()
            .div_(255)
        )
        with self.torch.inference_mode():
            logits = self.model((x - self.mean) / self.std)[0]
            if (
                tuple(logits.shape) != (1, 19, 512, 512)
                or not self.torch.isfinite(logits).all()
            ):
                raise RuntimeError("Unexpected head parser logits")
            return logits[0].argmax(0).numpy().astype(np.uint8)

    def regions(self, rgb):
        rgb = _rgb(rgb)
        height, width = rgb.shape[:2]
        source_sha = _array_sha(rgb)
        boxes, detection = self.detector.boxes(rgb)
        result, records = [], []
        for number, bbox in enumerate(boxes, 1):
            x0, y0, x1, y1 = _integer_box(bbox)
            if x1 > width or y1 > height:
                raise RuntimeError("Segmentation ROI exceeds its original RGB")
            native = np.array(rgb[y0:y1, x0:x1], copy=True, order="C")
            labels = self._parse(native)
            if (
                labels.shape != (512, 512)
                or labels.dtype != np.uint8
                or np.any(labels > 18)
            ):
                raise RuntimeError("Unexpected head parser label map")
            native_mask = (
                np.array(
                    Image.fromarray(
                        np.isin(labels, HEAD_LABELS).astype(np.uint8) * 255
                    ).resize((x1 - x0, y1 - y0), Image.Resampling.NEAREST)
                )
                > 0
            )
            full_mask = np.zeros((height, width), dtype=bool)
            full_mask[y0:y1, x0:x1] = native_mask
            ys, xs = np.nonzero(full_mask)
            tight = (
                [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                if len(xs)
                else None
            )
            record = dict(
                detection_number=number,
                segmentation_roi_native=list(bbox),
                segmentation_native_wh=[x1 - x0, y1 - y0],
                segmentation_input_wh=[512, 512],
                segmentation_rgb_sha256=_array_sha(native),
                label_pixels_512={str(i): int((labels == i).sum()) for i in range(19)},
                head_pixels_native=int(full_mask.sum()),
                mask_native_sha256=_array_sha(full_mask),
                bbox_native=tight,
                state="head_pixels_found" if tight else "no_head_pixels",
                inverse_label_mapping="PIL nearest to the exact native ROI; zero outside ROI",
                includes_all_long_hair=False,
            )
            records.append(record)
            if tight is not None:
                result.append(dict(bbox_native=tight, mask_native=full_mask))
        if _array_sha(rgb) != source_sha:
            raise RuntimeError("Head pixel extraction changed original RGB")
        return result, dict(
            state="head_pixels_found" if result else "no_head",
            reason="current_2d_head_labels"
            if result
            else "no_face_detection"
            if not boxes
            else "no_head_labels_in_detected_rois",
            source_wh=[width, height],
            source_rgb_sha256=source_sha,
            source_full_rgb_unchanged=True,
            device="cpu",
            cpu_threads=2,
            head_count=len(result),
            parser_calls=len(boxes),
            head_labels=list(HEAD_LABELS),
            excluded_labels=[0, 14, 15, 16],
            preprocessing="PIL RGB BILINEAR512, float32 /255, ImageNet mean/std; segmentation only, never SR input",
            geometry_model_used=False,
            masks_used_for_training_loss=False,
            detection=detection,
            records=records,
            quality_accepted=False,
        )


def _mask_alpha(mask, feather_native):
    mask4 = np.repeat(np.repeat(mask, 4, axis=0), 4, axis=1)
    if feather_native == 0:
        return mask4, mask4.astype(np.float32)
    from scipy.ndimage import distance_transform_edt

    # The zero border also feathers inward where the tight bbox touches head.
    distance = distance_transform_edt(np.pad(mask4, 1))[1:-1, 1:-1]
    alpha = np.minimum(distance / (4 * feather_native), 1).astype(np.float32)
    alpha[~mask4] = 0
    return mask4, alpha


def superresolve_head(
    rgb, bbox, callback, *, context_native=32, mask_native=None, feather_native=2
):
    """Return bbox4 RGB only; callback sees unmodified native context pixels.

    The callback owns SwinIR inference and temporary padding, and must return
    exactly 4x uint8 RGB for the context crop. Optional semantic masks limit SR
    to head pixels; outside the mask the original bbox Lanczos4 stays exact.
    """
    rgb = _rgb(rgb)
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = _integer_box(bbox)
    if x1 > width or y1 > height:
        raise ValueError("Head bbox lies beyond the original image")
    if type(context_native) is not int or context_native < 0:
        raise ValueError("context_native must be a nonnegative native pixel count")
    if (
        isinstance(feather_native, bool)
        or not isinstance(feather_native, (int, float))
        or not math.isfinite(feather_native)
        or feather_native < 0
    ):
        raise ValueError("feather_native must be finite and nonnegative")
    if mask_native is not None and (
        not isinstance(mask_native, np.ndarray)
        or mask_native.dtype != bool
        or mask_native.shape != (height, width)
    ):
        raise ValueError(
            "mask_native must be a boolean mask on the exact original RGB grid"
        )
    mask_sha = _array_sha(mask_native) if mask_native is not None else None
    selected_mask = (
        mask_native[y0:y1, x0:x1].copy() if mask_native is not None else None
    )
    source_sha = _array_sha(rgb)
    cx0 = max(0, ((x0 - context_native) // 8) * 8)
    cy0 = max(0, ((y0 - context_native) // 8) * 8)
    cx1 = min(width, math.ceil((x1 + context_native) / 8) * 8)
    cy1 = min(height, math.ceil((y1 + context_native) / 8) * 8)
    context = np.array(rgb[cy0:cy1, cx0:cx1], copy=True, order="C")
    context_sha = _array_sha(context)
    called = selected_mask is None or bool(selected_mask.any())
    prediction = callback(context) if called else None
    if called and (
        not isinstance(prediction, np.ndarray)
        or prediction.dtype != np.uint8
        or prediction.shape != ((cy1 - cy0) * 4, (cx1 - cx0) * 4, 3)
    ):
        raise ValueError("SR callback must return exact context4 uint8 RGB")
    crop_in_context = [
        (x0 - cx0) * 4,
        (y0 - cy0) * 4,
        (x1 - cx0) * 4,
        (y1 - cy0) * 4,
    ]
    a, b, c, d = crop_in_context
    output = prediction[b:d, a:c].copy() if called else None
    composition = {}
    if selected_mask is not None:
        base = np.array(
            Image.fromarray(rgb[y0:y1, x0:x1]).resize(
                ((x1 - x0) * 4, (y1 - y0) * 4), Image.Resampling.LANCZOS
            )
        )
        mask4, alpha = _mask_alpha(selected_mask, feather_native)
        if called:
            blended = np.rint(
                base.astype(np.float32) * (1 - alpha[..., None])
                + output.astype(np.float32) * alpha[..., None]
            ).astype(np.uint8)
            output = base.copy()
            output[mask4] = blended[mask4]
        else:
            output = base.copy()
        if not np.array_equal(output[~mask4], base[~mask4]):
            raise RuntimeError("Head composition changed pixels outside its mask")
        composition = dict(
            mask_native_sha256=mask_sha,
            mask_pixels_native=int(selected_mask.sum()),
            mask_pixels_4x=int(mask4.sum()),
            outside_mask_pixels_4x=int((~mask4).sum()),
            feather_native=feather_native,
            feather_direction="inside only; alpha zero outside head mask",
            outside_mask_reference="Original native bbox PIL Lanczos4",
            outside_mask_equal_reference=True,
            base_rgb_sha256=_array_sha(base),
        )
    if output.shape != ((y1 - y0) * 4, (x1 - x0) * 4, 3):
        raise RuntimeError("Head4 dimensions disagree with its exact source bbox")
    if _array_sha(rgb) != source_sha:
        raise RuntimeError("Head SR callback changed original full-resolution RGB")
    if mask_native is not None and _array_sha(mask_native) != mask_sha:
        raise RuntimeError("Head SR callback changed the original head mask")
    receipt = dict(
        bbox_native=[x0, y0, x1, y1],
        context_bbox_native=[cx0, cy0, cx1, cy1],
        output_bbox_in_context=crop_in_context,
        source_wh=[width, height],
        native_head_wh=[x1 - x0, y1 - y0],
        context_native_wh=[cx1 - cx0, cy1 - cy0],
        output_wh=[output.shape[1], output.shape[0]],
        context_native=context_native,
        scale=4,
        source_rgb_sha256=source_sha,
        native_head_rgb_sha256=_array_sha(rgb[y0:y1, x0:x1]),
        context_rgb_sha256=context_sha,
        sr_context_rgb_sha256=_array_sha(prediction) if called else None,
        output_rgb_sha256=_array_sha(output),
        no_source_downsampling=True,
        source_full_rgb_unchanged=True,
        full_frame_4x_generated=False,
        semantic_mask_used=mask_native is not None,
        blending_used=mask_native is not None and called,
        sr_inference_called=called,
        composition=composition,
        full_to_head4_pixel_corners=[[4, 0, -4 * x0], [0, 4, -4 * y0], [0, 0, 1]],
        context_policy="Native natural RGB, outward 8-grid context, no resize or artificial background; callback removes its inference padding",
    )
    return output, receipt


def crop_intrinsics(K, bbox, scale=4):
    """Map source pixel-corner K to an integer crop; world-to-camera stays fixed.

    K uses COLMAP pixel-corner coordinates. Integer-center camera conventions
    require conversion before this operation, rather than scaling cx/cy alone.
    This guarantees point projections, not native rasterizer covariance/crop
    equivalence for off-axis views.
    """
    x0, y0, _, _ = _integer_box(bbox)
    matrix = np.asarray(K, dtype=np.float64)
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or matrix[0, 0] <= 0
        or matrix[1, 1] <= 0
        or not np.allclose(matrix[2], [0, 0, 1], rtol=0, atol=1e-12)
    ):
        raise ValueError("Expected a finite pixel-corner intrinsic matrix")
    if type(scale) is not int or scale < 1:
        raise ValueError("Crop scale must be a positive integer")
    transform = np.array([[scale, 0, -scale * x0], [0, scale, -scale * y0], [0, 0, 1]])
    return transform @ matrix
