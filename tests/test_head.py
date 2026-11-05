"""Mixed-resolution inputs retain source pixels and crop-camera mappings."""

import json
import builtins
import hashlib
import sys
from types import ModuleType

import numpy as np
import pytest
from PIL import Image

import msv2.head as head
from msv2.head import (
    HeadPixelExtractor,
    crop_intrinsics,
    head_bbox,
    superresolve_head,
    verify_head_detector,
)


def nearest4(rgb):
    return np.repeat(np.repeat(rgb, 4, axis=0), 4, axis=1)


def test_thin_profile_head_expansion_uses_face_height_and_exclusive_edges():
    # A thin profile still needs space behind the visible face for its ear/head.
    assert head_bbox([20, 30, 26, 50], [100, 100]) == [6, 18, 40, 53]
    assert head_bbox([1, 2, 9, 12], [13, 17]) == [0, 0, 13, 14]


def test_head_context_keeps_native_one_pixel_detail_and_trims_only_context():
    rgb = np.zeros((47, 83, 3), np.uint8)
    rgb[:, ::2, 0] = 255
    rgb[::2, :, 1] = 200
    rgb[23, 31] = [17, 31, 47]
    original = rgb.copy()
    bbox = [29, 19, 42, 31]
    seen = []

    def callback(crop):
        seen.append(crop.copy())
        return nearest4(crop)

    output, receipt = superresolve_head(rgb, bbox, callback, context_native=8)
    assert receipt["context_bbox_native"] == [16, 8, 56, 40]
    assert np.array_equal(seen[0], original[8:40, 16:56])
    assert output.shape == (48, 52, 3)
    assert np.array_equal(output, nearest4(original[19:31, 29:42]))
    assert np.array_equal(output[16:20, 8:12], np.full((4, 4, 3), [17, 31, 47]))
    assert receipt["output_bbox_in_context"] == [52, 44, 104, 92]
    assert np.array_equal(rgb, original)
    assert receipt["full_frame_4x_generated"] is False
    assert receipt["semantic_mask_used"] is False
    assert receipt["blending_used"] is False
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize("bbox", [[0, 0, 3, 5], [26, 12, 29, 17], [0, 12, 4, 17]])
def test_edge_clipped_context_does_not_move_or_drop_last_head_pixel(bbox):
    rgb = np.random.default_rng(12).integers(0, 256, (17, 29, 3), dtype=np.uint8)
    output, receipt = superresolve_head(rgb, bbox, nearest4)
    x0, y0, x1, y1 = bbox
    assert receipt["context_bbox_native"] == [0, 0, 29, 17]
    assert np.array_equal(output, nearest4(rgb[y0:y1, x0:x1]))
    assert receipt["output_wh"] == [(x1 - x0) * 4, (y1 - y0) * 4]


def test_callback_cannot_mutate_the_original_full1_rgb():
    rgb = np.random.default_rng(82).integers(0, 256, (23, 39, 3), dtype=np.uint8)
    original = rgb.copy()

    def callback(crop):
        assert np.array_equal(crop, original)
        crop[:] = 211
        return nearest4(crop)

    output, receipt = superresolve_head(rgb, [3, 5, 11, 14], callback)
    assert np.array_equal(rgb, original)
    assert np.all(output == 211)
    assert receipt["source_full_rgb_unchanged"] is True


def test_crop_intrinsics_preserve_arbitrary_off_axis_point_projections():
    K = np.array([[1301, 0, 611.3], [0, 1279, 799.1], [0, 0, 1]])
    original = K.copy()
    bbox = [337, 241, 572, 428]
    transformed = crop_intrinsics(K, bbox)
    points = np.array([[0.1, -0.2, 2.0], [-0.3, 0.5, 4.0], [1.0, -0.8, 3.5]])
    source = points @ K.T
    source = source[:, :2] / source[:, 2:]
    head = points @ transformed.T
    head = head[:, :2] / head[:, 2:]
    assert np.allclose(head, (source - bbox[:2]) * 4, atol=2e-12, rtol=0)
    assert transformed[0, 2] == 4 * (K[0, 2] - bbox[0])
    assert transformed[1, 2] == 4 * (K[1, 2] - bbox[1])
    assert np.array_equal(K, original)
    assert np.array_equal(crop_intrinsics(K, [0, 0, 1280, 960], scale=1), K)


@pytest.mark.parametrize(
    "bbox", [[-1, 0, 4, 4], [2, 2, 2, 4], [0, 0, 20, 20], [0.5, 0, 4, 4]]
)
def test_invalid_head_bboxes_are_rejected(bbox):
    with pytest.raises(ValueError):
        superresolve_head(np.zeros((8, 8, 3), np.uint8), bbox, nearest4)


def test_sr_callback_must_return_exact_context4_rgb():
    rgb = np.zeros((17, 29, 3), np.uint8)
    with pytest.raises(ValueError, match="exact context4"):
        superresolve_head(rgb, [1, 2, 9, 12], lambda crop: crop)


def test_detector_pin_rejects_drift_before_optional_cv2_import(tmp_path):
    detector = tmp_path / "wrong.onnx"
    detector.write_bytes(b"not the pinned YuNet asset")
    with pytest.raises(RuntimeError, match="pinned YuNet"):
        verify_head_detector(detector)


def test_mask_limits_sr_and_feathers_only_inside_native_head_pixels():
    rgb = np.full((25, 31, 3), [31, 61, 101], np.uint8)
    original = rgb.copy()
    mask = np.zeros(rgb.shape[:2], bool)
    mask[6:19, 8:23] = True
    mask[11:13, 14:16] = False  # A true hole must remain original RGB.
    before = mask.copy()
    bbox = [6, 4, 25, 21]

    def callback(crop):
        assert np.array_equal(crop, original)
        return np.full((crop.shape[0] * 4, crop.shape[1] * 4, 3), 231, np.uint8)

    output, receipt = superresolve_head(rgb, bbox, callback, mask_native=mask)
    base = np.array(
        Image.fromarray(rgb[4:21, 6:25]).resize((76, 68), Image.Resampling.LANCZOS)
    )
    selected = nearest4(mask[4:21, 6:25])
    assert np.array_equal(output[~selected], base[~selected])
    assert np.all(
        output[20, 28] == 231
    )  # At least two native pixels from the hole/boundary.
    assert np.all(output[8, 8] > base[8, 8])
    assert np.all(output[8, 8] < 231)  # First inside pixel blends, never dilation.
    assert np.array_equal(rgb, original) and np.array_equal(mask, before)
    assert receipt["composition"]["outside_mask_equal_reference"]
    assert receipt["composition"]["feather_direction"].startswith("inside only")
    assert receipt["semantic_mask_used"] and receipt["sr_inference_called"]
    assert not receipt["full_frame_4x_generated"]
    json.dumps(receipt, allow_nan=False)


def test_empty_mask_bypasses_sr_and_returns_original_bbox_lanczos4():
    rgb = np.random.default_rng(81).integers(0, 256, (19, 27, 3), dtype=np.uint8)

    def forbidden(crop):
        raise AssertionError("Empty head must not run SwinIR")

    output, receipt = superresolve_head(
        rgb, [3, 2, 17, 15], forbidden, mask_native=np.zeros((19, 27), bool)
    )
    expected = np.array(
        Image.fromarray(rgb[2:15, 3:17]).resize((56, 52), Image.Resampling.LANCZOS)
    )
    assert np.array_equal(output, expected)
    assert not receipt["sr_inference_called"] and not receipt["blending_used"]
    assert receipt["sr_context_rgb_sha256"] is None


def test_zero_feather_keeps_exact_sr_inside_and_exact_original_outside():
    rgb = np.full((13, 17, 3), 47, np.uint8)
    mask = np.zeros((13, 17), bool)
    mask[4:9, 5:12] = True
    output, _ = superresolve_head(
        rgb,
        [2, 3, 15, 11],
        lambda crop: np.full_like(nearest4(crop), 173),
        mask_native=mask,
        feather_native=0,
    )
    selected = nearest4(mask[3:11, 2:15])
    assert np.all(output[selected] == 173)
    assert np.all(output[~selected] == 47)


@pytest.mark.parametrize("mask", [np.ones((4, 4), bool), np.zeros((8, 8), np.uint8)])
def test_mask_must_match_original_native_boolean_pixel_grid(mask):
    with pytest.raises(ValueError, match="exact original RGB grid"):
        superresolve_head(
            np.zeros((8, 8, 3), np.uint8), [1, 1, 6, 6], nearest4, mask_native=mask
        )


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_feather_rejects_invalid_pixel_radius(value):
    with pytest.raises(ValueError, match="feather_native"):
        superresolve_head(
            np.zeros((8, 8, 3), np.uint8), [1, 1, 6, 6], nearest4, feather_native=value
        )


def extractor_with_labels(boxes, labels, seen):
    extractor = HeadPixelExtractor.__new__(HeadPixelExtractor)

    class Detector:
        def boxes(self, rgb):
            return boxes, dict(state="heads_found" if boxes else "no_head")

    def parse(crop):
        seen.append(crop.copy())
        return labels.copy()

    extractor.detector = Detector()
    extractor._parse = parse
    return extractor


def test_head_pixels_map_back_to_native_grid_and_bbox_is_tight():
    rgb = np.random.default_rng(72).integers(0, 256, (91, 123, 3), dtype=np.uint8)
    original = rgb.copy()
    labels = np.zeros((512, 512), np.uint8)
    labels[128:384, 128:384] = 1
    labels[64:128, 128:384] = 17  # Hair is part of the head.
    labels[32:64, 128:384] = 18  # Hat is part of the head.
    labels[384:448, 128:384] = 14  # Neck is excluded.
    labels[448:480, 128:384] = 15  # Necklace is excluded.
    labels[480:512, 128:384] = 16  # Clothes are excluded.
    seen = []
    extractor = extractor_with_labels([[20, 10, 84, 74]], labels, seen)
    regions, metadata = extractor.regions(rgb)
    expected = np.zeros((91, 123), bool)
    expected[14:58, 36:68] = True
    assert len(regions) == 1 and regions[0]["bbox_native"] == [36, 14, 68, 58]
    assert np.array_equal(regions[0]["mask_native"], expected)
    assert np.array_equal(
        seen[0], rgb[10:74, 20:84]
    )  # Parser receives raw ROI, not resized SR.
    assert np.array_equal(rgb, original)
    assert metadata["source_full_rgb_unchanged"] and not metadata["geometry_model_used"]
    assert metadata["records"][0]["segmentation_input_wh"] == [512, 512]
    assert metadata["records"][0]["head_pixels_native"] == 44 * 32
    json.dumps(metadata, allow_nan=False)


def test_no_face_and_empty_head_labels_never_invent_a_head_bbox():
    rgb = np.zeros((40, 60, 3), np.uint8)
    seen = []
    extractor = extractor_with_labels([], np.zeros((512, 512), np.uint8), seen)
    regions, metadata = extractor.regions(rgb)
    assert regions == [] and seen == [] and metadata["parser_calls"] == 0
    assert metadata["reason"] == "no_face_detection"
    extractor = extractor_with_labels(
        [[4, 5, 36, 37]], np.full((512, 512), 16, np.uint8), seen
    )
    regions, metadata = extractor.regions(rgb)
    assert regions == [] and len(seen) == 1
    assert metadata["reason"] == "no_head_labels_in_detected_rois"


def test_independent_two_dimensional_rois_keep_source_pixel_coordinates():
    rgb = np.zeros((80, 100, 3), np.uint8)
    labels = np.full((512, 512), 9, np.uint8)  # All labels 1..13 are retained.
    regions, metadata = extractor_with_labels(
        [[2, 3, 18, 19], [61, 43, 93, 75]], labels, []
    ).regions(rgb)
    assert [row["bbox_native"] for row in regions] == [[2, 3, 18, 19], [61, 43, 93, 75]]
    assert not np.any(regions[0]["mask_native"] & regions[1]["mask_native"])
    assert metadata["head_count"] == 2


def test_private_parser_import_does_not_replace_swinir_models_or_global_importer(
    tmp_path, monkeypatch
):
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / "resnet.py").write_text(
        'MARKER = "verified private source"\n', encoding="utf-8"
    )
    (folder / "bisenet.py").write_text(
        "from models.resnet import MARKER\nfrom math import floor\nVALUE = (MARKER, floor(1.9))\n",
        encoding="utf-8",
    )
    public, public_resnet = ModuleType("models"), ModuleType("models.resnet")
    public_resnet.MARKER = "SwinIR namespace must remain unchanged"
    monkeypatch.setitem(sys.modules, "models", public)
    monkeypatch.setitem(sys.modules, "models.resnet", public_resnet)
    importer = builtins.__import__
    loaded = head._private_parser(tmp_path)
    assert loaded.VALUE == ("verified private source", 1)
    assert loaded.__name__.startswith("_msv2_headparser_")
    assert (
        sys.modules["models"] is public
        and sys.modules["models.resnet"] is public_resnet
    )
    assert builtins.__import__ is importer
    assert not list(tmp_path.rglob("__pycache__"))


def test_parser_pins_accept_canonical_lf_without_modifying_existing_source(
    tmp_path, monkeypatch
):
    folder = tmp_path / "models"
    folder.mkdir()
    pins = {}
    for name in ("bisenet.py", "resnet.py"):
        canonical = (name + "\n# source\n").encode()
        (folder / name).write_bytes(canonical.replace(b"\n", b"\r\n"))
        pins["models/" + name] = (len(canonical), hashlib.sha256(canonical).hexdigest())
    weight = tmp_path / "resnet18.pt"
    weight.write_bytes(b"complete checkpoint fixture")
    monkeypatch.setattr(head, "PARSER_SOURCE_FILES", pins)
    monkeypatch.setattr(head, "PARSER_WEIGHT_BYTES", weight.stat().st_size)
    monkeypatch.setattr(
        head, "PARSER_WEIGHT_SHA256", hashlib.sha256(weight.read_bytes()).hexdigest()
    )
    before = {path: path.read_bytes() for path in folder.iterdir()}
    assets = head.verify_head_parser(tmp_path, weight)
    assert all(path.read_bytes() == contents for path, contents in before.items())
    assert (
        assets["source_files"]["models/bisenet.py"]["canonical_lf_sha256"]
        == pins["models/bisenet.py"][1]
    )
    weight.write_bytes(b"tampered checkpoint")
    with pytest.raises(RuntimeError, match="pinned resnet18.pt"):
        head.verify_head_parser(tmp_path, weight)
    (folder / "bisenet.py").write_text("tampered source\n")
    with pytest.raises(RuntimeError, match="pinned yakhyo"):
        head.verify_head_parser(tmp_path, weight)


def test_bad_assets_are_rejected_before_any_torch_or_cv2_import(tmp_path, monkeypatch):
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / "bisenet.py").write_text("untrusted source\n")
    importer = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name in ("torch", "cv2"):
            raise AssertionError(
                "Optional inference import happened before asset verification"
            )
        return importer(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(RuntimeError, match="pinned yakhyo"):
        HeadPixelExtractor(tmp_path, tmp_path / "bad.pt", tmp_path / "bad.onnx")
