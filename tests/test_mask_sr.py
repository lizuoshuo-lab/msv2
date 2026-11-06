"""Whole-image composition preserves pixels outside real semantic support."""

import hashlib
import json

import numpy as np
from PIL import Image
import pytest

import msv2.mask_sr as compose_module
from msv2.mask_sr import compose_face


def nearest4(array):
    return np.repeat(np.repeat(array, 4, axis=0), 4, axis=1)


def reference(rgb):
    return np.array(
        Image.fromarray(rgb).resize(
            (rgb.shape[1] * 4, rgb.shape[0] * 4), Image.Resampling.LANCZOS
        )
    )


def region(mask, scope="face"):
    ys, xs = np.nonzero(mask)
    return dict(
        bbox_native=[
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ],
        mask_native=mask,
        mask_scope=scope,
    )


def test_full_image_reference_is_exact_outside_mask_holes_and_bbox_edges():
    rgb = np.random.default_rng(504).integers(0, 256, (23, 31, 3), dtype=np.uint8)
    original = rgb.copy()
    mask = np.zeros((23, 31), bool)
    mask[5:17, 7:23] = True
    mask[5:7, 7:10] = False  # Irregular mask boundary inside its tight bbox.
    mask[10:12, 13:15] = False
    selected = nearest4(mask)
    calls = []

    def changing_sr(crop):
        calls.append(crop.copy())
        return np.full(
            (crop.shape[0] * 4, crop.shape[1] * 4, 3), [227, 17, 91], np.uint8
        )

    output, receipt = compose_face(rgb, [region(mask)], changing_sr, feather_native=0)
    base = reference(rgb)
    assert output.shape == (92, 124, 3) and output.dtype == np.uint8
    assert np.array_equal(output[~selected], base[~selected])
    assert np.all(output[selected] == [227, 17, 91])
    assert np.array_equal(
        calls[0], original
    )  # Context is native RGB, never 512 or downscaled.
    assert np.array_equal(rgb, original)
    # A separately resized bbox differs at this outside-mask corner. The
    # central hole is checked above too; neither may be overwritten by a
    # bbox-wide Lanczos image.
    wrong_base = np.array(
        Image.fromarray(rgb[5:17, 7:23]).resize((64, 48), Image.Resampling.LANCZOS)
    )
    assert not np.array_equal(wrong_base[:8, :12], base[20:28, 28:40])
    assert receipt["outside_mask_equal_reference"]
    assert receipt["one_complete_training_image"] and receipt["added_views"] == 0
    assert receipt["full_frame_4x_generated"] and receipt["mask_scope"] == "face"
    assert receipt["union_mask_native_pixels"] == int(mask.sum())
    assert (
        receipt["union_mask_native_sha256"]
        == hashlib.sha256(mask.tobytes()).hexdigest()
    )
    json.dumps(receipt, allow_nan=False)


def test_feather_changes_only_inside_support_including_hole_boundaries():
    rgb = np.full((23, 31, 3), [31, 61, 101], np.uint8)
    mask = np.zeros((23, 31), bool)
    mask[5:17, 7:23] = True
    mask[10:12, 13:15] = False
    output, _ = compose_face(
        rgb, [region(mask)], lambda crop: np.full_like(nearest4(crop), 231)
    )
    base = reference(rgb)
    assert np.array_equal(output[~nearest4(mask)], base[~nearest4(mask)])
    assert np.all(output[20, 28] > base[20, 28])
    assert np.all(output[20, 28] < 231)  # First inside pixel feathers inward.
    assert np.all(output[32, 40] == 231)  # Core farther from outer edge and hole.


@pytest.mark.parametrize("regions", [None, []])
def test_no_face_is_pure_full_lanczos_without_any_sr_call(regions):
    rgb = np.random.default_rng(9).integers(0, 256, (17, 39, 3), dtype=np.uint8)

    def forbidden(crop):
        raise AssertionError("No support must not trigger SR")

    output, receipt = compose_face(rgb, regions, forbidden)
    assert np.array_equal(output, reference(rgb))
    assert receipt["sr_calls"] == receipt["union_mask_pixels"] == 0
    assert receipt["outside_mask_equal_reference"]
    assert not receipt["input_composite_mask"]


def test_empty_native_mask_is_a_recorded_noop():
    rgb = np.full((9, 13, 3), 53, np.uint8)
    item = dict(bbox_native=[1, 2, 6, 7], mask_native=np.zeros((9, 13), bool))
    output, receipt = compose_face(rgb, [item], None)
    assert np.array_equal(output, reference(rgb))
    assert receipt["regions"][0]["state"] == "empty_mask"
    assert not receipt["regions"][0]["sr_inference_called"]


@pytest.mark.parametrize(
    "bbox,context",
    [([0, 0, 3, 7], [0, 0, 16, 16]), ([34, 16, 41, 23], [24, 8, 41, 23])],
)
def test_non_square_border_rois_keep_native_context_and_correct_full_canvas_position(
    bbox, context
):
    rgb = np.random.default_rng(37).integers(0, 256, (23, 41, 3), dtype=np.uint8)
    mask = np.zeros((23, 41), bool)
    x0, y0, x1, y1 = bbox
    mask[y0:y1, x0:x1] = True
    seen = []

    def distinct_sr(crop):
        seen.append(crop.copy())
        return nearest4(crop) ^ np.array([255, 17, 39], np.uint8)

    output, receipt = compose_face(
        rgb, [region(mask)], distinct_sr, context_native=8, feather_native=0
    )
    a, b, c, d = context
    assert np.array_equal(seen[0], rgb[b:d, a:c])
    assert receipt["regions"][0]["context_bbox_native"] == context
    assert np.array_equal(
        output[4 * y0 : 4 * y1, 4 * x0 : 4 * x1],
        nearest4(rgb[y0:y1, x0:x1]) ^ np.array([255, 17, 39], np.uint8),
    )
    assert np.array_equal(output[~nearest4(mask)], reference(rgb)[~nearest4(mask)])


def test_multiple_masks_use_union_and_first_region_wins_overlaps_without_repeated_sr():
    rgb = np.full((21, 29, 3), 51, np.uint8)
    first, second = np.zeros((21, 29), bool), np.zeros((21, 29), bool)
    first[3:12, 3:15] = True
    second[7:18, 10:25] = True
    items = [region(first), region(second), region(first.copy())]

    def run():
        count = []

        def callback(crop):
            count.append(1)
            return np.full_like(
                nearest4(crop), [201, 17, 29] if len(count) == 1 else [13, 211, 41]
            )

        return (*compose_face(rgb, items, callback, feather_native=0), len(count))

    output, receipt, calls = run()
    assert calls == receipt["sr_calls"] == 2
    assert np.all(output[nearest4(first)] == [201, 17, 29])
    assert np.all(output[nearest4(second & ~first)] == [13, 211, 41])
    assert np.array_equal(
        output[~nearest4(first | second)], reference(rgb)[~nearest4(first | second)]
    )
    assert receipt["regions"][2]["state"] == "covered_by_previous_region"
    assert receipt["regions"][2]["owned_pixels_native"] == 0
    assert np.array_equal(run()[0], output)
    assert receipt["union_mask_pixels"] == int((first | second).sum())


@pytest.mark.parametrize("mutation", ["source", "mask", "bbox", "order"])
def test_callback_mutation_of_source_or_region_evidence_is_rejected(mutation):
    rgb = np.full((19, 27, 3), 81, np.uint8)
    mask = np.zeros((19, 27), bool)
    mask[3:13, 4:17] = True
    items = [region(mask)]

    def callback(crop):
        if mutation == "source":
            rgb[0, 0] = 0
        elif mutation == "mask":
            mask[3, 4] = False
        elif mutation == "bbox":
            items[0]["bbox_native"][0] += 1
        else:
            items.append(region(mask.copy()))
        return nearest4(crop)

    with pytest.raises(RuntimeError, match="changed"):
        compose_face(rgb, items, callback)


@pytest.mark.parametrize("bad", ["grid", "dtype", "bbox", "scope"])
def test_invalid_semantic_regions_fail_before_an_sr_callback(bad):
    rgb = np.full((13, 19, 3), 52, np.uint8)
    mask = np.zeros((13, 19), bool)
    mask[2:8, 4:12] = True
    item = region(mask)
    if bad == "grid":
        item["mask_native"] = mask[:9]
    elif bad == "dtype":
        item["mask_native"] = mask.astype(np.uint8)
    elif bad == "bbox":
        item["bbox_native"][0] -= 1
    else:
        item["mask_scope"] = "head"

    def forbidden(crop):
        raise AssertionError("Invalid evidence must fail before SR")

    with pytest.raises(ValueError):
        compose_face(rgb, [item], forbidden)


def test_color_float_work_is_strip_bounded_even_if_roi_covers_the_whole_image(
    monkeypatch,
):
    rgb = np.full((257, 137, 3), [37, 81, 121], np.uint8)
    mask = np.ones(rgb.shape[:2], bool)
    shapes, original = [], compose_module.np.rint

    def observe(value):
        shapes.append(value.shape)
        return original(value)

    monkeypatch.setattr(compose_module.np, "rint", observe)
    output, receipt = compose_face(
        rgb, [region(mask)], lambda crop: np.full_like(nearest4(crop), 191)
    )
    assert output.shape == (1028, 548, 3)
    assert shapes and all(shape[0] <= 128 and shape[-1] == 3 for shape in shapes)
    assert receipt["outside_mask_equal_reference"]
