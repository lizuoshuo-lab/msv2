"""Paired inputs preserve native parents and supervise only true ROI support."""

import hashlib
import json

import numpy as np
from PIL import Image, PngImagePlugin
import pytest

import msv2.paired_data as paired
from msv2.paired_data import (
    child_camera,
    child_roi,
    compose_parent_child,
    copy_parent_png,
)


def face(mask):
    yy, xx = np.nonzero(mask)
    return dict(
        mask_scope="face",
        mask_native=mask,
        bbox_native=[
            int(xx.min()),
            int(yy.min()),
            int(xx.max()) + 1,
            int(yy.max()) + 1,
        ],
    )


def full_reference(rgb):
    return np.array(
        Image.fromarray(rgb).resize(
            (rgb.shape[1] * 4, rgb.shape[0] * 4), Image.Resampling.LANCZOS
        )
    )


def reference_child(rgb, roi):
    x0, y0, x1, y1 = roi
    valid = full_reference(rgb)[
        y0 * 4 : min(y1, rgb.shape[0]) * 4, x0 * 4 : min(x1, rgb.shape[1]) * 4
    ]
    return np.pad(
        valid,
        (
            (0, max(0, y1 - rgb.shape[0]) * 4),
            (0, max(0, x1 - rgb.shape[1]) * 4),
            (0, 0),
        ),
        mode="edge",
    )


def test_parent_copy_keeps_png_original_bytes_and_ancillary_metadata(tmp_path):
    rgb = np.random.default_rng(4).integers(0, 256, (17, 29, 3), np.uint8)
    info = PngImagePlugin.PngInfo()
    info.add_text("capture", "untouched original metadata")
    source = tmp_path / "native.png"
    output = tmp_path / "parents/frame.png"
    Image.fromarray(rgb).save(source, pnginfo=info, compress_level=1)
    original = source.read_bytes()
    receipt = copy_parent_png(source, output)
    assert source.read_bytes() == output.read_bytes() == original
    assert receipt["source_png_sha256"] == hashlib.sha256(original).hexdigest()
    assert receipt["source_rgb_sha256"] == hashlib.sha256(rgb.tobytes()).hexdigest()
    assert receipt["wh"] == [29, 17] and receipt["source_bytes_preserved"]
    assert receipt["resized"] is receipt["reencoded"] is False
    with pytest.raises(FileExistsError):
        copy_parent_png(source, output)
    with pytest.raises(ValueError):
        copy_parent_png(source, source)


def test_parent_rejects_reencoded_mode_before_creating_output(tmp_path):
    source = tmp_path / "alpha.png"
    output = tmp_path / "copy.png"
    Image.new("RGBA", (5, 7)).save(source)
    with pytest.raises(ValueError, match="original RGB PNG"):
        copy_parent_png(source, output)
    assert not output.exists()


def test_child_is_only_roi_and_uses_raw_context_with_holes_and_no_global4canvas(
    monkeypatch,
):
    rgb = np.random.default_rng(32).integers(0, 256, (97, 129, 3), np.uint8)
    before = rgb.copy()
    mask = np.zeros(rgb.shape[:2], bool)
    mask[35:56, 51:72] = True
    mask[42:46, 58:63] = False
    mask[35:38, 51:54] = False
    region = face(mask)
    seen = []
    assert paired._box_resize_is_exact()
    original_resize = Image.Image.resize
    requested = []

    def record_resize(self, size, *args, **kwargs):
        requested.append(tuple(size))
        return original_resize(self, size, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "resize", record_resize)

    def sr(crop):
        seen.append(crop.copy())
        return np.full(
            (crop.shape[0] * 4, crop.shape[1] * 4, 3), [233, 17, 139], np.uint8
        )

    children, receipt = compose_parent_child(rgb, [region], sr, feather_native=0)
    assert len(children) == receipt["sr_calls"] == 1
    child = children[0]
    roi = child["roi_native"]
    assert roi == [40, 24, 80, 64]
    assert child["rgb"].shape == (160, 160, 3) and child["mask"].shape == (160, 160)
    assert tuple([129 * 4, 97 * 4]) not in requested
    np.testing.assert_array_equal(seen[0], before[0:88, 16:104])
    base = reference_child(before, roi)
    np.testing.assert_array_equal(child["rgb"][~child["mask"]], base[~child["mask"]])
    assert np.all(child["rgb"][child["mask"]] == [233, 17, 139])
    assert int(child["mask"].sum()) == int(mask.sum()) * 16
    np.testing.assert_array_equal(rgb, before)
    p = child["provenance"]
    assert p["sr_receipt"]["semantic_mask_used"] is False
    assert (
        p["sr_receipt"]["context_rgb_sha256"]
        == hashlib.sha256(seen[0].tobytes()).hexdigest()
    )
    assert p["outside_mask_equal_reference"] and p["child_loss_mask_required"]
    assert not p["registered_to_colmap"] and not p["ordinary_image_sampling"]
    assert receipt["sampling_group_count"] == 1 and receipt["parent_scale"] == 1
    assert receipt["parent_rgb_unchanged"] and not receipt["full_frame_4x_saved"]
    assert (
        not receipt["full_frame_4x_cached"]
        and not receipt["temporary_full4_reference_generated"]
    )
    json.dumps(receipt, allow_nan=False)
    json.dumps(p, allow_nan=False)


@pytest.mark.parametrize(
    "wh,bbox,expected",
    [
        ([29, 17], [25, 12, 29, 17], [16, 0, 32, 24]),
        ([129, 97], [51, 35, 72, 56], [40, 24, 80, 64]),
        ([29, 17], [0, 0, 3, 5], [0, 0, 16, 16]),
    ],
)
def test_roi_outward8_alignment_and_odd_physical_bounds(wh, bbox, expected):
    assert child_roi(bbox, wh) == expected
    assert all(v % 8 == 0 for v in expected)


@pytest.mark.parametrize("bbox", [[25, 12, 29, 17], [0, 0, 3, 5], [0, 13, 4, 17]])
def test_edge_padding_is_never_supervised_or_sent_as_fake_context(bbox):
    rgb = np.random.default_rng(121).integers(0, 256, (17, 29, 3), np.uint8)
    mask = np.zeros((17, 29), bool)
    x0, y0, x1, y1 = bbox
    mask[y0:y1, x0:x1] = True
    seen = []

    def sr(raw):
        seen.append(raw.copy())
        return np.repeat(np.repeat(raw, 4, 0), 4, 1)

    children, _ = compose_parent_child(rgb, [face(mask)], sr, feather_native=0)
    c = children[0]
    a, b, d, e = c["roi_native"]
    assert c["rgb"].shape == ((e - b) * 4, (d - a) * 4, 3)
    assert c["rgb"].shape[0] % 32 == c["rgb"].shape[1] % 32 == 0
    assert not c["mask"][max(0, (17 - b) * 4) :].any()
    assert not c["mask"][:, max(0, (29 - a) * 4) :].any()
    np.testing.assert_array_equal(seen[0], rgb)
    base = reference_child(rgb, c["roi_native"])
    np.testing.assert_array_equal(c["rgb"][~c["mask"]], base[~c["mask"]])


def test_inside_only_feather_preserves_holes_and_true_background():
    rgb = np.full((49, 67, 3), [31, 61, 101], np.uint8)
    mask = np.zeros((49, 67), bool)
    mask[15:35, 23:43] = True
    mask[24:26, 31:33] = False
    children, _ = compose_parent_child(
        rgb,
        [face(mask)],
        lambda raw: np.full((raw.shape[0] * 4, raw.shape[1] * 4, 3), 231, np.uint8),
    )
    c = children[0]
    base = reference_child(rgb, c["roi_native"])
    np.testing.assert_array_equal(c["rgb"][~c["mask"]], base[~c["mask"]])
    x0, y0, _, _ = c["roi_native"]
    edge = c["rgb"][(15 - y0) * 4, (23 - x0) * 4]
    core = c["rgb"][(19 - y0) * 4, (27 - x0) * 4]
    assert np.all(edge > [31, 61, 101]) and np.all(edge < 231)
    assert np.all(core == 231)


@pytest.mark.parametrize("regions", [None, []])
def test_no_face_leaves_only_the_native_sampling_group_and_no_sr(regions):
    rgb = np.zeros((17, 29, 3), np.uint8)

    def forbidden(raw):
        raise AssertionError("No face must not call SR")

    children, receipt = compose_parent_child(rgb, regions, forbidden)
    assert children == [] and receipt["child_count"] == receipt["sr_calls"] == 0
    assert (
        receipt["sampling_group_count"] == 1
        and receipt["union_native_mask_pixels"] == 0
    )


def test_empty_masks_and_fully_overlapped_faces_do_not_duplicate_sr_or_loss():
    rgb = np.full((65, 97, 3), 43, np.uint8)
    m1 = np.zeros((65, 97), bool)
    m1[15:36, 20:41] = True
    m2 = np.zeros((65, 97), bool)
    m2[28:48, 33:53] = True
    empty = dict(bbox_native=[1, 1, 2, 2], mask_native=np.zeros_like(m1))
    calls = []

    def sr(raw):
        calls.append(raw.shape)
        return np.full(
            (raw.shape[0] * 4, raw.shape[1] * 4, 3), 100 + len(calls) * 20, np.uint8
        )

    children, receipt = compose_parent_child(
        rgb, [empty, face(m1), face(m2), face(m1)], sr, feather_native=0
    )
    assert len(children) == len(calls) == 2
    assert [r["state"] for r in receipt["regions"]] == [
        "empty_mask",
        "child_generated",
        "child_generated",
        "covered_by_previous_region",
    ]
    assert sum(int(c["mask"].sum()) for c in children) == int((m1 | m2).sum()) * 16
    mapped = np.zeros((65 * 4, 97 * 4), np.uint8)
    for c in children:
        x0, y0, x1, y1 = c["roi_native"]
        validw = min(x1, 97) - x0
        validh = min(y1, 65) - y0
        mapped[y0 * 4 : (y0 + validh) * 4, x0 * 4 : (x0 + validw) * 4] += c["mask"][
            : validh * 4, : validw * 4
        ]
        base = reference_child(rgb, c["roi_native"])
        np.testing.assert_array_equal(c["rgb"][~c["mask"]], base[~c["mask"]])
    assert mapped.max() == 1
    np.testing.assert_array_equal(mapped, np.repeat(np.repeat(m1 | m2, 4, 0), 4, 1))


def test_pillow_reference_fallback_is_transient_and_identical(monkeypatch):
    rgb = np.random.default_rng(82).integers(0, 256, (41, 67, 3), np.uint8)
    mask = np.zeros((41, 67), bool)
    mask[15:26, 25:38] = True
    monkeypatch.setattr(paired, "_box_resize_is_exact", lambda: False)
    c, r = compose_parent_child(
        rgb, [face(mask)], lambda raw: np.repeat(np.repeat(raw, 4, 0), 4, 1)
    )
    assert r["temporary_full4_reference_generated"] and not r["full_frame_4x_cached"]
    base = reference_child(rgb, c[0]["roi_native"])
    np.testing.assert_array_equal(c[0]["rgb"][~c[0]["mask"]], base[~c[0]["mask"]])


@pytest.mark.parametrize("change", ["rgb", "mask", "bbox", "sequence"])
def test_callback_cannot_mutate_native_rgb_or_region_identity(change):
    rgb = np.zeros((33, 49, 3), np.uint8)
    mask = np.zeros((33, 49), bool)
    mask[12:20, 18:28] = True
    item = face(mask)
    regions = [item]

    def bad(raw):
        if change == "rgb":
            rgb[0, 0] = 1
        elif change == "mask":
            mask[0, 0] = True
        elif change == "bbox":
            item["bbox_native"][0] += 1
        else:
            regions.reverse()
            regions.append(item)
        return np.zeros((raw.shape[0] * 4, raw.shape[1] * 4, 3), np.uint8)

    with pytest.raises(RuntimeError, match="changed"):
        compose_parent_child(rgb, regions, bad)


def test_child_camera_projects_points_with_full_fov_and_warmup2_mapping():
    K = np.array([[117.0, 0, 65.2], [0, 119.0, 48.7], [0, 0, 1.0]])
    pose = np.eye(4)
    pose[:3, 3] = [0.2, -0.1, 2.0]
    roi = [40, 24, 80, 64]
    full = child_camera(K, pose, [129, 97], roi)
    half = child_camera(K, pose, [129, 97], roi, warmup_factor=2)
    xyz = np.array([[0.1, -0.2, 1.0], [-0.3, 0.1, 2.0], [0.15, 0.4, 3.0]]).T
    uv = K @ xyz
    uv = uv[:2] / uv[2:]
    cropped = np.asarray(full["K"]) @ xyz
    cropped = cropped[:2] / cropped[2:]
    np.testing.assert_allclose(
        cropped, 4 * (uv - np.asarray(roi[:2])[:, None]), rtol=0, atol=1e-12
    )
    np.testing.assert_array_equal(
        half["K"], np.diag([0.5, 0.5, 1]) @ np.asarray(full["K"])
    )
    assert full["full_wh"] == [516, 388] and half["full_wh"] == [258, 194]
    assert full["viewport"] == [160, 96, 320, 256] and half["viewport"] == [
        80,
        48,
        160,
        128,
    ]
    assert half["wh"] == [80, 80] and half["warmup_tile16_aligned"]
    assert full["w2c"] == half["w2c"] == pose.tolist()
    assert full["full_tan_fovx"] == half["full_tan_fovx"] == 129 / (2 * 117)
    assert full["requires_full_fov_roi_kernel"] and not full["geometry_parity_verified"]
    assert half["_msv2_roi"] == dict(
        full_wh=[258, 194], viewport_xy=[80, 48], wh=[80, 80]
    )
    assert full["full_tan_fovx"] != full["wh"][0] / (2 * np.asarray(full["K"])[0, 0])
    json.dumps(full, allow_nan=False)
    json.dumps(half, allow_nan=False)


def test_warmup_odd_parent_actual_wh_never_assumes_child_to_parent_ratio4():
    K = np.array([[117.0, 0, 65.2], [0, 119.0, 48.7], [0, 0, 1.0]])
    c = child_camera(
        K,
        np.eye(4),
        [129, 97],
        [40, 24, 80, 64],
        warmup_factor=2,
        current_parent_wh=[64, 48],
        current_full_wh=[258, 194],
    )
    assert c["full_original_wh"] == [516, 388] and c["full_current_wh"] == [258, 194]
    assert c["current_parent_wh"] == [64, 48]
    assert c["full_to_current_parent_scale"] == [258 / 64, 194 / 48]
    assert c["full_to_current_parent_scale"] != [4, 4]
    np.testing.assert_allclose(
        c["current_parent_K"], np.diag([64 / 129, 48 / 97, 1]) @ K
    )
    assert c["physical_bounds_native"] == [0, 0, 129, 97]
    assert c["physical_bounds_current_full"] == [0, 0, 258, 194]
    with pytest.raises(ValueError, match="noninteger"):
        child_camera(
            K, np.eye(4), [129, 97], [40, 24, 80, 64], current_full_wh=[257, 194]
        )
    with pytest.raises(ValueError, match="tile16"):
        child_camera(K, np.eye(4), [128, 96], [8, 8, 24, 24], current_full_wh=[128, 96])


def test_warmup_nearest_mask_preserves_holes_and_padding_without_new_support():
    rgb = np.zeros((17, 29, 3), np.uint8)
    m = np.zeros((17, 29), bool)
    m[10:17, 23:29] = True
    m[12:14, 25:27] = False
    children, _ = compose_parent_child(
        rgb, [face(m)], lambda raw: np.repeat(np.repeat(raw, 4, 0), 4, 1)
    )
    c = children[0]
    mask2 = np.array(
        Image.fromarray(c["mask"]).resize(
            (c["mask"].shape[1] // 2, c["mask"].shape[0] // 2), Image.Resampling.NEAREST
        )
    )
    x0, y0, x1, y1 = c["roi_native"]
    native = np.zeros((y1 - y0, x1 - x0), bool)
    native[: 17 - y0, : 29 - x0] = m[y0:17, x0:29]
    np.testing.assert_array_equal(mask2, np.repeat(np.repeat(native, 2, 0), 2, 1))
    assert mask2.shape[0] % 16 == mask2.shape[1] % 16 == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"context_native": 0},
        {"roi_context_native": 7},
        {"feather_native": -1},
        {"feather_native": float("nan")},
    ],
)
def test_invalid_preprocessing_parameters_fail_before_sr(kwargs):
    with pytest.raises(ValueError):
        compose_parent_child(np.zeros((17, 29, 3), np.uint8), [], None, **kwargs)


@pytest.mark.parametrize("roi", [[1, 0, 16, 16], [0, 0, 40, 24], [0, 24, 32, 32]])
def test_invalid_camera_viewport_cannot_silently_round_or_change_physical_bounds(roi):
    with pytest.raises(ValueError):
        child_camera(np.eye(3), np.eye(4), [29, 17], roi)


def test_invalid_face_scope_tight_bounds_and_callback_dtype():
    rgb = np.zeros((17, 29, 3), np.uint8)
    mask = np.zeros((17, 29), bool)
    mask[5:8, 8:12] = True
    region = face(mask)
    region["mask_scope"] = "head"
    with pytest.raises(ValueError, match="face masks"):
        compose_parent_child(rgb, [region], None)
    region = face(mask)
    region["bbox_native"][0] -= 1
    with pytest.raises(ValueError, match="tight"):
        compose_parent_child(rgb, [region], None)
    with pytest.raises(ValueError, match="exact context4"):
        compose_parent_child(
            rgb,
            [face(mask)],
            lambda raw: np.zeros((raw.shape[0] * 4, raw.shape[1] * 4, 3), np.float32),
        )
