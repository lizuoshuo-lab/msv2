"""CPU geometry, source-group split and immutable SfM-track contracts."""

from copy import deepcopy
import shutil
import struct
from typing import NamedTuple

import numpy as np
from PIL import Image
import pytest

from msv2 import data
from msv2.head_dataset import (
    build_head_cameras,
    depth_sampling_capacity,
    split_camera_infos,
)
from msv2.io import save_json, sha256


@pytest.fixture
def mixed(tmp_path):
    prep, variant = tmp_path / "prepared", tmp_path / "variant"
    sparse = prep / "dataset/sparse/0"
    images = prep / "dataset/images"
    output = variant / "dataset/images"
    sparse.mkdir(parents=True)
    images.mkdir(parents=True)
    output.mkdir(parents=True)
    sizes = [(24, 18), (18, 28), (32, 20)]
    camera_ids, image_ids = [5, 7, 9], [11, 26, 50]
    camera_map, poses, rows, receipts = {}, [], [], []
    for i, (width, height) in enumerate(sizes):
        name, frame = f"source_{i}.png", f"source_{i}"
        pixels = np.random.default_rng(i).integers(
            0, 256, (height, width, 3), dtype=np.uint8
        )
        Image.fromarray(pixels).save(images / name)
        shutil.copy2(images / name, output / name)
        camera = data._Camera(
            camera_ids[i],
            0 if i == 0 else 1,
            width,
            height,
            (70.0, 12.0, 9.0)
            if i == 0
            else (71.0 + i, 91.0 + i, width / 2 + 0.25, height / 2 - 0.125),
        )
        camera_map[camera.id] = camera
        q = (2**-0.5, 0.0, 0.0, 2**-0.5) if i == 2 else (1.0, 0.0, 0.0, 0.0)
        observations = [(3.25, 4.5, 7)] + ([(8.75, 6.5, -1)] if i == 0 else [])
        payload = b"".join(struct.pack("<ddq", *v) for v in observations)
        image = data._ImagePose(
            image_ids[i],
            q,
            (i / 10, -0.1, 0.3),
            camera.id,
            name,
            len(observations),
            payload,
        )
        poses.append(image)
        rows.append(
            dict(
                name=name,
                frame=frame,
                wh=[width, height],
                K=data._camera(camera),
                w2c=data._w2c(image),
                source_sha256=sha256(images / name),
                source_index=[0, 7, 17][i],
                time_seconds=[0.0, 0.292, 0.713][i],
                colmap_image_id=image.id,
                colmap_camera_id=camera.id,
            )
        )
        boxes = (
            [[2, 3, 18, 16], [0, 0, 4, 5]]
            if i == 0
            else ([[8, 2, 30, 18]] if i == 2 else [])
        )
        heads = []
        for number, box in enumerate(boxes, 1):
            head = f"{frame}_head_{number:03d}.png"
            with Image.open(images / name) as source:
                crop = source.crop(box)
                crop.resize(
                    (crop.width * 4, crop.height * 4), Image.Resampling.NEAREST
                ).save(output / head)
            heads.append(
                dict(name=head, bbox_native=box, output_sha256=sha256(output / head))
            )
        receipts.append(
            dict(
                name=name,
                native_sha256=rows[-1]["source_sha256"],
                output_sha256=sha256(output / name),
                heads=heads,
            )
        )
    data._write_cameras(sparse / "cameras.bin", camera_map)
    data._write_images(sparse / "images.bin", poses)
    point = struct.pack("<Q", 1) + struct.pack(
        "<Q3d3BdQ", 7, 0.1, 0.2, 3.0, 12, 34, 56, 0.25, 3
    )
    point += struct.pack("<6i", 11, 0, 26, 0, 50, 0)
    (sparse / "points3D.bin").write_bytes(point)
    save_json(prep / "camera-inputs.json", dict(records=rows))
    return prep, variant, receipts, rows, camera_map, point


def wire_images(path):
    """Independent inspection of IDs and raw COLMAP observation wire bytes."""
    result = {}
    with path.open("rb") as stream:
        (count,) = struct.unpack("<Q", stream.read(8))
        for _ in range(count):
            header = struct.unpack("<i7di", stream.read(64))
            name = bytearray()
            while True:
                char = stream.read(1)
                assert char
                if char == b"\0":
                    break
                name.extend(char)
            (n,) = struct.unpack("<Q", stream.read(8))
            result[name.decode()] = (header, n, stream.read(24 * n))
        assert stream.read(1) == b""
    return result


def set_point_count(prep, count):
    """Keep the original tracked point and add distinct untracked sparse points."""
    path = prep / "dataset/sparse/0/points3D.bin"
    original = path.read_bytes()[8:]
    (track_count,) = struct.unpack("<Q", original[43:51])
    original = original[: 51 + 8 * track_count]
    extra = b"".join(
        struct.pack("<Q3d3BdQ", 8 + i, 0.1, 0.2, 3.0, 12, 34, 56, 0.25, 0)
        for i in range(count - 1)
    )
    path.write_bytes(struct.pack("<Q", count) + original + extra)
    return path.read_bytes()


def replace_heads(prep, variant, receipts, source_index, boxes):
    """Create declared lossless 4x test images from the native pixels."""
    receipt = receipts[source_index]
    heads = []
    for i, box in enumerate(boxes, 1):
        name = receipt["name"].removesuffix(".png") + f"_head_{i:03d}.png"
        output = variant / "dataset/images" / name
        with Image.open(prep / "dataset/images" / receipt["name"]) as image:
            crop = image.crop(box)
            crop.resize(
                (crop.width * 4, crop.height * 4), Image.Resampling.NEAREST
            ).save(output)
        heads.append(dict(name=name, bbox_native=box, output_sha256=sha256(output)))
    receipt["heads"] = heads


def declare_input_mask(prep, variant, receipts, source_index=0, head_index=0):
    """Declare a native binary mask with a nonrectangular interior hole."""
    head = receipts[source_index]["heads"][head_index]
    with Image.open(prep / "dataset/images" / receipts[source_index]["name"]) as image:
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
    x0, y0, x1, y1 = head["bbox_native"]
    mask[y0:y1, x0:x1] = 255
    mask[(y0 + y1) // 2, (x0 + x1) // 2] = 0
    path = variant / "head-masks" / head["name"]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(path)
    head.update(mask_name=path.name, mask_sha256=sha256(path))
    return path


def test_mixed_binary_preserves_all_full_poses_ids_observations_and_point_tracks(mixed):
    prep, variant, receipts, sources, camera_map, point = mixed
    original = {p.name: p.read_bytes() for p in (prep / "dataset/sparse/0").iterdir()}
    result = build_head_cameras(prep, variant, receipts, 3)
    sparse = variant / "dataset/sparse/0"
    assert (
        result["source_frames"],
        result["full_views"],
        result["head_views"],
        result["total_views"],
    ) == (3, 3, 3, 6)
    assert result["initial_sparse_points"] == 1
    assert (sparse / "points3D.bin").read_bytes() == point
    assert result["initial_sparse_sha256"] == sha256(
        prep / "dataset/sparse/0/points3D.bin"
    )
    old = wire_images(prep / "dataset/sparse/0/images.bin")
    new = wire_images(sparse / "images.bin")
    assert len(new) == 6
    for name, identity in old.items():
        assert new[name] == identity
    for row in result["records"]:
        if row["view_type"] == "head":
            header, n, payload = new[row["name"]]
            assert n == 0 and payload == b""
            assert header[0] > 50 and header[-1] > 9
            assert header[1:8] == old[row["source_name"]][0][1:8]
            assert row["frame"] != row["source_frame"]
    read_cameras = data._read_cameras(sparse)
    assert all(read_cameras[cid] == camera for cid, camera in camera_map.items())
    assert all(c.model == 1 for cid, c in read_cameras.items() if cid not in camera_map)
    assert original == {
        p.name: p.read_bytes() for p in (prep / "dataset/sparse/0").iterdir()
    }


def test_head_projection_and_pixel_centers_match_crop_then4x_with_rotated_pose(mixed):
    prep, variant, receipts, sources, _, _ = mixed
    result = build_head_cameras(prep, variant, receipts, 3)
    head = next(
        r
        for r in result["records"]
        if r["source_name"] == "source_2.png" and r["view_type"] == "head"
    )
    K = np.asarray(sources[2]["K"])
    np.testing.assert_array_equal(
        head["K"], [[292.0, 0.0, 33.0], [0.0, 372.0, 31.5], [0.0, 0.0, 1.0]]
    )
    assert head["wh"] == [88, 64]
    point = np.array([0.1, 0.2, 4.0])
    rotated = np.array(
        [[-0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    ) @ point + [0.2, -0.1, 0.3]
    native = K @ rotated
    native = native[:2] / native[2]
    expected = 4 * (native - [8, 2])
    camera_point = np.asarray(head["w2c"]) @ np.r_[point, 1]
    projected = np.asarray(head["K"]) @ camera_point[:3]
    np.testing.assert_allclose(projected[:2] / projected[2], expected, atol=1e-12)
    integer_projection = np.asarray(head["K_integer_centers"]) @ camera_point[:3]
    np.testing.assert_allclose(
        integer_projection[:2] / integer_projection[2],
        4 * (native - 0.5 + 0.5 - [8, 2]) - 0.5,
        atol=1e-12,
    )


def test_source_group_split_order_lineage_and_missing_detection_full_only(mixed):
    prep, variant, receipts, sources, _, _ = mixed
    result = build_head_cameras(prep, variant, list(reversed(receipts)), 3)
    rows = result["records"]
    assert [r["source_frame"] for r in rows] == ["source_0"] * 3 + ["source_1"] + [
        "source_2"
    ] * 2
    assert {r["split"] for r in rows if r["source_frame"] == "source_0"} == {"test"}
    assert {r["split"] for r in rows if r["source_frame"] == "source_2"} == {"train"}
    for row in rows:
        source = next(r for r in sources if r["frame"] == row["source_frame"])
        assert (row["source_index"], row["time_seconds"]) == (
            source["source_index"],
            source["time_seconds"],
        )
        assert row["native_image"] == str(prep / "dataset/images" / row["source_name"])
        assert row["native_sha256"] == source["source_sha256"]
    assert receipts[1]["heads"] == []
    assert len([r for r in rows if r["source_frame"] == "source_1"]) == 1


@pytest.mark.parametrize(
    "box",
    [
        [-1, 0, 3, 4],
        [0, 0, 25, 4],
        [4, 0, 4, 4],
        [0, 5, 4, 2],
        [0.0, 0, 4, 4],
        [False, 0, 4, 4],
    ],
)
def test_invalid_bbox_rejected_before_any_model_write(mixed, box):
    prep, variant, receipts, _, _, _ = mixed
    receipts[0]["heads"][0]["bbox_native"] = box
    with pytest.raises(data.DataPreparationError, match="bbox"):
        build_head_cameras(prep, variant, receipts, 3)
    assert not (variant / "dataset/sparse/0").exists()


def test_no_detected_heads_keeps_only_native_views_and_reports_no_crop(mixed):
    prep, variant, receipts, _, _, _ = mixed
    for item in receipts:
        item["heads"] = []
    result = build_head_cameras(prep, variant, receipts, 3)
    assert result["crop_used"] is False
    assert result["head_views"] == 0
    assert result["total_views"] == result["source_frames"]
    assert all(row["view_type"] == "full" for row in result["records"])
    assert all(row["image_scale"] == 1 for row in result["records"])
    assert result["generated_head_view_count"] == 0
    assert result["skipped_head_views"] == []


def test_initial_depth_guard_repeats_after_train_removal_and_checks_heldout_heads(
    mixed,
):
    prep, variant, receipts, _, camera_map, _ = mixed
    points = set_point_count(prep, 36)
    replace_heads(prep, variant, receipts, 0, [[0, 0, 1, 2]])
    replace_heads(prep, variant, receipts, 1, [[0, 0, 1, 1], [0, 0, 1, 2]])
    original_receipts = deepcopy(receipts)
    result = build_head_cameras(prep, variant, receipts, 3)
    assert receipts == original_receipts
    assert result["generated_head_view_count"] == 4
    assert result["head_views"] == 1
    assert result["initial_depth_quota"] == 12
    assert result["initial_train_view_count"] == 3
    assert result["crop_used"] is True
    skipped = result["skipped_head_views"]
    assert [(s["frame"], s["P"], s["q_at_skip"], s["split"]) for s in skipped] == [
        ("source_1_head_001", 4, 7, "train"),
        ("source_0_head_001", 8, 9, "test"),
        ("source_1_head_002", 8, 9, "train"),
    ]
    assert {s["reason"] for s in skipped} == {"below_initial_native_depth_quota"}
    assert all(s["view_type"] == "head" for s in skipped)
    assert all(
        s["native_bbox"] and s["source_name"] and s["source_frame"] for s in skipped
    )
    assert all(np.prod(s["warmup_wh"]) == s["P"] for s in skipped)
    sparse = variant / "dataset/sparse/0"
    assert (sparse / "points3D.bin").read_bytes() == points
    original = wire_images(prep / "dataset/sparse/0/images.bin")
    actual = wire_images(sparse / "images.bin")
    assert all(actual[name] == identity for name, identity in original.items())
    assert len(actual) == 4
    cameras = data._read_cameras(sparse)
    assert len(cameras) == 4
    assert all(cameras[cid] == value for cid, value in camera_map.items())
    assert all((variant / "dataset/images" / s["name"]).is_file() for s in skipped)
    rows = result["records"]
    assert len([r for r in rows if r["view_type"] == "full"]) == 3
    for source in ("source_0", "source_1", "source_2"):
        assert len({r["split"] for r in rows if r["source_frame"] == source}) == 1


def test_initial_depth_guard_falls_back_to_all_full_without_removing_full_views(mixed):
    prep, variant, receipts, _, camera_map, _ = mixed
    points = set_point_count(prep, 1000)
    for i in range(3):
        replace_heads(prep, variant, receipts, i, [[0, 0, 1, 1]])
    before = deepcopy(receipts)
    result = build_head_cameras(prep, variant, receipts, 3)
    assert receipts == before
    assert result["generated_head_view_count"] == len(result["skipped_head_views"]) == 3
    assert (
        result["head_views"] == 0 and result["full_views"] == result["total_views"] == 3
    )
    assert result["initial_depth_quota"] == 500
    assert result["initial_train_view_count"] == 2
    assert result["crop_used"] is False
    assert all(row["view_type"] == "full" for row in result["records"])
    sparse = variant / "dataset/sparse/0"
    assert data._read_cameras(sparse) == camera_map
    assert wire_images(sparse / "images.bin") == wire_images(
        prep / "dataset/sparse/0/images.bin"
    )
    assert (sparse / "points3D.bin").read_bytes() == points
    # Full views are deliberately retained even if their pixel count is too
    # small; the actual native reinitialization has its separate capacity guard.
    assert any(
        round(r["wh"][0] / 2) * round(r["wh"][1] / 2) < 500 for r in result["records"]
    )


def test_initial_depth_guard_keeps_exact_quota_head_and_uses_train_denominator(mixed):
    prep, variant, receipts, _, _, _ = mixed
    set_point_count(prep, 12)
    for i in range(3):
        replace_heads(prep, variant, receipts, i, [[0, 0, 1, 1]])
    result = build_head_cameras(prep, variant, receipts, 3)
    # Two full and two head training views: floor(12/4) == 3, not 12/6.
    assert result["initial_depth_quota"] == 3
    assert result["head_views"] == 3 and result["skipped_head_views"] == []
    second = variant.parent / "boundary"
    shutil.copytree(variant / "dataset/images", second / "dataset/images")
    set_point_count(prep, 16)
    exact = build_head_cameras(prep, second, receipts, 3)
    assert exact["initial_depth_quota"] == 4
    assert exact["head_views"] == 3 and exact["skipped_head_views"] == []


def test_initial_depth_guard_does_not_hide_bad_sha_in_a_head_that_would_be_skipped(
    mixed,
):
    prep, variant, receipts, _, _, _ = mixed
    set_point_count(prep, 1000)
    replace_heads(prep, variant, receipts, 0, [[0, 0, 1, 1]])
    receipts[0]["heads"][0]["output_sha256"] = "0" * 64
    with pytest.raises(data.DataPreparationError, match="SHA256"):
        build_head_cameras(prep, variant, receipts, 3)
    assert not (variant / "dataset/sparse/0").exists()


def test_declared_native_head_mask_is_verified_and_records_only_input_provenance(mixed):
    prep, variant, receipts, sources, _, point = mixed
    mask = declare_input_mask(prep, variant, receipts)
    before = deepcopy(receipts)
    result = build_head_cameras(prep, variant, receipts, 3)
    assert receipts == before
    row = next(
        r for r in result["records"] if r["name"] == receipts[0]["heads"][0]["name"]
    )
    assert row["input_head_mask_sha256"] == sha256(mask)
    assert row["head_mask_native_wh"] == sources[0]["wh"]
    assert row["input_head_mask_source"] == "existing_digital_human_face_parsing"
    assert "input_head_mask_sha256" not in result["records"][0]
    assert "mask_name" not in row and "mask_path" not in row
    assert result["training_mask_used"] is False and result["mask_used"] is False
    assert result["input_head_mask_used"] and result["input_composite_mask"]
    assert (variant / "dataset/sparse/0/points3D.bin").read_bytes() == point


@pytest.mark.parametrize("missing", ["mask_name", "mask_sha256"])
def test_head_mask_provenance_fields_must_be_declared_together(mixed, missing):
    prep, variant, receipts, _, _, _ = mixed
    declare_input_mask(prep, variant, receipts)
    del receipts[0]["heads"][0][missing]
    with pytest.raises(
        data.DataPreparationError, match="both mask_name and mask_sha256"
    ):
        build_head_cameras(prep, variant, receipts, 3)
    assert not (variant / "dataset/sparse/0").exists()


@pytest.mark.parametrize(
    "corruption",
    [
        "parser512",
        "head_crop",
        "rgb_mode",
        "palette_mode",
        "jpeg",
        "nonbinary",
        "empty",
        "bbox",
        "sha",
        "missing",
        "corrupt",
    ],
)
def test_head_mask_rejects_non_native_masks_or_wrong_pixel_provenance_before_writes(
    mixed, corruption
):
    prep, variant, receipts, _, _, _ = mixed
    path = declare_input_mask(prep, variant, receipts)
    head = receipts[0]["heads"][0]
    with Image.open(path) as image:
        pixels = np.array(image)
    if corruption == "parser512":
        Image.new("L", (512, 512), 255).save(path)
    elif corruption == "head_crop":
        x0, y0, x1, y1 = head["bbox_native"]
        Image.fromarray(pixels[y0:y1, x0:x1]).save(path)
    elif corruption == "rgb_mode":
        Image.fromarray(pixels).convert("RGB").save(path)
    elif corruption == "palette_mode":
        Image.fromarray(pixels).convert("P").save(path)
    elif corruption == "jpeg":
        Image.fromarray(pixels).save(path, format="JPEG")
    elif corruption == "nonbinary":
        pixels[4, 4] = 128
        Image.fromarray(pixels).save(path)
    elif corruption == "empty":
        Image.fromarray(np.zeros_like(pixels)).save(path)
    elif corruption == "bbox":
        pixels[:, head["bbox_native"][2] - 1] = 0
        Image.fromarray(pixels).save(path)
    elif corruption == "sha":
        head["mask_sha256"] = "0" * 64
    elif corruption == "missing":
        path.unlink()
    else:
        path.write_bytes(b"not an image")
    if corruption not in ("sha", "missing"):
        head["mask_sha256"] = sha256(path)
    with pytest.raises(data.DataPreparationError, match="mask"):
        build_head_cameras(prep, variant, receipts, 3)
    assert not (variant / "dataset/sparse/0").exists()


def test_capacity_skipped_head_still_requires_valid_native_input_mask(mixed):
    prep, variant, receipts, _, _, _ = mixed
    set_point_count(prep, 1000)
    replace_heads(prep, variant, receipts, 0, [[0, 0, 1, 1]])
    path = declare_input_mask(prep, variant, receipts)
    # The 1x1 helper mask is empty after its hole; all declared masks are checked
    # before quota filtering, even when this tiny head could not enter training.
    receipts[0]["heads"][0]["mask_sha256"] = sha256(path)
    with pytest.raises(data.DataPreparationError, match="nonempty"):
        build_head_cameras(prep, variant, receipts, 3)
    assert not (variant / "dataset/sparse/0").exists()


@pytest.mark.parametrize("point_header", [b"", b"short", struct.pack("<Q", 0)])
def test_initial_depth_guard_rejects_truncated_or_empty_sparse_header(
    mixed, point_header
):
    prep, variant, receipts, _, _, _ = mixed
    (prep / "dataset/sparse/0/points3D.bin").write_bytes(point_header)
    with pytest.raises(data.DataPreparationError, match="point|sparse"):
        build_head_cameras(prep, variant, receipts, 3)
    assert not (variant / "dataset/sparse/0").exists()


def test_depth_sampling_capacity_boundary_float_zero_quota_and_dynamic_point_growth():
    views = [dict(frame="full", wh=[100, 100]), dict(frame="head", wh=[2, 4])]
    before = deepcopy(views)
    boundary = depth_sampling_capacity(16, views)
    assert boundary["passed"] and boundary["quota"] == 8
    assert boundary["scope"] == "pixel_count_upper_bound_only"
    assert depth_sampling_capacity(17.999, views)["passed"]
    assert depth_sampling_capacity(1, views)["quota"] == 0
    assert views == before
    actual = depth_sampling_capacity(18.0, views)
    assert actual == dict(
        passed=False,
        quota=9,
        num_depth=18.0,
        view_count=2,
        insufficient_views=[dict(frame="head", wh=[2, 4], pixel_count=8, quota=9)],
        scope="pixel_count_upper_bound_only",
    )


@pytest.mark.parametrize(
    "num_depth", [0, -1, float("nan"), float("inf"), True, "8", None]
)
def test_depth_sampling_capacity_rejects_invalid_depth_counts(num_depth):
    with pytest.raises(data.DataPreparationError, match="positive finite"):
        depth_sampling_capacity(num_depth, [dict(frame="full", wh=[10, 10])])


@pytest.mark.parametrize(
    "views",
    [
        [],
        None,
        [None],
        [dict(frame="", wh=[10, 10])],
        [dict(frame="a", wh=[10, 10]), dict(frame="a", wh=[10, 10])],
        [dict(frame="a", wh=[0, 10])],
        [dict(frame="a", wh=[-1, 10])],
        [dict(frame="a", wh=[1.5, 10])],
        [dict(frame="a", wh=[True, 10])],
        [dict(frame="a", wh=[10])],
        [dict(frame="a")],
    ],
)
def test_depth_sampling_capacity_rejects_ambiguous_views_or_invalid_dimensions(views):
    with pytest.raises(data.DataPreparationError):
        depth_sampling_capacity(16, views)


def test_declared_missing_head_image_is_not_silently_skipped(mixed):
    prep, variant, receipts, _, _, _ = mixed
    (variant / "dataset/images" / receipts[0]["heads"][0]["name"]).unlink()
    with pytest.raises(data.DataPreparationError, match="Missing declared"):
        build_head_cameras(prep, variant, receipts, 3)


@pytest.mark.parametrize(
    "corruption", ["source_sha", "head_sha", "head_size", "native_full_pixels"]
)
def test_pixel_identity_and_true_head4_dimensions_are_required(mixed, corruption):
    prep, variant, receipts, _, _, _ = mixed
    head = receipts[0]["heads"][0]
    if corruption == "source_sha":
        receipts[0]["native_sha256"] = "0" * 64
    elif corruption == "head_sha":
        head["output_sha256"] = "0" * 64
    elif corruption == "head_size":
        path = variant / "dataset/images" / head["name"]
        Image.new("RGB", (32, 26)).save(path)
        head["output_sha256"] = sha256(path)
    else:
        path = variant / "dataset/images" / receipts[0]["name"]
        Image.new("RGB", (24, 18), "red").save(path)
        receipts[0]["output_sha256"] = sha256(path)
    with pytest.raises(data.DataPreparationError):
        build_head_cameras(prep, variant, receipts, 3)


@pytest.mark.parametrize(
    "corruption", ["missing_source", "duplicate_source", "head_collision", "traversal"]
)
def test_view_identity_and_paths_cannot_be_ambiguous(mixed, corruption):
    prep, variant, receipts, _, _, _ = mixed
    if corruption == "missing_source":
        receipts.pop()
    elif corruption == "duplicate_source":
        receipts.append(deepcopy(receipts[0]))
    elif corruption == "head_collision":
        receipts[0]["heads"][0]["name"] = receipts[1]["name"]
    else:
        receipts[0]["heads"][0]["name"] = "../outside.png"
    with pytest.raises(data.DataPreparationError):
        build_head_cameras(prep, variant, receipts, 3)


def test_repeat_identical_build_preserves_bytes_and_different_existing_model_is_not_overwritten(
    mixed,
):
    prep, variant, receipts, _, _, _ = mixed
    build_head_cameras(prep, variant, receipts, 3)
    sparse = variant / "dataset/sparse/0"
    before = {p.name: p.read_bytes() for p in sparse.iterdir()}
    build_head_cameras(prep, variant, receipts, 3)
    assert before == {p.name: p.read_bytes() for p in sparse.iterdir()}
    path = sparse / "cameras.bin"
    path.write_bytes(path.read_bytes() + b"different")
    changed = path.read_bytes()
    with pytest.raises(data.DataPreparationError, match="Existing variant"):
        build_head_cameras(prep, variant, receipts, 3)
    assert path.read_bytes() == changed
    assert not list(sparse.glob("*.tmp"))


class Info(NamedTuple):
    image_name: str
    uid: int


def test_native_camera_info_partition_uses_group_split_and_full_only_extent(mixed):
    prep, variant, receipts, _, _, _ = mixed
    rows = build_head_cameras(prep, variant, receipts, 3)["records"]
    infos = [Info(row["frame"], 100 + i) for i, row in enumerate(rows)]
    train, test, full_train = split_camera_infos(list(reversed(infos)), rows)
    assert [c.image_name for c in train] == [
        r["frame"] for r in rows if r["split"] == "train"
    ]
    assert [c.image_name for c in test] == [
        r["frame"] for r in rows if r["split"] == "test"
    ]
    assert [c.image_name for c in full_train] == ["source_1", "source_2"]
    assert all(c in train for c in full_train)


@pytest.mark.parametrize(
    "corruption",
    [
        "duplicate_info",
        "missing_info",
        "extra_info",
        "group_leak",
        "no_full",
        "duplicate_row",
    ],
)
def test_native_partition_rejects_leaks_missing_or_duplicated_views(mixed, corruption):
    prep, variant, receipts, _, _, _ = mixed
    rows = build_head_cameras(prep, variant, receipts, 3)["records"]
    infos = [Info(row["frame"], i) for i, row in enumerate(rows)]
    if corruption == "duplicate_info":
        infos.append(infos[0])
    elif corruption == "missing_info":
        infos.pop()
    elif corruption == "extra_info":
        infos.append(Info("not_registered", 999))
    elif corruption == "group_leak":
        rows[1]["split"] = "train"
    elif corruption == "no_full":
        rows[3]["view_type"] = "head"
    else:
        rows.append(dict(rows[0]))
    with pytest.raises(data.DataPreparationError):
        split_camera_infos(infos, rows)
