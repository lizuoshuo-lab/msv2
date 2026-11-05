"""Mixed native full-image and 4x head-view cameras without new SfM tracks."""

from dataclasses import replace
import hashlib
import math
from numbers import Integral, Real
from pathlib import Path
import struct
import uuid

import numpy as np
from PIL import Image

from .data import (
    DataPreparationError,
    _Camera,
    _camera,
    _read_cameras,
    _read_images,
    _relative_name,
    _w2c,
    _write_cameras,
    _write_images,
    _write_points,
)
from .io import read_json, sha256


def _require(value, message):
    if not value:
        raise DataPreparationError(message)


def depth_sampling_capacity(num_depth, views):
    """Check only the pixel-count upper bound of uniform native depth sampling.

    Native sampling draws without replacement from each view. Passing this
    necessary check does not establish that its probability map has enough
    nonzero entries. ``wh`` must be the actual sampling-resolution dimensions.
    """
    _require(
        isinstance(num_depth, Real)
        and not isinstance(num_depth, bool)
        and (isinstance(num_depth, Integral) or math.isfinite(num_depth))
        and num_depth > 0,
        "num_depth must be a positive finite number",
    )
    _require(
        isinstance(views, (list, tuple)) and bool(views),
        "Depth sampling needs a nonempty list of views",
    )
    num_depth = int(num_depth) if isinstance(num_depth, Integral) else float(num_depth)
    quota = (
        num_depth // len(views)
        if isinstance(num_depth, int)
        else math.floor(num_depth / len(views))
    )
    seen, insufficient = set(), []
    for view in views:
        _require(isinstance(view, dict), "Depth sampling view must be a mapping")
        frame, wh = view.get("frame"), view.get("wh")
        _require(
            isinstance(frame, str) and bool(frame) and frame not in seen,
            "Depth sampling frame names must be nonempty and unique",
        )
        seen.add(frame)
        _require(
            isinstance(wh, (list, tuple))
            and len(wh) == 2
            and all(
                isinstance(v, Integral) and not isinstance(v, bool) and v > 0
                for v in wh
            ),
            "Depth sampling dimensions must be two positive integers",
        )
        wh = [int(v) for v in wh]
        pixels = wh[0] * wh[1]
        if pixels < quota:
            insufficient.append(
                dict(frame=frame, wh=wh, pixel_count=pixels, quota=quota)
            )
    return dict(
        passed=not insufficient,
        quota=quota,
        num_depth=num_depth,
        view_count=len(views),
        insufficient_views=insufficient,
        scope="pixel_count_upper_bound_only",
    )


def _head_capacity_filter(rows, point_count):
    """Keep all full views and screen heads to a stable initial depth quota."""
    accepted, skipped = list(rows), []
    while True:
        training = [
            dict(frame=row["frame"], wh=[round(v / 2) for v in row["wh"]])
            for row in accepted
            if row["split"] == "train"
        ]
        capacity = depth_sampling_capacity(point_count, training)
        quota = capacity["quota"]
        rejected = set()
        for row in accepted:
            if row["view_type"] != "head":
                continue
            wh = [round(v / 2) for v in row["wh"]]
            pixels = wh[0] * wh[1]
            if pixels < quota:
                rejected.add(row["frame"])
                skipped.append(
                    dict(
                        name=row["name"],
                        frame=row["frame"],
                        source_frame=row["source_frame"],
                        source_name=row["source_name"],
                        view_type="head",
                        split=row["split"],
                        native_bbox=list(row["native_bbox"]),
                        warmup_wh=wh,
                        P=pixels,
                        q_at_skip=quota,
                        reason="below_initial_native_depth_quota",
                    )
                )
        if not rejected:
            return accepted, skipped, quota, len(training)
        accepted = [row for row in accepted if row["frame"] not in rejected]


def _image(path, expected_sha, wh):
    _require(path.is_file(), "Missing declared dataset image: " + str(path))
    _require(sha256(path) == expected_sha, "Image SHA256 mismatch: " + str(path))
    with Image.open(path) as image:
        _require(
            image.format == "PNG"
            and image.mode == "RGB"
            and list(image.size) == list(wh),
            "Head/full image must retain the declared lossless RGB PNG dimensions: "
            + str(path),
        )


def _payload_sha(image):
    if isinstance(image.payload, bytes):
        return hashlib.sha256(image.payload).hexdigest()
    digest = hashlib.sha256()
    with image.payload.path.open("rb") as stream:
        stream.seek(image.payload.offset)
        left = image.payload.length
        while left:
            block = stream.read(min(left, 1024**2))
            _require(bool(block), "Truncated original image observations")
            digest.update(block)
            left -= len(block)
    return digest.hexdigest()


def _image_identity(image):
    return (
        image.id,
        image.qvec,
        image.tvec,
        image.camera_id,
        image.name,
        image.observations,
        _payload_sha(image),
    )


def _bbox(value, wh):
    _require(
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(type(v) is int for v in value),
        "Head bbox must contain four integer exclusive pixel corners",
    )
    x0, y0, x1, y1 = value
    _require(
        0 <= x0 < x1 <= wh[0] and 0 <= y0 < y1 <= wh[1],
        "Head bbox is empty or outside its original full image",
    )
    return [x0, y0, x1, y1]


def _head_input_mask(head, variant, native_wh, bbox):
    """Verify an optional native parsing mask as input provenance, not a loss."""
    declared = [key in head for key in ("mask_name", "mask_sha256")]
    if not any(declared):
        return {}
    _require(all(declared), "Head input mask needs both mask_name and mask_sha256")
    _require(
        isinstance(head["mask_name"], str)
        and bool(head["mask_name"])
        and isinstance(head["mask_sha256"], str)
        and bool(head["mask_sha256"]),
        "Head input mask needs a nonempty relative name and SHA256",
    )
    root = (variant / "head-masks").resolve()
    path = (root / _relative_name(head["mask_name"])).resolve()
    _require(
        root.is_relative_to(variant) and path.is_relative_to(root),
        "Head input mask path escaped the variant",
    )
    _require(path.is_file(), "Missing declared head input mask: " + str(path))
    _require(sha256(path) == head["mask_sha256"], "Head input mask SHA256 mismatch")
    try:
        with Image.open(path) as image:
            _require(
                image.format == "PNG"
                and image.mode == "L"
                and list(image.size) == list(native_wh),
                "Head input mask must be a native full-size L PNG",
            )
            pixels = np.asarray(image)
            _require(
                bool(np.all((pixels == 0) | (pixels == 255))),
                "Head input mask must contain only binary 0 and 255 pixels",
            )
            actual_bbox = image.getbbox()
            _require(actual_bbox is not None, "Head input mask must be nonempty")
            _require(
                list(actual_bbox) == bbox,
                "Head input mask tight bbox differs from its declared native bbox",
            )
    except (OSError, ValueError) as error:
        raise DataPreparationError(
            "Cannot decode native head input mask: " + str(path)
        ) from error
    return dict(
        input_head_mask_sha256=head["mask_sha256"],
        head_mask_native_wh=list(native_wh),
        input_head_mask_source="existing_digital_human_face_parsing",
    )


def _record(source, image, camera, prep, variant, native_sha, bbox, split, kind):
    matrix = np.asarray(_camera(camera), dtype=np.float64)
    integer_centers = matrix.copy()
    integer_centers[:2, 2] -= 0.5
    result = dict(source)
    result.update(
        name=image.name,
        frame=Path(image.name).stem,
        source_frame=source["frame"],
        source_name=source["name"],
        source_group=source["frame"],
        view_type=kind,
        native_bbox=bbox,
        native_image=str(prep / "dataset/images" / source["name"]),
        native_sha256=native_sha,
        image=str(variant / "dataset/images" / image.name),
        source_sha256=sha256(variant / "dataset/images" / image.name),
        wh=[camera.width, camera.height],
        K=matrix.tolist(),
        K_integer_centers=integer_centers.tolist(),
        w2c=_w2c(image),
        colmap_image_id=image.id,
        colmap_camera_id=image.camera_id,
        split=split,
        image_scale=1 if kind == "full" else 4,
        added_SfM_observations=False,
    )
    return result


def build_head_cameras(prep_dir, variant_dir, sr_records, holdout):
    """Build full1x/head4x COLMAP cameras and return training-input metadata.

    The caller prepares all image files and writes the returned JSON. Full PNG
    bytes, original image/camera IDs, poses and observation payloads stay intact.
    Head views get new IDs, the same pose and zero SfM observations. Source-group
    holdout is defined by the preparation manifest order, never mixed-view order.
    """
    prep, variant = Path(prep_dir).resolve(), Path(variant_dir).resolve()
    original = prep / "dataset/sparse/0"
    sparse = (variant / "dataset/sparse/0").resolve()
    _require(sparse != original.resolve(), "Do not overwrite prepared SfM inputs")
    _require(sparse.is_relative_to(variant), "Variant sparse path escaped its root")
    _require(type(holdout) is int and holdout >= 2, "holdout must be at least 2")
    sources = read_json(prep / "camera-inputs.json")["records"]
    _require(bool(sources), "Preparation has no original source frames")
    _require(isinstance(sr_records, (list, tuple)), "Expected SR source records")
    names = [row["name"] for row in sources]
    frames = [row["frame"] for row in sources]
    _require(
        len(set(names)) == len(names) and len(set(frames)) == len(frames),
        "Original source names and frames must be unique",
    )
    receipts = {}
    for item in sr_records:
        name = _relative_name(item["name"])
        _require(name not in receipts, "Duplicate SR source record: " + name)
        receipts[name] = item
    _require(set(receipts) == set(names), "Every original frame needs one SR record")
    cameras, images = _read_cameras(original), _read_images(original)
    original_camera_ids = set(cameras)
    by_name = {image.name: image for image in images}
    _require(set(by_name) == set(names), "Prepared image names disagree with manifest")
    _require((original / "points3D.bin").is_file(), "Binary sparse points are required")
    with (original / "points3D.bin").open("rb") as stream:
        header = stream.read(8)
    _require(len(header) == 8, "Truncated COLMAP sparse-point count")
    (initial_points,) = struct.unpack("<Q", header)
    _require(initial_points > 0, "COLMAP model contains no sparse points")
    next_image = max(image.id for image in images) + 1
    next_camera = max(cameras) + 1
    reserved_names = {name.casefold() for name in names}
    reserved_frames = {frame.casefold() for frame in frames}
    _require(len(reserved_names) == len(names), "Case-colliding source names")
    _require(len(reserved_frames) == len(frames), "Case-colliding source frames")
    mixed, output_images = [], []
    for index, source in enumerate(sources):
        name = _relative_name(source["name"])
        _require(source["frame"] == Path(name).stem, "Original frame/name mismatch")
        image = by_name[name]
        _require(image.camera_id in cameras, "Missing original camera")
        camera = cameras[image.camera_id]
        native_wh = [camera.width, camera.height]
        _require(source["wh"] == native_wh, "Original camera/image size mismatch")
        _require(
            np.allclose(source["K"], _camera(camera), rtol=0, atol=1e-10)
            and np.allclose(source["w2c"], _w2c(image), rtol=0, atol=1e-10),
            "Original manifest camera differs from its SfM binary",
        )
        for key, value in (
            ("colmap_image_id", image.id),
            ("colmap_camera_id", image.camera_id),
        ):
            _require(source.get(key, value) == value, "Original COLMAP ID mismatch")
        item = receipts[name]
        native_sha = source["source_sha256"]
        _require(item["native_sha256"] == native_sha, "SR source SHA256 mismatch")
        _require(
            item["output_sha256"] == native_sha,
            "Full image must be an unchanged native PNG copy, not a resized composite",
        )
        _image(prep / "dataset/images" / name, native_sha, native_wh)
        _image(variant / "dataset/images" / name, native_sha, native_wh)
        split = "test" if index % holdout == 0 else "train"
        mixed.append(
            _record(
                source,
                image,
                camera,
                prep,
                variant,
                native_sha,
                [0, 0, *native_wh],
                split,
                "full",
            )
        )
        output_images.append(image)
        heads = item.get("heads", [])
        _require(
            isinstance(heads, list), "heads must be a list; use [] for no detection"
        )
        for head in heads:
            head_name = _relative_name(head["name"])
            _require(
                head_name.casefold() not in reserved_names, "Duplicate head image name"
            )
            head_frame = Path(head_name).stem.casefold()
            _require(head_frame not in reserved_frames, "Duplicate head frame identity")
            reserved_names.add(head_name.casefold())
            reserved_frames.add(head_frame)
            bbox = _bbox(head["bbox_native"], native_wh)
            x0, y0, x1, y1 = bbox
            wh = [4 * (x1 - x0), 4 * (y1 - y0)]
            _image(variant / "dataset/images" / head_name, head["output_sha256"], wh)
            mask_provenance = _head_input_mask(head, variant, native_wh, bbox)
            crop = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]])
            matrix = np.diag([4.0, 4.0, 1.0]) @ crop @ np.asarray(_camera(camera))
            head_camera = _Camera(
                next_camera,
                1,
                *wh,
                (matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]),
            )
            head_image = replace(
                image,
                id=next_image,
                name=head_name,
                camera_id=next_camera,
                observations=0,
                payload=b"",
            )
            cameras[next_camera] = head_camera
            next_camera += 1
            next_image += 1
            mixed.append(
                _record(
                    source,
                    head_image,
                    head_camera,
                    prep,
                    variant,
                    native_sha,
                    bbox,
                    split,
                    "head",
                )
            )
            mixed[-1].update(mask_provenance)
            output_images.append(head_image)
    # Every declared image has been checked, including heads that may now be
    # rejected. A removed train head raises the per-view quota, so iterate; apply
    # the same final rule to held-out heads without removing any full image.
    generated_heads = len(mixed) - len(sources)
    mixed, skipped_heads, initial_quota, train_count = _head_capacity_filter(
        mixed, initial_points
    )
    accepted_ids = {row["colmap_image_id"] for row in mixed}
    output_images = [image for image in output_images if image.id in accepted_ids]
    accepted_cameras = {image.camera_id for image in output_images}
    cameras = {
        key: value
        for key, value in cameras.items()
        if key in accepted_cameras or key in original_camera_ids
    }
    # Validate all inputs before writing the variant model. Temporary files live
    # under this exact variant; readback completes before any final replacement.
    sparse.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary = {
        name: sparse / ("." + name + "." + token + ".tmp")
        for name in ("cameras.bin", "images.bin", "points3D.bin")
    }
    try:
        _write_cameras(temporary["cameras.bin"], cameras)
        _write_images(temporary["images.bin"], output_images)
        point_count = _write_points(original, temporary["points3D.bin"])
        _require(point_count == initial_points, "Initial sparse-point count changed")

        # Private readers use standard filenames, so read back staged files by
        # temporarily exposing a tiny path view, without copying model payloads.
        class Staged:
            def __truediv__(self, name):
                return temporary.get(name, sparse / (".__absent__" + name))

        checked_cameras = _read_cameras(Staged())
        checked_images = _read_images(Staged())
        _require(checked_cameras == cameras, "Mixed camera binary readback mismatch")
        _require(
            [_image_identity(i) for i in checked_images]
            == [_image_identity(i) for i in output_images],
            "Original observations or mixed image identities changed on readback",
        )
        point_sha = sha256(original / "points3D.bin")
        _require(
            sha256(temporary["points3D.bin"]) == point_sha,
            "Original sparse point/track bytes changed",
        )
        for name, staged in temporary.items():
            target = sparse / name
            if target.exists():
                _require(
                    sha256(target) == sha256(staged), "Existing variant SfM differs"
                )
        for name, staged in temporary.items():
            staged.replace(sparse / name)
    finally:
        for staged in temporary.values():
            staged.unlink(missing_ok=True)
    return dict(
        records=mixed,
        source_frames=len(sources),
        full_views=len(sources),
        head_views=len(mixed) - len(sources),
        generated_head_view_count=generated_heads,
        skipped_head_views=skipped_heads,
        initial_depth_quota=initial_quota,
        initial_train_view_count=train_count,
        initial_depth_capacity_scope="pixel_count_upper_bound_only",
        total_views=len(mixed),
        source_group_holdout=holdout,
        sr_region="head",
        scale=4,
        full_image_scale=1,
        head_image_scale=4,
        mask_used=False,
        training_mask_used=False,
        input_head_mask_used=len(mixed) > len(sources),
        input_composite_mask=len(mixed) > len(sources),
        crop_used=len(mixed) > len(sources),
        background_trained=True,
        full_native_PNG_bytes_unchanged=True,
        initial_sparse_points=point_count,
        initial_sparse_sha256=point_sha,
        camera_poses_sha256=sha256(sparse / "images.bin"),
        original_full_poses_and_observations_unchanged=True,
        derived_head_SfM_observations=0,
        scaled_intrinsics_readback_verified=True,
    )


def split_camera_infos(camera_infos, rows):
    """Repartition native CameraInfo objects by source-group split and row order."""
    names = [info.image_name for info in camera_infos]
    frames = [row["frame"] for row in rows]
    _require(len(names) == len(set(names)), "Duplicate native camera-info names")
    _require(len(frames) == len(set(frames)), "Duplicate mixed row frames")
    _require(set(names) == set(frames), "Camera-info/row names must match exactly")
    lookup = dict(zip(names, camera_infos))
    groups = {}
    train, test, full_train = [], [], []
    for row in rows:
        split, kind, source = row["split"], row["view_type"], row["source_frame"]
        _require(split in ("train", "test"), "Invalid source-group split")
        _require(kind in ("full", "head"), "Invalid mixed view type")
        group = groups.setdefault(source, dict(split=split, full=0))
        _require(group["split"] == split, "A source group leaked across train/test")
        if kind == "full":
            _require(row["frame"] == source, "Full-view/source frame mismatch")
            group["full"] += 1
        info = lookup[row["frame"]]
        (train if split == "train" else test).append(info)
        if split == "train" and kind == "full":
            full_train.append(info)
    _require(
        all(g["full"] == 1 for g in groups.values()), "Each group needs one full view"
    )
    _require(
        bool(full_train),
        "Native full training views are needed for scene normalization",
    )
    return train, test, full_train
