"""Native parent images and masked4x auxiliary children, without full4 inputs.

Children are not COLMAP images or independent sampling groups. Their mask is
required by the paired loss. Camera metadata retains the parent's full FoV;
ordinary crop rendering cannot substitute for the matching ROI rasterizer.
"""

from functools import lru_cache
import hashlib
import math
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, __version__ as pillow_version

from .head import _integer_box, _mask_alpha, _rgb, crop_intrinsics, superresolve_head
from .io import sha256


def _sha(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array))).hexdigest()


def _wh(wh):
    a = np.asarray(wh)
    if a.shape != (2,) or not np.issubdtype(a.dtype, np.integer) or np.any(a < 1):
        raise ValueError("Expected positive integer native width and height")
    return tuple(map(int, a))


def copy_parent_png(source, destination):
    """Copy the original PNG bytes, never an RGB re-encoding; refuse overwrite."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError("Parent output must be separate from its native source")
    if destination.exists():
        raise FileExistsError("Parent output already exists: " + str(destination))
    before = sha256(source)
    with Image.open(source) as image:
        if image.format != "PNG" or image.mode != "RGB":
            raise ValueError("The native parent must be an original RGB PNG")
        wh = list(image.size)
        rgb_sha = _sha(np.asarray(image))
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against a concurrent output appearing.
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)
    if sha256(source) != before or sha256(destination) != before:
        raise RuntimeError("Native parent changed during its byte-preserving copy")
    return dict(
        source_png_sha256=before,
        output_png_sha256=before,
        source_rgb_sha256=rgb_sha,
        wh=wh,
        source_bytes_preserved=True,
        resized=False,
        reencoded=False,
    )


def child_roi(bbox, source_wh, *, roi_context_native=8):
    """Expand to an8-grid; right/bottom padding preserves odd-source alignment.

    Only the padded child viewport can extend beyond the image, by at most7
    native pixels. These padded pixels must have a false loss mask. SR itself
    always receives natural, clipped native pixels from superresolve_head.
    """
    width, height = _wh(source_wh)
    x0, y0, x1, y1 = _integer_box(bbox)
    if x1 > width or y1 > height:
        raise ValueError("Face bbox exceeds its native parent")
    if type(roi_context_native) is not int or roi_context_native < 8:
        raise ValueError("A child requires at least8 native context pixels")
    return [
        max(0, ((x0 - roi_context_native) // 8) * 8),
        max(0, ((y0 - roi_context_native) // 8) * 8),
        math.ceil(min(width, x1 + roi_context_native) / 8) * 8,
        math.ceil(min(height, y1 + roi_context_native) / 8) * 8,
    ]


@lru_cache(maxsize=1)
def _box_resize_is_exact():
    """Check this Pillow implementation using full-input box filtering.

    Unlike crop-then-resize, Image.resize(box=...) can retain the Lanczos
    filter's source context. A version failing calibration uses a temporary
    full4 reference, immediately cropped and released, never cached or saved.
    """
    rng = np.random.default_rng(270419)
    for height, width, box in (
        (23, 31, (0, 0, 16, 16)),
        (23, 31, (8, 8, 24, 23)),
        (41, 67, (8, 8, 56, 40)),
        (17, 39, (32, 0, 39, 17)),
    ):
        image = Image.fromarray(rng.integers(0, 256, (height, width, 3), np.uint8))
        x0, y0, x1, y1 = box
        direct = image.resize(
            (4 * (x1 - x0), 4 * (y1 - y0)), Image.Resampling.LANCZOS, box=box
        )
        full = image.resize((4 * width, 4 * height), Image.Resampling.LANCZOS)
        cropped = full.crop(tuple(4 * v for v in box))
        equal = np.array_equal(np.asarray(direct), np.asarray(cropped))
        image.close()
        direct.close()
        cropped.close()
        full.close()
        if not equal:
            return False
    return True


def _reference_roi(rgb, roi):
    width, height = rgb.shape[1], rgb.shape[0]
    x0, y0, x1, y1 = roi
    valid = [x0, y0, min(x1, width), min(y1, height)]
    a, b, c, d = valid
    fallback = not _box_resize_is_exact()
    with Image.fromarray(rgb) as image:
        if fallback:
            full = image.resize((width * 4, height * 4), Image.Resampling.LANCZOS)
            try:
                resized = full.crop(tuple(v * 4 for v in valid))
            finally:
                full.close()
        else:
            resized = image.resize(
                ((c - a) * 4, (d - b) * 4), Image.Resampling.LANCZOS, box=valid
            )
        try:
            pixels = np.array(resized)
        finally:
            resized.close()
    pad_right, pad_bottom = (x1 - c) * 4, (y1 - d) * 4
    if pad_right or pad_bottom:
        pixels = np.pad(pixels, ((0, pad_bottom), (0, pad_right), (0, 0)), mode="edge")
    return pixels, valid, fallback


def _prepare(rgb, regions):
    if regions is None:
        regions = []
    if not isinstance(regions, (list, tuple)):
        raise ValueError("Regions must be an ordered native face-mask sequence")
    height, width = rgb.shape[:2]
    prepared = []
    for region in regions:
        if not isinstance(region, dict) or region.get("mask_scope", "face") != "face":
            raise ValueError("Paired children require2D face masks")
        bbox = _integer_box(region["bbox_native"])
        if bbox[2] > width or bbox[3] > height:
            raise ValueError("Face bbox exceeds its native parent")
        mask = region["mask_native"]
        if (
            not isinstance(mask, np.ndarray)
            or mask.dtype != bool
            or mask.shape != (height, width)
        ):
            raise ValueError("Face mask must use the complete native bool grid")
        ys, xs = np.nonzero(mask)
        if len(xs) and [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ] != list(bbox):
            raise ValueError("Face bbox must be its mask's tight exclusive bounds")
        prepared.append(dict(region=region, bbox=bbox, mask=mask, mask_sha=_sha(mask)))
    return regions, prepared


def _unchanged(rgb, source_sha, regions, prepared):
    if _sha(rgb) != source_sha:
        raise RuntimeError("Paired preprocessing changed native parent RGB")
    if len(regions) != len(prepared) or any(
        r is not p["region"] for r, p in zip(regions, prepared)
    ):
        raise RuntimeError("Paired preprocessing changed ordered face regions")
    for p in prepared:
        r = p["region"]
        if (
            r.get("mask_native") is not p["mask"]
            or tuple(r.get("bbox_native", ())) != p["bbox"]
            or r.get("mask_scope", "face") != "face"
            or _sha(p["mask"]) != p["mask_sha"]
        ):
            raise RuntimeError("Paired preprocessing changed a face bbox or mask")


def compose_parent_child(
    rgb, regions, callback, *, context_native=32, roi_context_native=8, feather_native=2
):
    """Return children and numeric provenance; parent RGB is never replaced.

    Each child contains ``rgb`` (ROI4 uint8), ``mask`` (ROI4 bool),
    ``roi_native`` (exclusive8-grid bounds), and ``provenance``. Only mask
    pixels participate in a paired loss. Children stay outside COLMAP and the
    image-sampling pool. Input order owns overlap pixels exactly once.
    """
    rgb = _rgb(rgb)
    if type(context_native) is not int or context_native < 8:
        raise ValueError("SR requires at least8 native context pixels")
    if type(roi_context_native) is not int or roi_context_native < 8:
        raise ValueError("Child ROI requires at least8 native context pixels")
    if (
        isinstance(feather_native, bool)
        or not isinstance(feather_native, (int, float))
        or not math.isfinite(feather_native)
        or feather_native < 0
    ):
        raise ValueError("feather_native must be finite and nonnegative")
    regions, prepared = _prepare(rgb, regions)
    source_sha = _sha(rgb)
    height, width = rgb.shape[:2]
    owned = np.zeros((height, width), bool)
    children, records = [], []
    for number, item in enumerate(prepared, 1):
        _unchanged(rgb, source_sha, regions, prepared)
        mask = item["mask"]
        support = mask & ~owned
        count = int(support.sum())
        if not count:
            records.append(
                dict(
                    region_number=number,
                    state="empty_mask"
                    if not mask.any()
                    else "covered_by_previous_region",
                    child_index=None,
                    sr_inference_called=False,
                    mask_pixels_native=int(mask.sum()),
                    owned_pixels_native=0,
                )
            )
            continue
        if not callable(callback):
            raise ValueError("Supported face pixels require an SR callback")
        prediction, sr = superresolve_head(
            rgb, item["bbox"], callback, context_native=context_native, mask_native=None
        )
        _unchanged(rgb, source_sha, regions, prepared)
        roi = child_roi(
            item["bbox"], [width, height], roi_context_native=roi_context_native
        )
        base, valid, fallback = _reference_roi(rgb, roi)
        output = base.copy()
        rx0, ry0, rx1, ry1 = roi
        x0, y0, x1, y1 = item["bbox"]
        mask4 = np.zeros(output.shape[:2], bool)
        full_mask4, alpha = _mask_alpha(mask[y0:y1, x0:x1], feather_native)
        support4 = np.repeat(np.repeat(support[y0:y1, x0:x1], 4, 0), 4, 1)
        a, b, c, d = (4 * (x0 - rx0), 4 * (y0 - ry0), 4 * (x1 - rx0), 4 * (y1 - ry0))
        target, reference = output[b:d, a:c], base[b:d, a:c]
        for top in range(0, target.shape[0], 128):
            bottom = min(top + 128, target.shape[0])
            blend = np.rint(
                reference[top:bottom].astype(np.float32)
                * (1 - alpha[top:bottom, :, None])
                + prediction[top:bottom].astype(np.float32) * alpha[top:bottom, :, None]
            ).astype(np.uint8)
            target[top:bottom][support4[top:bottom]] = blend[support4[top:bottom]]
        mask4[b:d, a:c] = support4
        if not np.array_equal(output[~mask4], base[~mask4]):
            raise RuntimeError("Child changed pixels outside its loss mask")
        provenance = dict(
            region_number=number,
            bbox_native=list(item["bbox"]),
            roi_native=roi,
            valid_roi_native=valid,
            source_wh=[width, height],
            output_wh=[output.shape[1], output.shape[0]],
            source_rgb_sha256=source_sha,
            native_mask_sha256=item["mask_sha"],
            mask_pixels_native=int(mask.sum()),
            owned_pixels_native=count,
            mask_pixels_4x=int(mask4.sum()),
            rgb_sha256=_sha(output),
            mask_sha256=_sha(mask4),
            base_rgb_sha256=_sha(base),
            sr_receipt=sr,
            sr_inference_called=True,
            context_native=context_native,
            roi_context_native=roi_context_native,
            feather_native=feather_native,
            padding_native=[0, 0, rx1 - valid[2], ry1 - valid[3]],
            padding_rgb="edge extension of valid complete-image Lanczos4 reference; loss mask false",
            outside_mask_equal_reference=True,
            mask_scope="face",
            child_loss_mask_required=True,
            registered_to_colmap=False,
            ordinary_image_sampling=False,
            reference_method="PIL full-source Lanczos4 box filter"
            if not fallback
            else "temporary full-source Lanczos4 then crop",
            temporary_full4_reference_generated=fallback,
            full_frame_4x_cached=False,
            full_frame_4x_saved=False,
            full_frame_4x_training_image_generated=False,
            native_roi_grid=8,
            warmup_factor2_tile16_aligned=True,
        )
        children.append(
            dict(
                rgb=output,
                mask=mask4,
                roi_native=roi,
                bbox_native=list(item["bbox"]),
                provenance=provenance,
            )
        )
        records.append(
            dict(
                region_number=number,
                state="child_generated",
                child_index=len(children) - 1,
                sr_inference_called=True,
                mask_pixels_native=int(mask.sum()),
                owned_pixels_native=count,
            )
        )
        owned |= mask
        del prediction, full_mask4, alpha, support4, base
    _unchanged(rgb, source_sha, regions, prepared)
    receipt = dict(
        schema="msv2-native-parent-masked4x-children-v1",
        mask_scope="face",
        source_wh=[width, height],
        source_rgb_sha256=source_sha,
        parent_rgb_unchanged=True,
        parent_scale=1,
        child_scale=4,
        child_count=len(children),
        sr_calls=len(children),
        regions=records,
        union_native_mask_sha256=_sha(owned),
        union_native_mask_pixels=int(owned.sum()),
        overlap_policy="input order; first face owns overlapping loss pixels",
        sampling_group_count=1,
        children_are_auxiliary=True,
        children_registered_to_colmap=False,
        children_in_ordinary_image_sampling=False,
        child_loss_mask_required=True,
        no_source_downsampling=True,
        full_frame_4x_cached=False,
        full_frame_4x_saved=False,
        full_frame_4x_training_image_generated=False,
        temporary_full4_reference_generated=any(
            c["provenance"]["temporary_full4_reference_generated"] for c in children
        ),
        pillow_version=pillow_version,
        quality_accepted=False,
    )
    return children, receipt


def child_camera(
    K,
    w2c,
    source_wh,
    roi_native,
    *,
    scale=4,
    warmup_factor=1,
    current_full_wh=None,
    current_parent_wh=None,
):
    """Keep full-parent FoV and exact ROI viewport at training/warmup scale.

    This is a data contract for a matching full-FoV ROI kernel, not proof that
    an ordinary crop camera has equivalent Gaussian covariance/statistics.
    """
    width, height = _wh(source_wh)
    roi = _integer_box(roi_native)
    if (
        any(v % 8 for v in roi)
        or roi[0] >= width
        or roi[1] >= height
        or roi[2] > math.ceil(width / 8) * 8
        or roi[3] > math.ceil(height / 8) * 8
    ):
        raise ValueError("Child viewport must use valid outward8-grid bounds")
    if (
        scale != 4
        or type(scale) is not int
        or warmup_factor not in (1, 2)
        or type(warmup_factor) is not int
    ):
        raise ValueError("Expected child scale4 and warmup factor1 or2")
    native_K = np.asarray(K, dtype=np.float64)
    child_K = crop_intrinsics(native_K, roi, scale=scale)
    transform = np.asarray(w2c, dtype=np.float64)
    if (
        transform.shape != (4, 4)
        or not np.isfinite(transform).all()
        or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-12, rtol=0)
    ):
        raise ValueError("Expected a finite world-to-camera4x4 matrix")
    factor = scale // warmup_factor
    full_wh = _wh(
        [width * factor, height * factor]
        if current_full_wh is None
        else current_full_wh
    )
    parent_wh = _wh(
        [max(1, round(width / warmup_factor)), max(1, round(height / warmup_factor))]
        if current_parent_wh is None
        else current_parent_wh
    )
    sx, sy = full_wh[0] / width, full_wh[1] / height
    exact_viewport = [sx * roi[0], sy * roi[1], sx * roi[2], sy * roi[3]]
    viewport = [round(v) for v in exact_viewport]
    if not np.allclose(viewport, exact_viewport, rtol=0, atol=1e-9):
        raise ValueError("Actual full WH maps the native ROI to noninteger edges")
    roi_wh = [viewport[2] - viewport[0], viewport[3] - viewport[1]]
    if any(v % 16 for v in (*viewport[:2], *roi_wh)):
        raise ValueError("Current ROI origin and WH must remain tile16 aligned")
    full_K = np.diag([sx, sy, 1.0]) @ native_K
    child_K = np.array([[1, 0, -viewport[0]], [0, 1, -viewport[1]], [0, 0, 1]]) @ full_K
    parent_K = np.diag([parent_wh[0] / width, parent_wh[1] / height, 1.0]) @ native_K
    integer_K = child_K.copy()
    integer_K[:2, 2] -= 0.5
    return dict(
        K=child_K.tolist(),
        K_integer_centers=integer_K.tolist(),
        w2c=transform.tolist(),
        wh=roi_wh,
        source_wh=[width, height],
        roi_native=list(roi),
        scale=scale,
        warmup_factor=warmup_factor,
        full_K=full_K.tolist(),
        full_wh=list(full_wh),
        full_original_wh=[width * scale, height * scale],
        full_current_wh=list(full_wh),
        native_parent_wh=[width, height],
        current_parent_wh=list(parent_wh),
        current_parent_K=parent_K.tolist(),
        full_to_current_parent_scale=[
            full_wh[0] / parent_wh[0],
            full_wh[1] / parent_wh[1],
        ],
        full_to_native_scale=[sx, sy],
        physical_bounds_native=[0, 0, width, height],
        physical_bounds_current_full=[0, 0, *full_wh],
        _msv2_roi=dict(full_wh=list(full_wh), viewport_xy=viewport[:2], wh=roi_wh),
        viewport=viewport,
        viewport_xy=viewport[:2],
        full_tan_fovx=float(width / (2 * native_K[0, 0])),
        full_tan_fovy=float(height / (2 * native_K[1, 1])),
        full_fovx=float(2 * math.atan(width / (2 * native_K[0, 0]))),
        full_fovy=float(2 * math.atan(height / (2 * native_K[1, 1]))),
        full_fov_source="native parent; never inferred from child dimensions",
        requires_full_fov_roi_kernel=True,
        projection_policy="Inherit parent's FoV/world/full_proj; crop K is calibration metadata only",
        geometry_parity_verified=False,
        warmup_tile16_aligned=(viewport[2] - viewport[0]) % 16 == 0
        and (viewport[3] - viewport[1]) % 16 == 0,
    )
