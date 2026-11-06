"""Compose face-only SR into one complete original-image Lanczos4 canvas.

The semantic mask changes input pixels only. It does not add a camera/view,
enter a training loss, resize the RGB supplied to SR, or change a renderer.
"""

import hashlib
import math

import numpy as np
from PIL import Image

from .head import _integer_box, _mask_alpha, _rgb, superresolve_head


def _sha(array):
    # No full-frame RGB float conversion or additional bytes copy for hashing.
    return hashlib.sha256(memoryview(np.ascontiguousarray(array))).hexdigest()


def _regions(rgb, regions):
    if regions is None:
        regions = []
    if not isinstance(regions, (list, tuple)):
        raise ValueError("regions must be an ordered list of native face masks")
    height, width = rgb.shape[:2]
    prepared = []
    for region in regions:
        if not isinstance(region, dict) or region.get("mask_scope", "face") != "face":
            raise ValueError(
                "compose_face requires face masks, not hair/hat head masks"
            )
        x0, y0, x1, y1 = bbox = _integer_box(region["bbox_native"])
        if x1 > width or y1 > height:
            raise ValueError("Face bbox exceeds the native source image")
        mask = region["mask_native"]
        if (
            not isinstance(mask, np.ndarray)
            or mask.dtype != bool
            or mask.shape != (height, width)
        ):
            raise ValueError(
                "Face mask must be boolean on the complete native RGB grid"
            )
        ys, xs = np.nonzero(mask)
        if len(xs) and [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ] != list(bbox):
            raise ValueError(
                "Face bbox must equal the native mask's tight exclusive bounds"
            )
        prepared.append(
            dict(
                region=region,
                mask_ref=mask,
                mask_sha256=_sha(mask),
                bbox=bbox,
                scope=region.get("mask_scope", "face"),
            )
        )
    return regions, prepared


def _unchanged(rgb, source_sha, regions, prepared):
    if _sha(rgb) != source_sha:
        raise RuntimeError("Face SR changed the original full RGB")
    if len(regions) != len(prepared) or any(
        region is not item["region"] for region, item in zip(regions, prepared)
    ):
        raise RuntimeError("Face SR changed the ordered input regions")
    for item in prepared:
        region = item["region"]
        if (
            region.get("mask_native") is not item["mask_ref"]
            or tuple(region.get("bbox_native", ())) != item["bbox"]
            or region.get("mask_scope", "face") != item["scope"]
            or _sha(item["mask_ref"]) != item["mask_sha256"]
        ):
            raise RuntimeError("Face SR changed the original mask or bbox")


def _outside_equal(output, reference, native_union):
    # Checking four rows per native row avoids a full-size boolean4 canvas.
    for row in range(native_union.shape[0]):
        outside = ~np.repeat(native_union[row], 4)
        if not np.array_equal(
            output[4 * row : 4 * row + 4, outside],
            reference[4 * row : 4 * row + 4, outside],
        ):
            return False
    return True


def compose_face(rgb, regions, callback, *, context_native=32, feather_native=2):
    """Return one complete full4 RGB; outside the face union is exactly base.

    Regions use native exclusive bounds and full-native boolean masks. The
    input sequence is deterministic priority order: the first mask owns every
    overlap pixel, and a completely covered later mask does not call SR.
    Feathering is inward on each original mask, never a dilation. Color float
    temporaries cover at most one ROI strip, not the complete image.
    """
    rgb = _rgb(rgb)
    if type(context_native) is not int or context_native < 0:
        raise ValueError("context_native must be a nonnegative integer")
    if (
        isinstance(feather_native, bool)
        or not isinstance(feather_native, (int, float))
        or not math.isfinite(feather_native)
        or feather_native < 0
    ):
        raise ValueError("feather_native must be finite and nonnegative")
    regions, prepared = _regions(rgb, regions)
    source_sha = _sha(rgb)
    height, width = rgb.shape[:2]
    reference = np.array(
        Image.fromarray(rgb).resize((width * 4, height * 4), Image.Resampling.LANCZOS)
    )
    output = reference.copy()
    union = np.zeros((height, width), bool)
    records, calls = [], 0
    for number, item in enumerate(prepared, 1):
        _unchanged(rgb, source_sha, regions, prepared)
        x0, y0, x1, y1 = item["bbox"]
        mask = item["mask_ref"][y0:y1, x0:x1].copy()
        owned = mask & ~union[y0:y1, x0:x1]
        record = dict(
            region_number=number,
            bbox_native=list(item["bbox"]),
            mask_scope="face",
            mask_native_sha256=item["mask_sha256"],
            mask_pixels_native=int(mask.sum()),
            owned_pixels_native=int(owned.sum()),
            context_bbox_native=None,
            sr_inference_called=False,
            sr_call_index=None,
            sr_receipt=None,
        )
        if not owned.any():
            record["state"] = (
                "empty_mask" if not mask.any() else "covered_by_previous_region"
            )
            records.append(record)
            continue
        if not callable(callback):
            raise ValueError("A face with supported pixels requires an SR callback")
        prediction, sr_receipt = superresolve_head(
            rgb, item["bbox"], callback, context_native=context_native, mask_native=None
        )
        _unchanged(rgb, source_sha, regions, prepared)
        mask4, alpha = _mask_alpha(mask, feather_native)
        owned4 = np.repeat(np.repeat(owned, 4, axis=0), 4, axis=1)
        target = output[4 * y0 : 4 * y1, 4 * x0 : 4 * x1]
        base = reference[4 * y0 : 4 * y1, 4 * x0 : 4 * x1]
        # Blend against the complete-image Lanczos reference, never against a
        # separately resized bbox or a previously composed overlapping region.
        for start in range(0, target.shape[0], 128):
            stop = min(start + 128, target.shape[0])
            support = owned4[start:stop]
            a = alpha[start:stop, :, None]
            blended = np.rint(
                base[start:stop].astype(np.float32) * (1 - a)
                + prediction[start:stop].astype(np.float32) * a
            ).astype(np.uint8)
            target[start:stop][support] = blended[support]
        union[y0:y1, x0:x1] |= mask
        calls += 1
        record.update(
            state="composed",
            context_bbox_native=sr_receipt["context_bbox_native"],
            sr_inference_called=True,
            sr_call_index=calls,
            sr_receipt=sr_receipt,
            feather_native=feather_native,
            mask_pixels_4x=int(mask4.sum()),
        )
        records.append(record)
        del prediction, alpha, mask4, owned4
    _unchanged(rgb, source_sha, regions, prepared)
    outside_equal = _outside_equal(output, reference, union)
    if not outside_equal:
        raise RuntimeError("Face SR changed a pixel outside the native face-mask union")
    union_sha, union_pixels = _sha(union), int(union.sum())
    output_sha = _sha(output)
    return output, dict(
        schema="msv2-one-full-image-face-sr-v1",
        mask_scope="face",
        source_wh=[width, height],
        output_wh=[width * 4, height * 4],
        scale=4,
        source_rgb_sha256=source_sha,
        base_full4_rgb_sha256=_sha(reference),
        output_rgb_sha256=output_sha,
        full_rgb_sha256=output_sha,
        union_mask_sha256=union_sha,
        union_mask_pixels=union_pixels,
        union_mask_native_sha256=union_sha,
        union_mask_native_pixels=union_pixels,
        union_mask_sha256_encoding="native HxW bool C-order bytes",
        union_mask_4x_pixels=union_pixels * 16,
        outside_mask_4x_pixels=(width * height - union_pixels) * 16,
        region_count=len(prepared),
        sr_calls=calls,
        regions=records,
        overlap_policy="input region order; first region owns each native mask pixel, no repeated blend",
        reference_method="complete original RGB PIL Lanczos4, not separately resized bbox RGB",
        feather_native=feather_native,
        feather_direction="inside each face mask only; zero alpha outside",
        no_source_downsampling=True,
        source_full_rgb_unchanged=True,
        one_complete_training_image=True,
        training_views_per_source=1,
        added_views=0,
        no_added_views=True,
        full_frame_4x_generated=True,
        outside_mask_equal_reference=outside_equal,
        training_mask_used=False,
        input_composite_mask=bool(union_pixels),
        quality_accepted=False,
    )
