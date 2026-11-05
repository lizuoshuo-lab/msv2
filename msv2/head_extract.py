"""Independent CPU head-pixel extraction; original image pixels remain intact."""

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from .io import now, read_json, save_json, sha256


SCHEMA = "msv2-head-pixel-extraction-v1"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def array_sha(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def generated_directory(path):
    manifest = path / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        return read_json(manifest).get("schema") == SCHEMA
    except (ValueError, AttributeError):
        return False


def image_sources(source):
    source = Path(source).expanduser().resolve()
    if not source.is_dir():
        raise ValueError("Head extraction requires an image or dataset directory")
    if generated_directory(source):
        raise ValueError("Head extraction output cannot be used as source images")
    candidates = [source / "images", source / "dataset/images"]
    candidates = [path for path in candidates if path.is_dir()]
    if len(candidates) > 1:
        raise ValueError("Choose the intended images directory explicitly")
    images = candidates[0] if candidates else source
    files = []

    def visit(folder):
        if generated_directory(folder):
            return
        for path in sorted(folder.iterdir(), key=lambda p: p.name):
            if path.is_dir() and not path.is_symlink():
                visit(path)
            elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                files.append(path)

    visit(images)
    if not files:
        raise ValueError("No source images found in " + str(images))
    return source, images, files


def native_region(region, shape):
    mask = np.asarray(region["mask_native"])
    if mask.dtype != np.bool_ or mask.shape != shape:
        raise ValueError("Head mask must be boolean in the complete native image size")
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("An extracted head region must contain native pixels")
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    supplied = region["bbox_native"]
    if (
        not isinstance(supplied, (list, tuple))
        or any(type(x) is not int for x in supplied)
        or list(supplied) != bbox
    ):
        raise ValueError("Head bbox must match the native mask's exclusive bounds")
    return mask, bbox


def extract_head(
    source,
    output,
    *,
    parser_source=None,
    parser_weight=None,
    detector_weight=None,
    extractor=None,
):
    source, images, paths = image_sources(source)
    supplied_output = Path(output).expanduser()
    if supplied_output.exists() or supplied_output.is_symlink():
        raise FileExistsError("Head extraction requires a new output directory")
    output = supplied_output.resolve()
    if output.is_relative_to(source) or output.is_relative_to(images.resolve()):
        raise ValueError("Head extraction output must be outside the source directory")
    sources = [(path, sha256(path)) for path in paths]
    if extractor is None:
        from .head import HeadPixelExtractor

        extractor = HeadPixelExtractor(parser_source, parser_weight, detector_weight)
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "manifest.json"
    manifest = dict(
        schema=SCHEMA,
        state="running",
        created=now(),
        source=str(source),
        image_directory=str(images),
        source_images=len(sources),
        method="existing_digital_human_face_parsing",
        assets=extractor.assets,
        device="cpu",
        original_RGB_resized=False,
        background_color_filled=False,
        mask_resolution="complete native source dimensions; binary0/255",
        cutout_resolution="native tight bbox; RGB unchanged, binary alpha",
        bbox_convention="integer pixel-corner edges, upper exclusive",
        source_files_unchanged=False,
        training_started=False,
        GPU_used=False,
        records=[],
    )
    save_json(manifest_path, manifest)
    try:
        for path, expected in sources:
            if sha256(path) != expected:
                raise RuntimeError("Source image changed: " + str(path))
            with Image.open(path) as image:
                rgb = np.array(image.convert("RGB"), copy=True)
            rgb_sha = array_sha(rgb)
            regions, metadata = extractor.regions(rgb)
            if array_sha(rgb) != rgb_sha or sha256(path) != expected:
                raise RuntimeError("Head extraction changed source RGB: " + str(path))
            relative = path.relative_to(images)
            folder = output / "frames" / relative
            folder.mkdir(parents=True, exist_ok=False)
            union = np.zeros(rgb.shape[:2], dtype=bool)
            heads = []
            for index, region in enumerate(regions, 1):
                mask, bbox = native_region(region, rgb.shape[:2])
                union |= mask
                x0, y0, x1, y1 = bbox
                crop = rgb[y0:y1, x0:x1]
                rgba = np.dstack((crop, mask[y0:y1, x0:x1].astype(np.uint8) * 255))
                cutout = folder / f"head_{index:03d}.png"
                Image.fromarray(rgba).save(cutout)
                heads.append(
                    dict(
                        head_number=index,
                        bbox_native=bbox,
                        mask_pixels=int(mask.sum()),
                        native_mask_array_sha256=array_sha(mask),
                        cutout_file=cutout.relative_to(output).as_posix(),
                        cutout_sha256=sha256(cutout),
                        cutout_wh=[x1 - x0, y1 - y0],
                        source_crop_rgb_sha256=array_sha(crop),
                        RGB_bitwise_original=True,
                    )
                )
            mask_file = folder / "mask.png"
            Image.fromarray(union.astype(np.uint8) * 255).save(mask_file)
            ys, xs = np.nonzero(union)
            mask_bbox = (
                [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                if len(xs)
                else None
            )
            manifest["records"].append(
                dict(
                    source_id=relative.as_posix(),
                    source_image=str(path),
                    source_file_sha256=expected,
                    source_png_sha256=expected
                    if path.suffix.lower() == ".png"
                    else None,
                    source_rgb_sha256=rgb_sha,
                    source_wh=[rgb.shape[1], rgb.shape[0]],
                    source_shape=list(rgb.shape),
                    state="head_pixels_found" if heads else "no_head",
                    head_count=len(heads),
                    heads=heads,
                    mask_file=mask_file.relative_to(output).as_posix(),
                    mask_sha256=sha256(mask_file),
                    mask_array_sha256=array_sha(union),
                    mask_pixels=int(union.sum()),
                    mask_bbox=mask_bbox,
                    method_metadata=metadata,
                )
            )
        if any(sha256(path) != expected for path, expected in sources):
            raise RuntimeError("Source images changed during head extraction")
        manifest.update(
            state="completed",
            finished=now(),
            source_files_unchanged=True,
            heads=sum(row["head_count"] for row in manifest["records"]),
            no_head_images=sum(
                row["state"] == "no_head" for row in manifest["records"]
            ),
        )
    except Exception as error:
        manifest.update(state="failed", finished=now(), error=str(error))
        save_json(manifest_path, manifest)
        raise
    save_json(manifest_path, manifest)
    return dict(
        state="completed",
        source_images=len(sources),
        heads=manifest["heads"],
        no_head_images=manifest["no_head_images"],
        output=str(output),
        manifest=str(manifest_path),
        device="cpu",
        training_started=False,
    )
