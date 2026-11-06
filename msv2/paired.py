"""One native source camera with jointly supervised high-resolution face ROIs.

The pinned Gaussian model and optimizer remain upstream implementations. This
adapter changes the render domains and the photometric loss aggregation only.
Children are never inserted into COLMAP or the random camera-sampling list.
"""

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from .io import sha256

LAYOUT = "parent1x_mask_child4x_joint"


def route(used=True):
    return dict(
        sr_region="face",
        mask_scope="face",
        mask_source="existing_digital_human_face_parsing",
        training_mask_used=used,
        input_composite_mask=used,
        input_face_mask_used=used,
        preprocessing_crop_used=used,
        view_layout=LAYOUT,
        additional_training_views=False,
        full_frame_4x_generated=False,
        one_complete_training_image_per_source=True,
        parent_scale=1,
        child_scale=4,
        joint_source_update=True,
        outside_mask_reference="Original native 1x RGB",
    )


def save_inputs(run, row, rgb, regions, children, composition, extraction, calls):
    from .paired_data import copy_parent_png

    run = Path(run)
    parent = run / "dataset/images" / row["image"]
    native = Path(row["native_image"])
    if parent.exists():
        assert sha256(parent) == row["sha256"]
    else:
        copy_parent_png(native, parent)
    assert sha256(parent) == row["sha256"]
    stem = Path(row["image"]).stem
    faces, saved_children = [], []

    def png(relative, array):
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        Image.fromarray(array).save(tmp, format="PNG", compress_level=3)
        tmp.replace(path)
        return str(relative), sha256(path)

    for i, region in enumerate(regions, 1):
        mask = region["mask_native"]
        if not mask.any():
            continue
        name, digest = png(
            Path("face-masks") / f"{stem}_face_{i:03d}.png", mask.astype(np.uint8) * 255
        )
        faces.append(
            dict(
                mask_name=Path(name).name,
                mask_sha256=digest,
                bbox_native=list(region["bbox_native"]),
                mask_pixels=int(mask.sum()),
            )
        )
    for i, child in enumerate(children, 1):
        name, digest = png(Path("children") / f"{stem}_child_{i:03d}.png", child["rgb"])
        mask_name, mask_digest = png(
            Path("children") / f"{stem}_child_{i:03d}_mask.png",
            child["mask"].astype(np.uint8) * 255,
        )
        saved_children.append(
            dict(
                image=name,
                image_sha256=digest,
                mask=mask_name,
                mask_sha256=mask_digest,
                roi_native=list(child["roi_native"]),
                wh=[child["rgb"].shape[1], child["rgb"].shape[0]],
                mask_pixels=int(child["mask"].sum()),
                provenance=child["provenance"],
            )
        )
    return dict(
        name=row["image"],
        native_sha256=row["sha256"],
        output_sha256=sha256(parent),
        face_count=len(faces),
        faces=faces,
        children=saved_children,
        face_extraction=extraction,
        composition=composition,
        swinir_calls=calls,
        **route(bool(children)),
    )


def verify_inputs(run, item, wh):
    run = Path(run)
    assert sha256(run / "dataset/images" / item["name"]) == item["native_sha256"]
    with Image.open(run / "dataset/images" / item["name"]) as im:
        assert list(im.size) == wh and im.mode == "RGB"
    for child in item["children"]:
        for key in ("image", "mask"):
            assert sha256(run / child[key]) == child[key + "_sha256"]
        x0, y0, x1, y1 = child["roi_native"]
        assert all(v % 8 == 0 for v in (x0, y0, x1, y1))
        assert child["wh"] == [(x1 - x0) * 4, (y1 - y0) * 4]
        with Image.open(run / child["mask"]) as mask:
            assert list(mask.size) == child["wh"] and mask.mode == "L"
            assert sum(mask.histogram()[1:255]) == 0
            assert mask.histogram()[255] == child["mask_pixels"]


def camera_children(row, item, run):
    from .paired_data import child_camera

    children = []
    for child in item["children"]:
        camera = child_camera(row["K"], row["w2c"], row["wh"], child["roi_native"])
        children.append(dict(child, camera=camera))
    row.update(
        children=children,
        source_frame=row["frame"],
        view_type="full",
        view_layout=LAYOUT,
        source_group_split=True,
    )
    # The full union is used only to select mother-loss pixels.
    union = np.zeros((row["wh"][1], row["wh"][0]), dtype=bool)
    for face in item["faces"]:
        with Image.open(Path(run) / "face-masks" / face["mask_name"]) as im:
            union |= np.asarray(im) != 0
    path = Path(run) / "face-masks" / (row["frame"] + "_union.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(union.astype(np.uint8) * 255).save(path)
    row["face_mask"] = str(path)
    row["face_mask_sha256"] = sha256(path)


class JointSupervision:
    """Load one source group and render all children in the same autograd graph."""

    def __init__(self, native_render, source_dir, run):
        from .roi import wrap_renderer

        self.native_render = native_render
        self.roi_render = wrap_renderer(native_render, source_dir)
        self.run = Path(run)
        self.cached_key = None
        self.cached = None
        self.last_stats = {}
        self.source_updates = 0

    def targets(self, camera):
        import torch
        from .paired_loss import prepare_ssim_support

        key = (camera.image_name, camera.level)
        if key == self.cached_key:
            return self.cached
        self.cached = None
        row = camera.row
        if sha256(row["face_mask"]) != row["face_mask_sha256"]:
            raise RuntimeError("Mother mask changed")
        with Image.open(row["face_mask"]) as image:
            native_mask = np.array(image) != 0
        mask = torch.from_numpy(native_mask).to("cuda")
        if mask.shape != (camera.height, camera.width):
            # Conservative coverage avoids dropping tiny faces during native
            # half-size warmup, including odd-width/height rounding.
            mask = (
                torch.nn.functional.adaptive_max_pool2d(
                    mask.float()[None, None], (camera.height, camera.width)
                )[0, 0]
                > 0
            )
        children = []
        for saved in row.get("children", []):
            if any(
                sha256(self.run / saved[key]) != saved[key + "_sha256"]
                for key in ("image", "mask")
            ):
                raise RuntimeError("Auxiliary child pixels or mask changed")
            wh = tuple(round(v / camera.level) for v in saved["wh"])
            with Image.open(self.run / saved["image"]) as image:
                rgb = np.array(image.resize(wh))
            with Image.open(self.run / saved["mask"]) as image:
                support = np.array(image.resize(wh, Image.Resampling.NEAREST)) != 0
            target = (
                torch.from_numpy(rgb)
                .to(device="cuda", dtype=torch.float32)
                .div_(255)
                .permute(2, 0, 1)
                .contiguous()
            )
            support = torch.from_numpy(support).to("cuda")
            x0, y0, x1, y1 = saved["roi_native"]
            fw, fh = [int(v * 4 / camera.level) for v in row["wh"]]
            origin = [int(x0 * 4 / camera.level), int(y0 * 4 / camera.level)]
            # Nonphysical ROI edges are excluded from SSIM centers. Any padding
            # beyond the full image is already false in the child support.
            edges = (x0 == 0, y0 == 0, x1 == row["wh"][0], y1 == row["wh"][1])
            child_camera = SimpleNamespace(**vars(camera))
            child_camera.image_width, child_camera.image_height = wh
            child_camera.width, child_camera.height = wh
            child_camera._msv2_roi = dict(
                full_wh=[fw, fh], viewport_xy=origin, wh=list(wh)
            )
            children.append(
                dict(
                    camera=child_camera,
                    target=target,
                    mask=support,
                    physical_edges=edges,
                    support_cache=prepare_ssim_support(support, physical_edges=edges),
                )
            )
        self.cached_key = key
        self.cached = (mask, prepare_ssim_support(~mask), children)
        return self.cached

    def render(self, camera, gaussians, pipe, background, **kwargs):
        import torch

        output = self.native_render(camera, gaussians, pipe, background, **kwargs)
        face_mask, support, children = self.targets(camera)
        rendered = []
        radii = output["radii"].clone()
        area = output["area_max"].clone()
        for child in children:
            cc = child["camera"]
            cc._msv2_shared_viewspace_points = output["viewspace_points"]
            result = self.roi_render(cc, gaussians, pipe, background, **kwargs)
            sx = cc._msv2_roi["full_wh"][0] / camera.width
            sy = cc._msv2_roi["full_wh"][1] / camera.height
            radii = torch.maximum(
                radii, torch.ceil(result["radii"] / min(sx, sy)).to(radii.dtype)
            )
            area = torch.maximum(area, result["area_max"] / (sx * sy))
            rendered.append(
                dict(
                    prediction=result["render"],
                    **{k: v for k, v in child.items() if k != "camera"},
                )
            )
        output.update(
            radii=radii,
            visibility_filter=(radii > 0).nonzero(),
            area_max=area,
            paired_children=rendered,
            face_mask=face_mask,
            parent_support=support,
        )
        return output

    def loss(self, camera, output, target, weight, native_l1, native_ssim):
        from fused_ssim import FusedSSIMMap
        from .paired_loss import paired_loss

        def ssim_map(a, b):
            return FusedSSIMMap.apply(0.01**2, 0.03**2, a, b, "same", True)

        result = paired_loss(
            output["render"],
            target,
            output["face_mask"],
            output["paired_children"],
            weight,
            ssim_map=ssim_map,
            native_l1=native_l1,
            native_fused_ssim=native_ssim,
            parent_support=output["parent_support"],
        )
        self.last_stats = dict(
            result["stats"],
            source=camera.image_name,
            level=camera.level,
            child_count=len(output["paired_children"]),
            parent_wh=[camera.width, camera.height],
            ndc_dummy_shared=True,
            child_radius_policy="conservative mother-pixel-unit bound",
            child_area_policy="ROI statistic normalized by full-to-mother pixel area",
        )
        self.source_updates += 1
        return result


def install_joint_training(module, joint, output_path):
    """Replace only render and loss sites in the verified native training loop.

    The resulting source is retained for review. No global loss monkeypatch or
    hidden call-order state is used; evaluation still calls the native losses.
    """
    source = inspect.getsource(module.training)
    render = "render_pkg = render_imp(viewpoint_cam, gaussians, pipe, background, culling=gaussians._culling[:,viewpoint_cam.uid])"
    loss = """Ll1 = l1_loss(image, gt_image)
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)"""
    if source.count(render) != 1 or source.count(loss) != 1:
        raise RuntimeError("Pinned native render/loss call sites changed")
    adapted = source.replace(
        render, render.replace("render_imp(", "_msv2_joint.render(")
    )
    adapted = adapted.replace(
        loss,
        """joint_result = _msv2_joint.loss(viewpoint_cam, render_pkg, gt_image, opt.lambda_dssim, l1_loss, fused_ssim)
        Ll1, ssim_value, loss = joint_result["l1"], joint_result["ssim"], joint_result["loss"]""",
    )
    output_path = Path(output_path)
    output_path.write_text(adapted, encoding="utf-8")
    module.__dict__["_msv2_joint"] = joint
    exec(compile(adapted, str(output_path), "exec"), module.__dict__)
    return dict(
        upstream_function_sha256=__import__("hashlib")
        .sha256(source.encode())
        .hexdigest(),
        adapted_sha256=sha256(output_path),
        changed_sites=["source render", "joint masked photometric loss"],
        model_architecture_unchanged=True,
        optimizer_and_schedule_unchanged=True,
        one_backward_per_source=True,
        one_optimizer_update_per_source=True,
    )
