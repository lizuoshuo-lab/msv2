"""Isolated adapter around unchanged, pinned upstream MSV2 training code.

Each stage runs in a separate process. Dimensions and paths come from a frozen
job file. Learned state is never reused between variants.
"""

from pathlib import Path
from types import SimpleNamespace
import argparse
import ast
import copy
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time

import numpy as np
from PIL import Image

from .io import now, read_json as read, save_json as save, sha256 as sha

NAMES = {1: "original", 2: "interpolation2x", 4: "swinir4x"}
WEIGHT_SHA = "99adfa91350a84c99e946c1eb3d8fce34bc28f57d807b09dc8fe40a316328c0a"
COMMIT = "5ec9202c2db7d900728bfc09733e85467cff885e"
SOURCE_ARCHIVE_SHA = "7ea4ae14faf45d18b62dbd85cdd64129e3739868f4ddc2a9e74c080d7b1cf3b4"


def verify_source(source):
    source = Path(source)
    receipt = source / ".msv2-source.json"
    if not receipt.is_file():
        raise RuntimeError(
            "Pinned source receipt missing; provision source with deploy.sh: "
            + str(source)
        )
    saved = read(receipt)
    if (
        saved.get("commit") != COMMIT
        or saved.get("archive_sha256") != SOURCE_ARCHIVE_SHA
    ):
        raise RuntimeError("Source version differs from the pinned upstream archive")
    wanted = (
        "msv2",
        "scene",
        "utils",
        "gaussian_renderer",
        "arguments",
        "config",
        "lpipsPyTorch",
    )
    for relative, expected in saved["files"].items():
        if Path(relative).parts[0] in wanted:
            path = source / relative
            if not path.is_file() or sha(path) != expected:
                raise RuntimeError("Pinned source integrity check failed: " + relative)
    for part in wanted:
        for path in (source / part).rglob("*"):
            if not path.is_file() or path.suffix in (".pyc", ".pyo"):
                continue
            relative = path.relative_to(source).as_posix()
            if relative not in saved["files"] or sha(path) != saved["files"][relative]:
                raise RuntimeError("Pinned source integrity check failed: " + relative)
    return saved


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save_image(path, tensor):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.rint(
        tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
    ).astype(np.uint8)
    Image.fromarray(array).save(path)


class Engine:
    def __init__(self, job_file):
        self.job_file = Path(job_file).resolve()
        self.job = read(self.job_file)
        self.config = self.job["options"]
        self.root = Path(self.job["run_dir"])
        self.prep = self.root / "prepared"
        self.record = self.root / "export"
        self.repo = Path(self.job["source_dir"])
        self.swinir = Path(self.job["swinir_source"])
        self.weight = Path(self.job["swinir_weight"])
        # Legacy frozen jobs without this option retain their full-image mode.
        self.sr_region = self.config.get("sr_region", "full")
        runtime_root = self.root.parent.parent
        self.head_detector_weight = Path(
            self.job.get(
                "head_detector_weight",
                runtime_root / "weights/face_detection_yunet_2023mar.onnx",
            )
        )
        self.head_parser_source = Path(
            self.job.get("head_parser_source", runtime_root / "vendor/head-parser")
        )
        self.head_parser_weight = Path(
            self.job.get(
                "head_parser_weight", runtime_root / "weights/face_parsing_resnet18.pt"
            )
        )
        self.iterations = self.config["iterations"]

    def runtime(self):
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = 16384 if hard == resource.RLIM_INFINITY else min(16384, hard)
        if soft != resource.RLIM_INFINITY and soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        sys.path.insert(0, str(self.root / "source"))
        import torch

        torch.set_num_threads(self.config["cpu_threads"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError("A supported NVIDIA CUDA device is required")
        return torch

    def native_rows(self):
        audit = read(self.prep / "pixel-lineage.json")
        if not audit["passed"]:
            raise ValueError("Input pixel lineage is not verified")
        rows = audit["records"]
        if len(rows) < 3:
            raise ValueError(
                "At least three registered views are required for training with a held-out view"
            )
        assert len({r["image"] for r in rows}) == len(rows)
        return rows

    def initialize(self):
        rows = self.native_rows()
        verify_source(self.repo)
        source = self.root / "source"
        if source.exists():
            raise FileExistsError("Run snapshot already exists; choose a new run name")
        source.mkdir()
        checks = []
        for part in (
            "msv2",
            "scene",
            "utils",
            "gaussian_renderer",
            "arguments",
            "config",
            "lpipsPyTorch",
        ):
            shutil.copytree(
                self.repo / part,
                source / part,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            for path in sorted((source / part).rglob("*.py")):
                relative = path.relative_to(source)
                assert sha(path) == sha(self.repo / relative)
                checks.append(dict(path=str(relative), sha256=sha(path)))
        shutil.copy2(source / "config/fast", self.root / "fast.config")
        for scale in NAMES:
            (self.root / NAMES[scale]).mkdir()
        plan = dict(
            state="preparing",
            created=now(),
            repository="https://github.com/fatPeter/mini-splatting2",
            commit=COMMIT,
            options=self.config,
            source_checks=checks,
            fast_sha256=sha(self.root / "fast.config"),
            frame_count=len(rows),
            full_rgb=True,
            mask_used=False,
            training_mask_used=False,
            input_composite_mask=False,
            input_head_mask_used=False,
            head_training_views_requested=self.sr_region == "head"
            and 4 in self.config["scales"],
            head_training_views=False,
            sr_region=self.sr_region,
            crop_used=False,
            background_trained=True,
            source_dimensions=sorted({tuple(r["wh"]) for r in rows}),
            fresh_gaussians=True,
            fresh_optimizer=True,
            old_fitted_state_loaded=False,
            training_algorithm_changes=[],
            training_completed=False,
            quality_accepted=False,
            warmup="Official half-size warmup follows preparation of native/enlarged inputs",
        )
        save(self.root / "plan.json", plan)
        print("SOURCE_SNAPSHOT_READY", len(rows), flush=True)

    def stage_pixels(self, scale):
        run = self.root / NAMES[scale]
        out = run / "dataset/images"
        out.mkdir(parents=True, exist_ok=True)
        rows = self.native_rows()
        items = []
        if scale == 4 and self.sr_region == "head":
            sr = read(run / "superresolution.json")
            native = {row["image"]: row for row in rows}
            records = {row["name"]: row for row in sr["records"]}
            assert sr["state"] == "completed" and set(records) == set(native)
            for name, row in native.items():
                item = records[name]
                assert item["native_sha256"] == row["sha256"]
                assert sha(out / name) == item["output_sha256"] == row["sha256"]
                items.append(
                    dict(
                        name=name,
                        image_sha256=item["output_sha256"],
                        native_sha256=row["sha256"],
                        view_type="full",
                    )
                )
                for head in item["heads"]:
                    path = out / head["name"]
                    assert sha(path) == head["output_sha256"]
                    x0, y0, x1, y1 = head["bbox_native"]
                    with Image.open(path) as image:
                        assert image.mode == "RGB" and image.size == (
                            4 * (x1 - x0),
                            4 * (y1 - y0),
                        )
                    items.append(
                        dict(
                            name=head["name"],
                            image_sha256=head["output_sha256"],
                            native_sha256=row["sha256"],
                            view_type="head",
                        )
                    )
            save(
                run / "pixels.json",
                dict(
                    scale=4,
                    count=len(items),
                    source_frames=len(rows),
                    view_count=len(items),
                    items=items,
                    full_rgb=True,
                    training_mask_used=False,
                    input_composite_mask=any(item["heads"] for item in sr["records"]),
                    input_head_mask_used=any(item["heads"] for item in sr["records"]),
                    full_frame_4x_generated=False,
                    mixed_resolution=True,
                    sr_region="head",
                    full_view_scale=1,
                    head_view_scale=4,
                ),
            )
            print(
                "PIXELS_READY", scale, len(items), "mixed-resolution views", flush=True
            )
            return
        for row in rows:
            src = self.prep / "dataset/images" / row["image"]
            assert sha(src) == row["sha256"]
            dst = out / src.name
            width, height = row["wh"]
            if scale == 4:
                assert dst.exists(), "Generate SwinIR images first"
            elif not dst.exists():
                if scale == 1:
                    shutil.copy2(src, dst)
                else:
                    with Image.open(src) as image:
                        image.resize(
                            (width * scale, height * scale), Image.Resampling.LANCZOS
                        ).save(dst, compress_level=3)
            with Image.open(dst) as image:
                assert image.mode == "RGB" and image.size == (
                    width * scale,
                    height * scale,
                )
            items.append(
                dict(name=src.name, native_sha256=row["sha256"], image_sha256=sha(dst))
            )
        save(
            run / "pixels.json",
            dict(
                scale=scale,
                count=len(rows),
                items=items,
                full_rgb=True,
                no_mask=True,
                training_mask_used=False,
                input_composite_mask=False,
                sr_region=self.sr_region if scale == 4 else None,
            ),
        )
        print("PIXELS_READY", scale, len(rows), flush=True)

    def options(self, scale):
        from arguments import ModelParams, OptimizationParams, PipelineParams

        parser = argparse.ArgumentParser()
        lp, op, pp = (
            ModelParams(parser),
            OptimizationParams(parser),
            PipelineParams(parser),
        )
        args = parser.parse_args([])
        expression = ast.parse(
            (self.root / "fast.config").read_text(), mode="eval"
        ).body
        assert isinstance(expression, ast.Call) and isinstance(
            expression.func, ast.Name
        )
        assert expression.func.id == "Namespace" and not expression.args
        vars(args).update(
            {item.arg: ast.literal_eval(item.value) for item in expression.keywords}
        )
        args.imp_metric = self.config["importance_metric"]
        args.iterations = self.iterations
        args.num_depth_factor, args.sampling_factor = 1, 0.6
        args.source_path = str(self.root / NAMES[scale] / "dataset")
        args.model_path = str(self.root / NAMES[scale] / "model")
        args.resolution, args.data_device, args.eval = 1, "cpu", True
        return args, lp.extract(args), op.extract(args), pp.extract(args)

    # NATIVE_ADAPTER_METHODS
    def superresolve(self):
        torch = self.runtime()
        assert sha(self.weight) == WEIGHT_SHA
        # This stage runs alone in the queue; tiles bound attention memory.
        torch.cuda.set_per_process_memory_fraction(self.config["gpu_memory_fraction"])
        sys.path.insert(0, str(self.swinir))
        official = import_file("main_test_swinir", self.swinir / "main_test_swinir.py")
        args = SimpleNamespace(
            task="real_sr",
            scale=4,
            large_model=True,
            model_path=str(self.weight),
            tile=self.config["sr_tile"],
            tile_overlap=self.config["sr_tile_overlap"],
        )
        model = official.define_model(args).eval().cuda()
        detector = None
        if self.sr_region == "head":
            from .head import HeadPixelExtractor

            detector = HeadPixelExtractor(
                self.head_parser_source,
                self.head_parser_weight,
                self.head_detector_weight,
            )
        run = self.root / NAMES[4]
        out = run / "dataset/images"
        out.mkdir(parents=True, exist_ok=True)
        p = run / "superresolution.json"
        state = (
            read(p)
            if p.exists()
            else dict(
                state="running",
                started=now(),
                records=[],
                weight_sha256=WEIGHT_SHA,
                precision="float32",
                tile=self.config["sr_tile"],
                tile_overlap=self.config["sr_tile_overlap"],
                sr_region=self.sr_region,
                training_mask_used=False,
                input_composite_mask=False,
                input_head_mask_used=False,
                head_mask_source="existing_digital_human_face_parsing"
                if self.sr_region == "head"
                else None,
                mixed_resolution=self.sr_region == "head",
            )
        )
        done = {r["name"]: r for r in state["records"]}
        assert state.get("sr_region", "full") == self.sr_region, (
            "Cannot mix SR input modes in one run"
        )
        if detector is not None:
            state["head_pixel_extractor"] = detector.assets

        def infer(rgb, crop=False):
            # Head callbacks receive unmodified original context pixels.
            # Padding serves only window-8 attention, never input rescaling.
            h, w = rgb.shape[:2]
            x = (
                torch.from_numpy(rgb.copy())
                .to(device="cuda", dtype=torch.float32)
                .permute(2, 0, 1)[None]
                / 255.0
            )
            if crop:
                hp, wp = (-h) % 8, (-w) % 8
                while x.shape[2] < h + hp:
                    x = torch.cat([x, torch.flip(x, [2])], 2)
                x = x[:, :, : h + hp, :]
                while x.shape[3] < w + wp:
                    x = torch.cat([x, torch.flip(x, [3])], 3)
                x = x[:, :, :, : w + wp]
            else:
                # Preserve the existing full-image control's exact padding.
                hp = (h // 8 + 1) * 8 - h
                wp = (w // 8 + 1) * 8 - w
                x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, : h + hp, :]
                x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, : w + wp]
            current = copy.copy(args)
            if crop and (
                max(h, w) <= self.config["sr_tile"]
                or min(h + hp, w + wp) <= self.config["sr_tile_overlap"]
            ):
                current.tile = None
            pred = official.test(x, model, current, 8)[..., : h * 4, : w * 4]
            assert pred.shape == (1, 3, h * 4, w * 4) and torch.isfinite(pred).all()
            output = np.rint(
                pred[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
            ).astype(np.uint8)
            return output, dict(
                source_wh=[w, h],
                output_wh=[w * 4, h * 4],
                tile=current.tile,
                window_padding=[wp, hp],
                crop_input=crop,
                raw_pixels=True,
            )

        with torch.inference_mode():
            for row in self.native_rows():
                src = self.prep / "dataset/images" / row["image"]
                dst = out / row["image"]
                if row["image"] in done:
                    assert done[row["image"]]["native_sha256"] == row["sha256"]
                    assert sha(dst) == done[row["image"]]["output_sha256"]
                    for head in done[row["image"]].get("heads", []):
                        assert sha(out / head["name"]) == head["output_sha256"]
                        assert (
                            sha(run / "head-masks" / head["mask_name"])
                            == head["mask_sha256"]
                        )
                    continue
                t = time.monotonic()
                assert sha(src) == row["sha256"]
                with Image.open(src) as image:
                    rgb = np.array(image.convert("RGB"))
                calls, head_metadata, heads = [], None, []
                if detector is None:
                    arr, call = infer(rgb)
                    calls.append(call)
                else:
                    from .head import superresolve_head

                    regions, head_metadata = detector.regions(rgb)

                    def callback(crop):
                        output, call = infer(crop, crop=True)
                        calls.append(call)
                        return output

                    for number, region in enumerate(regions, 1):
                        bbox, mask = region["bbox_native"], region["mask_native"]
                        head_rgb, receipt = superresolve_head(
                            rgb, bbox, callback, mask_native=mask, context_native=32
                        )
                        head_name = f"{src.stem}_head_{number:03d}.png"
                        head_path = out / head_name
                        temporary = head_path.with_name(head_path.name + ".tmp")
                        Image.fromarray(head_rgb).save(
                            temporary, format="PNG", compress_level=3
                        )
                        temporary.replace(head_path)
                        mask_path = run / "head-masks" / head_name
                        mask_path.parent.mkdir(parents=True, exist_ok=True)
                        mask_tmp = mask_path.with_name(mask_path.name + ".tmp")
                        Image.fromarray(mask.astype(np.uint8) * 255).save(
                            mask_tmp, format="PNG"
                        )
                        mask_tmp.replace(mask_path)
                        heads.append(
                            dict(
                                name=head_name,
                                bbox_native=bbox,
                                output_sha256=sha(head_path),
                                mask_name=head_name,
                                mask_sha256=sha(mask_path),
                                receipt=receipt,
                            )
                        )
                        del head_rgb
                    # The complete RGB remains byte-identical at native size.
                    # Head SR is an additional calibrated input, not a paste
                    # into a full-resolution-4 background canvas.
                    shutil.copy2(src, dst)
                    arr = None
                if arr is not None:
                    temporary = dst.with_name(dst.name + ".tmp")
                    Image.fromarray(arr).save(temporary, format="PNG", compress_level=3)
                    temporary.replace(dst)
                item = dict(
                    name=row["image"],
                    native_sha256=row["sha256"],
                    output_sha256=sha(dst),
                    seconds=time.monotonic() - t,
                    sr_region=self.sr_region,
                    swinir_calls=calls,
                    head_detection=head_metadata,
                    heads=heads,
                )
                state["records"].append(item)
                state.update(
                    updated=now(),
                    processed=len(state["records"]),
                    state="running",
                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
                )
                save(p, state)
                print(
                    "SWINIR_" + self.sr_region.upper(),
                    len(state["records"]),
                    json.dumps(item),
                    flush=True,
                )
                del arr, rgb
        if self.sr_region == "head":
            state.update(
                head_frames=sum(bool(r["heads"]) for r in state["records"]),
                no_head_frames=sum(not r["heads"] for r in state["records"]),
                source_frames=len(state["records"]),
                head_view_count=sum(len(r["heads"]) for r in state["records"]),
                view_count=len(state["records"])
                + sum(len(r["heads"]) for r in state["records"]),
                full_view_scale=1,
                head_view_scale=4,
                input_composite_mask=any(r["heads"] for r in state["records"]),
                input_head_mask_used=any(r["heads"] for r in state["records"]),
                full_frame_4x_generated=False,
            )
        state.update(state="completed", finished=now())
        save(p, state)
        self.stage_pixels(4)

    def stage_cameras(self, scale):
        self.runtime()
        assert read(self.prep / "state.json")["state"] == "completed"
        from scene.colmap_loader import read_intrinsics_binary
        import struct

        run = self.root / NAMES[scale]
        if scale == 4 and self.sr_region == "head":
            from .head_dataset import build_head_cameras

            inputs = build_head_cameras(
                self.prep,
                run,
                read(run / "superresolution.json")["records"],
                self.config["holdout"],
            )
            save(run / "training-inputs.json", inputs)
            train_heads = sum(
                row["view_type"] == "head" and row["split"] == "train"
                for row in inputs["records"]
            )
            plan = read(self.root / "plan.json")
            plan.update(
                crop_used=inputs["crop_used"],
                head_training_views=train_heads > 0,
                source_frames=inputs["source_frames"],
                mixed_view_count=inputs["total_views"],
                full_view_count=inputs["full_views"],
                head_view_count=inputs["head_views"],
                train_head_view_count=train_heads,
                test_head_view_count=inputs["head_views"] - train_heads,
                skipped_head_view_count=len(inputs.get("skipped_head_views", [])),
                input_composite_mask=inputs["head_views"] > 0,
                input_head_mask_used=inputs["head_views"] > 0,
            )
            save(self.root / "plan.json", plan)
            self._rgb_caches(run, inputs["records"], scale)
            return
        data = run / "dataset"
        sp = data / "sparse/0"
        sp.mkdir(parents=True, exist_ok=True)
        for name in ("images.bin", "points3D.bin"):
            shutil.copy2(self.prep / "dataset/sparse/0" / name, sp / name)
        cameras = read_intrinsics_binary(
            str(self.prep / "dataset/sparse/0/cameras.bin")
        )
        with (sp / "cameras.bin").open("wb") as f:
            f.write(struct.pack("<Q", len(cameras)))
            for cid, c in cameras.items():
                assert c.model in ("PINHOLE", "SIMPLE_PINHOLE")
                params = (
                    np.asarray(c.params)
                    if c.model == "PINHOLE"
                    else np.asarray(
                        [c.params[0], c.params[0], c.params[1], c.params[2]]
                    )
                )
                f.write(struct.pack("<iiQQ", cid, 1, c.width * scale, c.height * scale))
                f.write(struct.pack("<4d", *(params * scale)))
        loaded = read_intrinsics_binary(str(sp / "cameras.bin"))
        for cid, c in cameras.items():
            assert (
                loaded[cid].width == c.width * scale
                and loaded[cid].height == c.height * scale
            )
            params = (
                np.asarray(c.params)
                if c.model == "PINHOLE"
                else np.asarray([c.params[0], c.params[0], c.params[1], c.params[2]])
            )
            assert np.array_equal(loaded[cid].params, params * scale)
        assert sha(sp / "images.bin") == sha(self.prep / "dataset/sparse/0/images.bin")
        assert sha(sp / "points3D.bin") == sha(
            self.prep / "dataset/sparse/0/points3D.bin"
        )
        rows = copy.deepcopy(read(self.prep / "camera-inputs.json")["records"])
        assert len(rows) == len(self.native_rows())
        pixel = {r["name"]: r for r in read(run / "pixels.json")["items"]}
        for i, r in enumerate(rows):
            r["native_image"] = str(self.prep / "dataset/images" / r["name"])
            r["image"] = str(data / "images" / r["name"])
            r["wh"] = [int(v * scale) for v in r["wh"]]
            r["native_sha256"] = r["source_sha256"]
            r["source_sha256"] = pixel[r["name"]]["image_sha256"]
            r["K"] = (np.diag([scale, scale, 1.0]) @ np.asarray(r["K"])).tolist()
            r["split"] = "test" if i % self.config["holdout"] == 0 else "train"
            k = np.asarray(r["K"]).copy()
            k[:2, 2] -= 0.5
            r["K_integer_centers"] = k.tolist()
        save(
            run / "training-inputs.json",
            dict(
                records=rows,
                mask_used=False,
                training_mask_used=False,
                input_composite_mask=False,
                sr_region=self.sr_region if scale == 4 else None,
                crop_used=False,
                background_trained=True,
                scale=scale,
                initial_sparse_sha256=sha(sp / "points3D.bin"),
                camera_poses_sha256=sha(sp / "images.bin"),
                scaled_intrinsics_readback_verified=True,
            ),
        )
        self._rgb_caches(run, rows, scale)

    def _rgb_caches(self, run, rows, scale):
        # Exactly PILtoTorch's default RGB resize, including the official warmup level.
        for level in (1, 2):
            cache = run / f"rgb-cache-{level}"
            cache.mkdir(exist_ok=True)
            for i, r in enumerate(rows):
                dst = cache / (r["frame"] + ".npy")
                wh = (round(r["wh"][0] / level), round(r["wh"][1] / level))
                if dst.exists():
                    mapped = np.load(dst, mmap_mode="r", allow_pickle=False)
                    try:
                        assert mapped.shape == (wh[1], wh[0], 3)
                    finally:
                        mapped._mmap.close()
                    continue
                with Image.open(r["image"]) as im:
                    arr = np.array(im.resize(wh))
                np.save(dst, arr)
            print("CAMERA_CACHE_READY", scale, level, len(rows), flush=True)

    def module_for(self, scale):
        self.runtime()
        run = self.root / NAMES[scale]
        rows = read(run / "training-inputs.json")["records"]
        by_name = {r["frame"]: r for r in rows}
        # Match the native entrypoint import order (scene imports camera_utils).
        import scene
        import utils.camera_utils as cu

        def load_camera(args, uid, info, level):
            assert args.resolution == 1
            row = by_name[info.image_name]
            assert (info.width, info.height) == tuple(row["wh"])
            assert np.allclose(
                info.R, np.asarray(row["w2c"])[:3, :3].T
            ) and np.allclose(info.T, np.asarray(row["w2c"])[:3, 3])
            info.image.close()
            return LazyCamera(row, uid, int(level))

        cu.loadCam = load_camera
        native_json = cu.camera_to_JSON

        def json_camera(uid, info):
            out = native_json(uid, info)
            row = by_name[info.image_name]
            out.update(
                K=row["K"],
                cx=row["K"][0][2],
                cy=row["K"][1][2],
                pixel_centers="COLMAP .5 based",
                split=row["split"],
            )
            return out

        cu.camera_to_JSON = json_camera

        scene.camera_to_JSON = json_camera
        original_reader = scene.sceneLoadTypeCallbacks["Colmap"]

        def configured_reader(*args, **kwargs):
            kwargs["llffhold"] = self.config["holdout"]
            info = original_reader(*args, **kwargs)
            if scale == 4 and self.sr_region == "head":
                from .head_dataset import split_camera_infos
                from scene.dataset_readers import getNerfppNorm

                train, test, full_train = split_camera_infos(
                    info.train_cameras + info.test_cameras, rows
                )
                # Derived crops share source identities. They never move a
                # held-out source into training or change scene normalization.
                info = info._replace(
                    train_cameras=train,
                    test_cameras=test,
                    nerf_normalization=getNerfppNorm(full_train),
                )
            return info

        scene.sceneLoadTypeCallbacks["Colmap"] = configured_reader
        native_scene = scene.Scene

        class RecordedScene(native_scene):
            latest = None

            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                RecordedScene.latest = self

        module = import_file("native_msv2_train", self.root / "source/msv2/train.py")
        module.Scene = RecordedScene
        module.network_gui.try_connect = lambda: None
        # No GUI socket is needed for local batch runs. Training functions stay native.
        return module, RecordedScene

    def preflight(self, scale):
        torch = self.runtime()
        torch.cuda.set_per_process_memory_fraction(self.config["gpu_memory_fraction"])
        run = self.root / NAMES[scale]
        rows = read(run / "training-inputs.json")["records"]
        from scene.gaussian_model import GaussianModel
        from scene.dataset_readers import read_points3D_binary
        from utils.graphics_utils import BasicPointCloud
        from gaussian_renderer import render_imp
        from fused_ssim import fused_ssim

        args, data, opt, pipe = self.options(scale)
        xyz, rgb, _ = read_points3D_binary(str(run / "dataset/sparse/0/points3D.bin"))
        assert len(xyz) >= 4 and np.isfinite(xyz).all()
        normalization_rows = [r for r in rows if r["split"] == "train"]
        if scale == 4 and self.sr_region == "head":
            normalization_rows = [
                r for r in normalization_rows if r["view_type"] == "full"
            ]
        centers = np.asarray(
            [np.linalg.inv(r["w2c"])[:3, 3] for r in normalization_rows]
        )
        extent = float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)
        g = GaussianModel(0)
        g.create_from_pcd(BasicPointCloud(xyz, rgb / 255.0, np.zeros_like(xyz)), extent)
        g.training_setup(opt)
        checks = []
        indices = (0, len(rows) // 2, len(rows) - 1)
        if scale == 4 and self.sr_region == "head":
            groups = list(dict.fromkeys(row["source_frame"] for row in rows))
            selected = {groups[0], groups[len(groups) // 2], groups[-1]}
            indices = [
                i for i, row in enumerate(rows) if row["source_frame"] in selected
            ]
        for level in (2, 1):
            for index in indices:
                c = LazyCamera(rows[index], index, level)
                k = c.K
                w = np.asarray(c.row["w2c"])
                projection_points = xyz
                if scale == 4 and self.sr_region == "head":
                    # A textureless head can have no SfM feature inside its
                    # crop. Check calibrated rays as well as actual points;
                    # these algebraic probes never initialize Gaussians.
                    pixels = np.array(
                        [
                            [0.2 * c.width, 0.2 * c.height, 1],
                            [0.8 * c.width, 0.8 * c.height, 1],
                            [0.5 * c.width, 0.5 * c.height, 1],
                        ]
                    )
                    rays = pixels @ np.linalg.inv(k).T
                    probes = (rays - w[:3, 3]) @ w[:3, :3]
                    projection_points = np.concatenate([xyz, probes])
                q = projection_points @ w[:3, :3].T + w[:3, 3]
                good = q[:, 2] > c.znear
                q = q[good]
                xx = projection_points[good]
                expected = q @ k.T
                expected = expected[:, :2] / expected[:, 2:] - 0.5
                hp = (
                    torch.cat(
                        [
                            torch.as_tensor(xx, dtype=torch.float32, device="cuda"),
                            torch.ones((len(xx), 1), device="cuda"),
                        ],
                        1,
                    )
                    @ c.full_proj_transform
                )
                uv = (hp[:, :2] / (hp[:, 3:] + 1e-7) + 1) * torch.tensor(
                    [c.width, c.height], device="cuda"
                ) / 2 - 0.5
                inside = (
                    (expected[:, 0] >= 0)
                    & (expected[:, 0] < c.width)
                    & (expected[:, 1] >= 0)
                    & (expected[:, 1] < c.height)
                )
                if not np.any(inside):
                    raise ValueError(
                        "Preflight view has no visible sparse point: " + c.image_name
                    )
                error = float(
                    np.max(np.abs(uv.cpu().numpy()[inside] - expected[inside]))
                )
                assert error < 0.1, (index, level, error)
                assert sha(c.row["image"]) == c.row["source_sha256"]
                with Image.open(c.row["image"]) as im:
                    reference = np.asarray(im.resize((c.width, c.height)))
                assert np.array_equal(c.rgb, reference)
                checks.append(
                    dict(
                        frame=c.image_name,
                        level=level,
                        wh=[c.width, c.height],
                        projection_max_error_px=error,
                        cache_pixel_error=0,
                    )
                )
        # The native 4x run starts with half-size warmup, not full-size raw
        # SfM splats. Test that actual first stage; both resolutions retain
        # their pixel/projection checks above. This does not certify the
        # later full-size training working set after native refinement.
        backward_level = 2 if scale == 4 else 1
        c = LazyCamera(rows[len(rows) // 2], 0, backward_level)
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        output = render_imp(c, g, pipe, torch.zeros(3, device="cuda"))
        target = c.original_image.cuda()
        image = output["render"]
        l1 = (image - target).abs().mean()
        loss = 0.8 * l1 + 0.2 * (1 - fused_ssim(image[None], target[None]))
        loss.backward()
        torch.cuda.synchronize()
        assert torch.isfinite(loss) and torch.isfinite(g._xyz.grad).all()
        peak = torch.cuda.max_memory_reserved() / 1024**3
        save_image(run / "preflight.png", image)
        result = dict(
            passed=True,
            scale=scale,
            checks=checks,
            initial_points=len(xyz),
            native_loss=float(loss),
            native_backward_seconds=time.monotonic() - start,
            backward_level=backward_level,
            backward_wh=[c.width, c.height],
            full_target_wh=rows[0]["wh"],
            target_dimensions=sorted({tuple(row["wh"]) for row in rows}),
            mixed_resolution=scale == 4 and self.sr_region == "head",
            peak_reserved_gib=peak,
            model_and_loss_unmodified=True,
            training_completed=False,
        )
        save(run / "preflight.json", result)
        print("STANDARD_PREFLIGHT_PASSED", json.dumps(result), flush=True)

    def train(self, scale):
        torch = self.runtime()
        run = self.root / NAMES[scale]
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(self.config["gpu_memory_fraction"])
        assert read(run / "preflight.json")["passed"] and not (run / "model").exists()
        plan = read(self.root / "plan.json")
        assert sha(self.root / "fast.config") == plan["fast_sha256"]
        for source in plan["source_checks"]:
            assert sha(self.root / "source" / source["path"]) == source["sha256"], (
                source["path"]
            )
        shutil.copy2(__file__, run / "training-adapter.py")
        m, Scene = self.module_for(scale)
        from .memory import wrap_training_render

        m.render_imp = wrap_training_render(m.render_imp, self.repo)
        args, data, opt, pipe = self.options(scale)
        assert (
            args.imp_metric == self.config["importance_metric"]
            and args.warn_until_iter == 3000
            and args.iterations == self.iterations
        )
        seed = self.config["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.set_device(0)
        save(run / "resolved-options.json", vars(args))
        import resource

        state = dict(
            state="training",
            pid=os.getpid(),
            started=now(),
            recipe=self.config,
            quality_accepted=False,
            nofile_limits=list(resource.getrlimit(resource.RLIMIT_NOFILE)),
            gpu_allocator_limit_gib=total
            * self.config["gpu_memory_fraction"]
            / 1024**3,
            target_layout="contiguous CHW, unchanged float32 pixel values",
            allocator_cache_trim="after each full-resolution backward"
            if scale == 4
            else "native",
            native_sources_unchanged=True,
            image_buffer_offload=m.render_imp.memory_stats,
            adapter_sha256=sha(run / "training-adapter.py"),
        )
        save(run / "execution.json", state)
        native_report = m.training_report
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()

        def report(tb, iteration, l1, loss, *rest):
            native_report(tb, iteration, l1, loss, *rest)
            if (
                scale == 4
                and self.sr_region == "head"
                and iteration == args.depth_reinit_iter
            ):
                from .head_dataset import depth_sampling_capacity

                active_views = Scene.latest.getTrainCameras_warn_up(
                    iteration, args.warn_until_iter, scale=1.0, scale2=2.0
                )
                capacity = depth_sampling_capacity(
                    len(Scene.latest.gaussians.get_xyz) * args.num_depth_factor,
                    [
                        dict(frame=c.image_name, wh=[c.width, c.height])
                        for c in active_views
                    ],
                )
                capacity.update(iteration=iteration, native_sampler_unchanged=True)
                save(run / "depth-capacity.json", capacity)
                if not capacity["passed"]:
                    raise RuntimeError(
                        "Native depth sampling exceeds input-view pixel capacity; "
                        "see " + str(run / "depth-capacity.json")
                    )
            c = LazyCamera.last
            if scale == 4 and c is not None and c.level == 1:
                torch.cuda.empty_cache()
            if iteration % 100 == 0 or iteration in (
                1,
                1999,
                2001,
                2999,
                3001,
                8001,
                13001,
                self.iterations,
            ):
                g = Scene.latest.gaussians
                c = LazyCamera.last
                item = dict(
                    iteration=iteration,
                    points=len(g.get_xyz),
                    loss=float(loss),
                    l1=float(l1),
                    seconds=time.monotonic() - start,
                    updated=now(),
                    wh=[c.width, c.height],
                    image=c.image_name,
                    warmup_level=c.level,
                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
                )
                item["open_file_descriptors"] = len(os.listdir("/proc/self/fd"))
                item["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
                item["allocated_gib"] = torch.cuda.memory_allocated() / 1024**3
                item["reserved_gib"] = torch.cuda.memory_reserved() / 1024**3
                assert math.isfinite(item["loss"])
                save(run / "progress.json", item)
                with (run / "progress.jsonl").open("a") as f:
                    f.write(json.dumps(item) + "\n")

        m.training_report = report
        try:
            # Identical native training function, loss, optimizer, render and depth kernels.
            m.training(data, opt, pipe, [-1], [self.iterations], [], None, -1, args)
            torch.cuda.synchronize()
            scene = Scene.latest
            g = scene.gaussians
            assert all(
                torch.isfinite(v).all()
                for v in (
                    g._xyz,
                    g._opacity,
                    g._scaling,
                    g._rotation,
                    g._features_dc,
                    g._features_rest,
                )
            )
            ply = run / f"model/point_cloud/iteration_{self.iterations}/point_cloud.ply"
            assert ply.exists()
            restored = m.GaussianModel(3)
            restored.load_ply(str(ply))
            restored.active_sh_degree = g.active_sh_degree
            checks = []
            with torch.no_grad():
                cams = scene.getTestCameras()
                for i in (0, len(cams) // 2, len(cams) - 1):
                    c = cams[i]
                    a = m.render(c, g, pipe, torch.zeros(3, device="cuda"))["render"]
                    b = m.render(c, restored, pipe, torch.zeros(3, device="cuda"))[
                        "render"
                    ]
                    d = (a - b).abs()
                    checks.append(
                        dict(
                            frame=c.image_name,
                            max_abs=float(d.max()),
                            mean_abs=float(d.mean()),
                        )
                    )
                    save_image(run / "preview" / f"{c.image_name}.png", a)
                    del a, b, d
            parity = all(c["max_abs"] < 0.005 for c in checks)
            assert parity, checks
            save(
                run / "ply-parity.json",
                dict(passed=parity, checks=checks, sha256=sha(ply)),
            )
            state.update(
                state="training_completed",
                finished=now(),
                iterations=self.iterations,
                optimizer_updates=self.iterations - 1,
                points=len(g.get_xyz),
                seconds=time.monotonic() - start,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
                train_frames=len(scene.getTrainCameras()),
                test_frames=len(scene.getTestCameras()),
                ply_sha256=sha(ply),
            )
            save(run / "execution.json", state)
            print("STANDARD_TRAINING_COMPLETED", json.dumps(state), flush=True)
        except BaseException as e:
            state.update(state="failed", finished=now(), error=repr(e))
            save(run / "execution.json", state)
            raise


class LazyCamera:
    cached_id = None
    cached_tensor = None
    last = None

    def __init__(self, row, uid, level=1):
        import torch
        from utils.graphics_utils import getWorld2View2

        self.row = row
        self.uid = uid
        self.colmap_id = 1
        self.image_name = row["frame"]
        self.level = level
        self.image_width, self.image_height = [round(v / level) for v in row["wh"]]
        self.width = self.image_width
        self.height = self.image_height
        self.K = np.diag(
            [self.width / row["wh"][0], self.height / row["wh"][1], 1.0]
        ) @ np.asarray(row["K"])
        w = np.asarray(row["w2c"])
        self.R = w[:3, :3].T
        self.T = w[:3, 3]
        self.FoVx = 2 * math.atan(self.width / (2 * self.K[0, 0]))
        self.FoVy = 2 * math.atan(self.height / (2 * self.K[1, 1]))
        self.znear = 0.01
        self.zfar = 100.0
        self.world_view_transform = (
            torch.tensor(getWorld2View2(self.R, self.T)).transpose(0, 1).cuda()
        )
        p = np.zeros((4, 4))
        p[0, 0] = 2 * self.K[0, 0] / self.width
        p[1, 1] = 2 * self.K[1, 1] / self.height
        p[0, 2] = 2 * self.K[0, 2] / self.width - 1
        p[1, 2] = 2 * self.K[1, 2] / self.height - 1
        p[2, 2] = self.zfar / (self.zfar - self.znear)
        p[2, 3] = -self.zfar * self.znear / (self.zfar - self.znear)
        p[3, 2] = 1
        self.projection_matrix = torch.tensor(
            p, dtype=torch.float32, device="cuda"
        ).T.contiguous()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.rgb_path = (
            Path(row["image"]).parents[2]
            / f"rgb-cache-{level}"
            / (row["frame"] + ".npy")
        )
        header = np.load(self.rgb_path, mmap_mode="r", allow_pickle=False)
        try:
            assert (
                header.shape == (self.height, self.width, 3)
                and header.dtype == np.uint8
            )
        finally:
            header._mmap.close()

    @property
    def rgb(self):
        # Each read owns a uint8 array; no camera retains an open file mapping.
        with self.rgb_path.open("rb") as stream:
            return np.load(stream, allow_pickle=False)

    @property
    def original_image(self):
        import torch

        LazyCamera.last = self
        if LazyCamera.cached_id != id(self):
            LazyCamera.cached_tensor = None
            # Division (not multiplication by reciprocal) matches native PILtoTorch.
            LazyCamera.cached_tensor = (
                torch.from_numpy(self.rgb)
                .to(device="cuda", dtype=torch.float32)
                .div_(255.0)
                .permute(2, 0, 1)
                .contiguous()
            )
            LazyCamera.cached_id = id(self)
        return LazyCamera.cached_tensor
