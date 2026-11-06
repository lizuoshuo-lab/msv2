"""Tile-aligned viewport for the pinned MS rasterizer, in a separate extension.

Projection, covariance FoV clamp and screen-gradient units use the complete
camera. Only tile intersections, pixel outputs and saved image state use the
viewport. This adapter changes neither Gaussian parameters nor the optimizer.
"""

from functools import wraps
import hashlib
import importlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from .io import read_json, save_json, sha256

MODULE = "diff_gaussian_rasterization_msv2_roi"
COMMIT = "5ec9202c2db7d900728bfc09733e85467cff885e"
ARCHIVE_SHA = "7ea4ae14faf45d18b62dbd85cdd64129e3739868f4ddc2a9e74c080d7b1cf3b4"
SOURCE_REL = "submodules/diff-gaussian-rasterization_ms"
_GLM_SHA = "20336cf066d4eac1c968b47aac782df75b5d2eeadcf1d795c7a41a0ef624fdd0"
_CORE_SHA = {
    "LICENSE.md": "cd5c95b3cfff3acc1bd412420c770f88809331c3db6872df11a970147aa8e81f",
    "conv.cu": "2c8fb88c55e5724e1b0f6474b01d77522f239909463ade8911e73d07ce3bb9f2",
    "cuda_rasterizer/adam.cu": "e2c8142d12f974ff2ebf6e302aee140a377a5cc20a4b61909e045be37733abc0",
    "cuda_rasterizer/adam.h": "e20766de348bf7416dd48875b871052881c2fe80c978396993b6c1e56acb58cf",
    "cuda_rasterizer/auxiliary.h": "bc0d833c2d565123d00495d72625df181119d02a810b80b86696d1ca3465bec2",
    "cuda_rasterizer/backward.cu": "f06a04e5f2ef53fe1655a67d019c8c09a3c89fc0984066becba03f5a5277f462",
    "cuda_rasterizer/backward.h": "6b6efdbf5f35df5fe16ec44675e0e254d58788c0ee0253410937c9519747cec2",
    "cuda_rasterizer/config.h": "290e87d7d6a8552349d657aa180f11643d19ea8f6d7d9ff0437a0d896380d262",
    "cuda_rasterizer/forward.cu": "c60cfa1e0f6f1ebbb7983c129f9e2c833e91a81dd8b809c97a8b4b65d9fabb39",
    "cuda_rasterizer/forward.h": "147f511849cafe733c8422eb35a0295d9cfb41886a494528bfec999ad9025541",
    "cuda_rasterizer/rasterizer.h": "ac24d8085cec98e84b47f1d660aa4b850c1825f92ecb0f7b87006abef800517f",
    "cuda_rasterizer/rasterizer_impl.cu": "e329bd0a0a82a2637e30635301bc670d5baeb4859b26197f59e9ae0aed9960fd",
    "cuda_rasterizer/rasterizer_impl.h": "ae885b19ff77cb233ca8800f08bf4faf8065eb92532b0fb330663830ed910aa1",
    "diff_gaussian_rasterization_ms/__init__.py": "584049aba1b92ae371b10f7a8f2d29d85828592b00be7802f6e5544eafb0b711",
    "ext.cpp": "d77e1574203aaad4a9eb54e71205147cfb81f7bcaa3aa571e378b8d6cbe0db2e",
    "rasterize_points.cu": "2fed5abe3987aeb0dcf4e3a869262205f67bdaaecc0dfd666ccf8023bd3c61ab",
    "rasterize_points.h": "fc1b771cbc0e01cf8c837d2296bbc9d3a596e14d916299e05c0c99cc15ef96be",
    "setup.py": "9f958943fe64d395627f544d1c3b9f493a4a2f545319c4ced6c97f1be8d91bea",
}


def _patch_path():
    return Path(__file__).with_name("roi_backend.patch")


def backend_path(repo, backend_dir=None):
    chosen = backend_dir or os.environ.get("MSV2_ROI_BACKEND_DIR")
    base = Path(chosen or Path(repo).parent / "msv2-roi-backend").resolve()
    identity = dict(
        patch=sha256(_patch_path()),
        adapter=sha256(__file__),
        python=list(sys.version_info[:3]),
        abi=sys.implementation.cache_tag,
        torch=metadata.version("torch"),
    )
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    return base / f"v1-{key}"


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise RuntimeError("ROI source path escapes its independent directory")
    return path


def _verify_source(repo):
    repo = Path(repo).resolve()
    receipt = read_json(repo / ".msv2-source.json")
    if receipt.get("commit") != COMMIT or receipt.get("archive_sha256") != ARCHIVE_SHA:
        raise RuntimeError("ROI backend requires the pinned native repository")
    source = repo / SOURCE_REL
    files = dict(_CORE_SHA)
    headers = {
        path.relative_to(source).as_posix(): sha256(path)
        for path in sorted((source / "third_party/glm/glm").rglob("*"))
        if path.is_file()
    }
    headers["third_party/glm/copying.txt"] = sha256(
        source / "third_party/glm/copying.txt"
    )
    digest = hashlib.sha256(
        json.dumps(headers, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != _GLM_SHA:
        raise RuntimeError("ROI backend GLM source differs from the pinned source")
    files.update(headers)
    for name, expected in files.items():
        if (
            receipt.get("files", {}).get(f"{SOURCE_REL}/{name}") != expected
            or sha256(source / name) != expected
        ):
            raise RuntimeError(f"ROI backend source mismatch: {name}")
    return source, files


def _apply_patch(directory, patch):
    """Apply exact hunks only; no fuzzy matching, offsets or external shell."""
    lines = patch.splitlines(keepends=True)
    index = 0
    while index < len(lines):
        old = lines[index].removeprefix("--- a/").rstrip("\n")
        if not lines[index].startswith("--- a/"):
            raise RuntimeError("Invalid ROI patch file header")
        index += 1
        if lines[index] != f"+++ b/{old}\n" or old not in _CORE_SHA:
            raise RuntimeError("ROI patch may only change pinned source files")
        index += 1
        path = _inside(directory, old)
        source = path.read_text(encoding="utf-8").splitlines(keepends=True)
        result, cursor = [], 0
        while index < len(lines) and not lines[index].startswith("--- a/"):
            match = re.fullmatch(
                r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*\n", lines[index]
            )
            if match is None:
                raise RuntimeError("Invalid ROI patch hunk")
            start = int(match[1]) - 1
            old_count, new_count = int(match[2] or 1), int(match[4] or 1)
            if start < cursor or start > len(source):
                raise RuntimeError("Overlapping ROI patch hunks")
            result.extend(source[cursor:start])
            cursor = start
            consumed = emitted = 0
            index += 1
            while index < len(lines) and lines[index][:1] in (" ", "+", "-"):
                if lines[index].startswith("--- a/"):
                    break
                prefix, content = lines[index][0], lines[index][1:]
                if prefix in (" ", "-"):
                    if cursor >= len(source) or source[cursor] != content:
                        raise RuntimeError(
                            f"ROI patch context mismatch: {old}:{cursor + 1}"
                        )
                    cursor += 1
                    consumed += 1
                if prefix in (" ", "+"):
                    result.append(content)
                    emitted += 1
                index += 1
            if consumed != old_count or emitted != new_count:
                raise RuntimeError("ROI patch hunk length mismatch")
        result.extend(source[cursor:])
        path.write_text("".join(result), encoding="utf-8", newline="\n")


def prepare_backend_source(repo, backend_dir):
    """Create a fresh source copy and checked patch without compiling or GPU."""
    source, files = _verify_source(repo)
    directory = Path(backend_dir).resolve()
    if directory.exists():
        raise FileExistsError(f"ROI backend directory already exists: {directory}")
    if directory.is_relative_to(Path(repo).resolve()):
        raise ValueError("ROI backend must be separate from the native repository")
    directory.mkdir(parents=True)
    for name in files:
        destination = _inside(directory, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, destination)
    _apply_patch(directory, _patch_path().read_text(encoding="utf-8"))
    old_package = _inside(directory, "diff_gaussian_rasterization_ms")
    new_package = _inside(directory, MODULE)
    if new_package.exists():
        raise FileExistsError(new_package)
    old_package.rename(new_package)
    patched = {
        (f"{MODULE}/__init__.py" if name.endswith("/__init__.py") else name): sha256(
            directory
            / (f"{MODULE}/__init__.py" if name.endswith("/__init__.py") else name)
        )
        for name in files
    }
    receipt = dict(
        schema="msv2-roi-backend-v1",
        state="source_prepared",
        module=MODULE,
        source_commit=COMMIT,
        source_archive_sha256=ARCHIVE_SHA,
        original_source_files=files,
        patched_source_files=patched,
        patch_sha256=sha256(_patch_path()),
        adapter_sha256=sha256(__file__),
    )
    save_json(directory / "roi-backend.json", receipt)
    return receipt


def _runtime():
    import torch

    return dict(
        python=list(sys.version_info[:3]),
        torch=torch.__version__,
        cuda=torch.version.cuda,
    )


def ensure_backend(repo, backend_dir=None):
    """Explicit setup step. Reuse a verified build; never overwrite old source."""
    directory = backend_path(repo, backend_dir)
    if directory.exists():
        receipt = read_json(directory / "roi-backend.json")
        if receipt.get("state") == "built":
            return _verify_backend_at(repo, directory)
        _verify_backend_at(repo, directory, built=False)
    else:
        receipt = prepare_backend_source(repo, directory)
    attempts = receipt.get("build_attempts", [])
    if len(attempts) >= 2:
        raise RuntimeError(
            "ROI backend build already failed twice; retained logs require inspection"
        )
    log = directory / f"build-{len(attempts) + 1:03d}.log"
    attempt = dict(log=log.name)
    attempts.append(attempt)
    receipt["build_attempts"] = attempts
    try:
        with log.open("x", encoding="utf-8") as stream:
            subprocess.run(
                [sys.executable, "setup.py", "build_ext", "--inplace"],
                cwd=directory,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
    except Exception as error:
        attempt.update(state="failed", error=str(error))
        receipt["state"] = "build_failed"
        save_json(directory / "roi-backend.json", receipt)
        raise
    attempt["state"] = "completed"
    extensions = list((directory / MODULE).glob("_C*.so"))
    if len(extensions) != 1:
        raise RuntimeError("Independent ROI extension build did not produce one module")
    receipt.update(
        state="built",
        runtime=_runtime(),
        extension=extensions[0].relative_to(directory).as_posix(),
        extension_sha256=sha256(extensions[0]),
        build_log_sha256=sha256(log),
    )
    save_json(directory / "roi-backend.json", receipt)
    return _verify_backend_at(repo, directory)


def verify_backend(repo, backend_dir=None):
    """Read-only verification of a separately built, byte-bound backend."""
    return _verify_backend_at(repo, backend_path(repo, backend_dir))


def _verify_backend_at(repo, directory, *, built=True):
    _, original_files = _verify_source(repo)
    receipt = read_json(directory / "roi-backend.json")
    if (
        receipt.get("schema") != "msv2-roi-backend-v1"
        or receipt.get("state")
        not in (("built",) if built else ("source_prepared", "build_failed"))
        or receipt.get("module") != MODULE
        or receipt.get("source_commit") != COMMIT
        or receipt.get("source_archive_sha256") != ARCHIVE_SHA
        or receipt.get("patch_sha256") != sha256(_patch_path())
        or receipt.get("adapter_sha256") != sha256(__file__)
        or (built and receipt.get("runtime") != _runtime())
        or receipt.get("original_source_files") != original_files
    ):
        raise RuntimeError(
            "ROI backend receipt/code/runtime does not match this pipeline"
        )
    expected_paths = {
        f"{MODULE}/__init__.py" if name.endswith("/__init__.py") else name
        for name in original_files
    }
    if set(receipt.get("patched_source_files", {})) != expected_paths:
        raise RuntimeError("ROI backend source receipt is incomplete")
    for name, expected in receipt.get("patched_source_files", {}).items():
        if sha256(_inside(directory, name)) != expected:
            raise RuntimeError(f"ROI backend patched source changed: {name}")
    if built:
        extension = _inside(directory, receipt["extension"])
        if sha256(extension) != receipt.get("extension_sha256"):
            raise RuntimeError("ROI backend binary changed after setup")
    return receipt


def viewport(camera):
    """Validate current-render geometry without importing Torch or CUDA."""
    value = camera._msv2_roi
    fields = []
    for key in ("full_wh", "viewport_xy", "wh"):
        pair = value.get(key)
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in pair)
        ):
            raise ValueError(f"ROI {key} must contain two integers")
        fields.extend(pair)
    full_w, full_h, x, y, width, height = fields
    if (
        min(full_w, full_h, width, height) <= 0
        or x < 0
        or y < 0
        or x >= full_w
        or y >= full_h
    ):
        raise ValueError("ROI must intersect a positive full image")
    if any(number % 16 for number in (x, y, width, height)):
        raise ValueError(
            "ROI origin and output extent must align with native 16-pixel tiles"
        )
    if (width, height) != (camera.image_width, camera.image_height):
        raise ValueError("ROI dimensions differ from the camera output dimensions")
    return full_w, full_h, x, y, width, height


def wrap_renderer(native_render_imp, repo, backend_dir=None):
    """Dispatch ROI cameras to a private MS backend; mothers remain native.

    ROI camera FoV, world_view_transform and full_proj_transform must come from
    its complete mother camera. The optional shared dummy is the exact mother
    tensor: both renders report its gradient in the same complete-camera NDC.
    No cropped-FoV camera matrix or factor-four dummy scaling is used here.
    """
    if getattr(native_render_imp, "_msv2_roi_wrapper", False):
        raise RuntimeError("ROI renderer is already installed")
    directory = backend_path(repo, backend_dir)
    backend = None
    stats = dict(roi_calls=0, full_calls=0, last_roi_pixels=0, last_full_pixels=0)

    @wraps(native_render_imp)
    def render(
        camera,
        pc,
        pipe,
        bg_color,
        scaling_modifier=1.0,
        override_color=None,
        flag_max_count=True,
        culling=None,
    ):
        nonlocal backend
        if not hasattr(camera, "_msv2_roi"):
            stats["full_calls"] += 1
            return native_render_imp(
                camera,
                pc,
                pipe,
                bg_color,
                scaling_modifier=scaling_modifier,
                override_color=override_color,
                flag_max_count=flag_max_count,
                culling=culling,
            )
        full_w, full_h, x, y, width, height = viewport(camera)
        import torch

        if backend is None:
            _verify_backend_at(repo, directory)
            sys.path.insert(0, str(directory))
            try:
                backend = importlib.import_module(MODULE)
            finally:
                sys.path.remove(str(directory))
            if Path(backend.__file__).resolve() != directory / MODULE / "__init__.py":
                raise RuntimeError(
                    "ROI module was loaded from another backend directory"
                )
        points = pc.get_xyz
        dummy = getattr(camera, "_msv2_shared_viewspace_points", None)
        if dummy is None:
            dummy = torch.zeros_like(points, requires_grad=True) + 0
            if dummy.requires_grad:
                dummy.retain_grad()
        elif (
            dummy.shape != points.shape
            or dummy.device != points.device
            or dummy.dtype != points.dtype
        ):
            raise ValueError(
                "Shared mother NDC dummy must match the current Gaussian tensor"
            )
        settings = backend.GaussianRasterizationSettings(
            image_height=height,
            image_width=width,
            full_image_height=full_h,
            full_image_width=full_w,
            viewport_x=x,
            viewport_y=y,
            tanfovx=math.tan(camera.FoVx * 0.5),
            tanfovy=math.tan(camera.FoVy * 0.5),
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=camera.camera_center,
            prefiltered=False,
            debug=pipe.debug,
        )
        scales = rotations = covariance = None
        if pipe.compute_cov3D_python:
            covariance = pc.get_covariance(scaling_modifier)
        else:
            scales, rotations = pc.get_scaling, pc.get_rotation
        dc = shs = colors = None
        if override_color is not None:
            colors = override_color
        elif pipe.convert_SHs_python:
            eval_sh = importlib.import_module("utils.sh_utils").eval_sh
            features = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            directions = points - camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            directions = directions / directions.norm(dim=1, keepdim=True)
            colors = torch.clamp_min(
                eval_sh(pc.active_sh_degree, features, directions) + 0.5, 0
            )
        else:
            dc, shs = pc.get_features_dc, pc.get_features_rest
        if culling is None:
            culling = torch.zeros(
                points.shape[0], dtype=torch.bool, device=points.device
            )
        image, radii, area = backend.GaussianRasterizer(raster_settings=settings)(
            means3D=points,
            means2D=dummy,
            dc=dc,
            shs=shs,
            culling=culling,
            colors_precomp=colors,
            opacities=pc.get_opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=covariance,
            flag_max_count=flag_max_count,
        )
        stats.update(
            roi_calls=stats["roi_calls"] + 1,
            last_roi_pixels=width * height,
            last_full_pixels=full_w * full_h,
        )
        return dict(
            render=image,
            viewspace_points=dummy,
            visibility_filter=(radii > 0).nonzero(),
            radii=radii,
            area_max=area,
        )

    render._msv2_roi_wrapper = True
    render.roi_stats = stats
    return render
