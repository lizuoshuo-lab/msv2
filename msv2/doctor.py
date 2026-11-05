"""Bounded environment and native CUDA forward/backward checks."""

import importlib
import os
from pathlib import Path
import shutil
import sys
import tempfile

from .config import data_root
from .engine import LazyCamera, verify_source
from .io import now, save_json


def doctor(root=None, cpu_only=False):
    root = data_root(root)
    source = Path(os.environ.get("MSV2_SOURCE_DIR", root / "vendor/mini-splatting2"))
    result = dict(
        state="checking",
        checked_at=now(),
        python=sys.version,
        source=str(source),
        tools={},
        cuda_checked=not cpu_only,
        training_completed=False,
    )
    try:
        for name in ("ffmpeg", "ffprobe", "colmap"):
            executable = shutil.which(name)
            if executable is None:
                raise RuntimeError("Executable missing from PATH: " + name)
            result["tools"][name] = executable
        if not (source / "msv2/train.py").is_file():
            raise RuntimeError("Pinned MSV2 source is missing; run deploy.sh")
        verify_source(source)
        if not cpu_only:
            import numpy as np
            import torch

            sys.path.insert(0, str(source))
            for module in (
                "simple_knn._C",
                "diff_gaussian_rasterization",
                "diff_gaussian_rasterization_ms",
                "fused_ssim",
            ):
                importlib.import_module(module)
            if not torch.cuda.is_available():
                raise RuntimeError("NVIDIA CUDA device is unavailable")
            from scene.gaussian_model import GaussianModel
            from gaussian_renderer import render_imp
            from fused_ssim import fused_ssim
            from utils.graphics_utils import BasicPointCloud
            from arguments import OptimizationParams
            from argparse import ArgumentParser
            from types import SimpleNamespace

            parser = ArgumentParser()
            optimization = OptimizationParams(parser)
            options = optimization.extract(parser.parse_args([]))
            generator = np.random.default_rng(2026)
            xyz = generator.uniform(-0.25, 0.25, (32, 3))
            xyz[:, 2] += 2
            model = GaussianModel(0)
            model.create_from_pcd(
                BasicPointCloud(
                    xyz, generator.uniform(0, 1, (32, 3)), np.zeros_like(xyz)
                ),
                1.0,
            )
            model.training_setup(options)
            with tempfile.TemporaryDirectory(prefix="msv2-doctor-") as temporary:
                folder = Path(temporary)
                cache = folder / "rgb-cache-1"
                cache.mkdir()
                np.save(cache / "probe.npy", np.full((32, 32, 3), 127, dtype=np.uint8))
                row = dict(
                    frame="probe",
                    wh=[32, 32],
                    K=[[32, 0, 16], [0, 32, 16], [0, 0, 1]],
                    w2c=np.eye(4).tolist(),
                    image=str(folder / "dataset/images/probe.png"),
                )
                camera = LazyCamera(row, 0)
                pipe = SimpleNamespace(
                    convert_SHs_python=False, compute_cov3D_python=False, debug=False
                )
                image = render_imp(camera, model, pipe, torch.zeros(3, device="cuda"))[
                    "render"
                ]
                target = camera.original_image
                loss = 0.8 * (image - target).abs().mean() + 0.2 * (
                    1 - fused_ssim(image[None], target[None])
                )
                loss.backward()
                torch.cuda.synchronize()
                assert torch.isfinite(loss) and torch.isfinite(model._xyz.grad).all()
            result.update(
                torch=torch.__version__,
                cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(0),
                kernel_forward_backward=True,
                optimizer_updates=0,
                probe_loss=float(loss),
            )
        result["state"] = "passed"
        return result
    except Exception as error:
        result.update(state="failed", error=repr(error))
        raise
    finally:
        save_json(root / "doctor.json", result)
