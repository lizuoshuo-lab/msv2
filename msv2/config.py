"""Validated JSON options; image dimensions and frame counts come from inputs."""

from copy import deepcopy
import math
import os
from pathlib import Path

from .io import read_json

DEFAULTS = dict(
    sample_fps=3.0,
    frame_indices=None,
    scales=[1, 2, 4],
    camera_model="PINHOLE",
    single_camera=True,
    matching="sequential",
    cpu_threads=4,
    iterations=18000,
    seed=0,
    importance_metric="outdoor",
    holdout=8,
    gpu_memory_fraction=0.85,
    sr_region="full",
    sr_tile=256,
    sr_tile_overlap=32,
    upload_stable_seconds=30,
)


def data_root(value=None):
    return (
        Path(value or os.environ.get("MSV2_ROOT", Path.home() / "msv2"))
        .expanduser()
        .resolve()
    )


def load_config(path=None, overrides=None):
    supplied = read_json(path) if path else {}
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError("Unknown configuration options: " + ", ".join(sorted(unknown)))
    options = deepcopy(DEFAULTS)
    options.update(supplied)
    options.update(
        {key: value for key, value in (overrides or {}).items() if value is not None}
    )
    if (
        isinstance(options["sample_fps"], bool)
        or not math.isfinite(options["sample_fps"])
        or options["sample_fps"] <= 0
    ):
        raise ValueError("sample_fps must be positive and finite")
    indices = options["frame_indices"]
    if indices is not None and (
        not indices
        or any(type(i) is not int or i < 0 for i in indices)
        or indices != sorted(set(indices))
    ):
        raise ValueError("frame_indices must be increasing unique nonnegative integers")
    scales = options["scales"]
    if (
        not isinstance(scales, list)
        or not scales
        or len(set(scales)) != len(scales)
        or any(type(s) is not int or s not in (1, 2, 4) for s in scales)
    ):
        raise ValueError("scales must be a nonempty subset of [1, 2, 4]")
    if options["importance_metric"] not in ("outdoor", "indoor"):
        raise ValueError("importance_metric must be outdoor or indoor")
    # Accept the earlier experimental option without restoring crop cameras.
    if options["sr_region"] == "head":
        options["sr_region"] = "face"
    if options["sr_region"] not in ("face", "full"):
        raise ValueError("sr_region must be face or full")
    if options["matching"] not in ("sequential", "exhaustive"):
        raise ValueError("matching must be sequential or exhaustive")
    if (
        options["camera_model"] not in ("PINHOLE", "SIMPLE_PINHOLE")
        or type(options["single_camera"]) is not bool
    ):
        raise ValueError(
            "Use PINHOLE/SIMPLE_PINHOLE and a boolean single_camera option"
        )
    if type(options["seed"]) is not int or not 0 <= options["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if (
        not math.isfinite(options["upload_stable_seconds"])
        or options["upload_stable_seconds"] < 0
    ):
        raise ValueError("upload_stable_seconds must be finite and nonnegative")
    for key in ("cpu_threads", "iterations", "holdout", "sr_tile"):
        if type(options[key]) is not int or options[key] < 1:
            raise ValueError(key + " must be a positive integer")
    if options["iterations"] < 8001:
        raise ValueError(
            "iterations must be at least 8001 to complete the native 3000/8000 simplification schedule"
        )
    if options["holdout"] < 2:
        raise ValueError("holdout must be at least 2")
    if not 0 < options["gpu_memory_fraction"] <= 0.95:
        raise ValueError("gpu_memory_fraction must be in (0, 0.95]")
    if type(options["sr_tile_overlap"]) is not int:
        raise ValueError("sr_tile_overlap must be an integer")
    if (
        not 0 <= options["sr_tile_overlap"] < options["sr_tile"]
        or options["sr_tile"] % 8
    ):
        raise ValueError(
            "SwinIR tile must be divisible by 8, with overlap smaller than tile"
        )
    return options
