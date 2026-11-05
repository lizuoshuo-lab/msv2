"""Public command line; all paths and dataset-specific choices are explicit."""

import argparse
import json
import os
from pathlib import Path

from . import __version__
from .config import data_root, load_config


def default_config(root, supplied):
    if supplied:
        return Path(supplied)
    candidate = root / "config.json"
    return candidate if candidate.exists() else None


def parse_scales(value):
    try:
        return [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Use comma-separated scales such as 1,2,4"
        ) from error


def parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--root", default=argparse.SUPPRESS, help="data, environment and run directory"
    )
    result = argparse.ArgumentParser(
        prog="msv2",
        description="Video or COLMAP → Mini-Splatting2 → standard PLY and cameras",
        parents=[common],
    )
    result.add_argument("--version", action="version", version=__version__)
    commands = result.add_subparsers(dest="command", required=True)
    train = commands.add_parser(
        "train", parents=[common], help="start a fresh reconstruction"
    )
    train.add_argument("input")
    train.add_argument("--config")
    train.add_argument("--scales", type=parse_scales)
    train.add_argument("--iterations", type=int)
    train.add_argument(
        "--name", help="unique run name; an existing run is never overwritten"
    )
    queue = commands.add_parser(
        "watch", parents=[common], help="train finished uploads in videos/"
    )
    queue.add_argument("--config")
    commands.add_parser(
        "status", parents=[common], help="show runs, progress and output paths"
    )
    check = commands.add_parser(
        "doctor", parents=[common], help="verify tools and native CUDA kernels"
    )
    check.add_argument(
        "--cpu",
        action="store_true",
        help="check preprocessing tools only, without a CUDA claim",
    )
    prepare = commands.add_parser(
        "prepare",
        parents=[common],
        help="prepare and register a video or import COLMAP data",
    )
    prepare.add_argument("input")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--config")
    extract = commands.add_parser(
        "extract-head",
        parents=[common],
        help="extract native head masks and transparent pixels on CPU",
    )
    extract.add_argument(
        "input", help="image directory, dataset or prepared dataset directory"
    )
    extract.add_argument(
        "--output", required=True, help="new directory outside the source"
    )
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    root = data_root(getattr(args, "root", None))
    if args.command == "train":
        from .pipeline import run_pipeline

        options = load_config(
            default_config(root, args.config),
            dict(scales=args.scales, iterations=args.iterations),
        )
        _, state = run_pipeline(args.input, options, root, args.name)
        return 0 if state["state"] == "completed" else 1
    if args.command == "watch":
        from .queue import watch

        watch(root, args.config)
        return 0
    if args.command == "status":
        from .queue import status

        result = status(root)
    elif args.command == "doctor":
        from .doctor import doctor

        result = doctor(root, args.cpu)
    elif args.command == "extract-head":
        from .head_extract import extract_head

        result = extract_head(
            args.input,
            args.output,
            parser_source=os.environ.get(
                "MSV2_HEAD_PARSER_SOURCE", str(root / "vendor/head-parser")
            ),
            parser_weight=os.environ.get(
                "MSV2_HEAD_PARSER_WEIGHT",
                str(root / "weights/face_parsing_resnet18.pt"),
            ),
            detector_weight=os.environ.get(
                "MSV2_HEAD_DETECTOR_WEIGHT",
                str(root / "weights/face_detection_yunet_2023mar.onnx"),
            ),
        )
    else:
        from .data import prepare_colmap, prepare_video

        source = Path(args.input).expanduser().resolve()
        options = load_config(default_config(root, args.config))
        options["source_dir"] = os.environ.get(
            "MSV2_SOURCE_DIR", str(root / "vendor/mini-splatting2")
        )
        prepared = (prepare_colmap if source.is_dir() else prepare_video)(
            source, Path(args.output), options
        )
        result = dict(state="prepared", dataset=str(prepared), training_completed=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
