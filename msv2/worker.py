"""A stage boundary releases native CUDA state before the next stage."""

import argparse
from pathlib import Path

from .engine import Engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=[
            "prepare",
            "initialize",
            "pixels",
            "sr",
            "cameras",
            "preflight",
            "train",
            "export",
            "evaluate",
        ],
    )
    parser.add_argument("job")
    parser.add_argument("--scale", type=int, choices=[1, 2, 4], default=1)
    args = parser.parse_args()
    engine = Engine(args.job)
    from .pipeline import implementation_fingerprint

    snapshot = engine.root / "pipeline-source/msv2"
    if Path(__file__).parent.resolve() != snapshot.resolve():
        raise RuntimeError("Worker must run from its frozen pipeline snapshot")
    if implementation_fingerprint(snapshot) != engine.job["implementation_sha256"]:
        raise RuntimeError("Frozen pipeline source integrity check failed")
    if args.action == "prepare":
        from .data import prepare_colmap, prepare_video

        source = Path(engine.job["input"])
        options = dict(engine.config, source_dir=str(engine.repo))
        (prepare_colmap if source.is_dir() else prepare_video)(
            source, engine.prep, options
        )
    elif args.action == "initialize":
        engine.initialize()
    elif args.action == "export":
        from .export import export

        export(engine, args.scale)
    elif args.action == "evaluate":
        from .report import evaluate

        evaluate(engine, args.scale)
    elif args.action == "sr":
        engine.superresolve()
    else:
        getattr(
            engine,
            {"pixels": "stage_pixels", "cameras": "stage_cameras"}.get(
                args.action, args.action
            ),
        )(args.scale)


if __name__ == "__main__":
    main()
