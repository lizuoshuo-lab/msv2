"""One input, one immutable run, fresh training for every requested scale."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import traceback

from .config import data_root
from .engine import COMMIT, Engine, NAMES, import_file
from .io import now, read_json, run_logged, save_json, sha256

WEIGHT_NAME = "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"


@contextmanager
def process_lock(path):
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another worker owns " + str(path)) from error
        try:
            yield stream.fileno()
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def source_fingerprint(source):
    source = Path(source).resolve()
    if source.is_file():
        return sha256(source)
    if not source.is_dir():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    files = sorted(
        p
        for p in source.rglob("*")
        if p.is_file()
        and (
            p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
            or p.name
            in (
                "cameras.bin",
                "images.bin",
                "points3D.bin",
                "cameras.txt",
                "images.txt",
                "points3D.txt",
            )
        )
    )
    if not files:
        raise ValueError("No images or COLMAP model files found")
    for path in files:
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def implementation_fingerprint(package=None):
    digest = hashlib.sha256()
    for path in sorted(
        (Path(package) if package else Path(__file__).parent).glob("*.py")
    ):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def task_fingerprint(source_hash, options):
    value = dict(
        input=source_hash,
        options=options,
        upstream=COMMIT,
        implementation=implementation_fingerprint(),
    )
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def run_pipeline(source, options, root=None, name=None):
    root = data_root(root)
    source = Path(source).expanduser().resolve()
    if name is not None and (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", name) or ".." in name
    ):
        raise ValueError(
            "Run name must contain letters, digits, underscores, hyphens or dots"
        )
    input_hash = source_fingerprint(source)
    task_hash = task_fingerprint(input_hash, options)
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", source.stem).strip("-") or "capture"
    name = (
        name
        or f"{slug[:40]}-{task_hash[:10]}-{now().replace(':', '').replace('-', '').split('.')[0]}"
    )
    run = root / "runs" / name
    with process_lock(root / "queue/gpu.lock") as gpu_lock_fd:
        run.mkdir(parents=True, exist_ok=False)
        snapshot = run / "pipeline-source"
        shutil.copytree(
            Path(__file__).parent,
            snapshot / "msv2",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        code_sha = implementation_fingerprint(snapshot / "msv2")
        if code_sha != implementation_fingerprint():
            raise RuntimeError(
                "Pipeline source changed while creating its snapshot; retry after the update finishes"
            )
        job = dict(
            schema="msv2-job-v1",
            run_dir=str(run),
            input=str(source),
            input_sha256=input_hash,
            task_sha256=task_hash,
            options=options,
            created=now(),
            implementation_sha256=code_sha,
            source_dir=os.environ.get(
                "MSV2_SOURCE_DIR", str(root / "vendor/mini-splatting2")
            ),
            swinir_source=os.environ.get(
                "MSV2_SWINIR_SOURCE", str(root / "vendor/SwinIR")
            ),
            swinir_weight=os.environ.get(
                "MSV2_SWINIR_WEIGHT", str(root / "weights" / WEIGHT_NAME)
            ),
        )
        job_file = run / "job.json"
        save_json(job_file, job)
        state = dict(
            state="running",
            pid=os.getpid(),
            created=now(),
            input=str(source),
            input_sha256=input_hash,
            task_sha256=task_hash,
            phase="prepare",
            completed_scales=[],
            failures={},
            training_completed=False,
            quality_accepted=False,
        )

        def update(phase):
            state.update(phase=phase, updated=now())
            save_json(run / "state.json", state)

        def stage(action, scale=1):
            update(f"{NAMES[scale]}/{action}")
            env = os.environ.copy()
            env.update(
                OMP_NUM_THREADS=str(options["cpu_threads"]),
                PYTHONUNBUFFERED="1",
                PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
            )
            env["PYTHONPATH"] = str(snapshot) + os.pathsep + env.get("PYTHONPATH", "")
            run_logged(
                [
                    sys.executable,
                    "-s",
                    "-m",
                    "msv2.worker",
                    action,
                    str(job_file),
                    "--scale",
                    str(scale),
                ],
                run / "logs" / f"{NAMES[scale]}-{action}.log",
                cwd=snapshot,
                env=env,
                pass_fds=(gpu_lock_fd,),
            )

        try:
            update("prepare")
            stage("prepare")
            if source_fingerprint(source) != input_hash:
                raise RuntimeError(
                    "Input changed during preparation; finish the upload and retry"
                )
            engine = Engine(job_file)
            stage("initialize")
            state["frames"] = len(engine.native_rows())
            # All metrics and export checks share the original image domain,
            # even if only an enlarged variant is requested for training.
            stage("pixels", 1)
            stage("cameras", 1)
            for scale in options["scales"]:
                try:
                    if scale == 4:
                        stage("sr", 4)
                    elif scale != 1:
                        stage("pixels", scale)
                    if scale != 1:
                        stage("cameras", scale)
                    stage("preflight", scale)
                    stage("train", scale)
                    stage("export", scale)
                    stage("evaluate", scale)
                    state["completed_scales"].append(scale)
                except Exception as error:
                    state["failures"][str(scale)] = repr(error)
                    (run / "logs" / f"{NAMES[scale]}-error.txt").write_text(
                        traceback.format_exc(), encoding="utf-8"
                    )
                    # Finished variants remain deliverable if a larger variant
                    # cannot fit the target GPU; failures remain explicit.
                update("variant_finished")
            report = import_file("msv2.frozen_report", snapshot / "msv2/report.py")
            report.publish(engine, state["completed_scales"], state["failures"])
            complete = len(state["completed_scales"]) == len(options["scales"])
            state.update(
                state="completed"
                if complete
                else "partial"
                if state["completed_scales"]
                else "failed",
                finished=now(),
                training_completed=complete,
                report=str(run / "export/comparison.html"),
            )
            plan = read_json(run / "plan.json")
            plan.update(
                state=state["state"],
                training_completed=complete,
                finished=state["finished"],
            )
            save_json(run / "plan.json", plan)
        except BaseException as error:
            state.update(
                state="interrupted"
                if isinstance(error, (KeyboardInterrupt, SystemExit))
                else "failed",
                error=repr(error),
                finished=now(),
            )
            (run / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise
        finally:
            save_json(run / "state.json", state)
            print(
                json.dumps(
                    dict(
                        state=state["state"],
                        run=str(run),
                        completed_scales=state["completed_scales"],
                        failures=state["failures"],
                    ),
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return run, state
