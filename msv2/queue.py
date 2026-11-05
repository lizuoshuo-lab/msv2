"""Stable-upload queue with content/config deduplication and explicit failures."""

import os
from pathlib import Path
import time

from .config import data_root, load_config
from .io import now, read_json, save_json
from .pipeline import (
    process_lock,
    run_pipeline,
    source_fingerprint,
    task_fingerprint,
    implementation_fingerprint,
)
import sys

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


def stable_candidate(path, observed, current_time, stable_seconds):
    stat = path.stat()
    signature = (stat.st_size, stat.st_mtime_ns)
    previous = observed.get(str(path))
    if previous is None or previous["signature"] != signature:
        observed[str(path)] = dict(signature=signature, since=current_time)
        return False
    return stat.st_size > 0 and current_time - previous["since"] >= stable_seconds


def watch(root=None, config=None, poll_seconds=5):
    root = data_root(root)
    videos = root / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    config = Path(config) if config else root / "config.json"
    observed = {}
    startup_code = implementation_fingerprint()
    with process_lock(root / "queue/watcher.lock"):
        status = dict(
            state="watching", pid=os.getpid(), started=now(), active_input=None
        )
        save_json(root / "queue/watcher.json", status)
        print("MSV2_WATCHING", videos, flush=True)
        try:
            while True:
                if implementation_fingerprint() != startup_code:
                    # A running job finishes from its snapshot before the queue adopts an update.
                    os.execv(
                        sys.executable,
                        [
                            sys.executable,
                            "-s",
                            "-m",
                            "msv2",
                            "watch",
                            "--root",
                            str(root),
                            "--config",
                            str(config),
                        ],
                    )
                options = load_config(config if config.exists() else None)
                for path in sorted(videos.iterdir()):
                    if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
                        continue
                    try:
                        if not stable_candidate(
                            path,
                            observed,
                            time.monotonic(),
                            options["upload_stable_seconds"],
                        ):
                            continue
                        source_hash = source_fingerprint(path)
                    except OSError:
                        # A completed upload can be renamed or removed while a scan is in progress.
                        continue
                    task_hash = task_fingerprint(source_hash, options)
                    receipt = root / "queue/tasks" / (task_hash + ".json")
                    if receipt.exists():
                        continue
                    status.update(active_input=str(path), updated=now())
                    save_json(root / "queue/watcher.json", status)
                    save_json(
                        receipt,
                        dict(
                            state="running",
                            pid=os.getpid(),
                            input=str(path),
                            started=now(),
                        ),
                    )
                    try:
                        run, state = run_pipeline(path, options, root)
                        save_json(
                            receipt,
                            dict(
                                state=state["state"],
                                run=str(run),
                                input=str(path),
                                finished=now(),
                            ),
                        )
                    except BaseException as error:
                        interrupted = isinstance(error, (KeyboardInterrupt, SystemExit))
                        save_json(
                            receipt,
                            dict(
                                state="interrupted" if interrupted else "failed",
                                input=str(path),
                                error=repr(error),
                                finished=now(),
                            ),
                        )
                        if interrupted:
                            raise
                        print("MSV2_INPUT_FAILED", path, repr(error), flush=True)
                    status["active_input"] = None
                status.update(updated=now())
                save_json(root / "queue/watcher.json", status)
                time.sleep(poll_seconds)
        finally:
            status.update(state="stopped", updated=now())
            save_json(root / "queue/watcher.json", status)


def status(root=None):
    root = data_root(root)
    watcher = root / "queue/watcher.json"
    result = dict(
        root=str(root),
        videos=str(root / "videos"),
        watcher=read_json(watcher) if watcher.exists() else None,
        runs=[],
    )
    for path in sorted((root / "runs").glob("*/state.json"), reverse=True):
        item = read_json(path)
        item["run"] = str(path.parent)
        phase = item.get("phase", "").split("/")[0]
        progress = path.parent / phase / "progress.json"
        if progress.exists():
            item["progress"] = read_json(progress)
        result["runs"].append(item)
    return result
