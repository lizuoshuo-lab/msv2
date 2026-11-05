"""Failure boundaries that must hold before a run can be accepted."""

import os

import pytest

from msv2.engine import COMMIT, SOURCE_ARCHIVE_SHA, verify_source
from msv2.io import read_json, save_json, sha256


@pytest.mark.parametrize("change", ["modify", "remove", "add", "version"])
def test_pinned_source_rejects_drift(tmp_path, change):
    source = tmp_path / "source"
    training = source / "msv2/train.py"
    training.parent.mkdir(parents=True)
    training.write_text("fixed upstream training source\n")
    receipt = dict(
        commit=COMMIT,
        archive_sha256=SOURCE_ARCHIVE_SHA,
        files={"msv2/train.py": sha256(training)},
    )
    save_json(source / ".msv2-source.json", receipt)
    verify_source(source)
    if change == "modify":
        training.write_text("changed loss\n")
    elif change == "remove":
        training.unlink()
    elif change == "add":
        (training.parent / "injected.py").write_text("unexpected source\n")
    else:
        receipt["commit"] = "other version"
        save_json(source / ".msv2-source.json", receipt)
    with pytest.raises(RuntimeError):
        verify_source(source)


@pytest.mark.skipif(os.name != "posix", reason="Linux queue process locks")
def test_interrupted_queue_records_terminal_state(tmp_path, monkeypatch):
    from msv2 import queue

    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "capture.mp4").write_bytes(b"input upload")
    monkeypatch.setattr(queue, "stable_candidate", lambda *args: True)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(queue, "run_pipeline", interrupt)
    with pytest.raises(KeyboardInterrupt):
        queue.watch(tmp_path, poll_seconds=0)
    receipts = list((tmp_path / "queue/tasks").glob("*.json"))
    assert len(receipts) == 1
    assert read_json(receipts[0])["state"] == "interrupted"
    assert read_json(tmp_path / "queue/watcher.json")["state"] == "stopped"


@pytest.mark.skipif(os.name != "posix", reason="Linux queue process locks")
def test_upload_rename_does_not_stop_the_queue(tmp_path, monkeypatch):
    from msv2 import queue

    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "capture.mp4").write_bytes(b"input upload")

    def renamed(*args):
        raise FileNotFoundError("upload renamed during scan")

    def stop_after_scan(*args):
        raise KeyboardInterrupt()

    monkeypatch.setattr(queue, "stable_candidate", renamed)
    monkeypatch.setattr(queue.time, "sleep", stop_after_scan)
    with pytest.raises(KeyboardInterrupt):
        queue.watch(tmp_path, poll_seconds=0)
    assert read_json(tmp_path / "queue/watcher.json")["active_input"] is None
    assert not (tmp_path / "queue/tasks").exists()
