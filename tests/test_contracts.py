"""CPU contracts for full-resolution inputs, queue intake and task identity."""

import json
import math
import os
import shutil

import numpy as np
from PIL import Image, PngImagePlugin
import pytest

from msv2.config import load_config
from msv2.engine import Engine
from msv2.io import save_json, sha256
from msv2.pipeline import source_fingerprint, task_fingerprint
from msv2.queue import stable_candidate


def configured(tmp_path, values):
    path = tmp_path / "options.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    return load_config(path)


def line_image(width, height):
    """A one-pixel white line cannot survive a hidden spatial reduction intact."""
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[:, width // 2, :] = 255
    pixels[0, 0, 0] = 197
    pixels[-1, -1, 2] = 83
    return Image.fromarray(pixels)


@pytest.fixture
def pixel_engine(tmp_path):
    run = tmp_path / "independent-run"
    images = run / "prepared/dataset/images"
    images.mkdir(parents=True)
    records = []
    # Odd, different dimensions also represent different calibrated cameras;
    # neither has a relationship to the historic demo's 2704x1520 dimensions.
    for index, (width, height) in enumerate(((63, 35), (47, 29), (53, 27)), 1):
        name = f"frame_{index:06d}.png"
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("capture-note", f"camera-{index}: original full RGB")
        with line_image(width, height) as image:
            image.save(images / name, pnginfo=metadata, compress_level=9)
        records.append(
            dict(image=name, sha256=sha256(images / name), wh=[width, height])
        )
    save_json(
        run / "prepared/pixel-lineage.json",
        dict(passed=True, no_spatial_resizing=True, no_mask=True, records=records),
    )
    job = run / "job.json"
    save_json(
        job,
        dict(
            options=load_config(),
            run_dir=str(run),
            source_dir=str(tmp_path / "unused-upstream"),
            swinir_source=str(tmp_path / "unused-swinir"),
            swinir_weight=str(tmp_path / "unused-weight.pth"),
        ),
    )
    return Engine(job), records


def test_scale_one_preserves_all_original_file_bytes(pixel_engine):
    engine, records = pixel_engine
    originals = {
        row["image"]: (engine.prep / "dataset/images" / row["image"]).read_bytes()
        for row in records
    }
    engine.stage_pixels(1)
    for row in records:
        source = engine.prep / "dataset/images" / row["image"]
        target = engine.root / "original/dataset/images" / row["image"]
        # This checks PNG ancillary data as well as pixels: decoding and
        # recompressing an otherwise identical image is not byte preservation.
        assert target.read_bytes() == originals[row["image"]]
        assert source.read_bytes() == originals[row["image"]]
        assert sha256(target) == row["sha256"]


def test_scale_two_keeps_native_one_pixel_detail_and_pixel_phase(pixel_engine):
    engine, records = pixel_engine
    engine.stage_pixels(2)
    for row in records:
        width, height = row["wh"]
        target = engine.root / "interpolation2x/dataset/images" / row["image"]
        with Image.open(target) as image:
            assert image.format == "PNG"
            assert image.mode == "RGB"
            assert image.size == (width * 2, height * 2)
            pixels = np.asarray(image)
        scan = pixels[height, :, 1].astype(np.float64)
        peak_columns = np.flatnonzero(scan >= 220)
        expected_center = 2 * (width // 2) + 0.5
        # Direct 2x interpolation retains two high-contrast samples of the
        # native white line. Half-size preprocessing loses this contrast;
        # enlarging that reduced input cannot restore it.
        assert peak_columns.tolist() == [2 * (width // 2), 2 * (width // 2) + 1]
        centroid = float(np.dot(scan, np.arange(len(scan))) / scan.sum())
        assert centroid == pytest.approx(expected_center, abs=0.05)
        assert sha256(engine.prep / "dataset/images" / row["image"]) == row["sha256"]


@pytest.mark.parametrize(
    "values",
    [
        {"sample_fps": 0},
        {"sample_fps": -1},
        {"sample_fps": math.nan},
        {"sample_fps": math.inf},
        {"sample_fps": True},
        {"scales": []},
        {"scales": [1, 1]},
        {"scales": [3]},
        {"scales": [True]},
        {"frame_indices": []},
        {"frame_indices": [1, 0]},
        {"frame_indices": [0, 0]},
        {"frame_indices": [-1]},
        {"frame_indices": [0, 1.5]},
        {"frame_indices": [False]},
        {"seed": -1},
        {"seed": 2**32},
        {"seed": 0.5},
        {"seed": True},
        {"iterations": 0},
        {"iterations": 1},
        {"iterations": 3000},
        {"iterations": 8000},
        {"iterations": True},
        {"cpu_threads": 0},
        {"holdout": 1},
        {"gpu_memory_fraction": 0},
        {"gpu_memory_fraction": 1},
        {"gpu_memory_fraction": math.nan},
        {"sr_tile": 255},
        {"sr_tile_overlap": -1},
        {"sr_tile_overlap": 256},
        {"sr_tile_overlap": 1.5},
        {"sr_tile_overlap": True},
        {"upload_stable_seconds": -1},
        {"upload_stable_seconds": math.nan},
        {"upload_stable_seconds": math.inf},
        {"camera_model": "FISHEYE_8"},
        {"single_camera": "yes"},
        {"unexpected_configuration_key": 3},
    ],
)
def test_invalid_options_fail_before_preparation_or_gpu(tmp_path, values):
    with pytest.raises(ValueError):
        configured(tmp_path, values)


def test_valid_boundaries_allow_non_demo_configuration(tmp_path):
    options = configured(
        tmp_path,
        dict(
            sample_fps=1.25,
            frame_indices=[0, 7, 31, 100],
            scales=[4],
            iterations=8001,
            seed=2**32 - 1,
            holdout=3,
            single_camera=False,
            gpu_memory_fraction=0.95,
            sr_tile=16,
            sr_tile_overlap=0,
            upload_stable_seconds=0,
        ),
    )
    assert options["frame_indices"] == [0, 7, 31, 100]
    assert options["scales"] == [4]
    assert options["iterations"] == 8001
    assert options["single_camera"] is False
    assert "source_wh" not in options and "expected_frames" not in options


def test_default_options_are_independent_between_jobs():
    first = load_config()
    first["scales"].clear()
    second = load_config()
    assert second["scales"] == [1, 2, 4]
    assert second["sample_fps"] == 3
    assert second["iterations"] == 18000


def test_upload_requires_a_complete_stable_window_and_restarts_on_changes(tmp_path):
    video = tmp_path / "uploaded video.mp4"
    video.write_bytes(b"first upload chunk")
    observed = {}
    assert not stable_candidate(video, observed, 0.0, 30)
    assert not stable_candidate(video, observed, 29.999, 30)
    assert stable_candidate(video, observed, 30.0, 30)

    video.write_bytes(b"first upload chunk followed by another chunk")
    assert not stable_candidate(video, observed, 31.0, 30)
    assert not stable_candidate(video, observed, 60.999, 30)
    assert stable_candidate(video, observed, 61.0, 30)

    # Replacing bytes without changing size still restarts the window when
    # the upload modifies mtime, as SFTP and normal file writers do.
    old = video.stat()
    video.write_bytes(b"x" * old.st_size)
    os.utime(video, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000_000))
    assert not stable_candidate(video, observed, 62.0, 30)
    assert not stable_candidate(video, observed, 91.999, 30)
    assert stable_candidate(video, observed, 92.0, 30)


def test_an_empty_upload_is_never_a_training_candidate(tmp_path):
    video = tmp_path / "empty.mp4"
    video.write_bytes(b"")
    observed = {}
    assert not stable_candidate(video, observed, 0.0, 1)
    assert not stable_candidate(video, observed, 100.0, 1)


def colmap_fingerprint_fixture(root):
    images = root / "images"
    sparse = root / "sparse/0"
    images.mkdir(parents=True)
    sparse.mkdir(parents=True)
    for name, wh in [("camera-a.png", (63, 35)), ("camera-b.png", (47, 29))]:
        with line_image(*wh) as image:
            image.save(images / name)
    (sparse / "cameras.txt").write_text(
        "1 PINHOLE 63 35 50 50 31.5 17.5\n2 PINHOLE 47 29 40 39 23.5 14.5\n",
        encoding="utf-8",
    )
    (sparse / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 camera-a.png\n31.5 17.5 1\n"
        "2 1 0 0 0 -1 0 0 2 camera-b.png\n10.1666666667 14.5 1\n",
        encoding="utf-8",
    )
    (sparse / "points3D.txt").write_text(
        "1 0 0 3 128 128 128 0.1 1 0 2 0\n", encoding="utf-8"
    )
    return root


def test_dataset_content_identity_ignores_location_and_mtime(tmp_path):
    first = colmap_fingerprint_fixture(tmp_path / "first dataset")
    second = tmp_path / "second dataset"
    shutil.copytree(first, second)
    for path in second.rglob("*"):
        if path.is_file():
            old = path.stat()
            os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000_000))
    assert source_fingerprint(first) == source_fingerprint(second)


def test_dataset_identity_changes_when_pose_or_image_content_changes(tmp_path):
    dataset = colmap_fingerprint_fixture(tmp_path / "dataset")
    original = source_fingerprint(dataset)
    poses = dataset / "sparse/0/images.txt"
    original_poses = poses.read_bytes()
    poses.write_text(poses.read_text().replace("-1 0 0", "-1.1 0 0"), encoding="utf-8")
    assert source_fingerprint(dataset) != original
    poses.write_bytes(original_poses)
    assert source_fingerprint(dataset) == original
    image_path = dataset / "images/camera-a.png"
    with Image.open(image_path) as image:
        pixels = np.array(image)
    pixels[0, 0] = [12, 34, 56]
    Image.fromarray(pixels).save(image_path)
    assert source_fingerprint(dataset) != original


def test_content_and_options_define_separate_queue_tasks(tmp_path, monkeypatch):
    dataset = colmap_fingerprint_fixture(tmp_path / "dataset")
    digest = source_fingerprint(dataset)
    monkeypatch.setattr(
        "msv2.pipeline.implementation_fingerprint", lambda: "frozen-code-fixture"
    )
    original = load_config(overrides={"scales": [1]})
    same = load_config(overrides={"scales": [1]})
    enlarged = load_config(overrides={"scales": [2]})
    assert task_fingerprint(digest, original) == task_fingerprint(digest, same)
    assert task_fingerprint(digest, original) != task_fingerprint(digest, enlarged)
