"""One complete composite image and camera per source frame."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from msv2.config import load_config
from msv2.engine import Engine
from msv2.export import camera_input_mapping, camera_records
from msv2.io import read_json, save_json
from msv2.report import publish, route_metadata, view_summary


@pytest.mark.parametrize("region", ["face", "full"])
def test_region_is_an_explicit_input_choice(region):
    assert load_config()["sr_region"] == "full"
    assert load_config(overrides={"sr_region": region})["sr_region"] == region


def test_old_region_name_maps_to_whole_image_composition():
    assert load_config(overrides={"sr_region": "head"})["sr_region"] == "face"


@pytest.mark.parametrize("region", ["parse", "crop", "other"])
def test_unknown_regions_are_rejected(region):
    with pytest.raises(ValueError, match="sr_region"):
        load_config(overrides={"sr_region": region})


def test_legacy_job_without_region_retains_full_mode(tmp_path):
    job = tmp_path / "job.json"
    options = load_config()
    options.pop("sr_region")
    save_json(
        job,
        dict(
            options=options,
            run_dir=str(tmp_path / "run"),
            source_dir="unused",
            swinir_source="unused",
            swinir_weight="unused",
        ),
    )
    assert Engine(job).sr_region == "full"


def source_rows(image):
    return [
        dict(
            frame=f"frame_{i:06d}",
            name=f"frame_{i:06d}.png",
            native_image=str(image),
            split="test" if i == 1 else "train",
            wh=[100, 60],
            K=[[70.0, 0.0, 50.0], [0.0, 75.0, 30.0], [0.0, 0.0, 1.0]],
            w2c=np.eye(4).tolist(),
        )
        for i in range(1, 4)
    ]


def full4_rows(base):
    rows = deepcopy(base)
    for row in rows:
        row["wh"] = [4 * v for v in row["wh"]]
        row["K"] = (np.diag([4.0, 4.0, 1.0]) @ np.asarray(row["K"])).tolist()
    return rows


def sr_records(base, face_counts):
    return dict(
        state="completed",
        view_layout="full4x_mask_composite",
        mask_scope="face",
        records=[
            dict(name=r["name"], face_count=count)
            for r, count in zip(base, face_counts)
        ],
    )


@pytest.mark.parametrize(
    "mode,face_counts",
    [
        ("full", [0, 0, 0]),
        ("face", [1, 1, 1]),
        ("face", [0, 0, 0]),
        ("face", [1, 0, 0]),
    ],
)
def test_report_retains_source_frame_count_and_whole_image_training(
    tmp_path, mode, face_counts
):
    engine = SimpleNamespace(
        root=tmp_path / "run",
        record=tmp_path / "export",
        config=load_config(overrides={"sr_region": mode}),
        iterations=18000,
    )
    source = tmp_path / "source.png"
    source.write_bytes(b"report fixture, not a rendered image")
    base = source_rows(source)
    for scale, name in [(1, "original"), (2, "interpolation2x"), (4, "swinir4x")]:
        rows = full4_rows(base) if scale == 4 else deepcopy(base)
        save_json(engine.root / name / "training-inputs.json", dict(records=rows))
        save_json(
            engine.record / name / "delivery.json",
            dict(parity_passed=True, cameras=3, points=100),
        )
        save_json(
            engine.record / name / "metrics.json",
            dict(
                execution=dict(
                    state="training_completed", train_frames=2, test_frames=1
                ),
                means={"test": dict(psnr=20.0, ssim=0.8)},
            ),
        )
    save_json(
        engine.root / "swinir4x/superresolution.json", sr_records(base, face_counts)
    )
    publish(engine, [1, 2, 4], {})
    summary = read_json(engine.record / "comparison.json")
    variant = summary["variants"]["swinir4x"]
    assert summary["source_frames"] == 3
    assert (
        variant["source_frames"] == variant["view_count"] == variant["full_views"] == 3
    )
    assert variant["head_views"] == 0
    assert variant["train_views"] == 2 and variant["test_views"] == 1
    assert (
        not variant["additional_training_views"] and not variant["training_crop_used"]
    )
    assert not variant["training_mask_used"]
    assert variant["full_view_scale"] == 4 and variant["full_frame_4x_generated"]
    assert variant["input_composite_mask"] is (mode == "face" and sum(face_counts) > 0)
    if mode == "face":
        assert variant["input_route"] == "full4x_mask_composite"
        assert variant["face_train_frames"] == sum(n > 0 for n in face_counts[1:])
        assert variant["face_test_frames"] == int(face_counts[0] > 0)
        assert variant["outside_mask_reference"] == "Original full-image Lanczos4"
        if not sum(face_counts):
            assert "无面部" in variant["label"]
    assert summary["quality_accepted"] is False
    html = (engine.record / "comparison.html").read_text(encoding="utf-8")
    assert "每帧只有一套整图相机" in html
    assert "头框四倍视图" not in html


@pytest.mark.parametrize("defect", ["duplicate_source", "crop_view", "duplicate_name"])
def test_duplicate_or_cropped_training_views_are_rejected(defect):
    rows = source_rows("unused.png")
    if defect == "duplicate_source":
        rows[-1]["source_frame"] = rows[0]["frame"]
    elif defect == "crop_view":
        rows[-1]["view_type"] = "head"
    else:
        rows[-1]["name"] = rows[0]["name"]
    with pytest.raises(ValueError):
        view_summary(rows)
    with pytest.raises(ValueError):
        camera_input_mapping(rows, "full")


def test_composite_camera_is_full_frame_uniform_scale_with_same_pose():
    base = source_rows("unused.png")
    rows = full4_rows(base)
    cameras = camera_records(rows)
    assert len(cameras) == len(base) == 3
    point = np.asarray([0.2, -0.15, 2.0])
    for original, camera in zip(base, cameras):
        native_K = np.asarray(original["K"])
        K = np.asarray(camera["K"])
        assert camera["width"] == 400 and camera["height"] == 240
        a = native_K @ point
        b = K @ point
        assert np.allclose(b[:2] / b[2], 4 * a[:2] / a[2], atol=1e-12, rtol=0)
        assert np.array_equal(camera["position"], [0, 0, 0])
        assert np.array_equal(camera["rotation"], np.eye(3))
        assert "native_bbox" not in camera
    data = camera_input_mapping(rows, "face", sr=sr_records(base, [1, 0, 1]))
    assert data["full_frame"] and data["training_resolution"] == [400, 240]
    assert data["view_count"] == data["source_frames"] == 3
    assert data["view_layout"] == "full4x_mask_composite"
    assert data["input_face_mask_used"] and not data["training_mask_used"]
    assert not data["crop_views_present"] and not data["additional_training_views"]


@pytest.mark.parametrize(
    "defect", ["missing", "duplicate", "wrong_layout", "bad_count"]
)
def test_composite_provenance_cannot_mislabel_input_views(tmp_path, defect):
    base = source_rows("unused.png")
    rows = full4_rows(base)
    sr = sr_records(base, [1, 1, 1])
    if defect == "missing":
        sr["records"].pop()
    elif defect == "duplicate":
        sr["records"].append(deepcopy(sr["records"][0]))
    elif defect == "wrong_layout":
        sr["view_layout"] = "full1x_head4x"
    else:
        sr["records"][0]["face_count"] = -1
    with pytest.raises(ValueError):
        camera_input_mapping(rows, "face", sr=sr)
    engine = SimpleNamespace(root=tmp_path, config={"sr_region": "face"})
    save_json(tmp_path / "swinir4x/superresolution.json", sr)
    with pytest.raises(ValueError):
        route_metadata(engine, 4, rows)


def test_standard_camera_json_keeps_original_field_contract():
    rows = source_rows("unused.png")
    cameras = camera_records(rows)
    assert not any(
        key in cameras[0] for key in ("source_frame", "view_type", "native_bbox")
    )
    data = camera_input_mapping(rows)
    assert data["training_resolution"] == [100, 60] and data["view_layout"] == "full"
