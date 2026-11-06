"""Auxiliary ROI cameras never inflate source counts or standard exports."""

from copy import deepcopy
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from msv2.export import (
    camera_input_mapping,
    camera_records,
    paired_camera_records,
    write_colmap_cameras,
    write_scene_bundle,
)
from msv2.io import read_json, save_json
from msv2.paired_data import child_camera
from msv2.report import PAIRED_LAYOUT, paired_view_summary, publish, route_metadata


def fixture(source, child_counts=(1, 2, 0)):
    rows = []
    items = []
    for i, count in enumerate(child_counts, 1):
        name = f"frame_{i:06d}.png"
        frame = name[:-4]
        row = dict(
            frame=frame,
            name=name,
            source_frame=frame,
            native_image=str(source),
            image=str(source),
            split="test" if i == 1 else "train",
            wh=[129, 97],
            K=[[117.0, 0, 65.2], [0, 119.0, 48.7], [0, 0, 1.0]],
            w2c=np.eye(4).tolist(),
            source_sha256=f"native-{i}",
            native_sha256=f"native-{i}",
            children=[],
        )
        for n in range(count):
            roi = [32 + 8 * n, 24, 72 + 8 * n, 64]
            row["children"].append(
                dict(
                    roi_native=roi,
                    wh=[160, 160],
                    image=f"children/{frame}_child_{n + 1}.png",
                    image_sha256=f"child-{i}-{n}",
                    mask=f"children/{frame}_mask_{n + 1}.png",
                    mask_sha256=f"mask-{i}-{n}",
                    mask_pixels=1024,
                    camera=child_camera(row["K"], row["w2c"], row["wh"], roi),
                )
            )
        rows.append(row)
        items.append(
            dict(
                name=name,
                face_count=count,
                children=deepcopy(row["children"]),
                output_sha256=row["native_sha256"],
                composition=dict(source_wh=row["wh"]),
            )
        )
    return rows, dict(state="completed", view_layout=PAIRED_LAYOUT, records=items)


@pytest.mark.parametrize(
    "child_counts,train_masks,child_train,child_test",
    [
        ((1, 2, 0), True, 2, 1),
        ((1, 0, 0), False, 0, 1),
        ((0, 0, 0), False, 0, 0),
    ],
)
def test_source_group_counts_and_real_train_mask_scope(
    child_counts, train_masks, child_train, child_test
):
    rows, sr = fixture("unused.png", child_counts)
    summary = paired_view_summary(rows, sr)
    assert (
        summary["view_count"]
        == summary["source_frames"]
        == summary["random_camera_views"]
        == 3
    )
    assert summary["source_sampling_groups"] == 3 and summary["head_views"] == 0
    assert summary["train_views"] == 2 and summary["test_views"] == 1
    assert summary["auxiliary_train_children"] == child_train
    assert summary["auxiliary_test_children"] == child_test
    assert summary["training_mask_used"] is train_masks
    assert summary["training_crop_used"] is train_masks
    assert summary["full_view_scale"] == 1 and not summary["full_frame_4x_generated"]
    assert summary["joint_source_update"] and not summary["additional_training_views"]
    mapping = camera_input_mapping(rows, "face", sr=sr)
    assert mapping["mother_cameras"] == 3 and mapping["auxiliary_child_cameras"] == sum(
        child_counts
    )
    assert mapping["training_resolution"] == [129, 97]
    assert mapping["mask_used"] is train_masks and not mapping["crop_views_present"]
    assert len(camera_records(rows)) == 3
    assert mapping["view_layout"] == PAIRED_LAYOUT


def test_auxiliary_camera_metadata_keeps_group_split_fullfov_and_export_pose():
    rows, sr = fixture("unused.png")
    transform = np.diag([1.0, -1.0, -1.0, 1.0])
    world = paired_camera_records(rows)
    exported = paired_camera_records(rows, transform=transform)
    assert (
        world["mother_cameras"]
        == world["source_frames"]
        == world["random_sampling_views"]
        == 3
    )
    assert (
        world["auxiliary_child_cameras"] == 3
        and world["children_registered_to_colmap"] is False
    )
    for row, record, yup in zip(rows, world["records"], exported["records"]):
        assert (
            record["source_frame"] == row["frame"] and record["split"] == row["split"]
        )
        assert record["parent"]["wh"] == [129, 97]
        for child, rotated in zip(record["children"], yup["children"]):
            assert (
                child["split"] == record["split"]
                and child["source_frame"] == record["source_frame"]
            )
            assert child["same_update_as_parent"] and child["loss_mask_required"]
            assert child["ordinary_sampling_view"] is False
            c = child["camera"]
            assert c["full_wh"] == [516, 388] and c["wh"] == [160, 160]
            assert c["full_tan_fovx"] == 129 / (2 * 117)
            np.testing.assert_array_equal(
                rotated["camera"]["w2c"],
                np.asarray(c["w2c"]) @ np.linalg.inv(transform),
            )
            np.testing.assert_array_equal(rotated["camera"]["K"], c["K"])
    assert not exported["quality_accepted"] and not exported["targets_in_scene_zip"]


@pytest.mark.parametrize("defect", ["pose", "K", "wh", "viewport"])
def test_frozen_child_camera_tamper_is_rejected(defect):
    rows, _ = fixture("unused.png")
    c = rows[0]["children"][0]["camera"]
    if defect == "pose":
        c["w2c"][0][3] += 0.1
    elif defect == "K":
        c["K"][0][2] += 4
    elif defect == "wh":
        rows[0]["children"][0]["wh"][0] += 1
    else:
        c["viewport"][0] += 16
    with pytest.raises(ValueError, match="Auxiliary camera"):
        paired_camera_records(rows)


@pytest.mark.parametrize(
    "defect", ["missing", "duplicate", "full4mother", "PNG", "children"]
)
def test_pairing_provenance_cannot_relabel_extra_views_or_fourfold_mothers(defect):
    rows, sr = fixture("unused.png")
    if defect == "missing":
        sr["records"].pop()
    elif defect == "duplicate":
        sr["records"].append(deepcopy(sr["records"][0]))
    elif defect == "full4mother":
        rows[0]["wh"] = [516, 388]
    elif defect == "PNG":
        rows[0]["source_sha256"] = "reencoded"
    else:
        rows[0]["children"].clear()
    with pytest.raises(ValueError):
        camera_input_mapping(rows, "face", sr=sr)


@pytest.mark.parametrize("counts", [(1, 2, 0), (1, 0, 0), (0, 0, 0)])
def test_cpu_report_uses_native_metrics_and_truthful_joint_mask_description(
    tmp_path, counts
):
    source = tmp_path / "original.png"
    source.write_bytes(b"CPU report-only source fixture, never rendered")
    rows, sr = fixture(source, counts)
    engine = SimpleNamespace(
        root=tmp_path / "run",
        record=tmp_path / "record",
        config=dict(sr_region="face", scales=[4], importance_metric="outdoor"),
        iterations=18000,
    )
    save_json(engine.root / "original/training-inputs.json", dict(records=rows))
    save_json(engine.root / "swinir4x/training-inputs.json", dict(records=rows))
    save_json(engine.root / "swinir4x/superresolution.json", sr)
    save_json(
        engine.record / "swinir4x/metrics.json",
        dict(
            execution=dict(state="training_completed", train_frames=2, test_frames=1),
            means=dict(test=dict(psnr=20.0, ssim=0.8)),
        ),
    )
    save_json(
        engine.record / "swinir4x/delivery.json",
        dict(parity_passed=True, cameras=3, points=31),
    )
    route = route_metadata(engine, 4, rows)
    assert route["input_route"] == PAIRED_LAYOUT
    publish(engine, [4], {})
    data = read_json(engine.record / "comparison.json")
    assert data["frames"] == 3 and data["training_mask_used"] is bool(sum(counts[1:]))
    variant = data["variants"]["swinir4x"]
    assert variant["view_dimensions"] == [[129, 97]]
    assert variant["auxiliary_children"] == sum(counts)
    assert data["quality_accepted"] is False
    html = (engine.record / "comparison.html").read_text(encoding="utf-8")
    assert "同一次优化器更新" in html and "主相机数仍等于源帧数" in html
    assert "原生模型、损失与优化器保持原样" not in html
    assert "统一 K×4" not in html


@pytest.mark.parametrize("paired", [False, True])
def test_bundle_keeps_standard_parent_cameras_and_optional_auxiliary_json(
    tmp_path, paired
):
    rows, _ = fixture("unused.png")
    save_json(tmp_path / "cameras.json", camera_records(rows))
    write_colmap_cameras(tmp_path / "colmap", rows)
    for name in (
        "upright-parity.json",
        "native-ply-parity.json",
        "camera-input-mapping.json",
        "plan.json",
    ):
        save_json(tmp_path / name, dict(cpu_fixture_only=True))
    (tmp_path / "point_cloud.ply").write_bytes(
        b"CPU bundle identity fixture; not a Gaussian model"
    )
    save_json(tmp_path / "paired-cameras.json", paired_camera_records(rows))
    output = write_scene_bundle(tmp_path, 3, paired=paired)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert ("paired-cameras.json" in archive.namelist()) is paired
        assert archive.read("cameras.json") == (tmp_path / "cameras.json").read_bytes()
        assert not any(name.startswith("children/") for name in archive.namelist())
        cameras = archive.read("colmap/cameras.txt").decode()
        assert cameras.count("PINHOLE") == 3 and "129 97" in cameras
