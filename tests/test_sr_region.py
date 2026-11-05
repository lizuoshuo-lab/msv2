"""Source-group and export/report contracts for mixed full1x/head4x input."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from msv2.config import load_config
from msv2.engine import Engine
from msv2.export import camera_input_mapping, camera_records
from msv2.head import crop_intrinsics
from msv2.io import read_json, save_json
from msv2.report import publish, view_summary


@pytest.mark.parametrize("region", ["head", "full"])
def test_region_is_an_explicit_validated_input_choice(region):
    assert load_config()["sr_region"] == "full"
    assert load_config(overrides={"sr_region": region})["sr_region"] == region


@pytest.mark.parametrize("region", ["face", "parse"])
def test_unpublished_semantic_modes_are_not_runtime_options(region):
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


def mixed_rows(base, head_count):
    rows = []
    for original in base:
        row = deepcopy(original)
        row.update(source_frame=row["frame"], source_name=row["name"], view_type="full")
        rows.append(row)
    for row in rows[:head_count]:
        head = deepcopy(row)
        head.update(
            frame=row["frame"] + "_head_001",
            name=row["frame"] + "_head_001.png",
            view_type="head",
            bbox_native=[10, 5, 30, 25],
            wh=[80, 80],
        )
        head["K"] = crop_intrinsics(row["K"], head["bbox_native"]).tolist()
        rows.append(head)
    return rows


@pytest.mark.parametrize(
    "region,head_count,generated_count",
    [
        ("head", 0, 0),
        ("head", 3, 3),
        ("head", 0, 3),
        ("head", 1, 1),
        ("head", 2, 3),
        ("full", 0, 0),
    ],
)
def test_report_counts_real_sources_and_training_views(
    tmp_path, region, head_count, generated_count
):
    engine = SimpleNamespace(
        root=tmp_path / "run",
        record=tmp_path / "export",
        config=load_config(overrides={"sr_region": region}),
        iterations=18000,
    )
    source = tmp_path / "source.png"
    source.write_bytes(b"read-only report fixture")
    base = source_rows(source)
    for scale, name in [(1, "original"), (2, "interpolation2x"), (4, "swinir4x")]:
        records = (
            mixed_rows(base, head_count)
            if scale == 4 and region == "head"
            else deepcopy(base)
        )
        metadata = {"records": records}
        if scale == 4 and region == "head":
            metadata.update(
                generated_head_view_count=generated_count,
                initial_depth_quota=20,
                skipped_head_views=[
                    dict(
                        name=row["frame"] + "_head_001.png",
                        source_frame=row["frame"],
                        reason="below_initial_native_depth_quota",
                        P=4,
                        q_at_skip=20,
                    )
                    for row in base[head_count:generated_count]
                ],
            )
        save_json(engine.root / name / "training-inputs.json", metadata)
        save_json(
            engine.record / name / "delivery.json",
            dict(parity_passed=True, cameras=len(records), points=100),
        )
        save_json(
            engine.record / name / "metrics.json",
            dict(
                execution=dict(
                    state="training_completed",
                    train_frames=sum(r["split"] == "train" for r in records),
                    test_frames=sum(r["split"] == "test" for r in records),
                ),
                means={"test": {"psnr": 20, "ssim": 0.8}},
            ),
        )
    sr_records = [
        dict(
            name=row["name"],
            heads=[{"name": row["frame"] + "_head_001.png"}]
            if i < generated_count
            else [],
        )
        for i, row in enumerate(base)
    ]
    save_json(
        engine.root / "swinir4x/superresolution.json",
        dict(
            state="completed",
            sr_region=region,
            records=sr_records,
            head_mask_source="existing_digital_human_face_parsing"
            if region == "head"
            else None,
        ),
    )
    publish(engine, [1, 2, 4], {})
    summary = read_json(engine.record / "comparison.json")
    variant = summary["variants"]["swinir4x"]
    assert summary["source_frames"] == 3
    assert variant["source_frames"] == 3 and variant["view_count"] == 3 + head_count
    assert variant["full_views"] == 3 and variant["head_views"] == head_count
    assert variant["train_source_frames"] == 2 and variant["test_source_frames"] == 1
    assert variant["training_mask_used"] is False
    assert variant["input_composite_mask"] is (region == "head" and head_count > 0)
    assert variant["input_head_mask_used"] is (region == "head" and head_count > 0)
    assert variant["input_roi_composite"] is (region == "head" and head_count > 0)
    assert variant["full_frame_4x_generated"] is (region == "full")
    text = (engine.record / "comparison.html").read_text(encoding="utf-8")
    if region == "head":
        assert variant["full_view_scale"] == 1
        assert variant["generated_head_view_count"] == generated_count
        assert variant["skipped_head_view_count"] == generated_count - head_count
        assert "其余 Lanczos" not in text and "纯 Lanczos 四倍" not in text
        if head_count:
            assert variant["head_view_scale"] == 4
            assert variant["native_footprint_limitation"]
            if head_count == 1:
                assert variant["head_train_views"] == 0
                assert variant["head_test_views"] == 1
                assert variant["input_route"] == "full1x_head4x_holdout_only"
                assert variant["head_sr_training_used"] is False
                assert "头部 mask 仅留出评估" in variant["label"]
            else:
                assert variant["input_route"] == "full1x_head4x"
                assert variant["head_sr_training_used"] is True
        elif generated_count:
            assert "初始深度配额跳过" in variant["label"]
            assert "no_head" not in variant["label"]
            assert variant["input_route"] == "full1x_all_heads_capacity_skipped"
            assert variant["head_sr_training_used"] is False
            assert variant["skipped_head_reasons"] == {
                "below_initial_native_depth_quota": 3
            }
            assert "头框像素少于初始深度采样配额" in text
        else:
            assert "仅完整原图1倍" in variant["label"]
            assert variant["input_route"] == "full1x_no_head"
    else:
        assert variant["label"] == "整幅 SwinIR 超分四倍"
        assert variant["full_view_scale"] == 4


def test_source_group_split_cannot_be_misreported_as_held_out():
    base = source_rows("unused.png")
    rows = mixed_rows(base, 1)
    rows[-1]["split"] = "train"
    with pytest.raises(ValueError, match="source frame"):
        view_summary(rows)
    with pytest.raises(ValueError, match="source split"):
        camera_input_mapping(rows, "head")


def test_camera_exports_retain_mixed_resolution_and_source_binding():
    base = source_rows("unused.png")
    rows = mixed_rows(base, 1)
    cameras = camera_records(rows)
    full, head = cameras[0], cameras[-1]
    assert full["source_frame"] == head["source_frame"] == base[0]["frame"]
    assert full["view_type"] == "full" and head["view_type"] == "head"
    assert head["native_bbox"] == [10, 5, 30, 25]
    assert full["position"] == head["position"] and full["rotation"] == head["rotation"]
    assert head["K"] == crop_intrinsics(base[0]["K"], [10, 5, 30, 25]).tolist()
    meta = camera_input_mapping(rows, "head")
    assert meta["full_frame"] is False and meta["training_resolution"] is None
    assert meta["training_resolutions"] == [[80, 80], [100, 60]]
    assert meta["source_frames"] == 3 and meta["view_count"] == 4
    assert meta["view_layout"] == "full1x_head4x"
    assert meta["input_head_mask_used"] and meta["input_roi_composite"]
    assert meta["input_composite_mask"] and not meta["full_frame_4x_generated"]
    assert meta["training_mask_used"] is False


def test_legacy_camera_json_keeps_original_field_contract():
    rows = source_rows("unused.png")
    cameras = camera_records(rows)
    assert not any(
        key in cameras[0] for key in ("source_frame", "view_type", "native_bbox")
    )
    assert camera_input_mapping(rows)["training_resolution"] == [100, 60]
    assert "view_layout" not in camera_input_mapping(rows)


def test_test_only_head_cameras_are_not_claimed_as_training_crops():
    rows = mixed_rows(source_rows("unused.png"), 1)
    metadata = camera_input_mapping(rows, "head")
    assert metadata["crop_views_present"] is True
    assert metadata["training_crop_used"] is False
    assert metadata["head_train_views"] == 0 and metadata["head_test_views"] == 1
