"""CPU routing checks for full-frame face-mask composition, without CUDA claims."""

import struct
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from msv2.engine import Engine
from msv2.io import read_json, save_json, sha256


def prepared_engine(tmp_path, monkeypatch, boxes, *, mode="face", empty_masks=False):
    torch = pytest.importorskip("torch")
    import msv2.engine as engine_module
    import msv2.head as head_module

    assert not torch.cuda.is_initialized()
    monkeypatch.setattr(
        torch.cuda, "set_per_process_memory_fraction", lambda value: None
    )
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    original_to = torch.Tensor.to

    def cpu_to(self, *args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    rgb = np.random.default_rng(317).integers(0, 256, (53, 45, 3), np.uint8)
    run = tmp_path / "run"
    source = run / "prepared/dataset/images/frame_000001.png"
    source.parent.mkdir(parents=True)
    Image.fromarray(rgb).save(source)
    source_bytes = source.read_bytes()
    weight = tmp_path / "test-weight"
    weight.write_bytes(b"CPU test double, not a SwinIR checkpoint")
    monkeypatch.setattr(engine_module, "WEIGHT_SHA", sha256(weight))
    calls = []

    class Model:
        def eval(self):
            return self

        def cuda(self):
            return self

    def sr_test(tensor, model, options, window):
        assert tensor.device.type == "cpu" and window == 8
        assert options.tile == (None if mode != "full" else 256)
        actual = np.rint(tensor[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        # Each tested context reaches the image boundary. This checks the
        # actual original RGB before any model normalization/padding.
        assert np.array_equal(actual[:53, :45], rgb)
        calls.append(actual)
        return tensor.repeat_interleave(4, 2).repeat_interleave(4, 3)

    official = SimpleNamespace(define_model=lambda args: Model(), test=sr_test)
    monkeypatch.setattr(engine_module, "import_file", lambda name, path: official)
    masks = []
    for box in boxes:
        x0, y0, x1, y1 = box
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        if not empty_masks:
            mask[y0:y1, x0:x1] = True
        masks.append(mask)

    class Extractor:
        assets = {"test_double": True, "mask_scope": "face"}

        def __init__(self, *paths, mask_scope):
            assert mode != "full" and mask_scope == "face"

        def regions(self, value):
            assert np.array_equal(value, rgb)
            return [
                dict(bbox_native=box, mask_native=mask, mask_scope="face")
                for box, mask in zip(boxes, masks)
            ], {"mask_scope": "face"}

    monkeypatch.setattr(head_module, "HeadPixelExtractor", Extractor)
    job = run / "job.json"
    save_json(
        job,
        dict(
            run_dir=str(run),
            source_dir=str(tmp_path),
            swinir_source=str(tmp_path),
            swinir_weight=str(weight),
            options=dict(
                sr_region=mode,
                iterations=18000,
                gpu_memory_fraction=0.85,
                sr_tile=256,
                sr_tile_overlap=32,
                holdout=2,
            ),
        ),
    )
    lineage = []
    for number in (1, 2, 3):
        path = source.with_name(f"frame_{number:06d}.png")
        if number != 1:
            path.write_bytes(source_bytes)
        lineage.append(dict(image=path.name, sha256=sha256(path), wh=[45, 53]))
    save_json(run / "prepared/pixel-lineage.json", dict(passed=True, records=lineage))
    engine = Engine(job)
    monkeypatch.setattr(engine, "runtime", lambda: torch)
    return engine, rgb, source_bytes, calls, masks


@pytest.mark.parametrize(
    "boxes,empty_masks,expected_calls",
    [
        ([], False, 0),
        ([[5, 6, 31, 35]], False, 1),
        ([[0, 0, 25, 28], [18, 24, 43, 51]], False, 2),
        ([[5, 6, 31, 35], [5, 6, 31, 35]], False, 1),
        ([[5, 6, 31, 35]], True, 0),
    ],
)
def test_face_engine_has_one_full4_image_per_source_and_no_crop_views(
    tmp_path, monkeypatch, boxes, empty_masks, expected_calls
):
    engine, rgb, source_bytes, calls, masks = prepared_engine(
        tmp_path, monkeypatch, boxes, empty_masks=empty_masks
    )
    engine.superresolve()
    run = engine.root / "swinir4x"
    state = read_json(run / "superresolution.json")
    assert state["state"] == "completed"
    assert state["source_frames"] == state["view_count"] == 3
    assert state["view_layout"] == "full4x_mask_composite"
    assert state["mask_scope"] == "face" and state["full_frame_4x_generated"]
    assert not state["additional_training_views"] and not state["training_mask_used"]
    assert len(calls) == 3 * expected_calls
    assert len(list((run / "dataset/images").iterdir())) == 3
    union = np.logical_or.reduce(masks) if masks else np.zeros(rgb.shape[:2], bool)
    union4 = union.repeat(4, 0).repeat(4, 1)
    reference = np.array(
        Image.fromarray(rgb).resize((180, 212), Image.Resampling.LANCZOS)
    )
    for record in state["records"]:
        assert record["face_count"] == (0 if empty_masks else len(boxes))
        assert record["composition"]["added_views"] == 0
        assert record["composition"]["one_complete_training_image"]
        assert record["composition"]["outside_mask_equal_reference"]
        full = run / "dataset/images" / record["name"]
        with Image.open(full) as image:
            assert image.mode == "RGB" and image.size == (180, 212)
            actual = np.array(image)
        assert np.array_equal(actual[~union4], reference[~union4])
        if expected_calls == 0:
            assert np.array_equal(actual, reference)
        for face in record["faces"]:
            mask_file = run / "face-masks" / face["mask_name"]
            assert sha256(mask_file) == face["mask_sha256"]
            with Image.open(mask_file) as image:
                assert image.size == (45, 53) and image.mode == "L"
                assert image.getbbox() == tuple(face["bbox_native"])
        assert (
            engine.prep / "dataset/images" / record["name"]
        ).read_bytes() == source_bytes
    assert state["input_composite_mask"] is bool(union.any())
    assert state["input_face_mask_used"] is bool(union.any())
    pixels = read_json(run / "pixels.json")
    assert pixels["count"] == pixels["view_count"] == 3
    engine.superresolve()
    assert len(calls) == 3 * expected_calls
    if state["records"][0]["faces"]:
        # Auxiliary masks are part of the resume contract, never training RGBs.
        f = run / "face-masks" / state["records"][0]["faces"][0]["mask_name"]
        f.write_bytes(b"changed")
        with pytest.raises(AssertionError):
            engine.superresolve()
    assert not __import__("torch").cuda.is_initialized()


def test_full_swinir_control_keeps_full_image_inference_without_parser(
    tmp_path, monkeypatch
):
    engine, rgb, _, calls, _ = prepared_engine(tmp_path, monkeypatch, [], mode="full")
    engine.superresolve()
    state = read_json(engine.root / "swinir4x/superresolution.json")
    assert len(calls) == 3 and state["sr_region"] == "full"
    assert not state["input_composite_mask"]
    expected = rgb.repeat(4, 0).repeat(4, 1)
    for item in state["records"]:
        assert item["composition"] is None and item["faces"] == []
        with Image.open(
            engine.root / "swinir4x/dataset/images" / item["name"]
        ) as image:
            assert np.array_equal(np.array(image), expected)


def test_composite_cameras_scale_whole_image_K_and_keep_native_pose_sparse_points_split(
    tmp_path, monkeypatch
):
    from msv2.data import _read_cameras

    engine, _, _, _, _ = prepared_engine(tmp_path, monkeypatch, [[5, 6, 31, 35]])
    engine.superresolve()
    sparse = engine.prep / "dataset/sparse/0"
    sparse.mkdir(parents=True)
    (sparse / "images.bin").write_bytes(b"original unmodified pose/observation bytes")
    (sparse / "points3D.bin").write_bytes(b"original unmodified sparse-point bytes")
    K = [[75.0, 0.0, 12.25], [0.0, 79.0, 22.75], [0.0, 0.0, 1.0]]
    with (sparse / "cameras.bin").open("wb") as stream:
        stream.write(struct.pack("<QiiQQ4d", 1, 1, 1, 45, 53, 75.0, 79.0, 12.25, 22.75))
    rows = []
    for i, row in enumerate(engine.native_rows()):
        pose = np.eye(4)
        pose[0, 3] = 0.1 * i
        rows.append(
            dict(
                frame=row["image"][:-4],
                name=row["image"],
                wh=row["wh"],
                K=K,
                w2c=pose.tolist(),
                source_sha256=row["sha256"],
            )
        )
    save_json(engine.prep / "camera-inputs.json", dict(records=rows))
    save_json(engine.prep / "state.json", dict(state="completed"))
    save_json(engine.root / "plan.json", {})
    scene = ModuleType("scene")
    loader = ModuleType("scene.colmap_loader")

    def read_cameras(path):
        return {
            cid: SimpleNamespace(
                model="PINHOLE" if c.model == 1 else "SIMPLE_PINHOLE",
                width=c.width,
                height=c.height,
                params=c.params,
            )
            for cid, c in _read_cameras(Path(path).parent).items()
        }

    loader.read_intrinsics_binary = read_cameras
    monkeypatch.setitem(sys.modules, "scene", scene)
    monkeypatch.setitem(sys.modules, "scene.colmap_loader", loader)
    engine.stage_cameras(4)
    inputs = read_json(engine.root / "swinir4x/training-inputs.json")
    assert (
        len(inputs["records"]) == inputs["source_frames"] == inputs["view_count"] == 3
    )
    assert [r["split"] for r in inputs["records"]] == ["test", "train", "test"]
    for before, after in zip(rows, inputs["records"]):
        assert after["frame"] == before["frame"] and after["w2c"] == before["w2c"]
        assert after["wh"] == [180, 212]
        assert np.array_equal(after["K"], np.diag([4, 4, 1]) @ np.array(K))
        assert "view_type" not in after and "source_frame" not in after
    for name in ["images.bin", "points3D.bin"]:
        assert (engine.root / "swinir4x/dataset/sparse/0" / name).read_bytes() == (
            sparse / name
        ).read_bytes()
    assert not inputs["additional_training_views"] and inputs["full_frame_4x_generated"]
    assert inputs["view_layout"] == "full4x_mask_composite"
    Image.new("RGB", (20, 20)).save(
        engine.root / "swinir4x/dataset/images/auxiliary_face_crop.png"
    )
    with pytest.raises(AssertionError, match="Only complete"):
        engine.stage_pixels(4)
