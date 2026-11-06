"""CPU routing of native parents and auxiliary joint-loss children; no CUDA."""

import struct
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from msv2.engine import Engine
from msv2.io import read_json, save_json, sha256


def prepared_engine(
    tmp_path, monkeypatch, boxes, *, mode="face", empty_masks=False, frame_count=3
):
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
        assert options.tile == (None if mode not in ("full", None) else 256)
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
                **({"sr_region": mode} if mode is not None else {}),
                iterations=18000,
                gpu_memory_fraction=0.85,
                sr_tile=256,
                sr_tile_overlap=32,
                holdout=2,
            ),
        ),
    )
    lineage = []
    for number in range(1, frame_count + 1):
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
def test_parent_is_native_png_and_child_is_masked_auxiliary_roi(
    tmp_path, monkeypatch, boxes, empty_masks, expected_calls
):
    from msv2.paired_data import _reference_roi

    engine, rgb, source_bytes, calls, masks = prepared_engine(
        tmp_path, monkeypatch, boxes, empty_masks=empty_masks
    )
    engine.superresolve()
    run = engine.root / "swinir4x"
    state = read_json(run / "superresolution.json")
    assert state["state"] == "completed"
    assert state["source_frames"] == state["view_count"] == 3
    assert state["view_layout"] == "parent1x_mask_child4x_joint"
    assert state["parent_scale"] == 1 and state["child_scale"] == 4
    assert state["joint_source_update"] and not state["full_frame_4x_generated"]
    assert not state["additional_training_views"]
    assert state["training_mask_used"] is bool(expected_calls)
    assert len(calls) == 3 * expected_calls
    assert len(list((run / "dataset/images").iterdir())) == 3
    union = np.logical_or.reduce(masks) if masks else np.zeros(rgb.shape[:2], bool)
    for record in state["records"]:
        assert record["face_count"] == (0 if empty_masks else len(boxes))
        assert len(record["children"]) == expected_calls
        assert record["composition"]["sampling_group_count"] == 1
        assert record["composition"]["children_are_auxiliary"]
        assert not record["composition"]["children_registered_to_colmap"]
        assert not record["composition"]["children_in_ordinary_image_sampling"]
        assert not record["composition"]["full_frame_4x_training_image_generated"]
        parent = run / "dataset/images" / record["name"]
        assert parent.read_bytes() == source_bytes
        assert sha256(parent) == record["native_sha256"] == record["output_sha256"]
        with Image.open(parent) as image:
            assert image.mode == "RGB" and image.size == (45, 53)
            assert np.array_equal(np.asarray(image), rgb)
        seen = np.zeros((53 * 4, 45 * 4), bool)
        for saved in record["children"]:
            assert Path(saved["image"]).parent == Path("children")
            assert Path(saved["mask"]).parent == Path("children")
            x0, y0, x1, y1 = saved["roi_native"]
            with Image.open(run / saved["image"]) as image:
                actual = np.array(image)
                assert image.mode == "RGB" and list(image.size) == saved["wh"]
            assert saved["wh"] == [4 * (x1 - x0), 4 * (y1 - y0)]
            assert all(v % 8 == 0 for v in saved["roi_native"])
            with Image.open(run / saved["mask"]) as image:
                assert image.mode == "L" and list(image.size) == saved["wh"]
                mask = np.array(image) != 0
                assert sum(image.histogram()[1:255]) == 0
            reference = _reference_roi(rgb, saved["roi_native"])[0]
            assert np.array_equal(actual[~mask], reference[~mask])
            assert int(mask.sum()) == saved["mask_pixels"]
            # Odd-image padding never becomes a target loss pixel.
            valid_h, valid_w = min(y1, 53) - y0, min(x1, 45) - x0
            assert not mask[4 * valid_h :, :].any()
            assert not mask[:, 4 * valid_w :].any()
            local = mask[: 4 * valid_h, : 4 * valid_w]
            window = seen[4 * y0 : 4 * min(y1, 53), 4 * x0 : 4 * min(x1, 45)]
            assert not (window & local).any()
            window |= local
        assert np.array_equal(seen, union.repeat(4, 0).repeat(4, 1))
        for face in record["faces"]:
            with Image.open(run / "face-masks" / face["mask_name"]) as image:
                assert image.size == (45, 53) and image.mode == "L"
                assert image.getbbox() == tuple(face["bbox_native"])
        assert (
            engine.prep / "dataset/images" / record["name"]
        ).read_bytes() == source_bytes
    pixels = read_json(run / "pixels.json")
    assert pixels["count"] == pixels["view_count"] == 3
    assert pixels["view_layout"] == state["view_layout"]
    engine.superresolve()
    assert len(calls) == 3 * expected_calls
    assert not __import__("torch").cuda.is_initialized()


@pytest.mark.parametrize("mode", ["full", None])
def test_explicit_and_default_full_controls_stay_whole_image_4x(
    tmp_path, monkeypatch, mode
):
    engine, rgb, _, calls, _ = prepared_engine(tmp_path, monkeypatch, [], mode=mode)
    engine.superresolve()
    state = read_json(engine.root / "swinir4x/superresolution.json")
    assert len(calls) == 3 and state["sr_region"] == "full"
    assert state["view_layout"] == "full4x" and state["full_frame_4x_generated"]
    assert not state["input_composite_mask"] and not state["training_mask_used"]
    expected = rgb.repeat(4, 0).repeat(4, 1)
    for item in state["records"]:
        assert item["composition"] is None and item["faces"] == []
        assert "children" not in item
        with Image.open(
            engine.root / "swinir4x/dataset/images" / item["name"]
        ) as image:
            assert np.array_equal(np.array(image), expected)


@pytest.mark.parametrize("target", ["parent", "child_rgb", "child_mask", "face_mask"])
def test_resume_rejects_corrupted_parent_or_auxiliary_inputs(
    tmp_path, monkeypatch, target
):
    engine, _, _, _, _ = prepared_engine(tmp_path, monkeypatch, [[5, 6, 31, 35]])
    engine.superresolve()
    run = engine.root / "swinir4x"
    item = read_json(run / "superresolution.json")["records"][0]
    paths = {
        "parent": run / "dataset/images" / item["name"],
        "child_rgb": run / item["children"][0]["image"],
        "child_mask": run / item["children"][0]["mask"],
        "face_mask": run / "face-masks" / item["faces"][0]["mask_name"],
    }
    paths[target].write_bytes(b"corrupted original or auxiliary input")
    with pytest.raises(AssertionError):
        engine.superresolve()


def prepare_cameras(engine, monkeypatch):
    from msv2.data import _read_cameras

    sparse = engine.prep / "dataset/sparse/0"
    sparse.mkdir(parents=True)
    (sparse / "images.bin").write_bytes(b"original pose and observation bytes")
    (sparse / "points3D.bin").write_bytes(b"original sparse-point and track bytes")
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
    loader = ModuleType("scene.colmap_loader")
    loader.read_intrinsics_binary = lambda path: {
        cid: SimpleNamespace(
            model="PINHOLE" if c.model == 1 else "SIMPLE_PINHOLE",
            width=c.width,
            height=c.height,
            params=c.params,
        )
        for cid, c in _read_cameras(Path(path).parent).items()
    }
    monkeypatch.setitem(sys.modules, "scene", ModuleType("scene"))
    monkeypatch.setitem(sys.modules, "scene.colmap_loader", loader)
    return rows, sparse


@pytest.mark.parametrize("boxes", [[], [[5, 6, 31, 35]]])
def test_only_native_parent_cameras_enter_colmap_sampling_and_children_keep_calibration(
    tmp_path, monkeypatch, boxes
):
    from msv2.data import _read_cameras

    engine, _, source_bytes, _, _ = prepared_engine(tmp_path, monkeypatch, boxes)
    engine.superresolve()
    before, sparse = prepare_cameras(engine, monkeypatch)
    engine.stage_cameras(4)
    run = engine.root / "swinir4x"
    inputs = read_json(run / "training-inputs.json")
    assert (
        len(inputs["records"]) == inputs["source_frames"] == inputs["view_count"] == 3
    )
    assert [r["split"] for r in inputs["records"]] == ["test", "train", "test"]
    assert inputs["view_layout"] == "parent1x_mask_child4x_joint"
    assert (
        not inputs["additional_training_views"]
        and not inputs["full_frame_4x_generated"]
    )
    output_camera = _read_cameras(run / "dataset/sparse/0")[1]
    assert (output_camera.width, output_camera.height) == (45, 53)
    assert list(output_camera.params) == [75.0, 79.0, 12.25, 22.75]
    for original, row in zip(before, inputs["records"]):
        assert row["wh"] == original["wh"] == [45, 53]
        assert row["K"] == original["K"] and row["w2c"] == original["w2c"]
        assert row["source_frame"] == original["frame"] and row["view_type"] == "full"
        assert (
            row["source_group_split"] and row["source_sha256"] == row["native_sha256"]
        )
        assert Path(row["image"]).read_bytes() == source_bytes
        assert len(row["children"]) == len(boxes)
        for c in row["children"]:
            x0, y0, x1, y1 = c["roi_native"]
            expected = np.array(
                [[4.0, 0.0, -4 * x0], [0.0, 4.0, -4 * y0], [0.0, 0.0, 1.0]]
            ) @ np.array(row["K"])
            assert np.array_equal(c["camera"]["K"], expected)
            assert c["camera"]["full_wh"] == [180, 212]
            assert c["camera"]["w2c"] == row["w2c"]
            assert c["camera"]["requires_full_fov_roi_kernel"]
            assert c["camera"]["full_tan_fovx"] == 45 / (2 * 75)
            assert c["camera"]["full_tan_fovy"] == 53 / (2 * 79)
            assert c["camera"]["wh"] == [4 * (x1 - x0), 4 * (y1 - y0)]
        for level in (1, 2):
            cached = np.load(run / f"rgb-cache-{level}" / (row["frame"] + ".npy"))
            assert cached.shape == (round(53 / level), round(45 / level), 3)
    for name in ("images.bin", "points3D.bin"):
        assert (run / "dataset/sparse/0" / name).read_bytes() == (
            sparse / name
        ).read_bytes()
    Image.new("RGB", (20, 20)).save(
        run / "dataset/images/forbidden_independent_child.png"
    )
    with pytest.raises(AssertionError):
        engine.stage_pixels(4)


def test_sixty_five_sources_remain_sixty_five_sampling_groups(tmp_path, monkeypatch):
    engine, _, _, calls, _ = prepared_engine(
        tmp_path, monkeypatch, [[5, 6, 31, 35]], frame_count=65
    )
    engine.superresolve()
    before, _ = prepare_cameras(engine, monkeypatch)
    engine.stage_cameras(4)
    run = engine.root / "swinir4x"
    inputs = read_json(run / "training-inputs.json")
    assert len(calls) == 65
    assert len(inputs["records"]) == len(before) == 65
    assert sum(len(r["children"]) for r in inputs["records"]) == 65
    assert len(list((run / "dataset/images").iterdir())) == 65
    assert all(r["source_group_split"] for r in inputs["records"])
    assert len(set(r["source_frame"] for r in inputs["records"])) == 65


@pytest.mark.parametrize("position", [(0, 0), (44, 0), (0, 52), (44, 52), (22, 26)])
@pytest.mark.parametrize("level", [1, 2])
def test_real_joint_targets_cover_tiny_faces_at_odd_size_and_keep_child_domain(
    tmp_path, monkeypatch, position, level
):
    torch = pytest.importorskip("torch")
    from msv2.paired import JointSupervision
    from msv2.paired_loss import paired_loss

    original_to = torch.Tensor.to

    def cpu_to(self, *args, **kwargs):
        if args and args[0] == "cuda":
            args = ("cpu", *args[1:])
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    x, y = position
    mask = np.zeros((53, 45), np.uint8)
    mask[y, x] = 255
    mother_path = tmp_path / "mother-mask.png"
    Image.fromarray(mask).save(mother_path)
    # Real child ROI padding uses the outward8 grid. This fixture has one
    # source face pixel, hence16 child4 pixels or4 child2 warmup pixels.
    child_mask = np.zeros((56 * 4, 48 * 4), np.uint8)
    child_mask[4 * y : 4 * y + 4, 4 * x : 4 * x + 4] = 255
    Image.fromarray(child_mask).save(tmp_path / "child-mask.png")
    Image.new("RGB", (48 * 4, 56 * 4), (101, 111, 121)).save(tmp_path / "child.png")
    row = dict(
        wh=[45, 53],
        face_mask=str(mother_path),
        face_mask_sha256=sha256(mother_path),
        children=[
            dict(
                image="child.png",
                mask="child-mask.png",
                wh=[48 * 4, 56 * 4],
                image_sha256=sha256(tmp_path / "child.png"),
                mask_sha256=sha256(tmp_path / "child-mask.png"),
                roi_native=[0, 0, 48, 56],
            )
        ],
    )
    camera = SimpleNamespace(
        image_name="source",
        level=level,
        row=row,
        width=round(45 / level),
        height=round(53 / level),
    )
    joint = object.__new__(JointSupervision)
    joint.run, joint.cached_key, joint.cached = tmp_path, None, None
    actual, support, children = joint.targets(camera)
    # Independent geometric cell-overlap oracle: every resized pixel whose
    # native footprint intersects the tiny source face must be excluded.
    xx = np.arange(camera.width)
    yy = np.arange(camera.height)
    expected_x = np.maximum(xx * 45 / camera.width, x) < np.minimum(
        (xx + 1) * 45 / camera.width, x + 1
    )
    expected_y = np.maximum(yy * 53 / camera.height, y) < np.minimum(
        (yy + 1) * 53 / camera.height, y + 1
    )
    expected = expected_y[:, None] & expected_x[None, :]
    assert np.array_equal(actual.numpy(), expected)
    assert actual.any() and len(children) == 1
    assert int(children[0]["mask"].sum()) == (4 // level) ** 2
    assert children[0]["camera"]._msv2_roi["full_wh"] == [
        45 * 4 // level,
        53 * 4 // level,
    ]
    assert children[0]["physical_edges"] == (True, True, False, False)
    assert not children[0]["mask"][4 * 53 // level :, :].any()
    assert not children[0]["mask"][:, 4 * 45 // level :].any()
    parent_prediction = torch.full(
        (3, camera.height, camera.width), 0.4, requires_grad=True
    )
    high_prediction = (
        children[0]["target"].detach().clone().mul_(0.5).requires_grad_(True)
    )
    views = [
        dict(
            prediction=high_prediction,
            **{k: v for k, v in children[0].items() if k != "camera"},
        )
    ]
    result = paired_loss(
        parent_prediction,
        torch.zeros_like(parent_prediction),
        actual,
        views,
        0.2,
        parent_support=support,
        ssim_map=lambda a, b: a * 0 + 1,
    )
    result["loss"].backward()
    assert not result["stats"]["native_fallback"]
    assert torch.count_nonzero(parent_prediction.grad[:, actual]) == 0
    assert torch.count_nonzero(high_prediction.grad[:, ~children[0]["mask"]]) == 0
    assert joint.targets(camera) is joint.cached
    assert not torch.cuda.is_initialized()


def test_joint_render_shares_ndc_dummy_and_merges_visibility_in_parent_units():
    torch = pytest.importorskip("torch")
    from msv2.paired import JointSupervision
    from msv2.paired_loss import prepare_ssim_support

    dummy = torch.zeros((3, 3), requires_grad=True)
    parent_image = torch.zeros((3, 10, 12), requires_grad=True)
    child_image = torch.zeros((3, 16, 16), requires_grad=True)
    cc = SimpleNamespace(_msv2_roi=dict(full_wh=[48, 40]))
    child_mask = torch.ones((16, 16), dtype=torch.bool)
    joint = object.__new__(JointSupervision)
    joint.native_render = lambda *a, **k: dict(
        render=parent_image,
        viewspace_points=dummy,
        radii=torch.tensor([4, 0, 0], dtype=torch.int32),
        area_max=torch.tensor([10.0, 0.0, 0.0]),
    )

    def roi_render(camera, *args, **kwargs):
        assert camera._msv2_shared_viewspace_points is dummy
        return dict(
            render=child_image,
            radii=torch.tensor([0, 12, 0], dtype=torch.int32),
            area_max=torch.tensor([0.0, 64.0, 0.0]),
        )

    joint.roi_render = roi_render
    face = torch.zeros((10, 12), dtype=torch.bool)
    joint.targets = lambda camera: (
        face,
        prepare_ssim_support(~face),
        [
            dict(
                camera=cc,
                target=torch.zeros_like(child_image),
                mask=child_mask,
                physical_edges=(False,) * 4,
                support_cache=prepare_ssim_support(child_mask, (False,) * 4),
            )
        ],
    )
    out = joint.render(SimpleNamespace(width=12, height=10), None, None, None)
    assert out["viewspace_points"] is dummy
    assert torch.equal(out["radii"], torch.tensor([4, 3, 0], dtype=torch.int32))
    assert torch.equal(out["visibility_filter"], torch.tensor([[0], [1]]))
    assert torch.equal(out["area_max"], torch.tensor([10.0, 4.0, 0.0]))
    assert len(out["paired_children"]) == 1


@pytest.mark.parametrize("changed", ["image", "mask"])
def test_joint_target_cache_miss_rejects_child_tamper_and_accepts_restored_bytes(
    tmp_path, monkeypatch, changed
):
    torch = pytest.importorskip("torch")
    from msv2.paired import JointSupervision

    original_to = torch.Tensor.to

    def cpu_to(self, *args, **kwargs):
        if args and args[0] == "cuda":
            args = ("cpu", *args[1:])
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    face = np.zeros((16, 16), np.uint8)
    face[4, 4] = 255
    Image.fromarray(face).save(tmp_path / "face.png")
    Image.fromarray(face.repeat(4, 0).repeat(4, 1)).save(tmp_path / "mask.png")
    Image.new("RGB", (64, 64), (123, 111, 99)).save(tmp_path / "image.png")
    saved = dict(
        image="image.png",
        mask="mask.png",
        wh=[64, 64],
        roi_native=[0, 0, 16, 16],
        image_sha256=sha256(tmp_path / "image.png"),
        mask_sha256=sha256(tmp_path / "mask.png"),
    )
    row = dict(
        wh=[16, 16],
        face_mask=str(tmp_path / "face.png"),
        face_mask_sha256=sha256(tmp_path / "face.png"),
        children=[saved],
    )
    camera = SimpleNamespace(
        image_name="same-source", level=1, row=row, width=16, height=16
    )
    joint = object.__new__(JointSupervision)
    joint.run, joint.cached_key, joint.cached = tmp_path, None, None
    path = tmp_path / saved[changed]
    original = path.read_bytes()
    path.write_bytes(b"changed after preparation and before training-target load")
    with pytest.raises(RuntimeError, match="[Cc]hild"):
        joint.targets(camera)
    path.write_bytes(original)
    mask, _, children = joint.targets(camera)
    assert mask.sum() == 1 and len(children) == 1
    assert children[0]["mask"].sum() == 16
    assert sha256(path) == saved[changed + "_sha256"]
    assert not torch.cuda.is_initialized()
