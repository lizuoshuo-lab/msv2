"""Real Engine input stages with CPU tensors and a deterministic SR test double.

These tests certify data routing and original-pixel preservation, not SwinIR
quality or CUDA capacity. CUDA entry points are replaced before execution.
"""

from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from msv2.engine import Engine
from msv2.io import read_json, save_json, sha256


@pytest.mark.parametrize(
    "boxes", [[], [[5, 6, 31, 35]], [[0, 0, 25, 28], [18, 24, 43, 51]]]
)
def test_head_engine_keeps_full_native_pixels_and_adds_exact_crop_views(
    tmp_path, monkeypatch, boxes
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
        assert options.tile is None  # Small head/context must avoid tile-stride zero.
        actual = np.rint(tensor[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        assert np.array_equal(actual[:53, :45], rgb)
        calls.append(actual)
        return tensor.repeat_interleave(4, 2).repeat_interleave(4, 3)

    official = SimpleNamespace(define_model=lambda args: Model(), test=sr_test)
    monkeypatch.setattr(engine_module, "import_file", lambda name, path: official)

    class Extractor:
        assets = {"test_double": True}

        def __init__(self, *paths):
            pass

        def regions(self, value):
            assert np.array_equal(value, rgb)
            regions = []
            for box in boxes:
                x0, y0, x1, y1 = box
                mask = np.zeros(rgb.shape[:2], dtype=bool)
                mask[y0:y1, x0:x1] = True
                regions.append(dict(bbox_native=box, mask_native=mask))
            return regions, {"state": "head_found" if boxes else "no_head"}

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
                sr_region="head",
                iterations=18000,
                gpu_memory_fraction=0.85,
                sr_tile=256,
                sr_tile_overlap=32,
            ),
        ),
    )
    save_json(
        run / "prepared/pixel-lineage.json",
        dict(
            passed=True,
            records=[
                dict(image=source.name, sha256=sha256(source), wh=[45, 53]),
                # Engine requires three source views; deterministic identical
                # pixels remain separate source identities in this bounded test.
                dict(image="frame_000002.png", sha256=sha256(source), wh=[45, 53]),
                dict(image="frame_000003.png", sha256=sha256(source), wh=[45, 53]),
            ],
        ),
    )
    for number in (2, 3):
        source.with_name(f"frame_{number:06d}.png").write_bytes(source_bytes)
    engine = Engine(job)
    monkeypatch.setattr(engine, "runtime", lambda: torch)
    engine.superresolve()
    state = read_json(run / "swinir4x/superresolution.json")
    assert state["state"] == "completed"
    assert state["source_frames"] == 3
    assert state["view_count"] == 3 * (1 + len(boxes))
    assert len(calls) == 3 * len(boxes)
    for record in state["records"]:
        full = run / "swinir4x/dataset/images" / record["name"]
        assert full.read_bytes() == source_bytes
        assert len(record["heads"]) == len(boxes)
        for head, box in zip(record["heads"], boxes):
            x0, y0, x1, y1 = box
            expected = rgb[y0:y1, x0:x1].repeat(4, 0).repeat(4, 1)
            with Image.open(full.with_name(head["name"])) as image:
                result = np.asarray(image)
                assert result.shape == expected.shape
                # Two-native-pixel inward feather changes only the edge ramp.
                assert np.array_equal(result[8:-8, 8:-8], expected[8:-8, 8:-8])
            mask_path = run / "swinir4x/head-masks" / head["mask_name"]
            assert sha256(mask_path) == head["mask_sha256"]
            with Image.open(mask_path) as mask_image:
                assert mask_image.mode == "L" and mask_image.size == (45, 53)
                assert mask_image.getbbox() == tuple(box)
    # Resume must neither repeat SR nor silently change the native images.
    engine.superresolve()
    assert len(calls) == 3 * len(boxes)
    assert not torch.cuda.is_initialized()
