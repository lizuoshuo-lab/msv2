"""Native pixel identity and isolated-output contracts for CPU extraction."""

import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from msv2.cli import main
from msv2.head_extract import SCHEMA, array_sha, extract_head
from msv2.io import read_json, save_json, sha256


class StubExtractor:
    assets = {"parser": {"source_commit": "cpu-test-pinned-source"}}

    def __init__(self, masks):
        self.masks = iter(masks)
        self.calls = 0

    def regions(self, rgb):
        self.calls += 1
        result = []
        for mask in next(self.masks):
            ys, xs = np.nonzero(mask)
            result.append(
                dict(
                    mask_native=mask,
                    bbox_native=[
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max()) + 1,
                        int(ys.max()) + 1,
                    ],
                )
            )
        return result, {"source_rgb_sha256": array_sha(rgb), "device": "cpu"}


def image_file(path, wh=(13, 9)):
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.random.default_rng(83).integers(0, 256, (wh[1], wh[0], 3), dtype=np.uint8)
    Image.fromarray(rgb).save(path)
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def rectangle(wh, bbox):
    mask = np.zeros((wh[1], wh[0]), dtype=bool)
    x0, y0, x1, y1 = bbox
    mask[y0:y1, x0:x1] = True
    return mask


def test_multi_head_and_no_head_outputs_keep_native_rgb_and_dimensions(tmp_path):
    source, output = tmp_path / "source", tmp_path / "head-pixels"
    path = source / "head.png"
    rgb = image_file(path)
    nohead = source / "nohead.png"
    image_file(nohead, wh=(7, 5))
    before = {p: sha256(p) for p in (path, nohead)}
    first = rectangle((13, 9), [1, 2, 6, 7])
    first[3, 3] = False
    second = rectangle((13, 9), [8, 1, 13, 5])
    extractor = StubExtractor([[first, second], []])
    result = extract_head(source, output, extractor=extractor)
    manifest = read_json(result["manifest"])
    assert result["heads"] == 2 and result["no_head_images"] == 1
    assert extractor.calls == 2 and manifest["assets"] == extractor.assets
    assert manifest["source_files_unchanged"] is True
    assert manifest["GPU_used"] is False and manifest["training_started"] is False
    row, empty = manifest["records"]
    assert row["source_shape"] == [9, 13, 3] and row["source_wh"] == [13, 9]
    assert row["source_file_sha256"] == row["source_png_sha256"] == before[path]
    assert row["source_rgb_sha256"] == hashlib.sha256(rgb.tobytes()).hexdigest()
    with Image.open(output / row["mask_file"]) as im:
        assert im.mode == "L" and im.size == (13, 9)
        assert np.array_equal(np.array(im), (first | second).astype(np.uint8) * 255)
    for head, mask in zip(row["heads"], (first, second)):
        x0, y0, x1, y1 = head["bbox_native"]
        file = output / head["cutout_file"]
        with Image.open(file) as im:
            assert im.mode == "RGBA" and im.size == (x1 - x0, y1 - y0)
            cutout = np.array(im)
        # RGB beneath transparent pixels is preserved as well as visible RGB.
        assert np.array_equal(cutout[:, :, :3], rgb[y0:y1, x0:x1])
        assert np.array_equal(
            cutout[:, :, 3], mask[y0:y1, x0:x1].astype(np.uint8) * 255
        )
        assert head["cutout_sha256"] == sha256(file)
    assert empty["state"] == "no_head" and empty["heads"] == []
    assert empty["mask_bbox"] is None and empty["mask_pixels"] == 0
    with Image.open(output / empty["mask_file"]) as im:
        assert im.size == (7, 5) and im.getbbox() is None
    assert [p.name for p in (output / empty["mask_file"]).parent.iterdir()] == [
        "mask.png"
    ]
    assert all(sha256(p) == digest for p, digest in before.items())


@pytest.mark.parametrize("wrapper", ["", "images", "dataset/images"])
def test_directory_inputs_keep_camera_relative_identity_and_skip_generated_images(
    tmp_path, wrapper
):
    source = tmp_path / "capture"
    images = source / wrapper
    a, b = images / "camera_a/frame.png", images / "camera_b/frame.png"
    c = images / "camera_a/frame.jpg"
    for path in (a, b, c):
        image_file(path)
    generated = images / "older-extraction"
    image_file(generated / "not-a-source.png")
    save_json(generated / "manifest.json", {"schema": SCHEMA})
    result = extract_head(
        source, tmp_path / "output", extractor=StubExtractor([[], [], []])
    )
    records = read_json(result["manifest"])["records"]
    assert result["source_images"] == 3
    assert {r["source_id"] for r in records} == {
        "camera_a/frame.png",
        "camera_a/frame.jpg",
        "camera_b/frame.png",
    }
    assert len({r["mask_file"] for r in records}) == 3
    jpg = next(r for r in records if r["source_id"].endswith(".jpg"))
    assert jpg["source_png_sha256"] is None
    assert jpg["source_file_sha256"] == sha256(c)
    with Image.open(c) as im:
        assert (
            jpg["source_rgb_sha256"]
            == hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()
        )


@pytest.mark.parametrize("guard", ["existing", "inside", "empty", "generated"])
def test_output_and_source_guards_fail_before_model_use(tmp_path, guard):
    source, output = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    if guard != "empty":
        image_file(source / "frame.png")
    if guard == "existing":
        output.mkdir()
        (output / "keep.txt").write_text("original")
    elif guard == "inside":
        output = source / "generated-output"
    elif guard == "generated":
        save_json(source / "manifest.json", {"schema": SCHEMA})
    extractor = StubExtractor([])
    with pytest.raises((ValueError, FileExistsError)):
        extract_head(source, output, extractor=extractor)
    assert extractor.calls == 0
    if guard == "existing":
        assert (output / "keep.txt").read_text() == "original"
    else:
        assert not output.exists()


def test_bad_extractor_cannot_silently_change_native_source_pixels(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    path = source / "frame.png"
    image_file(path)
    before = sha256(path)

    class MutatingExtractor:
        assets = {}

        def regions(self, rgb):
            rgb[:] = 0
            return [], {}

    with pytest.raises(RuntimeError, match="changed source RGB"):
        extract_head(source, output, extractor=MutatingExtractor())
    assert sha256(path) == before
    manifest = read_json(output / "manifest.json")
    assert manifest["state"] == "failed"
    assert manifest["source_files_unchanged"] is False
    assert manifest["records"] == []


def test_invalid_native_mask_shape_is_rejected(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    image_file(source / "frame.png")
    invalid = rectangle((8, 6), [1, 1, 4, 3])
    with pytest.raises(ValueError, match="native image size"):
        extract_head(source, output, extractor=StubExtractor([[invalid]]))
    assert read_json(output / "manifest.json")["state"] == "failed"


def test_cli_extract_is_independent_of_training_configuration_and_uses_deployed_assets(
    tmp_path, monkeypatch, capsys
):
    import msv2.head_extract as module

    root = tmp_path / "runtime"
    root.mkdir()
    (root / "config.json").write_text('{"unknown_training_option": true}')
    chosen = {
        "MSV2_HEAD_PARSER_SOURCE": "/existing parser source",
        "MSV2_HEAD_PARSER_WEIGHT": "/existing parser.pt",
        "MSV2_HEAD_DETECTOR_WEIGHT": "/existing locator.onnx",
    }
    for name, value in chosen.items():
        monkeypatch.setenv(name, value)
    calls = []

    def stub(source, output, **assets):
        calls.append((source, output, assets))
        return {"state": "completed", "device": "cpu"}

    monkeypatch.setattr(module, "extract_head", stub)
    assert (
        main(
            [
                "extract-head",
                "source images",
                "--output",
                "new output",
                "--root",
                str(root),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["device"] == "cpu"
    assert calls == [
        (
            "source images",
            "new output",
            dict(
                parser_source=chosen["MSV2_HEAD_PARSER_SOURCE"],
                parser_weight=chosen["MSV2_HEAD_PARSER_WEIGHT"],
                detector_weight=chosen["MSV2_HEAD_DETECTOR_WEIGHT"],
            ),
        )
    ]
