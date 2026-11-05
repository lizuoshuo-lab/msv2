"""CPU tests for frame identity, display orientation and COLMAP boundaries."""

import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from msv2 import data
from msv2.io import read_json, run_logged, sha256


FFMPEG, FFPROBE = shutil.which("ffmpeg"), shutil.which("ffprobe")


def text_model(folder, names, dimensions, rotated=False):
    folder.mkdir(parents=True, exist_ok=True)
    camera_lines, image_lines = [], []
    for index, (name, (width, height)) in enumerate(zip(names, dimensions)):
        cid = index + 1
        if index == 0:
            camera_lines.append(
                f"{cid} SIMPLE_PINHOLE {width} {height} 70 {width / 2} {height / 2}"
            )
        else:
            camera_lines.append(
                f"{cid} PINHOLE {width} {height} 71 91 {width / 2} {height / 2}"
            )
        if rotated and index == 1:
            q = f"{2**-0.5:.17g} 0 0 {2**-0.5:.17g}"
            translation = "-1 2 3"
        else:
            q, translation = "1 0 0 0", f"{index / 10} 0 0"
        image_lines.extend([f"{cid} {q} {translation} {cid} {name}", "3.25 4.5 7"])
    (folder / "cameras.txt").write_text(
        "\n".join(camera_lines) + "\n", encoding="utf-8"
    )
    (folder / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    tracks = " ".join(f"{index + 1} 0" for index in range(len(names)))
    (folder / "points3D.txt").write_text(
        "7 0.1 0.2 3 12 34 56 0.25 " + tracks + "\n", encoding="utf-8"
    )


def binary_images(path):
    """Independent, minimal wire-format inspection for known fixture records."""
    result = []
    with path.open("rb") as stream:
        (count,) = struct.unpack("<Q", stream.read(8))
        for _ in range(count):
            pose = struct.unpack("<i7di", stream.read(64))
            name = bytearray()
            while True:
                value = stream.read(1)
                if value == b"\0":
                    break
                if not value:
                    raise AssertionError("Truncated test model")
                name.extend(value)
            (size,) = struct.unpack("<Q", stream.read(8))
            points = [struct.unpack("<ddq", stream.read(24)) for _ in range(size)]
            result.append((pose, name.decode("utf-8"), points))
    return result


class FrameSelectionTests(unittest.TestCase):
    def test_real_nonuniform_pts_and_nonzero_origin(self):
        rows = data.select_frames(["7", "7.17", "7.52", "7.91"], 3, duration="1")
        self.assertEqual([r["source_index"] for r in rows], [0, 1, 2])
        self.assertEqual([r["time_seconds"] for r in rows], [0.0, 0.17, 0.52])
        self.assertEqual(rows[1]["source_pts_seconds"], 7.17)

    def test_exact_decimal_tie_chooses_earlier_frame(self):
        rows = data.select_frames(["0", ".1", ".3"], 5, duration=".31")
        self.assertEqual([r["source_index"] for r in rows], [0, 1])
        self.assertEqual(rows[1]["target_time_seconds"], 0.2)

    def test_oversampling_and_duplicate_pts_are_deduplicated(self):
        rows = data.select_frames([1.0, 1.0, 1.5], 1000000, duration=".6")
        self.assertEqual([r["source_index"] for r in rows], [0, 2])
        self.assertEqual(rows[-1]["target_time_seconds"], 0.5)

    def test_explicit_source_indices_lock_identity(self):
        rows = data.select_frames([2.0, 2.0, 2.5], frame_indices=[1, 2])
        self.assertEqual([r["source_index"] for r in rows], [1, 2])
        self.assertIsNone(rows[0]["target_time_seconds"])
        for indices in ([1, 1], [2, 1], [-1], [3], [True]):
            with (
                self.subTest(indices=indices),
                self.assertRaises(data.DataPreparationError),
            ):
                data.select_frames([0, 1, 2], frame_indices=indices)

    def test_invalid_or_missing_time_information_is_rejected(self):
        for pts in ([], [0, float("nan")], [1, 0]):
            with self.subTest(pts=pts), self.assertRaises(data.DataPreparationError):
                data.select_frames(pts)

    def test_probe_rejects_declared_but_missing_decoded_frames(self):
        probe = dict(
            streams=[dict(width=40, height=24, nb_frames="3")],
            frames=[dict(best_effort_timestamp_time="0", width=40, height=24)],
        )
        result = SimpleNamespace(returncode=0, stderr="", stdout=json.dumps(probe))
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(data.subprocess, "run", return_value=result):
                with self.assertRaisesRegex(data.DataPreparationError, "nb_frames"):
                    data._probe_video(Path("input.mp4"), "ffprobe", Path(directory))


class ColmapImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.names = ["left/same.jpg", "right/same.png"]
        self.dimensions = [(20, 12), (12, 20)]
        self.pixels = {}
        for index, (name, size) in enumerate(zip(self.names, self.dimensions)):
            target = self.source / "images" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            pixels = np.random.default_rng(index).integers(
                0, 256, (size[1], size[0], 3), dtype=np.uint8
            )
            Image.fromarray(pixels).save(target)
            with Image.open(target) as image:
                self.pixels[name] = np.array(image.convert("RGB"))
        text_model(self.source / "sparse/0", self.names, self.dimensions, rotated=True)

    def test_text_import_preserves_pixels_pose_observations_and_tracks(self):
        original = (self.source / "sparse/0/images.txt").read_bytes()
        output = self.root / "prepared"
        dataset = data.prepare_colmap(self.source, output)
        rows = read_json(output / "camera-inputs.json")["records"]
        lineage = read_json(output / "pixel-lineage.json")["records"]
        self.assertEqual(dataset, output.resolve() / "dataset")
        self.assertEqual([r["frame"] for r in rows], ["frame_000001", "frame_000002"])
        self.assertEqual([r["wh"] for r in rows], [[20, 12], [12, 20]])
        self.assertEqual(
            rows[0]["K"], [[70.0, 0.0, 10.0], [0.0, 70.0, 6.0], [0.0, 0.0, 1.0]]
        )
        np.testing.assert_allclose(
            rows[1]["w2c"],
            [
                [0.0, -1.0, 0.0, -1.0],
                [1.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            atol=1e-14,
        )
        for row in lineage:
            self.assertIsNone(row["source_index"])
            self.assertIsNone(row["time_seconds"])
            with Image.open(dataset / "images" / row["name"]) as image:
                np.testing.assert_array_equal(
                    np.array(image), self.pixels[row["source_colmap_name"]]
                )
        images = binary_images(dataset / "sparse/0/images.bin")
        self.assertEqual([r[0][0] for r in images], [1, 2])
        self.assertEqual(
            [r[1] for r in images], ["frame_000001.png", "frame_000002.png"]
        )
        self.assertEqual([r[2] for r in images], [[(3.25, 4.5, 7)], [(3.25, 4.5, 7)]])
        with (dataset / "sparse/0/points3D.bin").open("rb") as stream:
            self.assertEqual(struct.unpack("<Q", stream.read(8)), (1,))
            point = struct.unpack("<Q3d3BdQ", stream.read(51))
            self.assertEqual(point, (7, 0.1, 0.2, 3.0, 12, 34, 56, 0.25, 2))
            self.assertEqual(struct.unpack("<4i", stream.read(16)), (1, 0, 2, 0))
        self.assertEqual((self.source / "sparse/0/images.txt").read_bytes(), original)
        self.assertEqual(read_json(output / "state.json")["state"], "completed")

    def test_binary_import_preserves_standard_model(self):
        first = data.prepare_colmap(self.source, self.root / "first")
        second = data.prepare_colmap(first, self.root / "second")
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            self.assertEqual(
                sha256(first / "sparse/0" / name), sha256(second / "sparse/0" / name)
            )

    def test_unregistered_images_and_distortion_are_not_silently_accepted(self):
        extra = self.source / "images/extra.png"
        Image.new("RGB", (20, 12), "white").save(extra)
        with self.assertRaisesRegex(data.DataPreparationError, "Every input frame"):
            data.prepare_colmap(self.source, self.root / "unregistered")
        self.assertEqual(
            read_json(self.root / "unregistered/state.json")["state"], "failed"
        )
        cameras = self.source / "sparse/0/cameras.txt"
        cameras.write_text(
            cameras.read_text().replace("SIMPLE_PINHOLE", "SIMPLE_RADIAL"),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(data.DataPreparationError, "undistort"):
            data.prepare_colmap(self.source, self.root / "distorted")

    def test_existing_output_is_preserved(self):
        target = self.root / "existing"
        target.mkdir()
        (target / "keep.txt").write_text("keep")
        with self.assertRaises(FileExistsError):
            data.prepare_colmap(self.source, target)
        self.assertEqual((target / "keep.txt").read_text(), "keep")


class ColmapBoundary:
    """Only the COLMAP process boundary is replaced; video decoding stays real."""

    def __init__(self, missing=False):
        self.images = None
        self.missing = missing
        self.calls = []

    def __call__(self, command, log, cwd=None, env=None):
        args = list(map(str, command))
        if args[0] != "fixture-colmap":
            return run_logged(command, log, cwd=cwd, env=env)
        self.calls.append(args)
        if env.get("CUDA_VISIBLE_DEVICES") != "":
            raise AssertionError("COLMAP must run without GPU visibility")
        if args[1] == "feature_extractor":
            self.images = Path(args[args.index("--image_path") + 1])
        elif args[1] == "mapper":
            paths = sorted(self.images.glob("*.png"))
            if self.missing:
                paths = paths[:-1]
            dimensions = []
            for path in paths:
                with Image.open(path) as image:
                    dimensions.append(image.size)
            text_model(
                Path(args[args.index("--output_path") + 1]) / "0",
                [p.name for p in paths],
                dimensions,
            )
        elif args[1] == "bundle_adjuster":
            shutil.copytree(
                args[args.index("--input_path") + 1],
                args[args.index("--output_path") + 1],
                dirs_exist_ok=True,
            )


@unittest.skipUnless(
    FFMPEG and FFPROBE, "FFmpeg and FFprobe are required for CPU video tests"
)
class VideoPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        images = self.root / "input_frames"
        images.mkdir()
        for index in range(6):
            pixels = np.zeros((24, 40, 3), dtype=np.uint8)
            pixels[:, :, :] = [index * 30, 50, 180]
            pixels[:10, :12] = [240, 10, 20]
            pixels[-8:, -14:] = [20, 230, 30]
            Image.fromarray(pixels).save(images / f"{index:03d}.png")
        self.video = self.root / "landscape.mp4"
        subprocess.run(
            [
                FFMPEG,
                "-v",
                "error",
                "-y",
                "-framerate",
                "6",
                "-i",
                str(images / "%03d.png"),
                "-c:v",
                "mpeg4",
                "-q:v",
                "2",
                "-threads",
                "1",
                str(self.video),
            ],
            check=True,
            capture_output=True,
        )

    def test_landscape_and_portrait_keep_complete_display_pixels(self):
        portrait = self.root / "portrait.mp4"
        subprocess.run(
            [
                FFMPEG,
                "-v",
                "error",
                "-y",
                "-i",
                str(self.video),
                "-c",
                "copy",
                "-metadata:s:v:0",
                "rotate=90",
                str(portrait),
            ],
            check=True,
            capture_output=True,
        )
        metadata = json.loads(
            subprocess.check_output(
                [FFPROBE, "-v", "error", "-show_streams", "-of", "json", str(portrait)],
                text=True,
            )
        )["streams"][0]
        has_rotation = any(
            abs(float(item.get("rotation", 0))) == 90
            for item in metadata.get("side_data_list", [])
        )
        if (
            not has_rotation
            and abs(float(metadata.get("tags", {}).get("rotate", 0))) != 90
        ):
            # New FFmpeg versions no longer create a display matrix from a tag.
            subprocess.run(
                [
                    FFMPEG,
                    "-v",
                    "error",
                    "-y",
                    "-display_rotation:v:0",
                    "90",
                    "-i",
                    str(self.video),
                    "-c",
                    "copy",
                    str(portrait),
                ],
                check=True,
                capture_output=True,
            )
        for index, (video, expected_wh) in enumerate(
            [(self.video, [40, 24]), (portrait, [24, 40])]
        ):
            with self.subTest(video=video.name):
                before = sha256(video)
                output = self.root / ("prepared_" + str(index))
                boundary = ColmapBoundary()
                with patch.object(data, "run_logged", side_effect=boundary):
                    dataset = data.prepare_video(
                        video,
                        output,
                        dict(cpu_threads=1),
                        colmap="fixture-colmap",
                        ffmpeg=FFMPEG,
                        ffprobe=FFPROBE,
                    )
                rows = read_json(output / "pixel-lineage.json")["records"]
                self.assertEqual([r["source_index"] for r in rows], [0, 2, 4])
                self.assertTrue(all(r["wh"] == expected_wh for r in rows))
                reference = self.root / ("reference_" + str(index))
                reference.mkdir()
                subprocess.run(
                    [
                        FFMPEG,
                        "-v",
                        "error",
                        "-i",
                        str(video),
                        "-vsync",
                        "0",
                        "-threads",
                        "1",
                        "-pix_fmt",
                        "rgb24",
                        str(reference / "%06d.png"),
                    ],
                    check=True,
                    capture_output=True,
                )
                for row in rows:
                    with Image.open(dataset / "images" / row["name"]) as actual:
                        with Image.open(
                            reference / f"{row['source_index'] + 1:06d}.png"
                        ) as expected:
                            np.testing.assert_array_equal(
                                np.array(actual), np.array(expected)
                            )
                self.assertEqual(sha256(video), before)
                self.assertEqual(
                    read_json(output / "camera-inputs.json")["registered"], 3
                )
                feature = next(c for c in boundary.calls if c[1] == "feature_extractor")
                match = next(c for c in boundary.calls if c[1] == "sequential_matcher")
                self.assertEqual(
                    feature[feature.index("--SiftExtraction.use_gpu") + 1], "0"
                )
                self.assertEqual(
                    feature[feature.index("--SiftExtraction.domain_size_pooling") + 1],
                    "1",
                )
                self.assertEqual(
                    match[match.index("--SequentialMatching.overlap") + 1], "10"
                )

    def test_locked_indices_and_all_frame_registration(self):
        boundary = ColmapBoundary()
        output = self.root / "locked"
        with patch.object(data, "run_logged", side_effect=boundary):
            data.prepare_video(
                self.video,
                output,
                dict(frame_indices=[1, 4], matching="exhaustive", cpu_threads=1),
                colmap="fixture-colmap",
                ffmpeg=FFMPEG,
                ffprobe=FFPROBE,
            )
        rows = read_json(output / "pixel-lineage.json")["records"]
        self.assertEqual([r["source_index"] for r in rows], [1, 4])
        self.assertAlmostEqual(rows[0]["time_seconds"], 1 / 6, delta=1e-6)
        self.assertTrue(any(c[1] == "exhaustive_matcher" for c in boundary.calls))
        failed = self.root / "missing_camera"
        with patch.object(data, "run_logged", side_effect=ColmapBoundary(missing=True)):
            with self.assertRaisesRegex(data.DataPreparationError, "Every input frame"):
                data.prepare_video(
                    self.video,
                    failed,
                    dict(cpu_threads=1),
                    colmap="fixture-colmap",
                    ffmpeg=FFMPEG,
                    ffprobe=FFPROBE,
                )
        self.assertEqual(read_json(failed / "state.json")["state"], "failed")
        self.assertFalse((failed / "camera-inputs.json").exists())


if __name__ == "__main__":
    unittest.main()
