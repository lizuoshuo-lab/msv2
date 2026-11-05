"""CPU preparation of full RGB video frames and calibrated COLMAP datasets."""

from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import subprocess
import traceback

import numpy as np
from PIL import Image

from .io import now, run_logged, save_json, sha256


class DataPreparationError(RuntimeError):
    """Input frames or calibration did not satisfy the data contract."""


@dataclass(frozen=True)
class _Camera:
    id: int
    model: int
    width: int
    height: int
    params: tuple


@dataclass(frozen=True)
class _FileSlice:
    path: Path
    offset: int
    length: int


@dataclass(frozen=True)
class _ImagePose:
    id: int
    qvec: tuple
    tvec: tuple
    camera_id: int
    name: str
    observations: int
    payload: object


def _rational(value):
    try:
        number = Fraction(str(value))
    except (ValueError, ZeroDivisionError, TypeError) as error:
        raise DataPreparationError(
            "Frame timestamps must be finite real PTS"
        ) from error
    return number


def _floor(value):
    return value.numerator // value.denominator


def _ceil(value):
    return -((-value.numerator) // value.denominator)


def select_frames(timestamps, sample_fps=3.0, frame_indices=None, duration=None):
    """Select nearest real-PTS source frames, with earlier-frame tie breaking.

    Source indices are zero based. Targets start at the first decoded frame's
    PTS. Duplicate selections are removed; each selected frame retains its
    nearest sampling target. Explicit indices must be increasing and unique.
    The Voronoi-cell calculation also handles oversampling without generating
    millions of duplicate sampling targets.
    """
    if not timestamps:
        raise DataPreparationError("The video contains no timestamped frames")
    pts = [_rational(value) for value in timestamps]
    if any(b < a for a, b in zip(pts, pts[1:])):
        raise DataPreparationError("Presentation timestamps must be nondecreasing")
    origin = pts[0]
    relative = [value - origin for value in pts]
    if frame_indices is not None:
        indices = list(frame_indices)
        if (
            not indices
            or any(type(i) is not int or not 0 <= i < len(pts) for i in indices)
            or indices != sorted(set(indices))
        ):
            raise DataPreparationError(
                "frame_indices must be increasing unique source indices in range"
            )
        return [
            dict(
                source_index=i,
                time_seconds=float(relative[i]),
                source_pts_seconds=float(pts[i]),
                target_time_seconds=None,
                sampling_error_seconds=None,
            )
            for i in indices
        ]
    if isinstance(sample_fps, bool):
        raise DataPreparationError("sample_fps must be positive and finite")
    rate = _rational(sample_fps)
    if rate <= 0:
        raise DataPreparationError("sample_fps must be positive and finite")
    endpoint = _rational(duration) if duration is not None else relative[-1]
    if endpoint < relative[-1]:
        raise DataPreparationError(
            "Video duration ends before its last presentation timestamp"
        )
    # A duration denotes an exclusive end; absent a last-frame duration, retain
    # a sample exactly at the final known presentation timestamp.
    last_target = (
        _ceil(endpoint * rate) - 1
        if endpoint > relative[-1]
        else _floor(relative[-1] * rate)
    )
    unique = []
    for index, value in enumerate(relative):
        if not unique or value != unique[-1][1]:
            unique.append((index, value))
    selected = []
    for position, (index, value) in enumerate(unique):
        low = (
            0
            if position == 0
            else _floor((unique[position - 1][1] + value) * rate / 2) + 1
        )
        high = (
            last_target
            if position + 1 == len(unique)
            else min(last_target, _floor((value + unique[position + 1][1]) * rate / 2))
        )
        low = max(0, low)
        if low > high:
            continue
        center = _floor(value * rate)
        candidates = {min(high, max(low, center)), min(high, max(low, center + 1))}
        target_index = min(
            candidates, key=lambda k: (abs(Fraction(k, 1) / rate - value), k)
        )
        target = Fraction(target_index, 1) / rate
        selected.append(
            dict(
                source_index=index,
                time_seconds=float(value),
                source_pts_seconds=float(pts[index]),
                target_time_seconds=float(target),
                sampling_error_seconds=float(value - target),
            )
        )
    return selected


def _fresh_output(output):
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preparation output must be empty: " + str(output))
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset/images").mkdir(parents=True)
    return output


def _failed(output, state, error):
    state.update(
        state="failed",
        finished=now(),
        error=str(error),
        traceback=traceback.format_exc(),
    )
    save_json(output / "state.json", state)


def _logged(output, state, stage, command, cpu_only=False):
    state.update(stage=stage, updated=now())
    state["commands"].append([str(value) for value in command])
    save_json(output / "state.json", state)
    environment = None
    if cpu_only:
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    run_logged(command, output / "logs" / (stage + ".log"), env=environment)


def _probe_video(video, ffprobe, output):
    # Full stream JSON retains rotation metadata on older FFprobe versions,
    # whose -show_entries parser has no stream_side_data section.
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_streams",
        "-show_frames",
        "-of",
        "json",
        str(video),
    ]
    result = subprocess.run(
        [str(v) for v in command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    log = output / "logs/ffprobe.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stderr, encoding="utf-8")
    if result.returncode or result.stderr.strip():
        raise DataPreparationError(
            "FFprobe failed or reported decoding errors; see " + str(log)
        )
    probe = json.loads(result.stdout)
    streams, frames = probe.get("streams", []), probe.get("frames", [])
    if len(streams) != 1 or not frames:
        raise DataPreparationError(
            "A decodable video stream with frame PTS is required"
        )
    stream = streams[0]
    declared = str(stream.get("nb_frames", ""))
    if declared.isdigit() and int(declared) > 0 and int(declared) != len(frames):
        raise DataPreparationError(
            "Decoded frame count differs from declared nb_frames; the video may be incomplete"
        )
    rotation = stream.get("tags", {}).get("rotate", 0)
    for side in stream.get("side_data_list", []):
        if "rotation" in side:
            rotation = side["rotation"]
            break
    rotation = float(rotation)
    if not math.isfinite(rotation) or abs(rotation / 90 - round(rotation / 90)) > 1e-6:
        raise DataPreparationError(
            "Only right-angle display rotations preserve the full pixel rectangle"
        )
    quarter_turn = int(round(rotation / 90)) % 4
    timestamps = []
    for frame in frames:
        value = frame.get("best_effort_timestamp_time", frame.get("pts_time"))
        if value is None or value == "N/A":
            raise DataPreparationError(
                "FFprobe did not provide real presentation timestamps for every frame"
            )
        timestamps.append(value)
        width = int(frame.get("width", stream["width"]))
        height = int(frame.get("height", stream["height"]))
        frame["display_wh"] = [height, width] if quarter_turn % 2 else [width, height]
    relative_last = _rational(timestamps[-1]) - _rational(timestamps[0])
    last_duration = frames[-1].get("duration_time", frames[-1].get("pkt_duration_time"))
    durations = [relative_last]
    if last_duration not in (None, "N/A"):
        durations.append(relative_last + max(Fraction(0), _rational(last_duration)))
    if stream.get("duration") not in (None, "N/A"):
        durations.append(_rational(stream["duration"]))
    return frames, timestamps, max(durations), rotation


def _rgb_array(path):
    with Image.open(path) as image:
        if "A" in image.getbands() or "transparency" in image.info:
            alpha = image.convert("RGBA").getchannel("A")
            if alpha.getextrema() != (255, 255):
                raise DataPreparationError(
                    "Transparent inputs do not contain a complete RGB background: "
                    + str(path)
                )
        return np.array(image.convert("RGB"))


def _record(path, source_index=None, time_seconds=None, **extra):
    pixels = _rgb_array(path)
    height, width = pixels.shape[:2]
    return dict(
        image=path.name,
        name=path.name,
        frame=path.stem,
        sha256=sha256(path),
        wh=[width, height],
        source_index=source_index,
        time_seconds=time_seconds,
        rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
        **extra,
    )


def _read_exact(stream, count):
    value = stream.read(count)
    if len(value) != count:
        raise DataPreparationError("Truncated COLMAP binary model")
    return value


def _relative_name(value):
    path = PurePosixPath(value.replace("\\", "/"))
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\0" in value
        or (path.parts and ":" in path.parts[0])
    ):
        raise DataPreparationError("COLMAP image names must be relative paths")
    return path.as_posix()


def _camera(camera):
    if camera.model not in (0, 1):
        raise DataPreparationError(
            "COLMAP input must be undistorted PINHOLE or SIMPLE_PINHOLE; undistort it first"
        )
    if camera.width < 1 or camera.height < 1 or not np.isfinite(camera.params).all():
        raise DataPreparationError("Invalid COLMAP camera dimensions or intrinsics")
    if camera.model == 0:
        focal, cx, cy = camera.params
        fx = fy = focal
    else:
        fx, fy, cx, cy = camera.params
    if min(fx, fy) <= 0:
        raise DataPreparationError("COLMAP focal lengths must be positive")
    return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]


def _read_cameras(folder):
    result = {}
    binary = folder / "cameras.bin"
    if binary.exists():
        with binary.open("rb") as stream:
            (count,) = struct.unpack("<Q", _read_exact(stream, 8))
            for _ in range(count):
                cid, model, width, height = struct.unpack(
                    "<iiQQ", _read_exact(stream, 24)
                )
                if model not in (0, 1):
                    raise DataPreparationError(
                        "COLMAP input must be undistorted PINHOLE or SIMPLE_PINHOLE; undistort it first"
                    )
                size = 3 if model == 0 else 4
                params = struct.unpack("<" + "d" * size, _read_exact(stream, 8 * size))
                if cid in result:
                    raise DataPreparationError("Duplicate COLMAP camera ID")
                result[cid] = _Camera(cid, model, width, height, params)
    else:
        for line in (folder / "cameras.txt").read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            models = {"SIMPLE_PINHOLE": 0, "PINHOLE": 1}
            if parts[1] not in models:
                raise DataPreparationError(
                    "COLMAP input must be undistorted PINHOLE or SIMPLE_PINHOLE; undistort it first"
                )
            cid, model = int(parts[0]), models[parts[1]]
            params = tuple(map(float, parts[4:]))
            if len(params) != (3 if model == 0 else 4) or cid in result:
                raise DataPreparationError("Invalid or duplicate COLMAP camera record")
            result[cid] = _Camera(cid, model, int(parts[2]), int(parts[3]), params)
    if not result:
        raise DataPreparationError("COLMAP model contains no cameras")
    for item in result.values():
        _camera(item)
    return result


def _read_images(folder):
    result = []
    binary = folder / "images.bin"
    if binary.exists():
        file_size = binary.stat().st_size
        with binary.open("rb") as stream:
            (count,) = struct.unpack("<Q", _read_exact(stream, 8))
            if count > file_size // 73:
                raise DataPreparationError("Invalid COLMAP image count")
            for _ in range(count):
                row = struct.unpack("<i7di", _read_exact(stream, 64))
                name = bytearray()
                while True:
                    char = _read_exact(stream, 1)
                    if char == b"\0":
                        break
                    name.extend(char)
                (observations,) = struct.unpack("<Q", _read_exact(stream, 8))
                offset, length = stream.tell(), observations * 24
                if offset + length > file_size:
                    raise DataPreparationError("Truncated COLMAP image observations")
                stream.seek(length, 1)
                result.append(
                    _ImagePose(
                        row[0],
                        tuple(row[1:5]),
                        tuple(row[5:8]),
                        row[8],
                        _relative_name(name.decode("utf-8")),
                        observations,
                        _FileSlice(binary, offset, length),
                    )
                )
    else:
        with (folder / "images.txt").open(encoding="utf-8") as stream:
            while True:
                line = stream.readline()
                if not line:
                    break
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                row = line.strip().split(maxsplit=9)
                if len(row) != 10:
                    raise DataPreparationError("Invalid COLMAP image pose line")
                points = stream.readline()
                if points == "":
                    raise DataPreparationError("Missing COLMAP 2D observations line")
                values = points.split()
                if len(values) % 3:
                    raise DataPreparationError("Invalid COLMAP 2D observations")
                payload = b"".join(
                    struct.pack(
                        "<ddq",
                        float(values[i]),
                        float(values[i + 1]),
                        int(values[i + 2]),
                    )
                    for i in range(0, len(values), 3)
                )
                result.append(
                    _ImagePose(
                        int(row[0]),
                        tuple(map(float, row[1:5])),
                        tuple(map(float, row[5:8])),
                        int(row[8]),
                        _relative_name(row[9]),
                        len(values) // 3,
                        payload,
                    )
                )
    if len({item.id for item in result}) != len(result) or len(
        {item.name for item in result}
    ) != len(result):
        raise DataPreparationError("Duplicate COLMAP image IDs or names")
    return result


def _w2c(image):
    q = np.asarray(image.qvec, dtype=np.float64)
    t = np.asarray(image.tvec, dtype=np.float64)
    if (
        not np.isfinite(q).all()
        or not np.isfinite(t).all()
        or abs(np.linalg.norm(q) - 1.0) > 1e-5
    ):
        raise DataPreparationError(
            "COLMAP pose must contain a finite unit quaternion and translation"
        )
    w, x, y, z = q
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, t
    return matrix.tolist()


def _write_cameras(path, cameras):
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(cameras)))
        for camera in sorted(cameras.values(), key=lambda item: item.id):
            stream.write(
                struct.pack(
                    "<iiQQ", camera.id, camera.model, camera.width, camera.height
                )
            )
            stream.write(struct.pack("<" + "d" * len(camera.params), *camera.params))


def _write_images(path, images):
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(images)))
        for image in images:
            stream.write(
                struct.pack(
                    "<i7di", image.id, *image.qvec, *image.tvec, image.camera_id
                )
            )
            stream.write(image.name.encode("utf-8") + b"\0")
            stream.write(struct.pack("<Q", image.observations))
            if isinstance(image.payload, bytes):
                stream.write(image.payload)
            else:
                with image.payload.path.open("rb") as source:
                    source.seek(image.payload.offset)
                    left = image.payload.length
                    while left:
                        chunk = _read_exact(source, min(left, 1024**2))
                        stream.write(chunk)
                        left -= len(chunk)


def _write_points(folder, output):
    binary = folder / "points3D.bin"
    if binary.exists():
        with binary.open("rb") as stream:
            (count,) = struct.unpack("<Q", _read_exact(stream, 8))
            for _ in range(count):
                row = struct.unpack("<Q3d3BdQ", _read_exact(stream, 51))
                if not np.isfinite(row[1:4]).all() or not math.isfinite(row[7]):
                    raise DataPreparationError("Nonfinite COLMAP sparse point")
                _read_exact(stream, row[8] * 8)
        shutil.copy2(binary, output)
    else:
        count = 0
        with output.open("wb") as target:
            target.write(struct.pack("<Q", 0))
            for line in (
                (folder / "points3D.txt").read_text(encoding="utf-8").splitlines()
            ):
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                row = line.split()
                if len(row) < 8 or (len(row) - 8) % 2:
                    raise DataPreparationError("Invalid COLMAP sparse-point record")
                xyz = tuple(map(float, row[1:4]))
                rgb, error = tuple(map(int, row[4:7])), float(row[7])
                if (
                    not np.isfinite(xyz).all()
                    or not math.isfinite(error)
                    or any(v < 0 or v > 255 for v in rgb)
                ):
                    raise DataPreparationError("Invalid COLMAP sparse-point values")
                track = list(map(int, row[8:]))
                target.write(
                    struct.pack(
                        "<Q3d3BdQ", int(row[0]), *xyz, *rgb, error, len(track) // 2
                    )
                )
                if track:
                    target.write(struct.pack("<" + "i" * len(track), *track))
                count += 1
            target.seek(0)
            target.write(struct.pack("<Q", count))
    if count < 1:
        raise DataPreparationError("COLMAP model contains no sparse points")
    return count


def _assert_registered(images, names):
    actual = {image.name for image in images}
    expected = set(names)
    if actual != expected:
        missing, extra = sorted(expected - actual), sorted(actual - expected)
        raise DataPreparationError(
            f"Every input frame must register; missing={missing}, unexpected={extra}"
        )


def _finalize(output, model_folder, records, rename=None):
    cameras, images = _read_cameras(model_folder), _read_images(model_folder)
    rename = rename or {row["name"]: row["name"] for row in records}
    _assert_registered(images, rename)
    lookup = {
        rename[image.name]: replace(image, name=rename[image.name]) for image in images
    }
    if len(lookup) != len(records):
        raise DataPreparationError("Image names must remain unique")
    ordered, camera_rows = [], []
    for row in records:
        image = lookup[row["name"]]
        if image.camera_id not in cameras:
            raise DataPreparationError("COLMAP pose references a missing camera")
        camera = cameras[image.camera_id]
        if row["wh"] != [camera.width, camera.height]:
            raise DataPreparationError(
                "COLMAP camera dimensions disagree with decoded full RGB pixels"
            )
        matrix = _w2c(image)
        ordered.append(image)
        camera_rows.append(
            dict(
                name=row["name"],
                frame=row["frame"],
                wh=row["wh"],
                K=_camera(camera),
                w2c=matrix,
                source_sha256=row["sha256"],
                source_index=row["source_index"],
                time_seconds=row["time_seconds"],
                colmap_image_id=image.id,
                colmap_camera_id=image.camera_id,
            )
        )
    sparse = output / "dataset/sparse/0"
    sparse.mkdir(parents=True, exist_ok=True)
    _write_cameras(sparse / "cameras.bin", cameras)
    _write_images(sparse / "images.bin", ordered)
    points = _write_points(model_folder, sparse / "points3D.bin")
    save_json(
        output / "camera-inputs.json",
        dict(
            records=camera_rows,
            registered=len(ordered),
            extracted=len(records),
            sparse_points=points,
            coordinate_contract="OpenCV world-to-camera; COLMAP pixel-corner coordinates",
            mask_used=False,
            crop_used=False,
            full_rgb=True,
        ),
    )
    return len(ordered), points


def _complete_model(folder, names):
    candidates = []
    for model in sorted(folder.iterdir()):
        if not model.is_dir() or not any(
            (model / ("images" + suffix)).exists() for suffix in (".bin", ".txt")
        ):
            continue
        images = _read_images(model)
        candidates.append((len(images), model, images))
        if {image.name for image in images} == set(names):
            return model
    if not candidates:
        raise DataPreparationError("COLMAP did not recover a camera model")
    _, model, images = max(candidates, key=lambda item: item[0])
    _assert_registered(images, names)
    return model


def prepare_video(
    video, output, options=None, colmap="colmap", ffmpeg="ffmpeg", ffprobe="ffprobe"
):
    """Prepare full display-oriented RGB PNGs, real PTS, and CPU SfM cameras.

    Returns ``output/dataset``; the two manifests and state are in ``output``.
    ``frame_indices`` locks exact zero-based source indices. No nominal-FPS,
    human-pose, or interpolated-camera fallback is used.
    """
    options = dict(options or {})
    video = Path(video).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    output = _fresh_output(output)
    state = dict(
        state="running",
        stage="probing",
        started=now(),
        video=str(video),
        video_sha256=sha256(video),
        commands=[],
    )
    save_json(output / "state.json", state)
    try:
        frames, timestamps, duration, rotation = _probe_video(video, ffprobe, output)
        selected = select_frames(
            timestamps,
            options.get("sample_fps", 3.0),
            options.get("frame_indices"),
            duration,
        )
        if len(selected) < 2:
            raise DataPreparationError(
                "At least two distinct sampled frames are required for camera recovery"
            )
        selection_file = output / "source-frame-selection.ffscript"
        expression = "+".join(f"eq(n,{row['source_index']})" for row in selected)
        selection_file.write_text("select='" + expression + "'", encoding="utf-8")
        threads = int(options.get("cpu_threads", 4))
        if threads < 1:
            raise DataPreparationError("cpu_threads must be positive")
        images = output / "dataset/images"
        _logged(
            output,
            state,
            "extract",
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-y",
                "-xerror",
                "-threads",
                str(threads),
                # Autorotation is enabled by default; its explicit option
                # takes different forms in older and newer FFmpeg releases.
                "-i",
                video,
                "-map",
                "0:v:0",
                "-an",
                "-sn",
                "-dn",
                "-filter_script:v",
                selection_file,
                "-vsync",
                "0",
                "-threads",
                str(threads),
                "-pix_fmt",
                "rgb24",
                "-compression_level",
                "3",
                images / "frame_%06d.png",
            ],
        )
        paths = sorted(images.glob("*.png"))
        if len(paths) != len(selected):
            raise DataPreparationError(
                "FFmpeg output count disagrees with selected source identities"
            )
        records = []
        for path, selection in zip(paths, selected):
            row = _record(path, **selection)
            if row["wh"] != frames[row["source_index"]]["display_wh"]:
                raise DataPreparationError(
                    "Decoded dimensions disagree with display orientation; refusing a resized/cropped input"
                )
            records.append(row)
        save_json(
            output / "pixel-lineage.json",
            dict(
                passed=True,
                source_type="video",
                source=str(video),
                video_sha256=state["video_sha256"],
                source_frames=len(frames),
                output_frames=len(records),
                records=records,
                sample_fps=options.get("sample_fps", 3.0),
                explicit_frame_indices=options.get("frame_indices"),
                time_origin_seconds=float(_rational(timestamps[0])),
                display_rotation_degrees=rotation,
                no_spatial_resizing=True,
                no_mask=True,
                no_crop=True,
                full_rgb=True,
                decode_policy="Display autorotation, source-index selection, RGB24 lossless PNG; no scale/crop/fps filter",
            ),
        )
        camera_model = options.get("camera_model", "PINHOLE")
        if camera_model not in ("PINHOLE", "SIMPLE_PINHOLE"):
            raise DataPreparationError(
                "Use PINHOLE or SIMPLE_PINHOLE for full undistorted image coordinates"
            )
        same_size = len({tuple(row["wh"]) for row in records}) == 1
        single_camera = bool(options.get("single_camera", True)) and same_size
        database = output / "work/database.db"
        database.parent.mkdir(parents=True, exist_ok=True)
        common = ["--database_path", database]
        _logged(
            output,
            state,
            "features",
            [
                colmap,
                "feature_extractor",
                *common,
                "--image_path",
                images,
                "--ImageReader.single_camera",
                str(int(single_camera)),
                "--ImageReader.camera_model",
                camera_model,
                "--SiftExtraction.use_gpu",
                "0",
                "--SiftExtraction.num_threads",
                str(threads),
                "--SiftExtraction.estimate_affine_shape",
                "1",
                "--SiftExtraction.domain_size_pooling",
                "1",
            ],
            cpu_only=True,
        )
        matching = options.get("matching", "sequential")
        if matching not in ("sequential", "exhaustive"):
            raise DataPreparationError("matching must be sequential or exhaustive")
        matcher = [
            colmap,
            matching + "_matcher",
            *common,
            "--SiftMatching.use_gpu",
            "0",
            "--SiftMatching.num_threads",
            str(threads),
            "--SiftMatching.guided_matching",
            "1",
        ]
        if matching == "sequential":
            matcher.extend(["--SequentialMatching.overlap", "10"])
        _logged(output, state, "matching", matcher, cpu_only=True)
        models = output / "work/models"
        models.mkdir()
        _logged(
            output,
            state,
            "mapping",
            [
                colmap,
                "mapper",
                *common,
                "--image_path",
                images,
                "--output_path",
                models,
                "--Mapper.num_threads",
                str(threads),
                "--Mapper.ba_refine_principal_point",
                "1",
            ],
            cpu_only=True,
        )
        selected_model = _complete_model(models, [row["name"] for row in records])
        adjusted = output / "work/bundle-adjusted"
        adjusted.mkdir()
        _logged(
            output,
            state,
            "bundle_adjustment",
            [
                colmap,
                "bundle_adjuster",
                "--input_path",
                selected_model,
                "--output_path",
                adjusted,
                "--BundleAdjustment.refine_principal_point",
                "1",
            ],
            cpu_only=True,
        )
        registered, points = _finalize(output, adjusted, records)
        state.update(
            state="completed",
            stage="camera_inputs",
            finished=now(),
            frames_extracted=len(records),
            frames_registered=registered,
            sparse_points=points,
            full_rgb=True,
            mask_used=False,
            crop_used=False,
        )
        save_json(output / "state.json", state)
        return output / "dataset"
    except BaseException as error:
        _failed(output, state, error)
        raise


def prepare_colmap(source, output, options=None):
    """Import calibrated RGB images and binary/text COLMAP without refitting.

    Images become flat, uniquely named lossless RGB PNGs. Image IDs, poses,
    intrinsics, 2D observations and sparse-point tracks are preserved; only
    image filenames change in the binary model. All source images must register.
    """
    source = Path(source).expanduser().resolve()
    if not (source / "images").is_dir() and (source / "dataset/images").is_dir():
        source = source / "dataset"
    if not (source / "images").is_dir() or not (source / "sparse/0").is_dir():
        raise DataPreparationError("Expected COLMAP images/ and sparse/0/")
    output = _fresh_output(output)
    state = dict(
        state="running",
        stage="importing_colmap",
        started=now(),
        source=str(source),
        commands=[],
    )
    save_json(output / "state.json", state)
    try:
        sparse = source / "sparse/0"
        _read_cameras(sparse)
        images = _read_images(sparse)
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        names = {
            path.relative_to(source / "images").as_posix()
            for path in (source / "images").rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        }
        _assert_registered(images, names)
        if len(images) < 2:
            raise DataPreparationError("At least two registered images are required")
        records, rename = [], {}
        for index, image in enumerate(sorted(images, key=lambda item: item.name)):
            original = source / "images" / image.name
            pixels = _rgb_array(original)
            target = output / "dataset/images" / f"frame_{index + 1:06d}.png"
            Image.fromarray(pixels).save(target, compress_level=3)
            row = _record(
                target,
                source_colmap_name=image.name,
                source_file_sha256=sha256(original),
            )
            records.append(row)
            rename[image.name] = target.name
        save_json(
            output / "pixel-lineage.json",
            dict(
                passed=True,
                source_type="colmap",
                source=str(source),
                records=records,
                source_frames=len(records),
                output_frames=len(records),
                no_spatial_resizing=True,
                no_mask=True,
                no_crop=True,
                full_rgb=True,
                decode_policy="Original decoded RGB, lossless PNG encoding; no extra orientation or image resampling",
            ),
        )
        registered, points = _finalize(output, sparse, records, rename)
        state.update(
            state="completed",
            stage="camera_inputs",
            finished=now(),
            frames_extracted=len(records),
            frames_registered=registered,
            sparse_points=points,
            full_rgb=True,
            mask_used=False,
            crop_used=False,
        )
        save_json(output / "state.json", state)
        return output / "dataset"
    except BaseException as error:
        _failed(output, state, error)
        raise
