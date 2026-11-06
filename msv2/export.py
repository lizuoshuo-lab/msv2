"""Portable 3DGS exports with independent covariance, SH and render checks."""

from pathlib import Path
from types import SimpleNamespace
import copy, json, shutil, zipfile
import numpy as np
from .engine import NAMES, LazyCamera
from .io import read_json as read, save_json as save, sha256 as sha


def write_colmap_cameras(folder, rows):
    """Write a standard COLMAP text model containing camera poses only."""
    from scipy.spatial.transform import Rotation

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cameras = ["# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]"]
    images = [
        "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME",
        "# Empty POINTS2D lines: this export contains camera poses, not feature tracks.",
    ]
    for index, row in enumerate(rows, 1):
        k = np.asarray(row["K"])
        w = np.asarray(row["w2c"])
        q = Rotation.from_matrix(w[:3, :3]).as_quat()[[3, 0, 1, 2]]
        parameters = [k[0, 0], k[1, 1], k[0, 2], k[1, 2]]
        cameras.append(
            f"{index} PINHOLE {row['wh'][0]} {row['wh'][1]} "
            + " ".join(format(float(v), ".17g") for v in parameters)
        )
        images.append(
            str(index)
            + " "
            + " ".join(format(float(v), ".17g") for v in [*q, *w[:3, 3]])
            + f" {index} {row['name']}"
        )
        images.append("")
    (folder / "cameras.txt").write_text("\n".join(cameras) + "\n", encoding="utf-8")
    (folder / "images.txt").write_text("\n".join(images) + "\n", encoding="utf-8")
    (folder / "points3D.txt").write_text(
        "# No feature tracks are included in the pose-only export.\n", encoding="utf-8"
    )


def camera_records(rows):
    result = []
    for i, r in enumerate(rows):
        w = np.asarray(r["w2c"])
        c = np.linalg.inv(w)
        k = np.asarray(r["K"])
        result.append(
            dict(
                id=i,
                img_name=r["frame"],
                width=r["wh"][0],
                height=r["wh"][1],
                position=c[:3, 3].tolist(),
                rotation=c[:3, :3].tolist(),
                fx=k[0, 0],
                fy=k[1, 1],
                cx=k[0, 2],
                cy=k[1, 2],
                K=r["K"],
                pixelConvention="pixel_corners",
                split=r["split"],
            )
        )
    return result


def camera_input_mapping(rows, sr_region=None, *, sr=None):
    """Record complete training views; a mask never creates another camera."""
    from .report import view_summary

    counts = view_summary(rows)
    dimensions = counts["view_dimensions"]
    face_mode = sr_region == "face"
    if face_mode:
        if (
            sr is None
            or sr.get("state") != "completed"
            or sr.get("view_layout") != "full4x_mask_composite"
        ):
            raise ValueError("Face composition requires its full-frame SR provenance")
        declared = {row["name"]: row for row in sr["records"]}
        if len(declared) != len(sr["records"]) or set(declared) != {
            row["name"] for row in rows
        }:
            raise ValueError("Composite source names differ from camera names")
        if any(
            type(item.get("face_count")) is not int or item["face_count"] < 0
            for item in declared.values()
        ):
            raise ValueError("Invalid applied face count")
        mask_used = any(item.get("face_count", 0) > 0 for item in declared.values())
    else:
        mask_used = False
    return dict(
        records=[
            {
                key: value
                for key, value in row.items()
                if key not in ("image", "native_image")
            }
            for row in rows
        ],
        mask_used=False,
        training_mask_used=False,
        full_frame=True,
        training_crop_used=False,
        crop_views_present=False,
        additional_training_views=False,
        coordinate_convention="COLMAP .5-based pixel centers",
        training_resolution=dimensions[0] if len(dimensions) == 1 else None,
        training_resolutions=dimensions,
        source_frames=counts["source_frames"],
        view_count=counts["view_count"],
        full_views=counts["full_views"],
        head_views=0,
        source_group_split=True,
        view_layout="full4x_mask_composite" if face_mode else "full",
        input_composite_mask=mask_used,
        input_face_mask_used=mask_used,
        input_roi_composite=mask_used,
        full_frame_4x_generated=sr_region in ("face", "full"),
        mask_scope="face" if face_mode else None,
        outside_mask_reference="Original full-image Lanczos4" if face_mode else None,
    )


def export(engine, scale):
    torch = engine.runtime()
    run = engine.root / NAMES[scale]
    out = engine.record / NAMES[scale]
    out.mkdir(parents=True, exist_ok=True)
    assert (
        read(run / "execution.json")["state"] == "training_completed"
        and read(run / "ply-parity.json")["passed"]
    )
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from utils.sh_utils import eval_sh
    from plyfile import PlyData
    from scipy.spatial.transform import Rotation

    rows = read(run / "training-inputs.json")["records"]
    world = out / "point_cloud_world.ply"
    shutil.copy2(
        run / f"model/point_cloud/iteration_{engine.iterations}/point_cloud.ply", world
    )
    save(out / "cameras_world.json", camera_records(rows))
    world_model = GaussianModel(3)
    world_model.load_ply(str(world))
    world_model.active_sh_degree = 3
    ply = PlyData.read(world)
    v = ply["vertex"].data
    assert len(v.dtype.names) == 62 and all(
        np.isfinite(v[k]).all() for k in v.dtype.names
    )
    # Exact axis sign change avoids roundoff in arbitrary-angle world rotations.
    # COLMAP's initialization camera uses +Y down; its up becomes viewer +Y.
    rotation = np.diag([1.0, -1.0, -1.0])
    assert abs(np.linalg.det(rotation) - 1) < 1e-7
    transform = np.eye(4)
    transform[:3, :3] = rotation
    xyz = world_model.get_xyz.detach().cpu().numpy()
    rotated = xyz @ rotation.T
    for j, key in enumerate(("x", "y", "z")):
        v[key] = rotated[:, j]
    raw_quat = np.stack([v[f"rot_{j}"] for j in range(4)], 1).copy()
    q = raw_quat.astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    new_quat = np.stack(
        [-raw_quat[:, 1], raw_quat[:, 0], -raw_quat[:, 3], raw_quat[:, 2]], axis=1
    )
    for j in range(4):
        v[f"rot_{j}"] = new_quat[:, j]
    # Analytic signs for the 180-degree X rotation, with an independent basis check.
    rng = np.random.default_rng(9327)
    directions = rng.normal(size=(96, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    def basis(d):
        tensor = torch.as_tensor(d, dtype=torch.float64, device="cuda")
        columns = []
        for j in range(16):
            coefficients = torch.zeros(
                (len(d), 1, 16), dtype=torch.float64, device="cuda"
            )
            coefficients[:, :, j] = 1
            columns.append(eval_sh(3, coefficients, tensor).cpu().numpy()[:, 0])
        return np.stack(columns, 1)

    signs = np.asarray(
        [1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1], dtype=np.float32
    )
    a = basis(directions @ rotation.T)
    b = basis(directions)
    change = np.diag(signs)
    assert np.max(np.abs(a @ change - b)) < 1e-12
    features = world_model.get_features.detach().cpu().numpy().transpose(0, 2, 1)
    new_features = features * signs[None, None, :]
    for channel in range(3):
        v[f"f_dc_{channel}"] = new_features[:, channel, 0]
        for j in range(15):
            v[f"f_rest_{channel * 15 + j}"] = new_features[:, channel, j + 1]
    yup = out / "point_cloud.ply"
    ply.write(yup)
    model = GaussianModel(3)
    model.load_ply(str(yup))
    model.active_sh_degree = 3
    rrows = copy.deepcopy(rows)
    for r in rrows:
        r["w2c"] = (np.asarray(r["w2c"]) @ np.linalg.inv(transform)).tolist()
    save(out / "cameras.json", camera_records(rrows))
    # Independent full covariance, SH and render validation after PLY readback.
    old_matrix = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    new_matrix = Rotation.from_quat(
        model._rotation.detach().cpu().numpy()[:, [1, 2, 3, 0]]
    ).as_matrix()
    scales2 = np.exp(2 * world_model._scaling.detach().cpu().numpy().astype(np.float64))
    old_cov = (old_matrix * scales2[:, None, :]) @ old_matrix.transpose(0, 2, 1)
    new_cov = (new_matrix * scales2[:, None, :]) @ new_matrix.transpose(0, 2, 1)
    expected = rotation[None] @ old_cov @ rotation.T[None]
    covariance_relative = float(
        np.max(
            np.linalg.norm(new_cov - expected, axis=(1, 2))
            / np.maximum(np.linalg.norm(expected, axis=(1, 2)), 1e-15)
        )
    )
    position_error = float(
        np.max(np.abs(model.get_xyz.detach().cpu().numpy() @ rotation - xyz))
    )
    # Bound inverse coordinate error by float32 PLY precision at actual world scale.
    position_tolerance = float(8 * np.finfo(np.float32).eps * np.max(np.abs(xyz)))
    testdirs = rng.normal(size=(len(v), 3))
    testdirs /= np.linalg.norm(testdirs, axis=1, keepdims=True)
    td = torch.as_tensor(testdirs, dtype=torch.float32, device="cuda")
    tr = torch.as_tensor(testdirs @ rotation.T, dtype=torch.float32, device="cuda")
    sh_error = float(
        (
            eval_sh(3, world_model.get_features.transpose(1, 2), td)
            - eval_sh(3, model.get_features.transpose(1, 2), tr)
        )
        .abs()
        .max()
    )
    assert (
        covariance_relative < 2e-6
        and position_error < position_tolerance
        and sh_error < 1e-4
    )
    native = read(engine.root / "original/training-inputs.json")["records"]
    pipe = SimpleNamespace(
        convert_SHs_python=False, compute_cov3D_python=False, debug=False
    )
    bg = torch.zeros(3, device="cuda")
    checks = []
    with torch.no_grad():
        for i in (0, len(native) // 2, len(native) - 1):
            original = native[i]
            rotated_row = copy.deepcopy(original)
            rotated_row["w2c"] = (
                np.asarray(original["w2c"]) @ np.linalg.inv(transform)
            ).tolist()
            x = render(LazyCamera(original, i), world_model, pipe, bg)["render"]
            y = render(LazyCamera(rotated_row, i), model, pipe, bg)["render"]
            d = (x - y).abs()
            checks.append(
                dict(
                    frame=original["frame"],
                    max_abs=float(d.max()),
                    mean_abs=float(d.mean()),
                )
            )
            del x, y, d
    passed = all(c["max_abs"] < 0.005 for c in checks)
    assert passed, checks
    save(
        out / "upright-parity.json",
        dict(
            passed=passed,
            transform=transform.tolist(),
            convention="COLMAP world rotated 180 degrees around X; camera poses rotate identically; not measured gravity",
            covariance_relative=covariance_relative,
            inverse_position_error=position_error,
            position_tolerance=position_tolerance,
            SH_error=sh_error,
            render_checks=checks,
        ),
    )
    save(
        out / "camera-input-mapping.json",
        camera_input_mapping(
            rows,
            engine.config.get("sr_region", "full") if scale == 4 else None,
            sr=read(run / "superresolution.json")
            if scale == 4 and engine.config.get("sr_region", "full") == "face"
            else None,
        ),
    )
    shutil.copy2(run / "ply-parity.json", out / "native-ply-parity.json")
    shutil.copy2(engine.root / "plan.json", out / "plan.json")
    write_colmap_cameras(out / "colmap", rrows)
    bundle = out / "scene.zip"
    with zipfile.ZipFile(
        bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3
    ) as z:
        z.write(yup, "point_cloud.ply")
        z.write(out / "cameras.json", "cameras.json")
        for path in sorted((out / "colmap").iterdir()):
            z.write(path, "colmap/" + path.name)
        for name in (
            "upright-parity.json",
            "native-ply-parity.json",
            "camera-input-mapping.json",
            "plan.json",
        ):
            z.write(out / name, name)
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None and len(json.loads(z.read("cameras.json"))) == len(
            rows
        )
    save(
        out / "delivery.json",
        dict(
            bundle=bundle.name,
            points=len(v),
            cameras=len(rows),
            sha256=sha(bundle),
            parity_passed=True,
            quality_accepted=False,
        ),
    )
    print("STANDARD_EXPORT_DONE", scale, len(v), len(rows), bundle, flush=True)
