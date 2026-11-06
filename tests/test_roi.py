from pathlib import Path
from types import SimpleNamespace

import pytest

from msv2 import roi


def camera(full=(67, 51), origin=(16, 16), size=(64, 48)):
    return SimpleNamespace(
        image_width=size[0],
        image_height=size[1],
        _msv2_roi=dict(full_wh=list(full), viewport_xy=list(origin), wh=list(size)),
    )


def test_viewport_preserves_odd_complete_camera_and_allows_padding():
    assert roi.viewport(camera()) == (67, 51, 16, 16, 64, 48)


@pytest.mark.parametrize(
    "cam",
    [
        camera(origin=(3, 16)),
        camera(size=(60, 48)),
        camera(origin=(-16, 0)),
        camera(origin=(80, 0)),
        camera(full=(0, 51)),
        camera(size=(0, 48)),
        camera(full=(True, 51)),
        camera(full=(67.0, 51)),
    ],
)
def test_viewport_rejects_invalid_or_shifted_tile_geometry(cam):
    with pytest.raises(ValueError):
        roi.viewport(cam)


def test_viewport_checks_camera_output_dimensions():
    cam = camera()
    cam.image_width = 32
    with pytest.raises(ValueError, match="camera output"):
        roi.viewport(cam)


def test_non_roi_dispatch_is_native_and_does_not_import_backend(monkeypatch):
    monkeypatch.setattr(roi.metadata, "version", lambda _: "pinned-cpu-test")
    calls = []

    def native(*args, **kwargs):
        calls.append((args, kwargs))
        return marker

    def forbidden(*args, **kwargs):
        pytest.fail("Native mother must not load the ROI backend")

    monkeypatch.setattr(roi, "verify_backend", forbidden)
    marker = object()
    cam = SimpleNamespace(image_width=101, image_height=99)
    wrapped = roi.wrap_renderer(native, "unused")
    assert wrapped(cam, "pc", "pipe", "bg", culling="same") is marker
    assert calls == [
        (
            (cam, "pc", "pipe", "bg"),
            dict(
                scaling_modifier=1.0,
                override_color=None,
                flag_max_count=True,
                culling="same",
            ),
        )
    ]
    assert wrapped.roi_stats["full_calls"] == 1
    with pytest.raises(RuntimeError, match="already installed"):
        roi.wrap_renderer(wrapped, "unused")


def test_backend_location_prefers_explicit_then_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(roi.metadata, "version", lambda _: "pinned-cpu-test")
    monkeypatch.setenv("MSV2_ROI_BACKEND_DIR", str(tmp_path / "env"))
    env_path = roi.backend_path(tmp_path / "repo")
    assert env_path.parent == tmp_path / "env"
    assert (
        roi.backend_path(tmp_path / "repo", tmp_path / "explicit").parent
        == tmp_path / "explicit"
    )
    assert (
        roi.backend_path(tmp_path / "repo", tmp_path / "explicit").name == env_path.name
    )
    monkeypatch.delenv("MSV2_ROI_BACKEND_DIR")
    assert roi.backend_path(tmp_path / "repo").parent == tmp_path / "msv2-roi-backend"


def test_backend_versions_change_without_touching_existing_base(monkeypatch, tmp_path):
    base = tmp_path / "backends"
    base.mkdir()
    marker = base / "old-build.txt"
    marker.write_bytes(b"preserved")
    patch = tmp_path / "patch"
    patch.write_bytes(b"first patch")
    monkeypatch.setattr(roi, "_patch_path", lambda: patch)
    monkeypatch.setattr(roi.metadata, "version", lambda _: "runtime-one")
    first = roi.backend_path(tmp_path / "repo", base)
    patch.write_bytes(b"new patch")
    second = roi.backend_path(tmp_path / "repo", base)
    assert first != second
    monkeypatch.setattr(roi.metadata, "version", lambda _: "runtime-two")
    third = roi.backend_path(tmp_path / "repo", base)
    assert third not in (first, second)
    assert first.parent == second.parent == third.parent == base
    assert list(base.iterdir()) == [marker]
    assert marker.read_bytes() == b"preserved"


def test_exact_patch_keeps_context_and_rejects_fuzzy_input(tmp_path):
    target = tmp_path / "setup.py"
    target.write_text("first\nold\nlast\n")
    patch = (
        "--- a/setup.py\n+++ b/setup.py\n@@ -1,3 +1,3 @@\n first\n-old\n+new\n last\n"
    )
    roi._apply_patch(tmp_path, patch)
    assert target.read_text() == "first\nnew\nlast\n"
    with pytest.raises(RuntimeError, match="context mismatch"):
        roi._apply_patch(tmp_path, patch)


@pytest.mark.parametrize(
    "patch",
    [
        "--- a/../outside\n+++ b/../outside\n@@ -1 +1 @@\n-old\n+new\n",
        "--- a/setup.py\n+++ b/setup.py\n@@ -1,2 +1,1 @@\n-old\n+new\n",
    ],
)
def test_patch_rejects_escape_and_inconsistent_hunks(tmp_path, patch):
    (tmp_path / "setup.py").write_text("old\n")
    with pytest.raises(RuntimeError):
        roi._apply_patch(tmp_path, patch)


def test_existing_backend_is_never_replaced(monkeypatch, tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("preserved")
    monkeypatch.setattr(roi, "_verify_source", lambda repo: (Path(repo), {}))
    with pytest.raises(FileExistsError):
        roi.prepare_backend_source(tmp_path / "native", tmp_path)
    assert marker.read_text() == "preserved"


def test_patch_separates_pixel_allocation_from_full_geometry():
    patch = roi._patch_path().read_text()
    # This is a source contract check, not CUDA numerical acceptance.
    assert "+\tconst float ddelx_dx = 0.5 * full_W;" in patch
    assert "+\tconst float ddely_dy = 0.5 * full_H;" in patch
    assert "+\tconst float focal_y = full_height /" in patch
    assert "+\tconst float focal_x = full_width /" in patch
    assert "(float)(pix.x + viewport_x)" in patch
    assert "(y - ty) * viewport_grid.x + (x - tx)" in patch
    assert "if (B == 0) return;" in patch
    # ROI ImageState keeps original local width * height allocation lines.
    assert "required<ImageState>(full_width * full_height)" not in patch
