"""CPU boundary checks for the pinned image-state storage adapter."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from msv2 import memory
from msv2.engine import COMMIT, SOURCE_ARCHIVE_SHA
from msv2.io import save_json, sha256


@pytest.fixture
def hook_spy(monkeypatch):
    # Only observe the API boundary. No tensor, CUDA or gradient behavior is
    # simulated, and reaching the hooks would fail these bypass/guard tests.
    hooks = Mock(side_effect=AssertionError("saved hooks must not be installed"))
    torch_api = SimpleNamespace(
        is_grad_enabled=Mock(return_value=True),
        autograd=SimpleNamespace(graph=SimpleNamespace(saved_tensors_hooks=hooks)),
    )
    monkeypatch.setitem(sys.modules, "torch", torch_api)
    return torch_api, hooks


@pytest.fixture
def pinned_modules(tmp_path, monkeypatch):
    vendor = tmp_path / "vendor"
    source = vendor / memory._RASTER_SOURCE
    installed = tmp_path / "installed/diff_gaussian_rasterization_ms/__init__.py"
    for path in (source, installed):
        path.parent.mkdir(parents=True)
        path.write_bytes(b"pinned rasterizer ABI fixture\n")
    # Pin this fixture's real file bytes, then exercise actual file hashing.
    pinned_hash = sha256(source)
    monkeypatch.setattr(memory, "_RASTER_SHA", pinned_hash)
    receipt = dict(
        commit=COMMIT,
        archive_sha256=SOURCE_ARCHIVE_SHA,
        files={memory._RASTER_SOURCE: pinned_hash},
    )
    save_json(vendor / ".msv2-source.json", receipt)
    importer = Mock(return_value=SimpleNamespace(__file__=str(installed)))
    monkeypatch.setattr(memory, "importlib", SimpleNamespace(import_module=importer))
    return vendor, source, installed, receipt, importer


@pytest.mark.parametrize("drift", ["receipt", "vendor", "installed"])
def test_abi_drift_stops_before_native_or_hooks(pinned_modules, hook_spy, drift):
    vendor, source, installed, receipt, importer = pinned_modules
    _, hooks = hook_spy
    if drift == "receipt":
        receipt["files"][memory._RASTER_SOURCE] = "0" * 64
        save_json(vendor / ".msv2-source.json", receipt)
    elif drift == "vendor":
        source.write_bytes(b"different saved tensor ordering\n")
    else:
        installed.write_bytes(b"installed rasterizer differs from vendor\n")
    native = Mock(_msv2_image_state_wrapper=False)
    with pytest.raises(RuntimeError, match="pinned|Installed rasterizer"):
        memory.wrap_training_render(native, vendor)
    hooks.assert_not_called()
    native.assert_not_called()
    if drift == "installed":
        importer.assert_called_once_with("diff_gaussian_rasterization_ms")
    else:
        importer.assert_not_called()


def test_abi_receipt_binds_verified_installed_file(pinned_modules):
    vendor, source, installed, _, importer = pinned_modules
    receipt = memory._verify_rasterizer(vendor)
    assert receipt["source"] == str(source)
    assert receipt["installed_module"] == str(installed.resolve())
    assert receipt["module_sha256"] == sha256(source) == sha256(installed)
    importer.assert_called_once_with("diff_gaussian_rasterization_ms")


@pytest.mark.parametrize(
    "width,height,grad_enabled", [(37, 23, True), (8192, 8192, False)]
)
def test_small_or_inference_render_is_unchanged_passthrough(
    monkeypatch, hook_spy, width, height, grad_enabled
):
    torch_api, hooks = hook_spy
    torch_api.is_grad_enabled.return_value = grad_enabled
    source, camera = object(), SimpleNamespace(image_width=width, image_height=height)
    verifier = Mock(return_value={"verified": "fixture"})
    monkeypatch.setattr(memory, "_verify_rasterizer", verifier)
    result, model, pipeline, culling = object(), object(), object(), object()
    native = Mock(return_value=result, _msv2_image_state_wrapper=False)
    wrapped = memory.wrap_training_render(native, source)
    assert (
        wrapped(camera, model, pipeline, culling=culling, scaling_modifier=0.75)
        is result
    )
    native.assert_called_once_with(
        camera, model, pipeline, culling=culling, scaling_modifier=0.75
    )
    verifier.assert_called_once_with(source)
    hooks.assert_not_called()
    assert wrapped.offload_stats["offload_count"] == 0
    assert wrapped.offload_stats["restored_count"] == 0
