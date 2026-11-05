"""Lossless storage scheduling for the pinned rasterizer's image-state buffer.

The native renderer, loss, gradients and optimizer are unchanged. Large images
temporarily keep the opaque image-state bytes on CPU between rendering and its
backward pass, avoiding overlap with the full-resolution SSIM working set.
"""

from dataclasses import dataclass
from functools import wraps
import importlib
import json
from pathlib import Path

from .io import read_json, sha256

_MIN_PIXELS = 32 * 1024**2
_MIN_BUFFER_BYTES = 1024**3
_RASTER_SOURCE = (
    "submodules/diff-gaussian-rasterization_ms/"
    "diff_gaussian_rasterization_ms/__init__.py"
)
_RASTER_SHA = "584049aba1b92ae371b10f7a8f2d29d85828592b00be7802f6e5544eafb0b711"


@dataclass(frozen=True)
class _ImageState:
    device: object
    cpu_bytes: object
    size: int


def _verify_rasterizer(source_dir):
    # This byte-buffer order is specific to the pinned upstream Function's
    # save_for_backward(..., geomBuffer, binningBuffer, imgBuffer, sampleBuffer).
    from .engine import COMMIT, SOURCE_ARCHIVE_SHA

    source_dir = Path(source_dir)
    receipt = read_json(source_dir / ".msv2-source.json")
    if (
        receipt.get("commit") != COMMIT
        or receipt.get("archive_sha256") != SOURCE_ARCHIVE_SHA
        or receipt.get("files", {}).get(_RASTER_SOURCE) != _RASTER_SHA
        or sha256(source_dir / _RASTER_SOURCE) != _RASTER_SHA
    ):
        raise RuntimeError("Image-state memory adapter requires the pinned rasterizer")
    rasterizer = importlib.import_module("diff_gaussian_rasterization_ms")
    installed = Path(rasterizer.__file__).resolve()
    if sha256(installed) != _RASTER_SHA:
        raise RuntimeError("Installed rasterizer Python differs from the pinned source")
    return dict(
        commit=COMMIT,
        archive_sha256=SOURCE_ARCHIVE_SHA,
        source=str(source_dir / _RASTER_SOURCE),
        installed_module=str(installed),
        module_sha256=_RASTER_SHA,
    )


def wrap_training_render(native_render, source_dir):
    """Wrap only native render_imp; small images take the original path.

    The returned callable exposes numeric ``offload_stats`` and JSON-compatible
    ``memory_stats``. Transfers are blocking and preserve every uint8 value,
    shape and CUDA device. Only the third of the pinned Function's four saved
    CUDA byte buffers is moved. Inference without gradients needs no offload.
    """
    import torch

    if getattr(native_render, "_msv2_image_state_wrapper", False):
        raise RuntimeError("The image-state memory adapter is already installed")
    receipt = _verify_rasterizer(source_dir)
    counters = dict(
        minimum_pixels=_MIN_PIXELS,
        minimum_buffer_bytes=_MIN_BUFFER_BYTES,
        selection=3,
        uint8_buffers_expected=4,
        render_calls=0,
        large_image_calls=0,
        offload_count=0,
        restored_count=0,
        offload_bytes=0,
        restored_bytes=0,
        last_buffer_bytes=0,
        max_buffer_bytes=0,
    )
    stats = dict(
        policy="Pinned imgBuffer uint8 bytes: synchronous CPU save / CUDA restore",
        rasterizer=receipt,
        counters=counters,
    )

    @wraps(native_render)
    def render(viewpoint_camera, *args, **kwargs):
        counters["render_calls"] += 1
        pixels = int(viewpoint_camera.image_width) * int(viewpoint_camera.image_height)
        if pixels < _MIN_PIXELS or not torch.is_grad_enabled():
            return native_render(viewpoint_camera, *args, **kwargs)

        saved = dict(byte_buffers=0, offloads=0, size=0)

        def pack(tensor):
            if tensor.is_cuda and tensor.dtype == torch.uint8:
                saved["byte_buffers"] += 1
                if saved["byte_buffers"] == 3:
                    size = tensor.numel() * tensor.element_size()
                    if (
                        tensor.ndim != 1
                        or not tensor.is_contiguous()
                        or size < _MIN_BUFFER_BYTES
                    ):
                        raise RuntimeError(
                            "Pinned rasterizer imgBuffer layout/size changed"
                        )
                    # Do not retain the original CUDA tensor in the packed
                    # payload. The complete opaque buffer is copied unchanged.
                    payload = _ImageState(
                        tensor.device,
                        tensor.detach().to("cpu", non_blocking=False, copy=True),
                        size,
                    )
                    saved["offloads"] += 1
                    saved["size"] = size
                    counters["offload_count"] += 1
                    counters["offload_bytes"] += size
                    counters["last_buffer_bytes"] = size
                    counters["max_buffer_bytes"] = max(
                        counters["max_buffer_bytes"], size
                    )
                    return payload
            # Sharing storage through detach preserves values while avoiding
            # a saved-hook payload retaining an autograd reference cycle.
            return tensor.detach()

        def unpack(payload):
            if not isinstance(payload, _ImageState):
                return payload
            restored = payload.cpu_bytes.to(
                payload.device, non_blocking=False, copy=True
            )
            if (
                restored.dtype != torch.uint8
                or restored.ndim != 1
                or not restored.is_contiguous()
                or restored.numel() != payload.size
            ):
                raise RuntimeError("Restored rasterizer image-state layout changed")
            counters["restored_count"] += 1
            counters["restored_bytes"] += payload.size
            return restored

        # The hooks apply only to rendering. SSIM and SparseAdam execute
        # outside this scope with their original implementation and tensors.
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            result = native_render(viewpoint_camera, *args, **kwargs)
        if saved["byte_buffers"] != 4 or saved["offloads"] != 1:
            raise RuntimeError(
                "Pinned rasterizer must save exactly four CUDA byte buffers "
                "and offload only its image-state buffer"
            )
        counters["large_image_calls"] += 1
        if (
            counters["large_image_calls"] <= 3
            or counters["large_image_calls"] % 100 == 0
        ):
            print(
                "MSV2_IMAGE_STATE_MEMORY "
                + json.dumps(
                    dict(
                        large_image_calls=counters["large_image_calls"],
                        offload_count=counters["offload_count"],
                        restored_count=counters["restored_count"],
                        buffer_bytes=saved["size"],
                        offload_bytes=counters["offload_bytes"],
                        restored_bytes=counters["restored_bytes"],
                        pixels=pixels,
                    ),
                    sort_keys=True,
                ),
                flush=True,
            )
        return result

    render.memory_stats = stats
    render.offload_stats = counters
    render._msv2_image_state_wrapper = True
    return render
