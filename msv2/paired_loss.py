"""Pixel-pooled supervision for one native parent and its high-resolution ROIs.

One call describes one source group. The caller owns rendering, source-group
sampling and the single backward/optimizer update. This changes loss domains
and pixel weights; it does not change a Gaussian model or the SSIM CUDA kernel.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


_RADIUS = 5
_ALL_PHYSICAL = (True, True, True, True)


def _edges(value):
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError("physical_edges must contain left, top, right, bottom")
    if any(type(edge) is not bool for edge in value):
        raise ValueError("physical_edges entries must be boolean")
    return tuple(value)


def _mask(mask, shape=None, device=None):
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.ndim != 2:
        raise ValueError("A support mask must be a two-dimensional boolean tensor")
    if min(mask.shape) < 1:
        raise ValueError("A support mask cannot have an empty image dimension")
    if shape is not None and tuple(mask.shape) != tuple(shape):
        raise ValueError("Support mask and RGB dimensions differ")
    if device is not None and mask.device != device:
        raise ValueError("Support mask and RGB must be on the same device")
    return mask


def safe_ssim_centers(support, physical_edges=_ALL_PHYSICAL):
    """Centers whose complete in-image 11x11 window is supervised.

    True physical edges permit the upstream SSIM kernel's zero RGB padding.
    Artificial ROI edges do not. No RGB is multiplied by a mask before SSIM.
    """
    _mask(support)
    left, top, right, bottom = _edges(physical_edges)
    height, width = support.shape
    with torch.inference_mode(False), torch.no_grad():
        padded = torch.zeros(
            (height + 10, width + 10), dtype=torch.bool, device=support.device
        )
        padded[5 : 5 + height, 5 : 5 + width] = support
        if left:
            padded[:, :5] = True
        if top:
            padded[:5, :] = True
        if right:
            padded[:, -5:] = True
        if bottom:
            padded[-5:, :] = True
        invalid = (~padded).to(dtype=torch.float32)[None, None]
        return F.max_pool2d(invalid, 11, stride=1)[0, 0] == 0


@dataclass(frozen=True)
class SSIMSupport:
    """A precomputed domain, safe-center map and exact pixel counts."""

    support: torch.Tensor
    centers: torch.Tensor
    pixels: int
    center_pixels: int
    physical_edges: tuple
    _support_version: int
    _centers_version: int

    def _validate(self):
        if (
            self.support._version != self._support_version
            or self.centers._version != self._centers_version
        ):
            raise ValueError("Precomputed SSIM support was modified in place")

    def to(self, device):
        """Transfer precomputed boolean maps without recomputing windows."""
        self._validate()
        with torch.inference_mode(False), torch.no_grad():
            support = self.support.to(device=device)
            centers = self.centers.to(device=device)
        return SSIMSupport(
            support,
            centers,
            self.pixels,
            self.center_pixels,
            self.physical_edges,
            support._version,
            centers._version,
        )


def prepare_ssim_support(support, physical_edges=_ALL_PHYSICAL):
    """Prepare once per source/level; do not mutate the returned maps."""
    _mask(support)
    edges = _edges(physical_edges)
    with torch.inference_mode(False), torch.no_grad():
        support = support.clone()
        centers = safe_ssim_centers(support, edges)
    return SSIMSupport(
        support,
        centers,
        int(support.sum()),
        int(centers.sum()),
        edges,
        support._version,
        centers._version,
    )


def _resolve_support(mask, edges, cache):
    if cache is None:
        return prepare_ssim_support(mask, edges)
    if not isinstance(cache, SSIMSupport):
        raise ValueError("support_cache must be produced by prepare_ssim_support")
    cache._validate()
    if (
        cache.physical_edges != edges
        or cache.support.device != mask.device
        or not torch.equal(cache.support, mask)
    ):
        raise ValueError("Precomputed SSIM support does not match this view")
    return cache


def _rgb(prediction, target, reference=None):
    if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise ValueError("Prediction and target must be RGB tensors")
    if (
        prediction.ndim != 3
        or prediction.shape[0] != 3
        or min(prediction.shape[1:]) < 1
    ):
        raise ValueError("RGB views must have shape 3 x H x W")
    if (
        target.shape != prediction.shape
        or target.dtype != prediction.dtype
        or target.device != prediction.device
    ):
        raise ValueError("Prediction and target shape, dtype and device must match")
    if not prediction.is_floating_point():
        raise ValueError("RGB tensors must be floating point")
    if reference is not None and (
        prediction.device != reference.device or prediction.dtype != reference.dtype
    ):
        raise ValueError("All views in a source group must share device and dtype")
    if not bool(torch.isfinite(prediction).all()) or not bool(
        torch.isfinite(target).all()
    ):
        raise ValueError("RGB views must be finite")


def _default_map(prediction, target):
    from fused_ssim import FusedSSIMMap

    return FusedSSIMMap.apply(0.01**2, 0.03**2, prediction, target, "same", True)


def _default_native_ssim(prediction, target):
    from fused_ssim import fused_ssim

    return fused_ssim(prediction, target)


def _map(prediction, target, callback):
    value = callback(prediction[None], target[None])
    if not isinstance(value, torch.Tensor) or value.shape != (1, *prediction.shape):
        raise ValueError("ssim_map must return the complete 1 x 3 x H x W map")
    if value.device != prediction.device or value.dtype != prediction.dtype:
        raise ValueError("ssim_map must preserve RGB dtype and device")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("ssim_map contains a nonfinite value")
    return value[0]


def paired_loss(
    parent_prediction,
    parent_target,
    face_mask,
    children,
    lambda_dssim,
    *,
    ssim_map=None,
    native_fused_ssim=None,
    native_l1=None,
    parent_support=None,
):
    """Return ``loss``, ``l1``, ``ssim`` tensors and JSON-compatible ``stats``.

    Children are dictionaries containing prediction/target/mask/physical_edges.
    Their masks must already have exclusive ownership of overlapping ROIs.
    Optional support_cache entries and parent_support avoid repeated pooling.
    L1 pools native outside-mask RGB samples and child inside-mask RGB samples.
    SSIM pools only safe-center RGB entries. If no safe centers exist, the SSIM
    loss is zero and L1 still supervises every boundary pixel in its domain.
    """
    _rgb(parent_prediction, parent_target)
    _mask(face_mask, parent_prediction.shape[1:], parent_prediction.device)
    if (
        isinstance(lambda_dssim, bool)
        or not isinstance(lambda_dssim, (int, float))
        or not math.isfinite(lambda_dssim)
        or not 0 <= lambda_dssim <= 1
    ):
        raise ValueError("lambda_dssim must be finite and between zero and one")
    if not isinstance(children, (list, tuple)):
        raise ValueError("children must be an ordered list or tuple")
    prepared = []
    for child in children:
        if not isinstance(child, dict):
            raise ValueError("Each child must be a view dictionary")
        prediction, target = child["prediction"], child["target"]
        _rgb(prediction, target, parent_prediction)
        mask = _mask(child["mask"], prediction.shape[1:], prediction.device)
        edges = _edges(child["physical_edges"])
        prepared.append((prediction, target, mask, edges, child.get("support_cache")))
    face_pixels = int(face_mask.sum())
    if face_pixels == 0:
        if any(bool(mask.any()) for _, _, mask, _, _ in prepared):
            raise ValueError(
                "Child supervision cannot exist without a parent face mask"
            )
        l1_function = (
            native_l1 if native_l1 is not None else lambda a, b: (a - b).abs().mean()
        )
        ssim_function = (
            native_fused_ssim if native_fused_ssim is not None else _default_native_ssim
        )
        l1 = l1_function(parent_prediction, parent_target)
        ssim = ssim_function(parent_prediction[None], parent_target[None])
        loss = (1 - lambda_dssim) * l1 + lambda_dssim * (1 - ssim)
        pixels = face_mask.numel()
        return dict(
            loss=loss,
            l1=l1,
            ssim=ssim,
            stats=dict(
                native_fallback=True,
                parent_outside_pixels=pixels,
                parent_face_pixels=0,
                child_inside_pixels=[],
                child_view_count=len(prepared),
                l1_rgb_samples=3 * pixels,
                parent_ssim_centers=pixels,
                child_ssim_centers=[],
                ssim_rgb_centers=3 * pixels,
                ssim_empty_support=False,
                lambda_dssim=float(lambda_dssim),
                reduction="exact native full-image fallback",
            ),
        )
    parent_domain = _resolve_support(~face_mask, _ALL_PHYSICAL, parent_support)
    domains = [parent_domain]
    for _, _, mask, edges, cache in prepared:
        domains.append(_resolve_support(mask, edges, cache))
    if not sum(domain.pixels for domain in domains[1:]):
        raise ValueError("A nonempty face mask requires child supervision")
    predictions = [parent_prediction] + [view[0] for view in prepared]
    targets = [parent_target] + [view[1] for view in prepared]
    l1_parts = [
        ((prediction - target).abs() * domain.support[None]).sum()
        for prediction, target, domain in zip(predictions, targets, domains)
    ]
    rgb_samples = 3 * sum(domain.pixels for domain in domains)
    l1_numerator = sum(l1_parts)
    l1 = l1_numerator / rgb_samples
    center_samples = 3 * sum(domain.center_pixels for domain in domains)
    callback = ssim_map if ssim_map is not None else _default_map
    if center_samples:
        map_parts = []
        for prediction, target, domain in zip(predictions, targets, domains):
            if domain.center_pixels:
                value = _map(prediction, target, callback)
                map_parts.append((value * domain.centers[None]).sum())
        ssim = sum(map_parts) / center_samples
    else:
        ssim = 1 + sum(prediction.sum() * 0 for prediction in predictions)
    loss = (1 - lambda_dssim) * l1 + lambda_dssim * (1 - ssim)
    stats = dict(
        native_fallback=False,
        parent_face_pixels=face_pixels,
        parent_outside_pixels=parent_domain.pixels,
        child_inside_pixels=[domain.pixels for domain in domains[1:]],
        child_view_count=len(prepared),
        l1_rgb_samples=rgb_samples,
        parent_ssim_centers=parent_domain.center_pixels,
        child_ssim_centers=[domain.center_pixels for domain in domains[1:]],
        ssim_rgb_centers=center_samples,
        ssim_empty_support=not bool(center_samples),
        parent_l1_sum=float(l1_parts[0].detach()),
        child_l1_sums=[float(value.detach()) for value in l1_parts[1:]],
        lambda_dssim=float(lambda_dssim),
        reduction="pixel-pooled L1; RGB-weighted safe-center SSIM",
        SSIM_window=11,
        no_cross_domain_SSIM_windows=True,
        physical_padding="native SSIM zero RGB outside true image only",
        boundary_supervision="L1 covers support including mask boundary",
        independent_child_sampling=False,
    )
    return dict(loss=loss, l1=l1, ssim=ssim, stats=stats)
