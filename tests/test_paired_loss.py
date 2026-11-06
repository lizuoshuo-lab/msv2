"""CPU loss/gradient and real 11x11 SSIM-domain regressions."""

import json

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from msv2.paired_loss import paired_loss, prepare_ssim_support, safe_ssim_centers  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def two_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def rgb(height, width, value=0.2, *, grad=True):
    return torch.full(
        (3, height, width), value, dtype=torch.float64, requires_grad=grad
    )


def child(prediction, mask, *, target=None, edges=(False, False, False, False)):
    return dict(
        prediction=prediction,
        target=torch.zeros_like(prediction) if target is None else target,
        mask=mask,
        physical_edges=edges,
    )


def ssim_map(a, b):
    # These are the exact pinned fused-ssim Gaussian coefficients. This CPU
    # reference tests domains/autograd without importing its CUDA extension.
    g = a.new_tensor(
        [
            0.001028380123898387,
            0.0075987582094967365,
            0.036000773310661316,
            0.10936068743467331,
            0.21300552785396576,
            0.26601171493530273,
            0.21300552785396576,
            0.10936068743467331,
            0.036000773310661316,
            0.0075987582094967365,
            0.001028380123898387,
        ]
    )
    kernel = torch.outer(g, g)[None, None].repeat(3, 1, 1, 1)

    def blur(value):
        return F.conv2d(value, kernel, padding=5, groups=3)

    mu_a, mu_b = blur(a), blur(b)
    var_a, var_b = blur(a * a) - mu_a.square(), blur(b * b) - mu_b.square()
    covariance = blur(a * b) - mu_a * mu_b
    return ((2 * mu_a * mu_b + 0.01**2) * (2 * covariance + 0.03**2)) / (
        (mu_a.square() + mu_b.square() + 0.01**2) * (var_a + var_b + 0.03**2)
    )


def native_ssim(a, b):
    return ssim_map(a, b).mean()


def native_l1(a, b):
    return (a - b).abs().mean()


def test_no_face_is_exact_native_value_gradient_and_callbacks():
    torch.manual_seed(9)
    prediction = torch.rand((3, 17, 23), dtype=torch.float64, requires_grad=True)
    target = torch.rand_like(prediction)
    calls = []

    def old_l1(a, b):
        calls.append(("l1", a is prediction, a.shape))
        return native_l1(a, b)

    def old_ssim(a, b):
        calls.append(("ssim", a.shape))
        return native_ssim(a, b)

    def forbidden_map(*args):
        raise AssertionError("No-face must use the original scalar SSIM")

    result = paired_loss(
        prediction,
        target,
        torch.zeros((17, 23), dtype=torch.bool),
        [],
        0.2,
        native_l1=old_l1,
        native_fused_ssim=old_ssim,
        ssim_map=forbidden_map,
    )
    actual_grad = torch.autograd.grad(result["loss"], prediction)[0]
    copied = prediction.detach().clone().requires_grad_(True)
    expected = 0.8 * native_l1(copied, target) + 0.2 * (
        1 - native_ssim(copied[None], target[None])
    )
    expected_grad = torch.autograd.grad(expected, copied)[0]
    assert torch.equal(result["loss"], expected)
    assert torch.equal(actual_grad, expected_grad)
    assert calls == [("l1", True, (3, 17, 23)), ("ssim", (1, 3, 17, 23))]
    assert result["stats"]["native_fallback"]


def test_l1_pools_actual_rgb_samples_and_excludes_both_wrong_domains():
    parent, high = rgb(2, 2, 2.0), rgb(2, 3, 4.0)
    face = torch.tensor([[False, True], [False, False]])
    inside = torch.tensor([[True, False, True], [False, False, False]])
    result = paired_loss(
        parent,
        torch.zeros_like(parent),
        face,
        [child(high, inside)],
        0.0,
        ssim_map=ssim_map,
    )
    assert result["l1"].item() == pytest.approx((3 * 2 + 2 * 4) / (3 + 2))
    assert result["stats"]["l1_rgb_samples"] == 15
    result["loss"].backward()
    assert torch.equal(parent.grad[:, face], torch.zeros((3, 1), dtype=parent.dtype))
    assert torch.all(parent.grad[:, ~face] == 1 / 15)
    assert torch.all(high.grad[:, inside] == 1 / 15)
    assert torch.equal(high.grad[:, ~inside], torch.zeros((3, 4), dtype=parent.dtype))


def test_multiple_children_are_pixel_weighted_not_view_means():
    parent = rgb(15, 17, 0.1)
    face = torch.zeros((15, 17), dtype=torch.bool)
    face[6:9, 6:9] = True
    a, b = rgb(17, 19, 0.3), rgb(23, 25, 0.7)
    ma, mb = (
        torch.ones((17, 19), dtype=torch.bool),
        torch.ones((23, 25), dtype=torch.bool),
    )
    result = paired_loss(
        parent,
        torch.zeros_like(parent),
        face,
        [child(a, ma), child(b, mb)],
        0.2,
        ssim_map=ssim_map,
    )
    expected = ((255 - 9) * 0.1 + 323 * 0.3 + 575 * 0.7) / (255 - 9 + 323 + 575)
    assert result["l1"].item() == pytest.approx(expected)
    domains = [
        safe_ssim_centers(~face),
        safe_ssim_centers(ma, (False,) * 4),
        safe_ssim_centers(mb, (False,) * 4),
    ]
    maps = [ssim_map(p[None], torch.zeros_like(p)[None])[0] for p in (parent, a, b)]
    expected_ssim = sum((m * d[None]).sum() for m, d in zip(maps, domains)) / (
        3 * sum(int(d.sum()) for d in domains)
    )
    assert torch.equal(result["ssim"], expected_ssim)
    assert result["stats"]["child_inside_pixels"] == [323, 575]
    json.dumps(result["stats"])


def test_safe_outside_removes_entire_11_window_near_face():
    support = torch.ones((25, 29), dtype=torch.bool)
    support[12, 14] = False
    centers = safe_ssim_centers(support)
    expected = torch.ones_like(support)
    expected[7:18, 9:20] = False
    assert torch.equal(centers, expected)


def test_physical_and_artificial_edge_padding_are_different():
    support = torch.ones((11, 11), dtype=torch.bool)
    assert int(safe_ssim_centers(support).sum()) == 121
    assert int(safe_ssim_centers(support, (False,) * 4).sum()) == 1
    left = safe_ssim_centers(support, (True, False, False, False))
    assert int(left.sum()) == 6 and left[5, :6].all()


def test_roi_touching_real_image_edge_has_same_centers_as_full_image():
    support = torch.zeros((40, 44), dtype=torch.bool)
    support[10:30, :15] = True
    full = safe_ssim_centers(support)
    local = safe_ssim_centers(support[8:34, :24], (True, False, False, False))
    assert torch.equal(local, full[8:34, :24])


def boundary_case(parent_value=0.2, child_value=0.4):
    parent, high = rgb(40, 44, parent_value), rgb(64, 64, child_value)
    face = torch.zeros((40, 44), dtype=torch.bool)
    face[12:28, 14:30] = True
    mask = torch.zeros((64, 64), dtype=torch.bool)
    mask[12:52, 12:52] = True
    return parent, high, face, mask


def test_ssim_never_reads_or_differentiates_across_domain_boundary():
    parent, high, face, mask = boundary_case()
    gt, high_gt = torch.full_like(parent, 0.31), torch.full_like(high, 0.23)
    result = paired_loss(
        parent, gt, face, [child(high, mask, target=high_gt)], 0.2, ssim_map=ssim_map
    )
    result["loss"].backward()
    assert torch.count_nonzero(parent.grad[:, face]) == 0
    assert torch.count_nonzero(high.grad[:, ~mask]) == 0
    changed_parent = parent.detach().clone()
    changed_parent[:, face] = 0.99
    changed_high = high.detach().clone()
    changed_high[:, ~mask] = 0.01
    changed = paired_loss(
        changed_parent,
        gt,
        face,
        [child(changed_high, mask, target=high_gt)],
        0.2,
        ssim_map=ssim_map,
    )
    assert torch.equal(changed["loss"], result["loss"])
    # Masking only SSIM centers would still mix the opposite domain through
    # its windows; demonstrate that the unsafe reference actually differs.
    naive = (ssim_map(parent[None], gt[None])[0] * (~face)[None]).sum()
    naive_changed = (ssim_map(changed_parent[None], gt[None])[0] * (~face)[None]).sum()
    assert abs((naive - naive_changed).item()) > 0.1


def test_mask_boundary_keeps_l1_when_ssim_center_is_excluded():
    parent, high, face, mask = boundary_case()
    result = paired_loss(
        parent,
        torch.zeros_like(parent),
        face,
        [child(high, mask)],
        0.2,
        ssim_map=ssim_map,
    )
    result["loss"].backward()
    assert not safe_ssim_centers(mask, (False,) * 4)[12, 12]
    assert torch.all(high.grad[:, 12, 12] > 0)
    assert not safe_ssim_centers(~face)[11, 14]
    assert torch.all(parent.grad[:, 11, 14] > 0)


def test_cached_and_uncached_losses_and_gradients_are_identical():
    parent, high, face, mask = boundary_case()
    p_cache = prepare_ssim_support(~face)
    c_cache = prepare_ssim_support(mask, (False,) * 4).to("cpu")
    uncached = paired_loss(
        parent,
        torch.zeros_like(parent),
        face,
        [child(high, mask)],
        0.2,
        ssim_map=ssim_map,
    )
    view = child(high, mask)
    view["support_cache"] = c_cache
    cached = paired_loss(
        parent,
        torch.zeros_like(parent),
        face,
        [view],
        0.2,
        ssim_map=ssim_map,
        parent_support=p_cache,
    )
    assert torch.equal(uncached["loss"], cached["loss"])
    a = torch.autograd.grad(uncached["loss"], (parent, high))
    b = torch.autograd.grad(cached["loss"], (parent, high))
    assert all(torch.equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("which", ["support", "centers"])
def test_modified_cache_is_rejected(which):
    parent, high, face, mask = boundary_case()
    cache = prepare_ssim_support(~face)
    getattr(cache, which)[0, 0] = False
    with pytest.raises(ValueError, match="modified in place"):
        paired_loss(
            parent,
            torch.zeros_like(parent),
            face,
            [child(high, mask)],
            0.2,
            ssim_map=ssim_map,
            parent_support=cache,
        )


def test_cache_for_wrong_domain_is_rejected():
    parent, high, face, mask = boundary_case()
    with pytest.raises(ValueError, match="does not match"):
        paired_loss(
            parent,
            torch.zeros_like(parent),
            face,
            [child(high, mask)],
            0.2,
            ssim_map=ssim_map,
            parent_support=prepare_ssim_support(face),
        )


def test_cache_built_in_inference_mode_remains_version_checked():
    with torch.inference_mode():
        cache = prepare_ssim_support(torch.ones((13, 17), dtype=torch.bool))
    assert cache.support._version >= 0 and cache.centers._version >= 0
    cache._validate()
    cache.centers[0, 0] = False
    with pytest.raises(ValueError, match="modified in place"):
        cache.to("cpu")


def test_empty_safe_ssim_domain_is_zero_ssim_loss_without_nan():
    parent, high = rgb(2, 2), rgb(2, 2, 0.5)
    face, mask = (
        torch.ones((2, 2), dtype=torch.bool),
        torch.ones((2, 2), dtype=torch.bool),
    )

    def forbidden(*args):
        raise AssertionError("No SSIM window is supervised")

    result = paired_loss(
        parent,
        torch.zeros_like(parent),
        face,
        [child(high, mask)],
        0.2,
        ssim_map=forbidden,
    )
    assert result["ssim"].item() == 1 and result["loss"].item() == pytest.approx(
        0.8 * 0.5
    )
    assert result["stats"]["ssim_empty_support"]
    result["loss"].backward()
    assert torch.isfinite(high.grad).all() and not parent.grad.any()


def test_no_face_ignores_empty_child_without_reading_ssim_map():
    parent, high = rgb(17, 19), rgb(9, 11)
    result = paired_loss(
        parent,
        torch.zeros_like(parent),
        torch.zeros((17, 19), dtype=torch.bool),
        [child(high, torch.zeros((9, 11), dtype=torch.bool))],
        0.2,
        native_fused_ssim=native_ssim,
    )
    result["loss"].backward()
    assert high.grad is None and result["stats"]["native_fallback"]


@pytest.mark.parametrize("weight", [True, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_weight_is_rejected(weight):
    parent = rgb(13, 17)
    with pytest.raises(ValueError, match="lambda_dssim"):
        paired_loss(
            parent,
            torch.zeros_like(parent),
            torch.zeros((13, 17), dtype=torch.bool),
            [],
            weight,
        )


@pytest.mark.parametrize(
    "bad_mask", [torch.ones((13, 17)), torch.ones((13, 16), dtype=torch.bool)]
)
def test_invalid_native_mask_is_rejected(bad_mask):
    parent = rgb(13, 17)
    with pytest.raises(ValueError, match="mask|dimensions"):
        paired_loss(parent, torch.zeros_like(parent), bad_mask, [], 0.2)


def test_nonempty_face_without_child_supervision_is_rejected():
    parent = rgb(13, 17)
    with pytest.raises(ValueError, match="requires child"):
        paired_loss(
            parent,
            torch.zeros_like(parent),
            torch.ones((13, 17), dtype=torch.bool),
            [],
            0.2,
        )


def test_child_cannot_add_pixels_without_parent_face():
    parent, high = rgb(13, 17), rgb(13, 17)
    with pytest.raises(ValueError, match="without a parent"):
        paired_loss(
            parent,
            torch.zeros_like(parent),
            torch.zeros((13, 17), dtype=torch.bool),
            [child(high, torch.ones((13, 17), dtype=torch.bool))],
            0.2,
        )


@pytest.mark.parametrize("bad_edges", [(True, False), (1, True, True, True)])
def test_invalid_physical_edge_contract_is_rejected(bad_edges):
    with pytest.raises(ValueError, match="physical_edges"):
        safe_ssim_centers(torch.ones((17, 19), dtype=torch.bool), bad_edges)


def test_wrong_map_shape_or_nonfinite_map_is_rejected():
    parent, high, face, mask = boundary_case()
    for callback in (lambda a, b: a.mean(), lambda a, b: a * float("nan")):
        with pytest.raises(ValueError, match="map|nonfinite"):
            paired_loss(
                parent,
                torch.zeros_like(parent),
                face,
                [child(high, mask)],
                0.2,
                ssim_map=callback,
            )


def test_mixed_view_dtype_and_nonfinite_rgb_are_rejected():
    parent, high, face, mask = boundary_case()
    with pytest.raises(ValueError, match="device and dtype"):
        paired_loss(
            parent,
            torch.zeros_like(parent),
            face,
            [child(high.float(), mask)],
            0.2,
            ssim_map=ssim_map,
        )
    parent = parent.detach()
    parent[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        paired_loss(
            parent,
            torch.zeros_like(parent),
            face,
            [child(high, mask)],
            0.2,
            ssim_map=ssim_map,
        )
