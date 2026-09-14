import numpy as np
import pytest
import torch

from scene_understanding.fusion import (
    REGION_FAR,
    REGION_MIDDLE,
    REGION_NEAR,
    FusionResult,
    class_depth_stats,
    create_mask,
    depth_regions,
    fuse,
    validate_inputs,
)

SEG = np.array([[0, 13, 13], [0, 0, 13], [11, 11, 11]], dtype=np.int64)
DEPTH = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)


@pytest.fixture
def fr():
    return fuse(SEG, DEPTH)


# 1. shape mismatch
def test_segmentation_depth_shape_mismatch():
    with pytest.raises(ValueError, match="identical spatial dimensions"):
        fuse(np.zeros((2, 3), dtype=np.int64), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="identical spatial dimensions"):
        validate_inputs(np.zeros((3, 3), dtype=np.int64), np.zeros((2, 2)))


def test_non_2d_inputs_rejected():
    with pytest.raises(ValueError, match="2D"):
        fuse(np.zeros((3,), dtype=np.int64), np.zeros((3, 3)))


# 2. empty masks
def test_all_void_segmentation_raises():
    with pytest.raises(ValueError, match="non-void"):
        fuse(np.full((3, 3), 255, dtype=np.int64), np.ones((3, 3)))


def test_all_nonfinite_depth_raises():
    with pytest.raises(ValueError, match="no finite"):
        fuse(SEG, np.full((3, 3), np.nan))


# 3. class pixel count
def test_class_pixel_count():
    assert class_depth_stats(SEG, DEPTH, 13)["pixel_count"] == 3


def test_fuse_per_class_pixel_count(fr):
    assert fr.per_class[13]["pixel_count"] == 3


# 4. class pixel ratio
def test_class_pixel_ratio():
    assert class_depth_stats(SEG, DEPTH, 13)["pixel_ratio"] == pytest.approx(1 / 3)


# 5. class mean depth
def test_class_mean_depth():
    assert class_depth_stats(SEG, DEPTH, 13)["mean_depth"] == pytest.approx(11 / 3)


# 6. class median depth
def test_class_median_depth():
    assert class_depth_stats(SEG, DEPTH, 13)["median_depth"] == 3.0


# 7. class min/max depth
def test_class_min_max_depth():
    stats = class_depth_stats(SEG, DEPTH, 13)
    assert stats["min_depth"] == 2.0
    assert stats["max_depth"] == 6.0


def test_road_stats():
    stats = class_depth_stats(SEG, DEPTH, 0)
    assert stats["pixel_count"] == 3
    assert stats["mean_depth"] == pytest.approx(10 / 3)
    assert stats["median_depth"] == 4.0


# 8. missing semantic class
def test_missing_class_returns_none():
    assert class_depth_stats(SEG, DEPTH, 1) is None


def test_missing_class_absent_from_fusion(fr):
    assert 1 not in fr.per_class


# 9. near/middle/far categorization
def test_depth_regions_tercile_layout(fr):
    counts = {name: fr.region_summary[name]["pixel_count"] for name in ("far", "middle", "near")}
    assert counts == {"far": 3, "middle": 3, "near": 3}
    for name in ("far", "middle", "near"):
        assert fr.region_summary[name]["pixel_ratio"] == pytest.approx(1 / 3)


def test_region_map_codes():
    region_map, _ = depth_regions(DEPTH)
    assert set(np.unique(region_map)) == {REGION_FAR, REGION_MIDDLE, REGION_NEAR}
    assert (region_map == REGION_NEAR).sum() == 3
    assert (region_map == REGION_FAR).sum() == 3


# 10. inverse-depth convention
def test_inverse_depth_convention_near_has_larger_values(fr):
    near = DEPTH[fr.region_map == REGION_NEAR]
    far = DEPTH[fr.region_map == REGION_FAR]
    assert near.mean() > far.mean()
    assert fr.thresholds[1] > fr.thresholds[0]
    assert near.min() >= fr.thresholds[1]
    assert far.max() <= fr.thresholds[0]


# create_mask helper
def test_create_mask():
    mask = create_mask(SEG, 13)
    assert mask.sum() == 3
    assert mask[0, 1] and mask[0, 2] and mask[1, 2]


def test_create_mask_bad_class_id():
    with pytest.raises(ValueError, match="class_id"):
        create_mask(SEG, -1)


# quantile validation
def test_invalid_quantiles_rejected():
    with pytest.raises(ValueError, match="quantiles"):
        depth_regions(DEPTH, quantiles=(0.9, 0.2))
    with pytest.raises(ValueError, match="quantiles"):
        depth_regions(DEPTH, quantiles=(0.2,))


# 14. multiple classes
def test_multiple_classes_present(fr):
    assert set(fr.per_class) == {0, 11, 13}


# 18. same-image spatial contract is enforced
def test_same_image_shape_contract():
    with pytest.raises(ValueError, match="identical spatial dimensions"):
        fuse(np.zeros((4, 4), dtype=np.int64), DEPTH)


# 19. invalid NaN depth is masked, not fatal
def test_nan_depth_pixels_are_excluded():
    dep = DEPTH.copy()
    dep[0, 1] = np.nan  # a 'car' pixel (its value was 2.0)
    fr = fuse(SEG, dep)
    assert fr.per_class[13]["pixel_count"] == 2
    assert fr.analyzed_mask.sum() == 8
    assert np.isnan(dep).sum() == 1
    assert fr.per_class[13]["mean_depth"] == pytest.approx((3 + 6) / 2)
    assert "pixel_count" in fr.per_class[0]


# 20. invalid segmentation values
def test_negative_class_ids_rejected():
    seg = SEG.copy()
    seg[0, 0] = -5
    with pytest.raises(ValueError, match="negative"):
        fuse(seg, DEPTH)


def test_non_integer_segmentation_rejected():
    with pytest.raises(ValueError, match="segmentation mask"):
        fuse(SEG.astype(np.float32), DEPTH)


def test_boolean_depth_rejected():
    with pytest.raises(ValueError, match="depth"):
        fuse(SEG, np.ones((3, 3), dtype=bool))


# torch tensor input support
def test_torch_inputs_accepted():
    fr = fuse(torch.from_numpy(SEG), torch.from_numpy(DEPTH))
    assert fr.per_class[13]["pixel_count"] == 3
    assert isinstance(fr, FusionResult)


# deterministic within fuse
def test_fuse_is_deterministic():
    a = fuse(SEG, DEPTH)
    b = fuse(SEG, DEPTH)
    assert np.array_equal(a.region_map, b.region_map)
    assert a.per_class == b.per_class
    assert a.region_summary == b.region_summary