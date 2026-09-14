from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from preprocessing.errors import DatasetError
from preprocessing.kitti import KittiDepthDataset


def _write_rgb(path, size=(16, 8)):
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.full((size[1], size[0], 3), 120, dtype=np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _write_depth(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint16)).save(path)


def _make_frame(root, stem, depth_value_mm, img_size=(16, 8), depth_array=None):
    _write_rgb(root / "images" / f"{stem}.png", size=img_size)
    if depth_array is None:
        depth = np.full((8, 16), depth_value_mm, dtype=np.uint16)
    else:
        depth = depth_array
    _write_depth(root / "depth" / f"{stem}.png", depth)


@pytest.fixture
def depth_pairs(tmp_path):
    _make_frame(tmp_path, "0000000001", 5000)   # 5.0 m
    _make_frame(tmp_path, "0000000002", 12345)  # 12.345 m
    _make_frame(tmp_path, "0000000003", 0)      # invalid
    return tmp_path


def test_mm_pct_is_converted_to_metres(depth_pairs):
    ds = KittiDepthDataset(depth_pairs, image_size=None, scale_mm=1000.0)
    by_stem = {}
    for idx in range(len(ds)):
        sample = ds[idx]
        stem = sample["depth_path"].split("/")[-1][: -len(".png")]
        by_stem[stem] = sample
    assert by_stem["0000000001"]["depth"][0, 0] == pytest.approx(5.0)
    assert by_stem["0000000002"]["depth"][0, 0] == pytest.approx(12.345)


def test_zero_depth_is_marked_invalid(depth_pairs):
    ds = KittiDepthDataset(depth_pairs, image_size=None)
    by_stem = {}
    for idx in range(len(ds)):
        sample = ds[idx]
        stem = sample["depth_path"].split("/")[-1][: -len(".png")]
        by_stem[stem] = sample
    assert bool(by_stem["0000000001"]["valid_mask"][0, 0]) is True
    assert bool(by_stem["0000000003"]["valid_mask"][0, 0]) is False
    assert float(by_stem["0000000003"]["depth"][0, 0]) == 0.0


def test_depth_cap_flags_far_pixels_invalid(tmp_path):
    depth = np.zeros((8, 16), dtype=np.uint16)
    depth[:, :8] = 60_000  # 60 m
    depth[:, 8:] = 10_000  # 10 m
    _make_frame(tmp_path, "0000000001", 0, depth_array=depth)
    ds = KittiDepthDataset(tmp_path, image_size=None, depth_cap_m=50.0)
    sample = ds[0]
    assert sample["valid_mask"][0, 0].item() is False  # > cap -> invalid
    assert sample["depth"][0, 8].item() == pytest.approx(10.0)
    assert sample["valid_mask"][0, 8].item() is True


def test_pairs_matched_by_stem_not_by_order(tmp_path):
    _make_frame(tmp_path, "0000000001", 1000)
    _make_frame(tmp_path, "0000000002", 2000)
    _make_frame(tmp_path, "0000000003", 3000)

    stray = np.full((8, 16), 9000, dtype=np.uint16)
    _write_depth(tmp_path / "depth" / "0000000099.png", stray)
    ds = KittiDepthDataset(tmp_path, image_size=None)
    stems = {Path(s["depth_path"]).name for s in ds}
    assert stems == {"0000000001.png", "0000000002.png", "0000000003.png"}


def test_images_without_depth_are_dropped_when_skipping(tmp_path):
    _make_frame(tmp_path, "0000000001", 1000)
    _write_rgb(tmp_path / "images" / "0000000002.png")
    ds = KittiDepthDataset(tmp_path, image_size=None, missing_policy="skip")
    assert len(ds) == 1
    assert "0000000001" in ds[0]["depth_path"]


def test_images_without_depth_raise_by_default(tmp_path):
    _make_frame(tmp_path, "0000000001", 1000)
    _write_rgb(tmp_path / "images" / "0000000002.png")
    with pytest.raises(DatasetError, match="no depth pair"):
        KittiDepthDataset(tmp_path, image_size=None, missing_policy="error")


def test_nested_subdirectories_are_supported(tmp_path):
    _write_rgb(tmp_path / "images" / "drive_0001" / "0000000001.png")
    depth = np.full((8, 16), 4000, dtype=np.uint16)
    _write_depth(tmp_path / "depth" / "drive_0001" / "0000000001.png", depth)
    ds = KittiDepthDataset(tmp_path, image_size=None)
    assert len(ds) == 1
    assert ds[0]["depth"][0, 0].item() == pytest.approx(4.0)


def test_split_file_selects_entries_in_order(tmp_path):
    _make_frame(tmp_path, "0000000001", 1000)
    _make_frame(tmp_path, "0000000002", 2000)
    _make_frame(tmp_path, "0000000003", 3000)
    (tmp_path / "splits").mkdir(parents=True)
    (tmp_path / "splits" / "val.txt").write_text(
        "# comment line\n0000000003\n0000000001\n", encoding="utf-8"
    )
    ds = KittiDepthDataset(tmp_path, split="val", image_size=None)
    names = [Path(s["depth_path"]).name for s in ds]
    assert names == ["0000000003.png", "0000000001.png"]


def test_split_entry_can_be_relative_path(tmp_path):
    _make_frame(tmp_path, "0000000001", 1000)
    _make_frame(tmp_path, "0000000002", 2000)
    (tmp_path / "splits").mkdir(parents=True)
    (tmp_path / "splits" / "val.txt").write_text(
        "images/0000000002.png\n", encoding="utf-8"
    )
    ds = KittiDepthDataset(tmp_path, split="val", image_size=None)
    assert len(ds) == 1
    assert "0000000002" in ds[0]["depth_path"]


def test_split_missing_entry_raises_by_default(tmp_path):
    _make_frame(tmp_path, "0000000001", 1000)
    (tmp_path / "splits").mkdir(parents=True)
    (tmp_path / "splits" / "val.txt").write_text("0000000001\nmissing_frame\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="missing its"):
        KittiDepthDataset(tmp_path, split="val", image_size=None)


def test_split_missing_entry_skips_when_configured(tmp_path):
    _make_frame(tmp_path, "0000000001", 1000)
    (tmp_path / "splits").mkdir(parents=True)
    (tmp_path / "splits" / "val.txt").write_text("0000000001\nmissing_frame\n", encoding="utf-8")
    ds = KittiDepthDataset(tmp_path, split="val", image_size=None, missing_policy="skip")
    assert len(ds) == 1


def test_max_samples_truncates(depth_pairs):
    ds = KittiDepthDataset(depth_pairs, image_size=None, max_samples=2)
    assert len(ds) == 2


def test_invalid_max_samples_raises(depth_pairs):
    with pytest.raises(DatasetError):
        KittiDepthDataset(depth_pairs, max_samples=-1)
    with pytest.raises(DatasetError):
        KittiDepthDataset(depth_pairs, max_samples="10")


def test_no_pairs_raises(tmp_path):
    _make_frame(tmp_path, "0000000001", 1000)
    (tmp_path / "splits").mkdir(parents=True)
    (tmp_path / "splits" / "val.txt").write_text("", encoding="utf-8")
    with pytest.raises(DatasetError, match="No KITTI depth pairs"):
        KittiDepthDataset(tmp_path, split="val", image_size=None)


def test_empty_images_dir_raises(tmp_path):
    _make_frame(tmp_path, "0000000001", 1000)
    (tmp_path / "images" / "0000000001.png").unlink()
    with pytest.raises(DatasetError, match="No images found"):
        KittiDepthDataset(tmp_path, image_size=None)


def test_depth_resize_recomputes_valid_mask(tmp_path):
    depth = np.zeros((4, 8), dtype=np.uint16)
    depth[:, 4:] = 40_000  # 40 m on the right half
    _write_rgb(tmp_path / "images" / "0000000001.png", size=(8, 4))
    _write_depth(tmp_path / "depth" / "0000000001.png", depth)
    ds = KittiDepthDataset(tmp_path, image_size=(8, 16))
    sample = ds[0]
    assert sample["depth"].shape == (8, 16)
    assert sample["valid_mask"].shape == (8, 16)
    assert bool(sample["valid_mask"][0, 0]) is False
    assert bool(sample["valid_mask"][0, 15]) is True
    assert float(sample["depth"][0, 0]) == 0.0
    assert sample["depth"][0, 15].item() == pytest.approx(40.0)


def test_scale_mm_is_configurable(tmp_path):
    _make_frame(tmp_path, "0000000001", 25_000)
    ds = KittiDepthDataset(tmp_path, image_size=None, scale_mm=100.0)
    assert ds[0]["depth"][0, 0].item() == pytest.approx(250.0)


def test_from_config_builds_dataset(tmp_path):
    from utils.config import load_config

    _make_frame(tmp_path, "0000000001", 5000)
    cfg = load_config("pipeline", overrides={"data": {"kitti": {"root": str(tmp_path)}}})
    ds = KittiDepthDataset.from_config(cfg)
    assert len(ds) == 1