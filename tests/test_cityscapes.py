from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from preprocessing.cityscapes import (
    CITYSCAPES_LABEL_ID_TO_TRAIN_ID,
    IGNORE_INDEX,
    NUM_CLASSES,
    CityscapesDataset,
    labels_to_train_ids,
)
from preprocessing.errors import DatasetError


def _write_png(path, array, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode=mode).save(path)


def _make_frame(root, stem, layout="flat", split="train", with_label=True, label_value=26):
    if layout == "official":
        img_dir, lbl_dir = root / "leftImg8bit", root / "gtFine"
    else:
        img_dir, lbl_dir = root / "images", root / "labels"
    img_area = img_dir / split if layout == "official" else img_dir
    lbl_area = lbl_dir / split if layout == "official" else lbl_dir

    rgb = np.full((8, 16, 3), 60, dtype=np.uint8)
    _write_png(img_area / f"{stem}_leftImg8bit.png", rgb, "RGB")
    if with_label:
        label = np.full((8, 16), label_value, dtype=np.uint8)
        _write_png(lbl_area / f"{stem}_gtFine_labelIds.png", label, "L")


@pytest.fixture
def flat_frames(tmp_path):
    _make_frame(tmp_path, "frame_0001", label_value=26)  # labelId car
    _make_frame(tmp_path, "frame_0002", label_value=7)   # labelId road
    _make_frame(tmp_path, "frame_0003", label_value=0)   # labelId unlabeled
    return tmp_path


def test_mapping_table_is_complete():
    assert NUM_CLASSES == 19
    assert IGNORE_INDEX == 255
    assert len(CITYSCAPES_LABEL_ID_TO_TRAIN_ID) == 19
    assert CITYSCAPES_LABEL_ID_TO_TRAIN_ID[7] == 0      # road
    assert CITYSCAPES_LABEL_ID_TO_TRAIN_ID[26] == 13    # car
    assert CITYSCAPES_LABEL_ID_TO_TRAIN_ID[33] == 18    # bicycle


def test_labels_remap_to_train_ids():
    labels = np.asarray([[0, 7], [18, 26]], dtype=np.int64)  # labelIds
    remapped = labels_to_train_ids(labels)
    assert remapped.tolist() == [[255, 0], [255, 13]]


def test_loads_all_samples_and_maps_train_ids(flat_frames):
    ds = CityscapesDataset(flat_frames, image_size=None, max_samples=None)
    assert len(ds) == 3

    by_value = {}
    for idx in range(len(ds)):
        sample = ds[idx]
        assert tuple(sample["image"].shape) == (3, 8, 16)
        assert tuple(sample["label"].shape) == (8, 16)
        by_value[int(sample["label"][0, 0])] = sample["label_path"]

    assert by_value[13] is not None   # car
    assert by_value[0] is not None    # road
    assert by_value[255] is not None  # unlabeled -> ignore


def test_max_samples_truncates(flat_frames):
    ds = CityscapesDataset(flat_frames, image_size=None, max_samples=2)
    assert len(ds) == 2


def test_max_samples_zero_allows_empty_loader(flat_frames):
    assert len(CityscapesDataset(flat_frames, image_size=None, max_samples=0)) == 0


def test_invalid_max_samples_raises(flat_frames):
    with pytest.raises(DatasetError):
        CityscapesDataset(flat_frames, max_samples=-1)
    with pytest.raises(DatasetError):
        CityscapesDataset(flat_frames, max_samples="10")


def test_missing_label_raises_by_default(tmp_path):
    _make_frame(tmp_path, "frame_0001", with_label=True)
    _make_frame(tmp_path, "frame_0002", with_label=False)
    with pytest.raises(DatasetError, match="no matching label"):
        CityscapesDataset(tmp_path, image_size=None)


def test_missing_label_skipped_when_configured(tmp_path):
    _make_frame(tmp_path, "frame_0001", with_label=True)
    _make_frame(tmp_path, "frame_0002", with_label=False)
    ds = CityscapesDataset(tmp_path, image_size=None, missing_policy="skip")
    assert len(ds) == 1
    assert "frame_0001" in ds[0]["image_path"]


def test_labels_optional_when_not_required(tmp_path):
    _make_frame(tmp_path, "frame_0001", with_label=True)
    _make_frame(tmp_path, "frame_0002", with_label=False)
    ds = CityscapesDataset(
        tmp_path,
        image_size=None,
        require_labels=False,
        missing_policy="error",
    )
    assert len(ds) == 2
    label_paths = [s["label_path"] for s in ds]
    present = [p for p in label_paths if p is not None]
    assert len(present) == 1
    assert "frame_0001" in present[0]
    assert all(s["label"] is None for s in ds if s["label_path"] is None)


def test_pairs_are_matched_by_stem_not_order(tmp_path):
    _make_frame(tmp_path, "frame_b", label_value=11)  # labelId building
    _make_frame(tmp_path, "frame_a", label_value=8)   # labelId sidewalk
    labels = sorted((tmp_path / "labels").rglob("*_gtFine_labelIds.png"))
    images = sorted((tmp_path / "images").rglob("*_leftImg8bit.png"))
    assert images[0].name != labels[0].name  # order differs on disk

    ds = CityscapesDataset(tmp_path, image_size=None)
    pairs = {
        Path(s["image_path"]).name: Path(s["label_path"]).name for s in ds
    }
    assert pairs["frame_a_leftImg8bit.png"] == "frame_a_gtFine_labelIds.png"
    assert pairs["frame_b_leftImg8bit.png"] == "frame_b_gtFine_labelIds.png"


def test_supports_official_split_layout(tmp_path):
    for split in ("train", "val"):
        _make_frame(tmp_path, "frame_0001", layout="official", split=split, label_value=26)
    train = CityscapesDataset(tmp_path, split="train", image_size=None)
    val = CityscapesDataset(tmp_path, split="val", image_size=None)
    assert len(train) == 1
    assert len(val) == 1
    assert "train" in train[0]["image_path"]
    assert "val" in val[0]["image_path"]


def test_ignore_label_suffix_files_like_instance_ids(tmp_path):
    img_area = tmp_path / "images"
    lbl_area = tmp_path / "labels"
    img_area.mkdir(parents=True)
    lbl_area.mkdir(parents=True)

    _make_frame(tmp_path, "frame_0001", label_value=21)
    rb = np.full((8, 16), 5, dtype=np.uint8)
    _write_png(lbl_area / "frame_0001_gtFine_instanceIds.png", rb, "L")

    ds = CityscapesDataset(tmp_path, image_size=None)
    assert len(ds) == 1
    assert ds[0]["label_path"].endswith("_gtFine_labelIds.png")


def test_no_images_raises(tmp_path):
    (tmp_path / "images").mkdir(parents=True)
    (tmp_path / "labels").mkdir(parents=True)
    with pytest.raises(DatasetError, match="No Cityscapes pairs"):
        CityscapesDataset(tmp_path)


def test_images_are_normalised_with_imagenet_stats(flat_frames):
    ds = CityscapesDataset(flat_frames, image_size=None)
    sample = ds[0]
    expected = (60.0 / 255.0 - 0.485) / 0.229  # channel 0, source pixel 60
    assert sample["image"][0, 0, 0] == pytest.approx(expected)


def test_ambiguous_labels_raise(tmp_path):
    _make_frame(tmp_path, "frame_0001", label_value=26)
    img_b = tmp_path / "images" / "city_b"
    lbl_b = tmp_path / "labels" / "city_b"
    img_b.mkdir(parents=True, exist_ok=True)
    lbl_b.mkdir(parents=True, exist_ok=True)
    rgb = np.full((8, 16, 3), 60, dtype=np.uint8)
    _write_png(img_b / "frame_0001_leftImg8bit.png", rgb, "RGB")
    label = np.full((8, 16), 7, dtype=np.uint8)
    _write_png(lbl_b / "frame_0001_gtFine_labelIds.png", label, "L")
    with pytest.raises(DatasetError, match="Ambiguous labels"):
        CityscapesDataset(tmp_path)


def test_from_config_builds_dataset(tmp_path):
    from utils.config import load_config

    _make_frame(tmp_path, "frame_0001", label_value=7)
    cfg = load_config("pipeline", overrides={"data": {"cityscapes": {"root": str(tmp_path)}}})
    ds = CityscapesDataset.from_config(cfg)
    assert len(ds) == 1


def test_from_config_max_samples_override(tmp_path):
    from utils.config import load_config

    _make_frame(tmp_path, "frame_0001", label_value=7)
    _make_frame(tmp_path, "frame_0002", label_value=7)
    cfg = load_config(
        "pipeline",
        overrides={"data": {"cityscapes": {"root": str(tmp_path), "max_samples": 1}}},
    )
    assert len(CityscapesDataset.from_config(cfg)) == 1