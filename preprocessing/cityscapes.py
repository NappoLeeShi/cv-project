"""Cityscapes segmentation dataset loader.

Resolves the ``leftImg8bit``/``gtFine`` (or legacy ``images``/``labels``) layout
either in the official split layout (``<root>/<images>/<split>/...``) or as a
flat directory. Images and labels are paired by the shared *stem* of their file
name (the part before ``_leftImg8bit`` / ``_gtFine_labelIds``), never by sorting
two independent lists.

Label maps are stored as raw ``labelIds`` (0..33) and remapped to *trainId*
(0..18, 255 = ignore) using the official lookup from the Cityscapes scripts
<https://github.com/mcordts/cityscapesScripts> ``helpers/labels.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from preprocessing.errors import DatasetError
from preprocessing.transforms import ImageTransform, label_to_tensor, resize_nearest
from utils.config import get

LOGGER = logging.getLogger(__name__)

NUM_CLASSES = 19
IGNORE_INDEX = 255

# labelId -> trainId, extracted from the official Cityscapes labels.py
# (trainId entries with ignoreInEval=True and the void classes map to IGNORE_INDEX).
CITYSCAPES_LABEL_ID_TO_TRAIN_ID = {
    7: 0,    # road
    8: 1,    # sidewalk
    11: 2,   # building
    12: 3,   # wall
    13: 4,   # fence
    17: 5,   # pole
    19: 6,   # traffic light
    20: 7,   # traffic sign
    21: 8,   # vegetation
    22: 9,   # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
}

TRAIN_ID_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]

_IMAGE_SUFFIX = "_leftImg8bit.png"
_LABEL_SUFFIXES = ("_gtFine_labelIds.png", "_gtCoarse_labelIds.png")
_AUTO_DIRS = {
    "image": ("leftImg8bit", "images"),
    "label": ("gtFine", "labels"),
}

_TRAIN_ID_LOOKUP = np.zeros(256, dtype=np.int64)
_TRAIN_ID_LOOKUP[:] = IGNORE_INDEX
for _label_id, _train_id in CITYSCAPES_LABEL_ID_TO_TRAIN_ID.items():
    _TRAIN_ID_LOOKUP[_label_id] = _train_id


def labels_to_train_ids(label_ids: np.ndarray) -> np.ndarray:
    """Remap a ``labelIds`` array to Cityscapes ``trainId`` space."""
    return _TRAIN_ID_LOOKUP[label_ids]


class CityscapesDataset(Dataset):
    """Cityscapes segmentation dataset producing ``{image, label, ...}`` dicts.

    Args:
        root: Dataset root containing the image and label directories.
        split: Sub-directory name (``train``, ``val``, ``test``) or ``None`` for
            a flat layout.
        image_dir / label_dir: Directory names relative to ``root``, or absolute
            paths. When ``None`` the conventional names are auto-detected.
        image_size: Target ``[H, W]``; ``None`` keeps the source resolution.
        max_samples: Maximum number of pairs to serve; ``None`` disables the
            limit. Applied deterministically on the discovery order.
        require_labels: When ``True`` every image needs a matching label.
        missing_policy: ``"error"`` (raise if a pair is missing) or ``"skip"``
            (drop the offending image after a warning).
        label_suffix: Suffix that marks files under the label directory as
            semantic labels (everything else there is ignored).
    """

    def __init__(
        self,
        root: str | Path,
        split: str | None = "train",
        image_dir: str | Path | None = None,
        label_dir: str | Path | None = None,
        image_size: Sequence[int] | None = (512, 1024),
        max_samples: int | None = None,
        require_labels: bool = True,
        missing_policy: str = "error",
        label_suffix: str = "_gtFine_labelIds.png",
    ) -> None:
        if missing_policy not in ("error", "skip"):
            raise DatasetError(f"missing_policy must be 'error' or 'skip', got: {missing_policy!r}")
        if max_samples is not None and (not isinstance(max_samples, int) or max_samples < 0):
            raise DatasetError(f"max_samples must be a non-negative int or None, got: {max_samples!r}")

        self.root = Path(root).expanduser()
        self.split = split
        self.image_size = tuple(image_size) if image_size is not None else None
        self.max_samples = max_samples
        self.require_labels = require_labels
        self.missing_policy = missing_policy

        self.image_dir = self._resolve_dir(self.root, image_dir, "image")
        self.label_dir = self._resolve_dir(self.root, label_dir, "label")
        self.image_scan_dir = self._resolve_scan_dir(self.image_dir, self.split)
        self.label_scan_dir = self._resolve_scan_dir(self.label_dir, self.split)

        self.transform = ImageTransform(size=tuple(image_size)) if image_size is not None else ImageTransform()

        self.label_map = self._build_label_map(self.label_scan_dir, label_suffix)
        self.samples = self._build_samples(self.image_scan_dir, self.label_map)

        if len(self.samples) == 0 and self.max_samples != 0:
            raise DatasetError(
                f"No Cityscapes pairs found under {self.image_scan_dir} "
                f"(split={split!r})"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        image_path, label_path = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        label: torch.Tensor | None = None
        if label_path is not None:
            label_pil = Image.open(label_path).convert("L")
            label_pil = resize_nearest(label_pil, self.image_size)
            label = label_to_tensor(labels_to_train_ids(np.asarray(label_pil, dtype=np.int64)))

        sample = {
            "image": self.transform(image),
            "label": label,
            "image_path": str(image_path),
            "label_path": str(label_path) if label_path is not None else None,
        }
        return sample

    @classmethod
    def from_config(cls, cfg: dict, split: str | None = None) -> "CityscapesDataset":
        """Build a dataset from the ``data.cityscapes`` config block."""
        block = get(cfg, "data.cityscapes", {}) or {}
        image_size = get(cfg, "data.image_size") or (512, 1024)
        return cls(
            root=get(block, "root", "data/cityscapes"),
            split=split if split is not None else get(block, "split", "train"),
            image_dir=get(block, "image_dir"),
            label_dir=get(block, "label_dir"),
            image_size=image_size,
            max_samples=get(block, "max_samples"),
            require_labels=get(block, "require_labels", True),
            missing_policy=get(block, "missing_policy", "error"),
        )

    @staticmethod
    def _resolve_dir(root: Path, name: str | Path | None, kind: str) -> Path:
        root = Path(root)
        if name is not None:
            candidate = Path(name).expanduser()
            if not candidate.is_absolute():
                candidate = root / candidate
            return candidate

        for candidate_name in _AUTO_DIRS[kind]:
            candidate = root / candidate_name
            if candidate.is_dir():
                return candidate
        expected = ", ".join(_AUTO_DIRS[kind])
        raise DatasetError(
            f"Could not locate a {kind} directory under {root}; "
            f"expected one of: {expected}"
        )

    @staticmethod
    def _resolve_scan_dir(directory: Path, split: str | None) -> Path:
        if split is not None:
            candidate = directory / split
            if candidate.is_dir():
                return candidate
        if directory.is_dir():
            return directory
        raise DatasetError(f"Directory does not exist: {directory}")

    @staticmethod
    def _build_label_map(label_dir: Path, label_suffix: str) -> dict[str, Path]:
        label_map: dict[str, Path] = {}
        for path in sorted(label_dir.rglob("*.png")):
            name = path.name
            key = None
            if label_suffix and name.endswith(label_suffix):
                key = name[: -len(label_suffix)]
            else:
                for suffix in _LABEL_SUFFIXES:
                    if name.endswith(suffix):
                        key = name[: -len(suffix)]
                        break
            if key is None:
                continue
            if key in label_map:
                raise DatasetError(f"Ambiguous labels for {key}: {label_map[key]} and {path}")
            label_map[key] = path
        return label_map

    def _build_samples(
        self,
        image_dir: Path,
        label_map: dict[str, Path],
    ) -> list[tuple[Path, Path | None]]:
        samples: list[tuple[Path, Path | None]] = []
        missing: list[Path] = []

        for image_path in sorted(image_dir.rglob(f"*{_IMAGE_SUFFIX}")):
            key = image_path.name[: -len(_IMAGE_SUFFIX)]
            label_path = label_map.get(key)
            if label_path is None:
                if self.require_labels:
                    missing.append(image_path)
                    continue
                samples.append((image_path, None))
                continue
            samples.append((image_path, label_path))

        if missing:
            count = len(missing)
            preview = ", ".join(str(p) for p in missing[:5])
            if self.missing_policy == "error":
                raise DatasetError(
                    f"{count} image(s) have no matching label under "
                    f"{self.label_scan_dir} (first: {preview}...). Set "
                    f"missing_policy='skip' to drop them."
                )
            LOGGER.warning("Dropping %d image(s) without labels: %s ...", count, preview)

        if self.max_samples is not None:
            samples = samples[: self.max_samples]
        return samples