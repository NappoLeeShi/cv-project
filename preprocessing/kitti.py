"""KITTI depth dataset loader.

Layout expectations (flat directories under ``root``)::

    <root>/images/<stem>.png
    <root>/depth/<stem>.png

Depth PNGs are single-channel 16-bit values in millimetres; the loader converts
them to float32 metres (``value / scale_mm``). Depth ``0`` marks invalid pixels
(occlusion sparsity), exposed as a boolean ``valid_mask`` alongside ``depth``.

Optional split files ``<root>/splits/<split>.txt`` list one sample per line.
A line is either a bare stem (``0000001234``) or a relative path (``0000001234.png``
or ``subdir/0000001234.png``); the stem is derived from the path. Files with
``#`` prefixes and blank lines are ignored. When no split file exists for the
requested split, every stem present in both directories is used.

Images and depths are paired by stem, never by sorting two lists.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from preprocessing.errors import DatasetError
from preprocessing.transforms import DepthTransform, ImageTransform
from utils.config import get

LOGGER = logging.getLogger(__name__)


class KittiDepthDataset(Dataset):
    """KITTI depth dataset producing ``{image, depth, valid_mask, ...}`` dicts.

    Args:
        root: Dataset root holding ``images/`` and ``depth/``, plus an optional
            ``splits/`` directory.
        split: Split name (``train``, ``val``, ``test``) used to select a split
            file ``splits/<split>.txt``. When no file exists the split is
            ignored and all paired stems are used.
        image_dir / depth_dir: Directory names relative to ``root``, or absolute
            paths.
        image_size: Target ``[H, W]``; ``None`` keeps the source resolution.
        scale_mm: Divisor converting raw pixel values to metres.
        depth_cap_m: Pixels above this distance (in metres) are flagged invalid.
        max_samples: Maximum number of pairs to serve; ``None`` disables the
            limit. Applied deterministically on the discovery order.
        missing_policy: ``"error"`` (raise if an entry cannot be resolved) or
            ``"skip"`` (drop it after a warning).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "val",
        image_dir: str | Path = "images",
        depth_dir: str | Path = "depth",
        image_size: Sequence[int] | None = (512, 1024),
        scale_mm: float = 1000.0,
        depth_cap_m: float | None = None,
        max_samples: int | None = None,
        missing_policy: str = "error",
    ) -> None:
        if missing_policy not in ("error", "skip"):
            raise DatasetError(f"missing_policy must be 'error' or 'skip', got: {missing_policy!r}")
        if max_samples is not None and (not isinstance(max_samples, int) or max_samples < 0):
            raise DatasetError(f"max_samples must be a non-negative int or None, got: {max_samples!r}")

        self.root = Path(root).expanduser()
        self.split = split
        self.image_size = tuple(image_size) if image_size is not None else None
        self.scale_mm = float(scale_mm)
        self.depth_cap_m = float(depth_cap_m) if depth_cap_m is not None else None
        self.max_samples = max_samples
        self.missing_policy = missing_policy

        if self.scale_mm <= 0:
            raise DatasetError(f"scale_mm must be positive, got: {self.scale_mm}")

        self.image_dir = self._resolve_dir(root, image_dir, "images")
        self.depth_dir = self._resolve_dir(root, depth_dir, "depth")

        self.image_map = self._build_stem_map(self.image_dir, "image")
        self.depth_map = self._build_stem_map(self.depth_dir, "depth")

        self.image_transform = ImageTransform(size=tuple(image_size)) if image_size is not None else ImageTransform()
        self.depth_transform = DepthTransform(size=tuple(image_size)) if image_size is not None else DepthTransform()

        self.samples = self._build_samples()

        if len(self.samples) == 0:
            raise DatasetError(f"No KITTI depth pairs found under {self.root} (split={split!r})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        image_path, depth_path = self.samples[idx]

        image = Image.open(image_path).convert("RGB")
        depth = self._read_depth(depth_path)

        depth_tensor, valid = self.depth_transform(depth)

        return {
            "image": self.image_transform(image),
            "depth": depth_tensor,
            "valid_mask": valid,
            "image_path": str(image_path),
            "depth_path": str(depth_path),
        }

    @classmethod
    def from_config(cls, cfg: dict, split: str | None = None) -> "KittiDepthDataset":
        """Build a dataset from the ``data.kitti`` config block."""
        block = get(cfg, "data.kitti", {}) or {}
        image_size = get(cfg, "data.image_size") or (512, 1024)
        return cls(
            root=get(block, "root", "data/kitti"),
            split=split if split is not None else get(block, "split", "val"),
            image_dir=get(block, "image_dir", "images"),
            depth_dir=get(block, "depth_dir", "depth"),
            image_size=image_size,
            scale_mm=get(block, "scale_mm", 1000.0),
            depth_cap_m=get(block, "depth_cap_m"),
            max_samples=get(block, "max_samples"),
            missing_policy=get(block, "missing_policy", "error"),
        )

    @staticmethod
    def _resolve_dir(root: Path, name: str | Path, default_name: str) -> Path:
        candidate = Path(name).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        return candidate

    @staticmethod
    def _build_stem_map(directory: Path, kind: str) -> dict[str, Path]:
        stem_map: dict[str, Path] = {}
        for path in sorted(directory.rglob("*.png")):
            rel = path.relative_to(directory)
            stem = str(rel.with_suffix("")).replace("\\", "/")
            if stem in stem_map:
                raise DatasetError(f"Ambiguous {kind} paths for stem {stem!r}: {stem_map[stem]} and {path}")
            stem_map[stem] = path
        return stem_map

    def _read_split_ids(self) -> list[str] | None:
        split_path = self.root / "splits" / f"{self.split}.txt"
        if not split_path.is_file():
            return None
        ids: list[str] = []
        for raw in split_path.read_text(encoding="utf-8").splitlines():
            token = raw.strip()
            if not token or token.startswith("#"):
                continue
            ids.append(token)
        return ids

    @staticmethod
    def _stem_from_tree_id(token: str) -> str:
        rel = Path(token)
        name = rel.name if rel.suffix else str(rel)
        if name.endswith(".png"):
            name = name[: -len(".png")]
        return name.replace("\\", "/")

    def _build_samples(self) -> list[tuple[Path, Path]]:
        ids = self._read_split_ids()

        if ids is None:
            if not self.image_map:
                raise DatasetError(f"No images found in {self.image_dir}")
            common = sorted(set(self.image_map) & set(self.depth_map))
            unmatched = [stem for stem in sorted(self.image_map) if stem not in self.depth_map]
            if unmatched:
                if self.missing_policy == "error":
                    raise DatasetError(
                        f"{len(unmatched)} image(s) under {self.image_dir} have no "
                        f"depth pair (first: {', '.join(unmatched[:5])} ...)"
                    )
                LOGGER.warning(
                    "%d image(s) under %s have no depth pair; dropping them.",
                    len(unmatched),
                    self.image_dir,
                )
            samples = [(self.image_map[stem], self.depth_map[stem]) for stem in common]
            if self.max_samples is not None:
                samples = samples[: self.max_samples]
            return samples

        samples: list[tuple[Path, Path]] = []
        skipped = 0
        for token in ids:
            stem = self._stem_from_tree_id(token)
            image_path = self.image_map.get(stem)
            depth_path = self.depth_map.get(stem)
            if image_path is None or depth_path is None:
                if self.missing_policy == "error":
                    raise DatasetError(
                        f"Split entry {token!r} (stem {stem!r}) is missing its "
                        f"{'image' if image_path is None else 'depth'} file under "
                        f"split={self.split!r}"
                    )
                skipped += 1
                continue
            samples.append((image_path, depth_path))

        if skipped:
            LOGGER.warning("Skipped %d unresolved split entries for split %r", skipped, self.split)

        if self.max_samples is not None:
            samples = samples[: self.max_samples]
        return samples

    def _read_depth(self, path: Path) -> np.ndarray:
        img = Image.open(path)
        # 16-bit single-channel PNGs are decoded as mode 'I' (values 0..65535);
        # keep them as int64 so no information is lost, then scale in float32.
        array = np.asarray(img, dtype=np.int64)
        if array.ndim == 3:
            array = array[..., 0]
        depth = array.astype(np.float32) / self.scale_mm
        if self.depth_cap_m is not None:
            depth = np.where(depth <= self.depth_cap_m, depth, 0.0)
        return depth