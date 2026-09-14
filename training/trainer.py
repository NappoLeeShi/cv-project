"""Reusable U-Net trainer.

The trainer owns the training and validation loops, per-epoch metric tracking
and best-checkpoint saving. It is deliberately CLI-agnostic: callers (such as
``training/train_unet.py``) build the model, dataloaders, optimizer and loss,
then call :meth:`UNetTrainer.fit`.

* Segmentation metrics are reused from ``evaluation/segmentation_metrics.py``;
  no IoU formulas are re-implemented here.
* The checkpoint uses the ``{"model_state_dict": ...}`` structure already
  understood by ``models/unet/inference.py`` (which also accepts a raw
  ``state_dict``).
* Loss is ``CrossEntropyLoss(ignore_index=255)`` by default; the Cityscapes
  preprocessing already maps ignored classes to ``255``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.segmentation_metrics import evaluate_segmentation
from utils.device import resolve_device
from utils.logger import get_logger

LOGGER = get_logger("training.trainer")

DEFAULT_IGNORE_INDEX = 255
DEFAULT_NUM_CLASSES = 19


def _infer_num_classes(model: nn.Module) -> int | None:
    """Best-effort class count from the final ``1x1`` U-Net conv head."""
    last: Any = getattr(model, "out", None)
    if isinstance(last, nn.Conv2d):
        return int(last.out_channels)
    return None


class UNetTrainer:
    """Train a segmentation model through dataloaders of ``{image, label}``.

    Batches may be dictionaries with ``image``/``label`` keys (the format the
    existing :class:`~preprocessing.cityscapes.CityscapesDataset` produces) or
    ``(images, labels)`` tuples.

    The best model is selected on validation mIoU and saved to
    ``checkpoint_dir / checkpoint_filename``. Per-epoch metrics are exposed
    through :attr:`history` and can be persisted with :meth:`save_history`.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        criterion: nn.Module | None = None,
        num_epochs: int = 1,
        device: str | torch.device = "auto",
        num_classes: int | None = None,
        ignore_index: int = DEFAULT_IGNORE_INDEX,
        checkpoint_dir: str | Path = "checkpoints",
        checkpoint_filename: str = "unet_cityscapes.pth",
        history_path: str | Path | None = None,
        seed: int | None = None,
        config: dict | None = None,
        resume: str | Path | None = None,
        mixed_precision: bool = False,
    ) -> None:
        if not isinstance(num_epochs, int) or isinstance(num_epochs, bool) or num_epochs < 1:
            raise ValueError(f"num_epochs must be a positive int, got {num_epochs!r}")

        self.device = resolve_device(device)
        self.mixed_precision = bool(mixed_precision) and self.device.type == "cuda"
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = (
            criterion if criterion is not None else nn.CrossEntropyLoss(ignore_index=ignore_index)
        )
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        else:
            self.optimizer = optimizer
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.mixed_precision)

        self.num_epochs = int(num_epochs)
        if num_classes is None:
            num_classes = _infer_num_classes(self.model) or DEFAULT_NUM_CLASSES
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_filename = str(checkpoint_filename)
        self.history_path = Path(history_path) if history_path is not None else None
        self.seed = int(seed) if seed is not None else None
        self.config = config

        self.epoch = 0
        self.best_val_miou: float = -1.0
        self.best_epoch: int | None = None
        self.best_val_loss: float | None = None
        self.history: list[dict[str, Any]] = []

        if resume is not None:
            self.load_checkpoint(resume)

    # ------------------------------------------------------------------ props

    @property
    def checkpoint_path(self) -> Path:
        """Absolute-ish filesystem path of the best-checkpoint destination."""
        return self.checkpoint_dir / self.checkpoint_filename

    @property
    def num_parameters(self) -> int:
        """Total number of trainable model parameters."""
        return int(sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    # ------------------------------------------------------------------ fit

    def fit(self) -> dict[str, Any]:
        """Run all remaining epochs (respecting ``resume``) and return a summary."""
        start = self.epoch + 1
        for epoch in range(start, self.num_epochs + 1):
            train_loss = self._train_one_epoch(epoch)
            val_metrics = self.validate()
            self._record_epoch(epoch, train_loss, val_metrics)
            if val_metrics is not None:
                self._consider_save(epoch, val_metrics)
        return self.summary()

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss: float = 0.0
        num_batches = 0
        for batch in self.train_loader:
            images, labels = self._split_batch(batch)
            if labels is None:
                raise RuntimeError("Training batch has no labels; labels are required to train.")
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.mixed_precision):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.detach().cpu().item())
            num_batches += 1

        if num_batches == 0:
            raise RuntimeError("Training loader produced no batches (empty dataset?).")
        return float(total_loss / num_batches)

    @torch.no_grad()
    def validate(self) -> dict[str, float] | None:
        """Run one validation pass; ``None`` when no validation loader is set.

        Returns ``loss`` (averaged over batches), ``pixel_accuracy`` and
        ``miou`` (batch-averaged over the existing segmentation metrics).
        """
        if self.val_loader is None:
            return None
        self.model.eval()

        total_loss: float = 0.0
        pixel_accuracies: list[float] = []
        mious: list[float] = []
        num_batches = 0

        for batch in self.val_loader:
            images, labels = self._split_batch(batch)
            if labels is None:
                continue
            images, labels = images.to(self.device), labels.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)
            total_loss += float(loss.item())

            predictions = logits.argmax(dim=1)
            metrics = evaluate_segmentation(
                predictions,
                labels,
                num_classes=self.num_classes,
                ignore_index=self.ignore_index,
            )
            pixel_accuracies.append(float(metrics["pixel_accuracy"]))
            mious.append(float(metrics["miou"]))
            num_batches += 1

        if num_batches == 0:
            return {"loss": float(total_loss), "pixel_accuracy": 0.0, "miou": 0.0}
        return {
            "loss": float(total_loss / num_batches),
            "pixel_accuracy": float(np.mean(pixel_accuracies)),
            "miou": float(np.mean(mious)),
        }

    # ------------------------------------------------------------- bookkeeping

    def _record_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: dict[str, float] | None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": None,
            "val_pixel_accuracy": None,
            "val_miou": None,
        }
        if val_metrics is not None:
            record["val_loss"] = float(val_metrics["loss"])
            record["val_pixel_accuracy"] = float(val_metrics["pixel_accuracy"])
            record["val_miou"] = float(val_metrics["miou"])

        self.history.append(record)
        self.epoch = int(epoch)

        if val_metrics is not None:
            LOGGER.info(
                "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f | "
                "Val Pixel Acc: %.4f | Val mIoU: %.4f",
                epoch,
                self.num_epochs,
                train_loss,
                val_metrics["loss"],
                val_metrics["pixel_accuracy"],
                val_metrics["miou"],
            )
        else:
            LOGGER.info("Epoch %d/%d | Train Loss: %.4f", epoch, self.num_epochs, train_loss)
        return record

    def _consider_save(self, epoch: int, val_metrics: dict[str, float]) -> Path | None:
        """Save a checkpoint when the current validation mIoU improves the best."""
        val_miou = float(val_metrics["miou"])
        if val_miou <= self.best_val_miou:
            return None
        self.best_val_miou = val_miou
        self.best_epoch = int(epoch)
        self.best_val_loss = float(val_metrics["loss"])
        path = self._save_checkpoint(epoch, val_metrics)
        LOGGER.info(
            "Best checkpoint saved (epoch %d, val_miou %.4f): %s",
            epoch,
            val_miou,
            path,
        )
        return path

    def _save_checkpoint(self, epoch: int, val_metrics: dict[str, float]) -> Path:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_path
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": (
                self.optimizer.state_dict() if self.optimizer is not None else None
            ),
            "epoch": int(epoch),
            "best_epoch": int(self.best_epoch) if self.best_epoch is not None else int(epoch),
            "val_loss": float(val_metrics["loss"]),
            "val_miou": float(val_metrics["miou"]),
            "val_pixel_accuracy": float(val_metrics["pixel_accuracy"]),
            "num_classes": self.num_classes,
            "ignore_index": self.ignore_index,
            "seed": self.seed,
            "config": self.config,
        }
        torch.save(state, path)
        return path

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Resume model/optimizer state plus epoch and best metrics from a checkpoint."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            optimizer_state = checkpoint.get("optimizer_state_dict")
            if optimizer_state and self.optimizer is not None:
                self.optimizer.load_state_dict(optimizer_state)
            self.epoch = int(checkpoint.get("epoch", 0))
            self.best_val_miou = float(checkpoint.get("val_miou", -1.0))
            self.best_epoch = checkpoint.get("best_epoch")
            self.best_val_loss = checkpoint.get("val_loss")
            return checkpoint
        if isinstance(checkpoint, dict) and all(
            isinstance(value, torch.Tensor) for value in checkpoint.values()
        ):
            self.model.load_state_dict(checkpoint)
            return checkpoint
        raise ValueError(
            f"Unrecognized resume checkpoint format in {path}: expected either a raw "
            f"state_dict or a dict containing a 'model_state_dict' key."
        )

    # -------------------------------------------------------------- history

    def to_history_dict(self) -> dict[str, Any]:
        """JSON-serializable training history summary."""
        return {
            "epochs": list(self.history),
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_miou": self.best_val_miou,
        }

    def save_history(self, path: str | Path | None = None) -> Path:
        """Persist :meth:`to_history_dict` as JSON to ``path`` (or the configured one)."""
        target = Path(path) if path is not None else self.history_path
        if target is None:
            raise ValueError("No history path configured; pass path=...")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_history_dict(), handle, indent=2)
        return target

    def summary(self) -> dict[str, Any]:
        """Training summary: best epoch/metrics plus artifact paths."""
        data = self.to_history_dict()
        data["checkpoint_path"] = str(self.checkpoint_path)
        data["history_path"] = str(self.history_path) if self.history_path is not None else None
        return data

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _split_batch(batch: Any) -> tuple[Any, Any]:
        """Extract ``(images, labels)`` from a dict batch or a 2-tuple batch."""
        if isinstance(batch, dict):
            return batch.get("image"), batch.get("label")
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]
        raise TypeError(
            f"Unsupported batch type {type(batch).__name__}; expected a dict with "
            f"'image'/'label' keys or an (images, labels) tuple."
        )