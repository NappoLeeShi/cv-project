"""U-Net training entry point (Step 12).

Usage::

    python -m training.train_unet --config unet
    python -m training.train_unet --config configs/unet.yaml \\
        --epochs 20 --batch-size 1 --mixed-precision --device auto

Defaults target ~4 GB GPUs (batch 1 @ 256x512). Restore the original
higher-memory settings (batch 4 @ 512x1024) with ``--image-size 512 1024
--batch-size 4`` on a larger GPU. CLI arguments override the YAML
configuration values. The Cityscapes dataset is never downloaded; it must
already exist under ``data/cityscapes`` and training fails fast with a clear
error when it is missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from models.unet.model import UNet
from preprocessing.cityscapes import CityscapesDataset
from utils.config import CONFIG_DIR, ConfigError, get, load_config, resolve_path
from utils.device import resolve_device
from utils.logger import get_logger, setup_logging
from utils.seed import set_seed

from training.trainer import UNetTrainer

DEFAULT_IGNORE_INDEX = 255

LOGGER = get_logger("training.train_unet")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training.train_unet",
        description="Train the existing U-Net on Cityscapes (no downloads).",
    )
    parser.add_argument(
        "--config",
        default="configs/unet.yaml",
        help="Config name (e.g. 'unet') or a path to a U-Net YAML file.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        metavar=("H", "W"),
        default=None,
        help="Target [H, W] image size for images and masks, e.g. '256 512' "
        "or '512 1024'. Overrides data.image_size from the config.",
    )
    parser.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable FP16 AMP (autocast + GradScaler) on CUDA; "
        "--no-mixed-precision disables it. Defaults to training.mixed_precision.",
    )
    parser.add_argument("--device", type=str, default=None, help="'auto' (default), 'cpu', 'cuda'.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where checkpoints and the training history are written.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a previous U-Net checkpoint (model + optimizer + epoch).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing best checkpoint.",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="Cap the number of training samples (development/smoke runs).",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=None,
        help="Cap the number of validation samples (development/smoke runs).",
    )
    return parser


def _apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply shallow top-level section overrides (``training`` / ``env``)."""
    merged = dict(config)
    for section, values in (overrides or {}).items():
        if isinstance(values, dict) and isinstance(merged.get(section), dict):
            merged[section] = {**merged[section], **values}
        else:
            merged[section] = values
    return merged


def _load_config(config_arg: str, overrides: dict[str, Any]) -> dict[str, Any]:
    """Load a training config by name or by an explicit YAML path."""
    path = Path(config_arg)
    if path.suffix == ".yaml":
        if not path.is_file():
            fallback = CONFIG_DIR / path.name
            if not fallback.is_file():
                raise ConfigError(f"Configuration file not found: {config_arg}")
            path = fallback
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ConfigError(f"Configuration root must be a mapping, got: {type(data).__name__}")
        return _apply_overrides(data, overrides)
    return load_config(config_arg, overrides=overrides)


def _training_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Collect non-``None`` CLI training arguments into an override dict."""
    overrides: dict[str, Any] = {}
    training: dict[str, Any] = {}
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.batch_size is not None:
        training["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        training["learning_rate"] = args.learning_rate
    if args.num_workers is not None:
        training["num_workers"] = args.num_workers
    if args.seed is not None:
        training["seed"] = args.seed
    if args.train_samples is not None:
        training["train_max_samples"] = args.train_samples
    if args.val_samples is not None:
        training["val_max_samples"] = args.val_samples
    if args.mixed_precision is not None:
        training["mixed_precision"] = bool(args.mixed_precision)
    if training:
        overrides["training"] = training
    if args.image_size is not None:
        overrides["data"] = {"image_size": list(args.image_size)}
    if args.device is not None:
        overrides["env"] = {"device": args.device}
    return overrides


def build_datasets(config: dict[str, Any]) -> tuple[CityscapesDataset, CityscapesDataset]:
    """Build the real Cityscapes train/validation datasets; fail fast when missing."""
    root = resolve_path(get(config, "data.cityscapes_root", "data/cityscapes"))
    if not root.is_dir():
        raise FileNotFoundError(
            f"Cityscapes dataset not found at {root}. Training never downloads data; "
            f"obtain the dataset manually first."
        )
    image_size = get(config, "data.image_size", (512, 1024))
    split_dir = get(config, "data.split_dir", {}) or {}
    return (
        CityscapesDataset(
            root=root,
            split=split_dir.get("train", "train"),
            image_dir=get(config, "data.image_dir"),
            label_dir=get(config, "data.label_dir"),
            image_size=image_size,
            max_samples=get(config, "training.train_max_samples"),
        ),
        CityscapesDataset(
            root=root,
            split=split_dir.get("val", "val"),
            image_dir=get(config, "data.image_dir"),
            label_dir=get(config, "data.label_dir"),
            image_size=image_size,
            max_samples=get(config, "training.val_max_samples"),
        ),
    )


def build_dataloader(
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    device,
) -> DataLoader:
    """Build a streaming DataLoader (``persistent_workers`` only with workers)."""
    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    pin_memory = bool("cuda" in str(device) and torch.cuda.is_available())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def resolve_output_paths(
    config: dict[str, Any], output_dir: str | None
) -> tuple[Path, Path]:
    """Resolve the checkpoint directory and history file from config or ``--output-dir``."""
    if output_dir is not None:
        out = resolve_path(output_dir)
        return out, out / "unet_training_history.json"
    checkpoint_dir = resolve_path(get(config, "checkpoint.dir", "checkpoints"))
    history_path = resolve_path(
        get(config, "training.history_path", "outputs/analysis/unet_training_history.json")
    )
    return checkpoint_dir, history_path


def _validate_and_report(
    trainer: UNetTrainer,
    config: dict[str, Any],
    checkpoint_dir: Path,
) -> None:
    checkpoint_file = trainer.checkpoint_path
    if trainer.history_path is not None:
        trainer.save_history()
    LOGGER.info("Training finished.")
    LOGGER.info("Best epoch: %s", trainer.best_epoch)
    if trainer.best_val_loss is not None:
        LOGGER.info("Best validation loss: %.4f", trainer.best_val_loss)
    LOGGER.info("Best validation mIoU: %.4f", trainer.best_val_miou)
    LOGGER.info("Checkpoint: %s", checkpoint_file)
    LOGGER.info("History: %s", trainer.history_path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = _load_config(args.config, _training_kwargs(args))
    except (ConfigError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    seed = int(get(config, "training.seed", 42))
    set_seed(seed)

    setup_logging(level="INFO")
    logger = get_logger("training.train_unet")
    device = resolve_device(get(config, "env.device", "auto"))

    model = UNet.from_config(config)

    epochs = int(get(config, "training.epochs", 20))
    batch_size = int(get(config, "training.batch_size", 1))
    learning_rate = float(get(config, "training.learning_rate", 1e-4))
    weight_decay = float(get(config, "training.weight_decay", 1e-5))
    num_workers = int(get(config, "training.num_workers", 2))
    num_classes = int(get(config, "model.num_classes", 19))
    mixed_precision = bool(get(config, "training.mixed_precision", False))

    checkpoint_dir, history_path = resolve_output_paths(config, args.output_dir)
    checkpoint_file = checkpoint_dir / get(config, "checkpoint.filename", "unet_cityscapes.pth")
    if args.resume is None and checkpoint_file.exists() and not args.overwrite:
        print(
            f"error: checkpoint already exists: {checkpoint_file} (pass --overwrite to replace it).",
            file=sys.stderr,
        )
        return 1

    try:
        train_dataset, val_dataset = build_datasets(config)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    train_loader = build_dataloader(
        train_dataset, batch_size, num_workers, shuffle=True, seed=seed, device=device
    )
    val_loader = build_dataloader(
        val_dataset, batch_size, num_workers, shuffle=False, seed=seed, device=device
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=DEFAULT_IGNORE_INDEX)

    trainer = UNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=epochs,
        device=device,
        num_classes=num_classes,
        ignore_index=DEFAULT_IGNORE_INDEX,
        checkpoint_dir=checkpoint_dir,
        checkpoint_filename=checkpoint_file.name,
        history_path=history_path,
        seed=seed,
        config=config,
        resume=args.resume,
        mixed_precision=mixed_precision,
    )

    logger.info("device        : %s (cuda_available=%s)", device, torch.cuda.is_available())
    logger.info("seed          : %d", seed)
    logger.info("dataset (root): %s", train_dataset.root)
    logger.info("train samples : %d", len(train_dataset))
    logger.info("val samples   : %d", len(val_dataset))
    logger.info("num classes   : %d (ignore_index=%d)", num_classes, DEFAULT_IGNORE_INDEX)
    logger.info("batch size    : %d", batch_size)
    logger.info("learning rate : %g", learning_rate)
    logger.info("weight decay  : %g", weight_decay)
    logger.info("epochs        : %d", epochs)
    logger.info("num workers   : %d", num_workers)
    logger.info("mixed prec.   : %s", mixed_precision)
    logger.info("image size    : %s", get(config, "data.image_size", (256, 512)))
    logger.info("model params  : %d", trainer.num_parameters)

    try:
        trainer.fit()
    except KeyboardInterrupt:
        print("error: training interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report clearly
        print(f"error: training failed: {exc}", file=sys.stderr)
        return 1

    _validate_and_report(trainer, config, checkpoint_dir)
    logger.info("Done. Full training command for a real run:")
    logger.info("    python -m training.train_unet --config %s", args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())