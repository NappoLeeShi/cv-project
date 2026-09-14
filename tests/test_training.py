"""Step 12 training tests.

Training tests use tiny synthetic tensors on CPU only. They exercise the real
optimization loop (a real forward/backward/step) and verify checkpoint
compatibility with the existing U-Net inference loader — no Cityscapes files,
no internet, no CUDA requirement.
"""

import json

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.unet.inference import UNetInference
from models.unet.model import UNet
from training.trainer import UNetTrainer
from utils.seed import set_seed


class TinySegDataset(Dataset):
    """Deterministic synthetic segmentation dataset: ``{image, label}`` dicts.

    ``~15%`` of the pixels are set to ``ignore=255`` so the loss handles ignored
    classes exactly like the real Cityscapes preprocessing.
    """

    def __init__(self, n=8, num_classes=8, h=32, w=32, seed=0, ignore=255):
        gen = torch.Generator().manual_seed(seed)
        self.num_classes = num_classes
        self.ignore = ignore
        self.images = torch.randn(n, 3, h, w, generator=gen)
        self.labels = torch.randint(0, num_classes, (n, h, w), generator=gen)
        mask = torch.rand(n, h, w, generator=gen) < 0.15
        self.labels[mask] = ignore

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {"image": self.images[idx], "label": self.labels[idx]}


def make_loader(dataset, batch_size=4, shuffle=True, seed=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def make_trainer(tmp_path, **kwargs):
    model = kwargs.pop("model", UNet(num_classes=8, base_channels=4))
    train_loader = kwargs.pop("train_loader", make_loader(TinySegDataset(n=8)))
    val_loader = kwargs.pop(
        "val_loader", make_loader(TinySegDataset(n=8), shuffle=False, seed=1)
    )
    defaults = {
        "checkpoint_dir": tmp_path,
        "checkpoint_filename": "unet_cityscapes.pth",
        "num_epochs": 1,
        "device": "cpu",
    }
    defaults.update(kwargs)
    return UNetTrainer(model=model, train_loader=train_loader, val_loader=val_loader, **defaults)


# --------------------------------------------------------------------------- 1
def test_trainer_construction(tmp_path):
    trainer = make_trainer(tmp_path)
    assert isinstance(trainer, UNetTrainer)
    assert trainer.model is not None
    assert trainer.criterion.ignore_index == 255
    assert isinstance(trainer.optimizer, torch.optim.Adam)
    assert trainer.device.type == "cpu"


# --------------------------------------------------------------------------- 2
def test_one_tiny_epoch_performs_real_optimization_step(tmp_path):
    set_seed(0)
    model = UNet(num_classes=8, base_channels=4)
    before = {k: v.detach().clone() for k, v in model.named_parameters()}

    trainer = UNetTrainer(
        model=model,
        train_loader=make_loader(TinySegDataset(n=8)),
        val_loader=make_loader(TinySegDataset(n=8), shuffle=False, seed=1),
        checkpoint_dir=tmp_path,
        num_epochs=1,
        device="cpu",
    )
    trainer.fit()

    assert torch.isfinite(torch.tensor(trainer.history[0]["train_loss"]))
    changed = [
        name
        for name, param in model.named_parameters()
        if not torch.equal(before[name], param.detach())
    ]
    assert changed, "At least one parameter must change after an optimizer step"

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "Backward pass must populate gradients"


# --------------------------------------------------------------------------- 3
def test_validation_runs_and_returns_metrics(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.fit()
    record = trainer.history[0]
    assert record["val_loss"] is not None
    assert 0.0 <= record["val_pixel_accuracy"] <= 1.0
    assert 0.0 <= record["val_miou"] <= 1.0


# --------------------------------------------------------------------------- 4
def test_cross_entropy_ignores_index_255():
    logits = torch.randn(2, 19, 16, 16)
    targets = torch.randint(0, 19, (2, 16, 16), dtype=torch.long)
    mask = torch.rand(2, 16, 16) < 0.3
    targets[mask] = 255

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    expected = criterion(logits, targets).item()

    safe_targets = targets.where(targets != 255, torch.zeros_like(targets))
    manual = torch.nn.functional.cross_entropy(logits, safe_targets, reduction="none")
    manual = manual[targets != 255].mean().item()
    assert expected == pytest.approx(manual, abs=1e-6)

    ds = TinySegDataset(n=4, num_classes=19)
    loader = make_loader(ds, batch_size=4)
    trainer = UNetTrainer(
        model=UNet(num_classes=19, base_channels=4),
        train_loader=loader,
        val_loader=loader,
        device="cpu",
    )
    assert trainer.criterion.ignore_index == 255


# --------------------------------------------------------------------------- 5
def test_metrics_present_in_history(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.fit()
    record = trainer.history[0]
    for key in ("epoch", "train_loss", "val_loss", "val_pixel_accuracy", "val_miou"):
        assert key in record
    assert record["train_loss"] >= 0.0
    assert record["val_miou"] >= 0.0


# --------------------------------------------------------------------------- 6
def test_checkpoint_is_created(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.fit()
    path = trainer.checkpoint_path
    assert path.is_file()
    assert path.name == "unet_cityscapes.pth"


# --------------------------------------------------------------------------- 7
def test_checkpoint_contains_required_fields(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.fit()
    checkpoint = torch.load(trainer.checkpoint_path, map_location="cpu", weights_only=True)
    for key in (
        "model_state_dict",
        "optimizer_state_dict",
        "epoch",
        "val_loss",
        "val_miou",
        "config",
        "num_classes",
    ):
        assert key in checkpoint, f"missing checkpoint field {key}"
    assert set(checkpoint["model_state_dict"].keys()) == set(
        trainer.model.state_dict().keys()
    )
    assert checkpoint["num_classes"] == 8


# --------------------------------------------------------------------------- 8
def test_best_checkpoint_logic(tmp_path):
    model = UNet(num_classes=8, base_channels=4)
    trainer = make_trainer(tmp_path, model=model, val_loader=make_loader(TinySegDataset(n=4)))

    first = trainer._consider_save(1, {"loss": 2.0, "pixel_accuracy": 0.5, "miou": 0.25})
    assert first is not None and first.is_file()
    assert trainer.best_val_miou == 0.25
    assert trainer.best_epoch == 1
    assert trainer.best_val_loss == 2.0

    improved = trainer._consider_save(2, {"loss": 1.0, "pixel_accuracy": 0.6, "miou": 0.5})
    assert improved is not None
    assert trainer.best_epoch == 2
    assert trainer.best_val_loss == 1.0

    worse = trainer._consider_save(3, {"loss": 3.0, "pixel_accuracy": 0.4, "miou": 0.3})
    assert worse is None
    assert trainer.best_epoch == 2
    assert trainer.best_val_miou == 0.5

    checkpoint = torch.load(trainer.checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["val_miou"] == 0.5
    assert checkpoint["epoch"] == 2
    assert checkpoint["best_epoch"] == 2


# --------------------------------------------------------------------------- 9
def test_training_history_is_json_serializable(tmp_path):
    trainer = make_trainer(tmp_path, history_path=tmp_path / "history.json")
    trainer.fit()
    history = trainer.to_history_dict()
    payload = json.dumps(history)
    assert isinstance(payload, str)
    assert set(history.keys()) == {"epochs", "best_epoch", "best_val_loss", "best_val_miou"}
    saved = trainer.save_history()
    assert saved.is_file()
    assert json.loads(saved.read_text())["best_epoch"] == history["best_epoch"]


# --------------------------------------------------------------------------- 10
def test_deterministic_seed_reproduces_training(tmp_path):
    losses = []
    for run in range(2):
        set_seed(7)
        model = UNet(num_classes=8, base_channels=4)
        loader = DataLoader(
            TinySegDataset(n=8),
            batch_size=4,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(7),
        )
        trainer = UNetTrainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            checkpoint_dir=tmp_path / f"run{run}",
            num_epochs=1,
            device="cpu",
            seed=7,
        )
        trainer.fit()
        losses.append(trainer.history[0]["train_loss"])
    assert losses[0] == pytest.approx(losses[1], abs=1e-6)


# --------------------------------------------------------------------------- 11
def test_missing_dataset_raises_clear_error(tmp_path):
    from training.train_unet import build_datasets
    from utils.config import load_config

    root = tmp_path / "no_cityscapes_here"
    config = load_config(
        "unet", overrides={"data": {"cityscapes_root": str(root)}}
    )
    with pytest.raises(FileNotFoundError, match="Cityscapes dataset not found"):
        build_datasets(config)


def test_missing_config_returns_nonzero_exit():
    from training.train_unet import main

    assert main(["--config", "definitely_missing_config"]) == 1


# --------------------------------------------------------------------------- 12
def test_checkpoint_loadable_by_existing_unet_inference(tmp_path):
    trainer = make_trainer(
        tmp_path,
        model=UNet(num_classes=8, base_channels=4),
        val_loader=make_loader(TinySegDataset(n=8), shuffle=False, seed=1),
    )
    trainer.fit()

    inferencer = UNetInference(
        model=UNet(num_classes=8, base_channels=4),
        checkpoint=trainer.checkpoint_path,
        device="cpu",
    )
    prediction = inferencer.predict(torch.zeros(3, 32, 32))
    assert tuple(prediction.shape) == (32, 32)
    assert prediction.dtype == torch.long
    assert prediction.min().item() >= 0
    assert prediction.max().item() <= 7


def test_raw_state_dict_checkpoint_also_loads_for_resume(tmp_path):
    model = UNet(num_classes=8, base_channels=4)
    state_path = tmp_path / "raw.pt"
    torch.save(model.state_dict(), state_path)
    trainer = make_trainer(tmp_path, model=UNet(num_classes=8, base_channels=4))
    trainer.load_checkpoint(state_path)
    assert trainer.epoch == 0


def test_history_values_are_json_scalars(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.fit()
    record = trainer.history[0]
    for value in record.values():
        assert not torch.is_tensor(value)
        assert not isinstance(value, np.ndarray)


def test_mixed_precision_flag_is_noop_on_cpu(tmp_path):
    trainer = make_trainer(tmp_path, mixed_precision=True)
    assert trainer.mixed_precision is False
    assert trainer.scaler.is_enabled() is False
    trainer.fit()
    assert torch.isfinite(torch.tensor(trainer.history[0]["train_loss"]))