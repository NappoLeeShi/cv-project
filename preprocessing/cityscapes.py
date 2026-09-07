from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np


class CityscapesDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform

        self.images = sorted(self.image_dir.rglob("*_leftImg8bit.png"))

        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]

        # Cityscapes:
        # xxx_leftImg8bit.png
        # xxx_gtFine_labelIds.png

        label_path = Path(
            str(image_path)
            .replace("leftImg8bit", "gtFine")
            .replace("_leftImg8bit.png", "_gtFine_labelIds.png")
        )

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)

        if self.transform:
            image = self.transform(image)

        # Label không resize/interpolate như ảnh RGB
        label = torch.from_numpy(
            np.array(label, dtype=np.int64)
        )

        return image, label