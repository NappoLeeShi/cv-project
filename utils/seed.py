"""Deterministic seeding for reproducible runs.

Call :func:`set_seed` once at process start (orchestrated from ``main.py``)
to make Python, NumPy and PyTorch (CPU and CUDA) RNGs reproducible.

"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed all relevant RNGs.

    Also enables deterministic cuDNN behaviour (at the cost of some speed).
    """
    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False