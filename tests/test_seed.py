import random

import numpy as np
import torch

from utils.seed import set_seed


def test_seed_reproducible_python_numpy_torch():
    set_seed(123)
    seq1 = [random.random(), np.random.rand(), torch.randint(0, 1000, (4,)).tolist()]

    set_seed(123)
    seq2 = [random.random(), np.random.rand(), torch.randint(0, 1000, (4,)).tolist()]

    assert seq1 == seq2


def test_seed_changes_sequence():
    set_seed(1)
    first = random.random()

    set_seed(2)
    second = random.random()

    assert first != second


def test_seed_default_style_known_value():
    set_seed(0)
    value = random.random()
    set_seed(0)
    assert random.random() == value


def test_seed_does_not_raise_on_cpu():
    set_seed(7)
    x = torch.randn(3, 3)
    assert x.shape == (3, 3)
    assert torch.isfinite(x).all()