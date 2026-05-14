import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gapsbi.utils.seeding import set_all_seeds


def test_set_all_seeds_reproducible() -> None:
    set_all_seeds(42)
    py_1 = random.random()
    np_1 = np.random.rand(4)
    torch_1 = torch.rand(4)

    set_all_seeds(42)
    py_2 = random.random()
    np_2 = np.random.rand(4)
    torch_2 = torch.rand(4)

    assert py_1 == py_2
    assert np.allclose(np_1, np_2)
    assert torch.allclose(torch_1, torch_2)


def test_set_all_seeds_sets_cudnn_flags() -> None:
    set_all_seeds(7)

    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
