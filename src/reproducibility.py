"""Utilities that keep diagnostics from changing later training randomness."""

import random
from contextlib import contextmanager

import numpy as np
import torch


@contextmanager
def preserve_rng_state():
    """Restore Python, NumPy, CPU Torch, and CUDA RNG states on exit."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None
    )
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
