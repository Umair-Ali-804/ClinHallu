from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic_cudnn: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    if deterministic_cudnn:
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """Pass to `DataLoader(..., worker_init_fn=worker_init_fn)` so each
    DataLoader worker process gets an independent, seed-derived RNG state
    (otherwise NumPy/random calls inside a Dataset -- e.g. any future
    stochastic augmentation -- can be IDENTICAL across workers)."""
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
