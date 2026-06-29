"""
Utilities for reproducible experiments.

This module sets random seeds for Python, NumPy and PyTorch,
and optionally enforces deterministic execution.

Usage
-----
from utils.seed import set_seed

set_seed(
    seed=42,
    deterministic=True,
    benchmark=False
)
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(
    seed: int,
    deterministic: bool = True,
    benchmark: bool = False,
) -> None:
    """
    Set random seeds for reproducible experiments.

    Parameters
    ----------
    seed : int
        Random seed.

    deterministic : bool
        Force deterministic algorithms.

    benchmark : bool
        Enable CuDNN benchmark mode.
        Should normally be False for reproducibility.
    """

    # ------------------------------------------------------------------
    # Python
    # ------------------------------------------------------------------
    random.seed(seed)

    # ------------------------------------------------------------------
    # Hash seed
    # ------------------------------------------------------------------
    os.environ["PYTHONHASHSEED"] = str(seed)

    # ------------------------------------------------------------------
    # NumPy
    # ------------------------------------------------------------------
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # PyTorch CPU
    # ------------------------------------------------------------------
    torch.manual_seed(seed)

    # ------------------------------------------------------------------
    # PyTorch CUDA
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # CuDNN
    # ------------------------------------------------------------------
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    # ------------------------------------------------------------------
    # Deterministic Algorithms (PyTorch >= 1.8)
    # ------------------------------------------------------------------
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Older PyTorch versions may not support this
            pass


def seed_worker(worker_id: int) -> None:
    """
    Seed each DataLoader worker.

    Pass this function as worker_init_fn.

    Example
    -------
    DataLoader(
        dataset,
        worker_init_fn=seed_worker,
        generator=generator
    )
    """

    worker_seed = torch.initial_seed() % (2**32)

    np.random.seed(worker_seed)

    random.seed(worker_seed)


def create_generator(seed: int) -> torch.Generator:
    """
    Create a seeded PyTorch Generator.

    This ensures DataLoader shuffling is reproducible.

    Returns
    -------
    torch.Generator
    """

    generator = torch.Generator()

    generator.manual_seed(seed)

    return generator