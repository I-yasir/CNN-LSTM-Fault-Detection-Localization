"""
Main entry point for the Fault Detection project.

Phase 1 Responsibilities
------------------------
1. Load project configuration.
2. Set random seed.
3. Initialize logger.
4. Select compute device.
5. Verify dataset file exists.
6. Print project summary.

Dataset loading, splitting, normalization and training
will be added in later phases.
"""

from pathlib import Path

import torch

from utils.config import load_config
from utils.logger import get_logger
from utils.seed import set_seed


def get_device(device_cfg: str) -> torch.device:
    """
    Resolve the computation device.

    Parameters
    ----------
    device_cfg : str
        One of:
            - auto
            - cpu
            - cuda

    Returns
    -------
    torch.device
    """

    device_cfg = device_cfg.lower()

    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_cfg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")

    return torch.device("cpu")


def main():

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------
    cfg = load_config("configuration/config.yaml")

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------
    set_seed(
        seed=cfg.project.seed,
        deterministic=cfg.project.deterministic,
        benchmark=cfg.project.benchmark,
    )

    # ---------------------------------------------------------
    # Logger
    # ---------------------------------------------------------
    logger = get_logger(
        cfg.logging.directory,
        cfg.logging.level,
    )

    logger.info("Configuration loaded successfully.")
    logger.info("Random seed set to %d.", cfg.project.seed)

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------
    device = get_device(cfg.project.device)

    logger.info("Using device: %s", device)

    # ---------------------------------------------------------
    # Dataset File Check
    # ---------------------------------------------------------
    dataset_path = Path(cfg.dataset.file)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{dataset_path}"
        )

    logger.info("Dataset found.")

    # ---------------------------------------------------------
    # Project Summary
    # ---------------------------------------------------------
    logger.info("=" * 70)
    logger.info("Project Summary")
    logger.info("=" * 70)

    logger.info("Project       : %s", cfg.project.name)
    logger.info("Version       : %s", cfg.project.version)
    logger.info("Dataset       : %s", dataset_path)
    logger.info("Signal Shape  : (%d, %d)",
                cfg.dataset.signal_length,
                cfg.dataset.num_channels)
    logger.info("Classes       : %d", cfg.dataset.num_classes)
    logger.info("Device        : %s", device)
    logger.info("Seed          : %d", cfg.project.seed)

    logger.info("=" * 70)
    logger.info("Phase 1 initialization completed successfully.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()