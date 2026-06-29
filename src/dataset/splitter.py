"""
Dataset splitting utilities.

Responsibilities
----------------
- Create reproducible train/validation/test splits.
- Save split indices.
- Load existing split indices.
- Never load signal data into memory.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

from utils.io import load_npz, save_npz


class DatasetSplitter:
    """
    Train / Validation / Test splitter.

    Notes
    -----
    Only the sample indices are generated and saved.
    Signal data is never loaded into memory.
    """

    def __init__(self, cfg, logger=None):

        self.cfg = cfg
        self.logger = logger

        self.dataset_path = Path(cfg.dataset.file)
        self.split_path = Path(cfg.split.save_path)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _dataset_size(self) -> int:
        """
        Determine dataset size without loading the dataset.
        """

        with h5py.File(self.dataset_path, "r") as h5:
            return len(h5[self.cfg.dataset.keys.signals])

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def create(self, overwrite: bool = False) -> dict:
        """
        Create train/validation/test split.

        Parameters
        ----------
        overwrite : bool
            Overwrite an existing split.

        Returns
        -------
        dict
            Dictionary containing indices.
        """

        if self.split_path.exists() and not overwrite:

            if self.logger:
                self.logger.info(
                    "Split already exists. Loading existing indices."
                )

            return self.load()

        n_samples = self._dataset_size()

        indices = np.arange(n_samples)

        train_idx, temp_idx = train_test_split(
            indices,
            train_size=self.cfg.split.train,
            shuffle=self.cfg.split.shuffle,
            random_state=self.cfg.project.seed,
        )

        val_ratio = (
            self.cfg.split.validation
            / (self.cfg.split.validation + self.cfg.split.test)
        )

        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_ratio,
            shuffle=self.cfg.split.shuffle,
            random_state=self.cfg.project.seed,
        )

        save_npz(
            self.split_path,
            train=train_idx,
            validation=val_idx,
            test=test_idx,
        )

        if self.logger:
            self.logger.info("Dataset split created.")
            self.logger.info("Train samples      : %d", len(train_idx))
            self.logger.info("Validation samples : %d", len(val_idx))
            self.logger.info("Test samples       : %d", len(test_idx))

        return {
            "train": train_idx,
            "validation": val_idx,
            "test": test_idx,
        }

    def load(self) -> dict:
        """
        Load previously saved split indices.
        """

        if not self.split_path.exists():
            raise FileNotFoundError(
                f"Split file not found:\n{self.split_path}"
            )

        split = load_npz(self.split_path)

        return {
            "train": split["train"],
            "validation": split["validation"],
            "test": split["test"],
        }

    def summary(self) -> None:
        """
        Print split summary.
        """

        split = self.load()

        total = (
            len(split["train"])
            + len(split["validation"])
            + len(split["test"])
        )

        if self.logger:

            self.logger.info("=" * 60)
            self.logger.info("Dataset Split Summary")
            self.logger.info("=" * 60)

            self.logger.info(
                "Train      : %6d (%.2f%%)",
                len(split["train"]),
                len(split["train"]) / total * 100,
            )

            self.logger.info(
                "Validation : %6d (%.2f%%)",
                len(split["validation"]),
                len(split["validation"]) / total * 100,
            )

            self.logger.info(
                "Test       : %6d (%.2f%%)",
                len(split["test"]),
                len(split["test"]) / total * 100,
            )

            self.logger.info("=" * 60)