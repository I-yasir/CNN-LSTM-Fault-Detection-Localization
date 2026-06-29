
"""
Normalization utilities.

Features
--------
- Compute normalization statistics from training data only.
- Memory efficient (streaming).
- Save/load normalization parameters.
- On-the-fly transformation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from utils.io import load_npz, save_npz


class BaseNormalizer(ABC):
    """
    Abstract base class for all normalizers.
    """

    @abstractmethod
    def fit(self, dataset):
        pass

    @abstractmethod
    def transform(self, signal):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @classmethod
    @abstractmethod
    def load(cls, path):
        pass

    def __call__(self, signal):
        return self.transform(signal)


class ZScoreNormalizer(BaseNormalizer):
    """
    Channel-wise Z-score normalization.

    Statistics are computed ONLY from the training dataset.
    """

    def __init__(self):

        self.mean = None
        self.std = None
        self.fitted = False

    # ------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------

    def fit(self, dataset):
        """
        Compute channel-wise mean and std.

        Parameters
        ----------
        dataset : LazyFaultDataset
            Training dataset only.
        """

        n_channels = dataset.cfg.dataset.num_channels

        channel_sum = np.zeros(n_channels, dtype=np.float64)
        channel_sq_sum = np.zeros(n_channels, dtype=np.float64)

        total_samples = 0

        for i in range(len(dataset)):

            signal = dataset.get_raw_signal(i)

            # (L,C)

            channel_sum += signal.sum(axis=0)

            channel_sq_sum += np.square(signal).sum(axis=0)

            total_samples += signal.shape[0]

        self.mean = channel_sum / total_samples

        variance = (
            channel_sq_sum / total_samples
            - np.square(self.mean)
        )

        variance = np.maximum(variance, 1e-12)

        self.std = np.sqrt(variance)

        self.fitted = True

        return self

    # ------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------

    def transform(self, signal):
        """
        Normalize one signal.

        Parameters
        ----------
        signal : ndarray

        Shape
        -----
        (C,L)
        """

        if not self.fitted:
            raise RuntimeError(
                "Normalizer has not been fitted."
            )

        return (
            signal
            - self.mean[:, None]
        ) / self.std[:, None]

    # ------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------

    def save(self, path):

        if not self.fitted:
            raise RuntimeError(
                "Cannot save an unfitted normalizer."
            )

        save_npz(
            path,
            mean=self.mean,
            std=self.std,
        )

    @classmethod
    def load(cls, path):

        data = load_npz(path)

        obj = cls()

        obj.mean = data["mean"]

        obj.std = data["std"]

        obj.fitted = True

        return obj

    # ------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------

    @property
    def statistics(self):

        if not self.fitted:
            raise RuntimeError(
                "Normalizer has not been fitted."
            )

        return {
            "mean": self.mean,
            "std": self.std,
        }

    def __repr__(self):

        if not self.fitted:
            return "ZScoreNormalizer(Not Fitted)"

        return (
            f"ZScoreNormalizer("
            f"channels={len(self.mean)})"
        )

