"""
Lazy-loading PyTorch Dataset for fault classification and localization.

Features
--------
- Never loads the entire dataset into memory.
- One HDF5 handle per DataLoader worker.
- Supports split indices.
- Optional transform pipeline.
- Returns channel-first tensors (C, L).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class LazyFaultDataset(Dataset):
    """
    Lazy-loading HDF5 dataset.

    Parameters
    ----------
    cfg : Config
        Project configuration.

    indices : np.ndarray
        Sample indices belonging to this split.

    transform : callable, optional
        Applied to the signal after loading.
    """

    def __init__(
        self,
        cfg,
        indices: np.ndarray,
        transform: Optional[Callable] = None,
    ):

        self.cfg = cfg

        self.indices = np.asarray(indices, dtype=np.int64)

        self.transform = transform

        self.dataset_path = Path(cfg.dataset.file)

        self.signal_key = cfg.dataset.keys.signals
        self.class_key = cfg.dataset.keys.fault_class
        self.location_key = cfg.dataset.keys.fault_location

        # Opened lazily (one handle per DataLoader worker)
        self._h5 = None
        self._signals = None
        self._classes = None
        self._locations = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open(self):
        """
        Lazily open the HDF5 file.
        """

        if self._h5 is not None:
            return

        self._h5 = h5py.File(self.dataset_path, "r")

        self._signals = self._h5[self.signal_key]
        self._classes = self._h5[self.class_key]
        self._locations = self._h5[self.location_key]

    def _read_signal(self, sample_idx: int) -> np.ndarray:
        """
        Read one raw signal from disk.

        Returns
        -------
        np.ndarray
            Shape: (sequence_length, channels)
        """

        self._open()

        signal = self._signals[sample_idx]

        return signal.astype(np.float32, copy=False)

    def _preprocess_signal(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Internal preprocessing.

        Current operations:
        - Convert to channel-first format.

        Returns
        -------
        np.ndarray
            Shape: (channels, sequence_length)
        """

        signal = signal.T

        if self.transform is not None:
            signal = self.transform(signal)

        return signal

    def _read_label(
        self,
        sample_idx: int,
    ):

        self._open()

        fault_class = int(self._classes[sample_idx])

        fault_location = float(self._locations[sample_idx])

        return fault_class, fault_location

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_raw_signal(
        self,
        index: int,
    ) -> np.ndarray:
        """
        Return a raw signal without preprocessing.

        Shape
        -----
        (sequence_length, channels)

        Used by:
        - Normalization
        - Dataset inspection
        - Future preprocessing modules
        """

        sample_idx = int(self.indices[index])

        return self._read_signal(sample_idx)

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, index):

        sample_idx = int(self.indices[index])

        signal = self._read_signal(sample_idx)

        signal = self._preprocess_signal(signal)

        fault_class, fault_location = self._read_label(sample_idx)

        signal = torch.from_numpy(signal)

        fault_class = torch.tensor(
            fault_class,
            dtype=torch.long,
        )

        fault_location = torch.tensor(
            fault_location,
            dtype=torch.float32,
        )

        return signal, fault_class, fault_location

    # ------------------------------------------------------------------
    # Resource Management
    # ------------------------------------------------------------------

    def close(self):
        """
        Close the HDF5 file.
        """

        if self._h5 is not None:

            self._h5.close()

            self._h5 = None
            self._signals = None
            self._classes = None
            self._locations = None

    def __del__(self):

        self.close()

