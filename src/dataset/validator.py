
"""
Dataset validation utilities.

Responsibilities
----------------
- Verify HDF5 file structure.
- Validate dataset shapes.
- Validate label ranges.
- Validate fault location range.
- Detect NaN/Inf values.
- Print dataset summary.

The validator is memory-efficient and never loads the
entire signal dataset into RAM.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


class DatasetValidator:
    """
    Validate an HDF5 fault dataset.
    """

    def __init__(self, cfg, logger=None):

        self.cfg = cfg
        self.logger = logger

        self.dataset_path = Path(cfg.dataset.file)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, message: str):
        if self.logger:
            self.logger.info(message)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self):
        """
        Validate the dataset.

        Raises
        ------
        FileNotFoundError
        KeyError
        ValueError
        """

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found:\n{self.dataset_path}"
            )

        with h5py.File(self.dataset_path, "r") as h5:

            keys = self.cfg.dataset.keys

            # ----------------------------------------------------------
            # Required datasets
            # ----------------------------------------------------------
            required = [
                keys.signals,
                keys.fault_class,
                keys.fault_location,
            ]

            for key in required:
                if key not in h5:
                    raise KeyError(f"Missing dataset: '{key}'")

            signals = h5[keys.signals]
            fault_class = h5[keys.fault_class]
            fault_location = h5[keys.fault_location]

            # ----------------------------------------------------------
            # Dimensions
            # ----------------------------------------------------------
            if signals.ndim != 3:
                raise ValueError(
                    f"signals must be 3-dimensional, got {signals.shape}"
                )

            if fault_class.ndim != 1:
                raise ValueError(
                    f"fault_class must be 1-dimensional, got {fault_class.shape}"
                )

            if fault_location.ndim != 1:
                raise ValueError(
                    f"fault_location must be 1-dimensional, got {fault_location.shape}"
                )

            n_samples = signals.shape[0]

            if len(fault_class) != n_samples:
                raise ValueError(
                    "signals and fault_class have different lengths."
                )

            if len(fault_location) != n_samples:
                raise ValueError(
                    "signals and fault_location have different lengths."
                )

            # ----------------------------------------------------------
            # Signal shape
            # ----------------------------------------------------------
            if signals.shape[1] != self.cfg.dataset.signal_length:
                raise ValueError(
                    f"Expected signal length "
                    f"{self.cfg.dataset.signal_length}, "
                    f"got {signals.shape[1]}"
                )

            if signals.shape[2] != self.cfg.dataset.num_channels:
                raise ValueError(
                    f"Expected {self.cfg.dataset.num_channels} channels, "
                    f"got {signals.shape[2]}"
                )

            # ----------------------------------------------------------
            # Validate class labels
            # ----------------------------------------------------------
            classes = fault_class[:]

            if np.isnan(classes).any():
                raise ValueError("NaN values detected in fault_class.")

            if np.isinf(classes).any():
                raise ValueError("Inf values detected in fault_class.")

            if classes.min() < 0:
                raise ValueError("Negative class labels detected.")

            if classes.max() >= self.cfg.dataset.num_classes:
                raise ValueError(
                    "Class labels exceed configured num_classes."
                )

            # ----------------------------------------------------------
            # Validate locations
            # ----------------------------------------------------------
            locations = fault_location[:]

            if np.isnan(locations).any():
                raise ValueError("NaN values detected in fault_location.")

            if np.isinf(locations).any():
                raise ValueError("Inf values detected in fault_location.")

            if np.any(
                locations < self.cfg.dataset.location.minimum
            ):
                raise ValueError(
                    "Fault locations below configured minimum."
                )

            if np.any(
                locations > self.cfg.dataset.location.maximum
            ):
                raise ValueError(
                    "Fault locations above configured maximum."
                )

            # ----------------------------------------------------------
            # Validate signals (chunked)
            # ----------------------------------------------------------
            chunk_size = 512

            for start in range(0, n_samples, chunk_size):

                stop = min(start + chunk_size, n_samples)

                batch = signals[start:stop]

                if np.isnan(batch).any():
                    raise ValueError(
                        f"NaN values detected in signals "
                        f"(samples {start}-{stop-1})."
                    )

                if np.isinf(batch).any():
                    raise ValueError(
                        f"Inf values detected in signals "
                        f"(samples {start}-{stop-1})."
                    )

            self._summary(
                signals.shape,
                n_samples,
                classes,
                locations,
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _summary(
        self,
        signal_shape,
        samples,
        classes,
        locations,
    ):

        self._log("=" * 70)
        self._log("Dataset Validation Successful")
        self._log("=" * 70)

        self._log(f"Dataset          : {self.dataset_path.name}")
        self._log(f"Samples          : {samples}")
        self._log(f"Signal Shape     : {signal_shape}")
        self._log(f"Signal Length    : {signal_shape[1]}")
        self._log(f"Channels         : {signal_shape[2]}")
        self._log(f"Classes Present  : {len(np.unique(classes))}")
        self._log(
            f"Location Range   : "
            f"{locations.min():.2f} - {locations.max():.2f}"
        )

        self._log("=" * 70)
