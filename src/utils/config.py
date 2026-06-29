from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigNode:
    """
    Dictionary with attribute-style access.

    Example
    -------
    cfg.training.epochs
    cfg.model.cnn.channels
    """

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                value = ConfigNode(value)
            setattr(self, key, value)

    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigNode):
                value = value.to_dict()
            result[key] = value
        return result

    def __repr__(self):
        return repr(self.to_dict())


class Config:
    """
    Loads and manages project configuration.

    Responsibilities
    ----------------
    - Load YAML configuration
    - Resolve project-relative paths
    - Create output directories automatically
    - Provide attribute-style access
    """

    def __init__(self, config_path: str | Path):

        self.project_root = Path(__file__).resolve().parents[2]

        self.config_path = Path(config_path)

        if not self.config_path.is_absolute():
            self.config_path = self.project_root / self.config_path

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found:\n{self.config_path}"
            )

        with open(self.config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        self.cfg = ConfigNode(config)

        self._resolve_paths()

        self._create_output_directories()

    def _resolve_path(self, path: str) -> Path:
        """
        Convert relative path to absolute project path.
        """
        p = Path(path)

        if p.is_absolute():
            return p

        return self.project_root / p

    def _resolve_paths(self):

        # Dataset
        self.cfg.dataset.file = self._resolve_path(
            self.cfg.dataset.file
        )

        # Split
        self.cfg.split.save_path = self._resolve_path(
            self.cfg.split.save_path
        )

        # Normalization
        self.cfg.normalization.save_path = self._resolve_path(
            self.cfg.normalization.save_path
        )

        # Outputs
        self.cfg.checkpoint.directory = self._resolve_path(
            self.cfg.checkpoint.directory
        )

        self.cfg.logging.directory = self._resolve_path(
            self.cfg.logging.directory
        )

        self.cfg.report.directory = self._resolve_path(
            self.cfg.report.directory
        )

        self.cfg.figures.directory = self._resolve_path(
            self.cfg.figures.directory
        )

        self.cfg.outputs.predictions = self._resolve_path(
            self.cfg.outputs.predictions
        )

        self.cfg.outputs.metrics = self._resolve_path(
            self.cfg.outputs.metrics
        )

        self.cfg.outputs.history = self._resolve_path(
            self.cfg.outputs.history
        )

        self.cfg.outputs.report = self._resolve_path(
            self.cfg.outputs.report
        )

    def _create_output_directories(self):

        directories = [
            self.cfg.checkpoint.directory,
            self.cfg.logging.directory,
            self.cfg.report.directory,
            self.cfg.figures.directory,
            self.cfg.outputs.predictions,
            self.cfg.split.save_path.parent,
            self.cfg.normalization.save_path.parent,
        ]

        for directory in directories:
            Path(directory).mkdir(
                parents=True,
                exist_ok=True
            )

    def __getattr__(self, item: str) -> Any:
        return getattr(self.cfg, item)

    def __repr__(self):
        return repr(self.cfg)


def load_config(config_path: str | Path) -> Config:
    """
    Convenience function.

    Example
    -------
    cfg = load_config("configuration/config.yaml")
    """
    return Config(config_path)