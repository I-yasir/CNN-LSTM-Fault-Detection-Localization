"""
Generic file I/O utilities.

Supported formats
-----------------
- NumPy (.npy, .npz)
- JSON
- YAML
- Pickle
- PyTorch checkpoints

These utilities are intentionally generic and contain
no project-specific logic.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    path : str | Path

    Returns
    -------
    Path
        Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------
# NumPy
# ---------------------------------------------------------------------

def save_npz(file_path: str | Path, **arrays: Any) -> None:
    """
    Save multiple arrays into a compressed NPZ file.

    Example
    -------
    save_npz(
        "split_indices.npz",
        train=train_idx,
        val=val_idx,
        test=test_idx,
    )
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)

    np.savez_compressed(file_path, **arrays)


def load_npz(file_path: str | Path) -> dict:
    """
    Load an NPZ file as a dictionary.
    """
    file_path = Path(file_path)

    with np.load(file_path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


# ---------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------

def save_json(file_path: str | Path, data: dict) -> None:
    """
    Save dictionary as JSON.
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False,
        )


def load_json(file_path: str | Path) -> dict:
    """
    Load JSON file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------

def save_yaml(file_path: str | Path, data: dict) -> None:
    """
    Save dictionary as YAML.
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)

    with open(file_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
        )


def load_yaml(file_path: str | Path) -> dict:
    """
    Load YAML file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------
# Pickle
# ---------------------------------------------------------------------

def save_pickle(file_path: str | Path, obj: Any) -> None:
    """
    Save object using pickle.
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)

    with open(file_path, "wb") as f:
        pickle.dump(
            obj,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def load_pickle(file_path: str | Path) -> Any:
    """
    Load pickle object.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------
# PyTorch
# ---------------------------------------------------------------------

def save_checkpoint(
    file_path: str | Path,
    checkpoint: dict,
) -> None:
    """
    Save a PyTorch checkpoint.

    Example
    -------
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_score": best_score,
    }
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)

    torch.save(checkpoint, file_path)


def load_checkpoint(
    file_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict:
    """
    Load a PyTorch checkpoint.
    """
    return torch.load(
        file_path,
        map_location=map_location,
    )