"""
General utilities:
- seed setting
- device selection
- YAML config loading
- metrics computation helpers
"""

from __future__ import annotations
import os
import random
from typing import Dict, Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Ensure reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(device_pref: str = "auto") -> torch.device:
    """
    Select device:
    - 'auto': prefer CUDA, then MPS (Apple), else CPU
    - 'cuda'/'mps'/'cpu': force choice
    """
    if device_pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device(device_pref)


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(*paths: str) -> None:
    """Create folders if missing."""
    for p in paths:
        os.makedirs(p, exist_ok=True)