from __future__ import annotations

import os
import random
from typing import Iterable

import numpy as np
import torch


def select_device(preference: Iterable[str]) -> torch.device:
    for name in preference:
        name = name.lower()
        if name == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if name == "cpu":
            return torch.device("cpu")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
