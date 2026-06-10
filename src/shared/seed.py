"""Reproducible seeding for the PRT redesign workspace.

Call set_all_seeds(s) at the top of every training script.
"""
import os
import random
import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Lock RNG state across Python, NumPy, and PyTorch (CPU + CUDA + cuDNN)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
