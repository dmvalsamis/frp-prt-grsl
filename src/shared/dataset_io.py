"""Data loading for the PRT (Persistence-Residual Transformer) pipeline.

Reads the assembled dataset cube ``dataset_v3.h5`` plus its manifest and
train-derived normalization statistics.

Path resolution (in order):
  1. Environment variable ``FRP_DATA_DIR`` (point this at the directory that
     holds ``dataset_v3.h5``); falls back to
  2. the repository ``data/`` directory (two levels up from this file).
The licensed source cube ``dataset_v3.h5`` is NOT redistributed in this repo
(see ``data/README.md`` for acquisition). The manifest and normalization JSON
ARE shipped under ``data/``.

TEST-SET SEAL: the held-out test split is governed by the SHA-256-anchored
access protocol (``configs/test_access_protocol.md``); the project's cumulative
test-set accesses are fixed at 3. ``load_split('test')`` raises by default and
must be explicitly unsealed with ``unsealed=True``.
"""
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


_ENV_DIR = os.environ.get("FRP_DATA_DIR")
DATA_DIR = Path(_ENV_DIR) if _ENV_DIR else (Path(__file__).resolve().parents[2] / "data")
H5_PATH = DATA_DIR / "dataset_v3.h5"
MANIFEST_PATH = DATA_DIR / "dataset_v3_manifest.csv"
NORM_PATH = DATA_DIR / "normalization_params_v3.json"

FRP_LAG_1H_IDX = 5  # feature index of the frp_lag_1h channel in the input tensor

VAL_EVENTS = [
    "alexandroupolis_2023",
    "caceres_2023",
    "mandra_2023",
    "rhodes_2023",
    "split_2017",
]
TEST_EVENTS = ["corinthos_2024", "valmaior_2024", "varnavas_2024", "volos_2023"]


def load_split(split: str, unsealed: bool = False):
    """Load (inputs, targets) for a split. Returns numpy arrays."""
    if split == "test" and not unsealed:
        raise RuntimeError(
            "Test split is sealed. Held-out test access is budgeted and "
            "pre-declared (see configs/test_access_protocol.md). Pass "
            "unsealed=True only for one of the three authorized evaluations."
        )
    with h5py.File(H5_PATH, "r") as f:
        inputs = f[f"{split}/inputs"][:]
        targets = f[f"{split}/targets"][:]
    return inputs.astype(np.float32), targets.astype(np.float32)


def load_metadata():
    with h5py.File(H5_PATH, "r") as f:
        return json.loads(f["metadata"][()])


def load_manifest(split: str | None = None):
    df = pd.read_csv(MANIFEST_PATH)
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    return df


def load_norm_params():
    with open(NORM_PATH) as f:
        return json.load(f)


def get_frp_norm():
    """Returns (mean, std) of frp_lag_1h used to denormalize persistence."""
    p = load_norm_params()
    return float(p["frp_lag_1h"]["mean"]), float(p["frp_lag_1h"]["std"])
