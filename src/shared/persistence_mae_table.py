"""Precompute MAE_persistence[event, horizon] on fire-active train samples.

The SSAT loss divides the model's per-event MAE by this table to mirror the
evaluation metric (SS = 1 - MAE_model / MAE_persistence).

Computed ONCE before training and held fixed. Persistence here means the same
thing as in the eval harness: predict frp_lag_1h (denormalized) for all
horizons.
"""
import numpy as np
import pandas as pd

from .dataset_io import FRP_LAG_1H_IDX, get_frp_norm


def compute_persistence_mae_table(
    inputs: np.ndarray,
    targets: np.ndarray,
    manifest: pd.DataFrame,
    horizon: int = 12,
    fire_active_only: bool = True,
) -> dict:
    """Compute MAE_persistence[event][horizon_idx] in raw MW.

    Args:
        inputs: (N, L, F) z-score normalized; frp_lag_1h is index 5.
        targets: (N, horizon) raw MW.
        manifest: DataFrame with event_name column, indexed same as inputs/targets.
        horizon: number of forecast steps.
        fire_active_only: if True, restrict to samples where ANY target step > 0.

    Returns:
        dict[event_name] -> np.ndarray of shape (horizon,) with MAE per horizon.
    """
    frp_mean, frp_std = get_frp_norm()
    last_ts = inputs.shape[1] - 1

    # Persistence prediction: denormalized frp_lag_1h at the last lookback step,
    # broadcast across all horizons.
    pers_raw = np.clip(
        inputs[:, last_ts, FRP_LAG_1H_IDX] * frp_std + frp_mean, 0.0, None
    )  # (N,)
    pers_pred = np.broadcast_to(pers_raw[:, None], targets.shape).copy()  # (N, H)

    abs_err = np.abs(pers_pred - targets).astype(np.float32)  # (N, H)

    if fire_active_only:
        mask = (targets > 0).any(axis=1)
    else:
        mask = np.ones(targets.shape[0], dtype=bool)

    table = {}
    for event in manifest["event_name"].unique():
        ev_mask = (manifest["event_name"].values == event) & mask
        if ev_mask.sum() == 0:
            continue
        table[event] = abs_err[ev_mask].mean(axis=0)  # (H,)
    return table


def summarize_table(table: dict, horizon_1based: int = 6) -> None:
    """Print a sanity summary at a specific horizon (default t+6)."""
    h_idx = horizon_1based - 1
    print(f"  Persistence MAE table at t+{horizon_1based} (fire-active, raw MW):")
    for event, mae in sorted(table.items(), key=lambda kv: kv[1][h_idx]):
        print(f"    {event:32s}: {mae[h_idx]:>9.1f} MW")
