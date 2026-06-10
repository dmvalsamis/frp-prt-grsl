"""Tier-1 engineered wind features.

Computes 5 derived per-timestep features from existing raw channels, z-scores
them with train-only statistics, and appends them to the input array.

Input: (N, L, 32) z-scored inputs (existing dataset_v3 feature layout).
Output: (N, L, 37) with 5 new features appended.

The 5 features (all per-timestep, all eventually z-scored):

  1. shear_signed     : ws850(t) - ws10(t)               [m/s]
       Vertical wind-speed difference between 850 hPa and 10 m. Positive
       means aloft winds dominate — the synoptic Etesian regime that
       Phase 5A flagged as the rhodes failure mode.

  2. wind_var_12h     : std12h(u10) + std12h(v10)        [m/s]
       Rolling 12-hour standard deviation of surface wind components.
       Captures gustiness / turbulence without the circular-statistics
       trap of direct wind-direction variance.

  3. ws850_max_12h    : max12h(ws850)                    [m/s]
       Rolling 12-hour max of upper-level wind speed. Captures peak-vs-mean
       asymmetry — sustained high winds vs transient peaks.

  4. hdw_product      : vpd_kPa(t) * ws10(t)             [kPa·m/s]
       Hot-dry-windy product. Classical blowup indicator.

  5. wind_frp_coupling: ws850(t) * frp_lag_1h(t)         [m/s · MW]
       Wind-fire joint condition. "Already burning AND it's windy."

Rolling windows use min(available, 12) — at the start of the lookback
where fewer than 12 hours are available, the rolling stat uses what is
available. No look-ahead.
"""
from __future__ import annotations

import numpy as np


# Indices into the existing 32-feature layout (from dataset-io skill)
FRP_LAG_1H_IDX = 5
U10_IDX = 18
V10_IDX = 19
WS10_IDX = 25
WS850_IDX = 27
VPD_IDX = 28

NEW_FEATURE_NAMES = [
    "shear_signed",
    "wind_var_12h",
    "ws850_max_12h",
    "hdw_product",
    "wind_frp_coupling",
]


def _denormalize(inputs_z: np.ndarray, idx: int, name: str, norm_params: dict) -> np.ndarray:
    m = float(norm_params[name]["mean"])
    s = float(norm_params[name]["std"])
    return inputs_z[:, :, idx] * s + m


def _rolling_max(x: np.ndarray, w: int = 12) -> np.ndarray:
    """Causal rolling max over time axis. x: (N, L) -> (N, L)."""
    N, L = x.shape
    out = np.empty_like(x)
    for t in range(L):
        start = max(0, t - w + 1)
        out[:, t] = x[:, start : t + 1].max(axis=1)
    return out


def _rolling_std_sum(u: np.ndarray, v: np.ndarray, w: int = 12) -> np.ndarray:
    """Causal rolling (std(u) + std(v)) over time axis."""
    N, L = u.shape
    out = np.zeros_like(u)
    for t in range(L):
        start = max(0, t - w + 1)
        seg_u = u[:, start : t + 1]
        seg_v = v[:, start : t + 1]
        if seg_u.shape[1] > 1:
            out[:, t] = seg_u.std(axis=1) + seg_v.std(axis=1)
        # else 0 (only one timestep visible; no variance)
    return out


def _compute_raw_new_features(
    inputs_z: np.ndarray, norm_params: dict
) -> np.ndarray:
    """Compute the 5 features in physical units. Shape (N, L, 5)."""
    u10 = _denormalize(inputs_z, U10_IDX, "u10", norm_params)
    v10 = _denormalize(inputs_z, V10_IDX, "v10", norm_params)
    ws10 = _denormalize(inputs_z, WS10_IDX, "wind_speed_10m", norm_params)
    ws850 = _denormalize(inputs_z, WS850_IDX, "wind_speed_850", norm_params)
    vpd = _denormalize(inputs_z, VPD_IDX, "vpd_kPa", norm_params)
    frp_lag = _denormalize(inputs_z, FRP_LAG_1H_IDX, "frp_lag_1h", norm_params)
    frp_lag_pos = np.clip(frp_lag, 0.0, None)

    f1_shear = ws850 - ws10
    f2_wind_var = _rolling_std_sum(u10, v10, w=12)
    f3_ws850_max = _rolling_max(ws850, w=12)
    f4_hdw = vpd * ws10
    f5_wind_frp = ws850 * frp_lag_pos

    return np.stack([f1_shear, f2_wind_var, f3_ws850_max, f4_hdw, f5_wind_frp], axis=2)


def build_wind_features(
    inputs_z: np.ndarray,
    norm_params: dict,
    train_stats: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Extend (N, L, 32) -> (N, L, 37) with z-scored derived wind features.

    Args:
        inputs_z: (N, L, 32) z-scored inputs.
        norm_params: load_norm_params() dict — used to denormalize raw channels.
        train_stats: dict from a prior call on TRAIN data containing 'means'
            and 'stds' arrays of shape (5,). If None, computed from this data
            (only do this for the train split) and returned.

    Returns:
        (extended_inputs (N, L, 37) float32, train_stats dict).
    """
    raw_new = _compute_raw_new_features(inputs_z, norm_params)  # (N, L, 5)

    if train_stats is None:
        flat = raw_new.reshape(-1, 5)
        means = flat.mean(axis=0)
        stds = flat.std(axis=0)
        stds = np.where(stds < 1e-6, 1.0, stds)
        train_stats = {
            "feature_names": NEW_FEATURE_NAMES,
            "means": means.tolist(),
            "stds": stds.tolist(),
        }
    else:
        means = np.array(train_stats["means"], dtype=np.float32)
        stds = np.array(train_stats["stds"], dtype=np.float32)

    new_z = ((raw_new - means[None, None, :]) / stds[None, None, :]).astype(np.float32)
    extended = np.concatenate([inputs_z.astype(np.float32), new_z], axis=2)
    return extended, train_stats
