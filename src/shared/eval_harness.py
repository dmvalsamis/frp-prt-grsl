"""Val-only evaluation harness for the PRT redesign.

Computes per-event SS_fire_t6 (and other reporting horizons) on the val split.
Returns event-level skill scores, cross-event mean/std, and the same metrics
excluding caceres_2023 (the structural OOD event per evaluation-standards).

TEST split is NOT touched here.
"""
import numpy as np
import pandas as pd
import torch


REPORT_HORIZONS_1BASED = [1, 3, 6, 12]


@torch.no_grad()
def predict_split(model, inputs: np.ndarray, batch_size: int = 256, device=None) -> np.ndarray:
    """Run model in eval mode on a split; return (N, H) raw-MW predictions."""
    model.eval()
    X = torch.tensor(inputs, dtype=torch.float32)
    out = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i : i + batch_size]
        if device is not None:
            xb = xb.to(device)
        pb = model(xb).detach().cpu().numpy()
        out.append(pb)
    return np.concatenate(out, axis=0).astype(np.float32)


def compute_per_event_ss(
    preds: np.ndarray,
    targets: np.ndarray,
    persistence_pred: np.ndarray,
    manifest: pd.DataFrame,
    events: list[str],
    horizon_1based: int = 6,
) -> dict:
    """Compute per-event SS_fire on fire-active samples at a single horizon.

    SS = 1 - MAE_model / MAE_persistence, evaluated per-event.

    Args:
        preds: (N, H) raw MW.
        targets: (N, H) raw MW.
        persistence_pred: (N, H) raw MW (same persistence pred used as denominator).
        manifest: DataFrame with event_name column, len N.
        events: list of event names to evaluate.
        horizon_1based: which horizon (1-based) to report SS at.

    Returns:
        dict[event] -> {"n_fire": int, "ss": float, "mae_model": float, "mae_pers": float}
        For events with n_fire == 0, ss is NaN.
    """
    h = horizon_1based - 1
    out = {}
    ev_arr = manifest["event_name"].values
    fire = (targets > 0).any(axis=1)
    for ev in events:
        m = (ev_arr == ev) & fire
        n_fire = int(m.sum())
        if n_fire == 0:
            out[ev] = {"n_fire": 0, "ss": float("nan"), "mae_model": float("nan"), "mae_pers": float("nan")}
            continue
        mae_m = float(np.abs(preds[m, h] - targets[m, h]).mean())
        mae_p = float(np.abs(persistence_pred[m, h] - targets[m, h]).mean())
        ss = 1.0 - mae_m / mae_p if mae_p > 0 else float("nan")
        out[ev] = {"n_fire": n_fire, "ss": ss, "mae_model": mae_m, "mae_pers": mae_p}
    return out


def aggregate_ss(per_event: dict, exclude: list[str] | None = None) -> dict:
    """Cross-event mean/std of SS_fire, optionally excluding a list of events."""
    exclude = set(exclude or [])
    vals = [v["ss"] for ev, v in per_event.items() if ev not in exclude and not np.isnan(v["ss"])]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n_events": 0}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "n_events": len(vals),
    }


def make_persistence_pred(inputs: np.ndarray, frp_mean: float, frp_std: float,
                           frp_lag_1h_idx: int = 5, horizon: int = 12) -> np.ndarray:
    """Build the persistence prediction array (N, H) used as SS denominator."""
    last_ts = inputs.shape[1] - 1
    pers_raw = np.clip(inputs[:, last_ts, frp_lag_1h_idx] * frp_std + frp_mean, 0.0, None)
    return np.broadcast_to(pers_raw[:, None], (inputs.shape[0], horizon)).astype(np.float32)
