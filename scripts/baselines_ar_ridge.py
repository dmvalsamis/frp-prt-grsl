"""
09_baselines_v3.py — Phase 3A (v3): Baseline Establishment on Val Split of dataset_v3.h5

Val split: 5 events — alexandroupolis_2023, caceres_2023, mandra_2023,
                       rhodes_2023, split_2017

Key differences from 09_baselines_v2.py:
  - Reads dataset_v3.h5 + dataset_v3_manifest.csv + normalization_params_v3.json
  - Outputs: baselines_v3_val_results.json, baselines_v3_val_comparison.png
  - Prints three-way comparison table: v1 (4 events), v2 (5 events), v3 (5 events)
  - Training set is now 25 events (up from 21 in v2) — norm params differ
"""

import json
import warnings
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
HDF5_PATH = ROOT / "data/processed/dataset_v3.h5"
MANIFEST_PATH = ROOT / "data/processed/dataset_v3_manifest.csv"
NORM_PATH = ROOT / "data/processed/normalization_params_v3.json"
V1_RESULTS_PATH = ROOT / "data/processed/baselines_val_results.json"
V2_RESULTS_PATH = ROOT / "data/processed/baselines_v2_val_results.json"
OUT_JSON = ROOT / "data/processed/baselines_v3_val_results.json"
OUT_CSV = ROOT / "data/processed/baselines_v3_val_table.csv"
OUT_PLOT = ROOT / "data/plots/baselines_v3_val_comparison.png"

# ---------------------------------------------------------------------------
# Feature index map (canonical — same 32 features as v1/v2)
# ---------------------------------------------------------------------------
FEAT = {
    "hour_sin": 0, "hour_cos": 1, "doy_sin": 2, "doy_cos": 3,
    "log_frp": 4,
    "frp_lag_1h": 5, "frp_lag_2h": 6, "frp_lag_3h": 7,
    "frp_lag_6h": 8, "frp_lag_12h": 9, "frp_lag_24h": 10,
    "frp_rolling_6h_mean": 11, "frp_rolling_6h_std": 12,
    "frp_trend_6h": 13, "cumulative_frp": 14, "hours_since_first_fire": 15,
    "is_fire": 16, "sat_fraction": 17,
    "u10": 18, "v10": 19, "t2m_K": 20, "d2m_K": 21, "blh_m": 22,
    "u850": 23, "v850": 24, "wind_speed_10m": 25, "wind_dir_10m": 26,
    "wind_speed_850": 27, "vpd_kPa": 28,
    "precip_1h_mm": 29, "precip_24h_mm": 30, "delta_wind_3h": 31,
}
LAST_TS = 47  # index of last timestep in 48h lookback window (0-indexed)

# v3 val split has 5 events (unchanged from v2)
VAL_EVENTS = [
    "alexandroupolis_2023",
    "caceres_2023",
    "mandra_2023",
    "rhodes_2023",
    "split_2017",
]
HORIZONS = [1, 3, 6, 12]  # t+k steps to report


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data():
    print("Loading HDF5 and manifest ...")
    with h5py.File(HDF5_PATH, "r") as f:
        h5_meta = json.loads(f["metadata"][()])
        feature_names = h5_meta["feature_names"]
        actual_lookback = h5_meta.get("LOOKBACK", "unknown")
        print(f"  HDF5 LOOKBACK={actual_lookback}  HORIZON={h5_meta.get('HORIZON', '?')}")
        print(f"  Features ({len(feature_names)}): {feature_names[:5]}...")

        # Verify feature index map matches actual feature order
        for feat, idx in FEAT.items():
            if idx < len(feature_names) and feature_names[idx] != feat:
                print(f"  WARNING: FEAT['{feat}']={idx} but HDF5 pos {idx} = '{feature_names[idx]}'")

        train_inputs = f["train/inputs"][:]
        train_targets = f["train/targets"][:]
        val_inputs = f["val/inputs"][:]
        val_targets = f["val/targets"][:]

    manifest = pd.read_csv(MANIFEST_PATH)
    manifest["input_start_utc"] = pd.to_datetime(manifest["input_start_utc"])
    manifest["target_end_utc"] = pd.to_datetime(manifest["target_end_utc"])
    # t = last observed timestep = target_end_utc - HORIZON hours
    manifest["t_last"] = manifest["target_end_utc"] - pd.Timedelta(hours=12)

    train_manifest = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val_manifest = manifest[manifest["split"] == "val"].reset_index(drop=True)

    with open(NORM_PATH) as f:
        norm_params = json.load(f)

    # Verify val events present in manifest
    val_events_in_manifest = sorted(val_manifest["event_name"].unique())
    print(f"  Val events in manifest: {val_events_in_manifest}")
    for ev in VAL_EVENTS:
        if ev not in val_events_in_manifest:
            print(f"  ERROR: {ev} not found in val manifest!")

    return train_inputs, train_targets, val_inputs, val_targets, train_manifest, val_manifest, norm_params


# ---------------------------------------------------------------------------
# Denormalization helpers
# ---------------------------------------------------------------------------
def denorm(arr, feature_name, norm_params):
    """Denormalize a 1-D or N-D array using stored mean/std."""
    p = norm_params[feature_name]
    return arr * p["std"] + p["mean"]


def get_val_feature(val_inputs, feature_name, timestep=LAST_TS):
    """Extract normalized feature at given timestep; shape (N,)."""
    return val_inputs[:, timestep, FEAT[feature_name]]


def get_val_feature_raw(val_inputs, feature_name, norm_params, timestep=LAST_TS):
    """Extract and denormalize feature at given timestep; shape (N,)."""
    normed = get_val_feature(val_inputs, feature_name, timestep)
    return denorm(normed, feature_name, norm_params)


# ---------------------------------------------------------------------------
# Fire-active mask
# ---------------------------------------------------------------------------
def fire_active_mask(targets):
    """True where at least one target step is > 0."""
    return targets.max(axis=1) > 0.0


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------
def mae(pred, true):
    return np.mean(np.abs(pred - true))


def rmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2))


def compute_metrics_per_event(preds, targets, val_manifest, norm_params):
    """
    preds: (N, 12) float32 predictions in raw MW
    targets: (N, 12) float32 raw MW
    Returns dict keyed by event name.
    """
    fire_mask = fire_active_mask(targets)
    results = {}

    for event in VAL_EVENTS:
        ev_mask = val_manifest["event_name"] == event
        ev_idx = ev_mask[ev_mask].index.to_numpy()

        if len(ev_idx) == 0:
            print(f"  WARNING: {event} has 0 samples in val manifest — skipping")
            results[event] = {
                "MAE_all": {f"+{k}": float("nan") for k in range(1, 13)},
                "MAE_fire": {f"+{k}": float("nan") for k in range(1, 13)},
                "RMSE": {f"+{k}": float("nan") for k in [6, 12]},
                "n_samples_all": 0,
                "n_samples_fire": 0,
            }
            continue

        p_ev = preds[ev_idx]
        t_ev = targets[ev_idx]
        fm_ev = fire_mask[ev_idx]

        n_all = len(ev_idx)
        n_fire = int(fm_ev.sum())

        mae_all = {}
        mae_fire = {}
        rmse_dict = {}

        for k in range(1, 13):
            step = k - 1
            p_k = p_ev[:, step]
            t_k = t_ev[:, step]

            mae_all[f"+{k}"] = float(mae(p_k, t_k))

            if n_fire > 0:
                mae_fire[f"+{k}"] = float(mae(p_k[fm_ev], t_k[fm_ev]))
            else:
                mae_fire[f"+{k}"] = float("nan")

        for k in [6, 12]:
            step = k - 1
            p_k = p_ev[:, step]
            t_k = t_ev[:, step]
            rmse_dict[f"+{k}"] = float(rmse(p_k, t_k))

        results[event] = {
            "MAE_all": mae_all,
            "MAE_fire": mae_fire,
            "RMSE": rmse_dict,
            "n_samples_all": n_all,
            "n_samples_fire": n_fire,
        }

    return results


def aggregate_metrics(per_event):
    """Compute mean ± std (ddof=1) across all val events for all 12 steps."""
    agg = {}
    for metric_key in ["MAE_all", "MAE_fire"]:
        mean_key = f"{metric_key}_mean"
        std_key = f"{metric_key}_std"
        agg[mean_key] = {}
        agg[std_key] = {}
        for k in range(1, 13):
            step_key = f"+{k}"
            vals = []
            for event in VAL_EVENTS:
                v = per_event[event][metric_key].get(step_key, float("nan"))
                if not np.isnan(v):
                    vals.append(v)
            agg[mean_key][step_key] = float(np.mean(vals)) if vals else float("nan")
            agg[std_key][step_key] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")

    return agg


def compute_skill_scores(per_event, persistence_per_event):
    """SS = 1 - MAE_baseline / MAE_persistence, at t+6, MAE_fire."""
    ss_per_event = {}
    for event in VAL_EVENTS:
        base_mae = per_event[event]["MAE_fire"].get("+6", float("nan"))
        pers_mae = persistence_per_event[event]["MAE_fire"].get("+6", float("nan"))
        if np.isnan(base_mae) or np.isnan(pers_mae) or pers_mae == 0:
            ss_per_event[event] = float("nan")
        else:
            ss_per_event[event] = float(1.0 - base_mae / pers_mae)

    # Mean, std, excl caceres
    vals_all = [v for v in ss_per_event.values() if not np.isnan(v)]
    vals_excl = [ss_per_event[e] for e in VAL_EVENTS
                 if e != "caceres_2023" and not np.isnan(ss_per_event.get(e, float("nan")))]
    return {
        "per_event": ss_per_event,
        "mean": float(np.mean(vals_all)) if vals_all else float("nan"),
        "std": float(np.std(vals_all, ddof=1)) if len(vals_all) > 1 else float("nan"),
        "excl_caceres": float(np.mean(vals_excl)) if vals_excl else float("nan"),
    }


# ---------------------------------------------------------------------------
# Baseline 1: Persistence
# ---------------------------------------------------------------------------
def baseline_persistence(val_inputs, val_manifest, norm_params):
    print("  Baseline 1: Persistence")
    frp_t = get_val_feature_raw(val_inputs, "frp_lag_1h", norm_params)
    is_fire = val_inputs[:, LAST_TS, FEAT["is_fire"]]

    N = len(frp_t)
    preds = np.zeros((N, 12), dtype=np.float32)
    for k in range(12):
        pred_k = np.where(is_fire > 0, frp_t, 0.0)
        preds[:, k] = np.clip(pred_k, 0, None)

    return preds


# ---------------------------------------------------------------------------
# Baseline 2: Diurnal Persistence (frp_lag_24h)
# ---------------------------------------------------------------------------
def baseline_diurnal_persistence(val_inputs, norm_params):
    print("  Baseline 2: Diurnal Persistence")
    frp_24h = get_val_feature_raw(val_inputs, "frp_lag_24h", norm_params)

    N = len(frp_24h)
    preds = np.zeros((N, 12), dtype=np.float32)
    for k in range(12):
        preds[:, k] = np.clip(frp_24h, 0, None)

    return preds


# ---------------------------------------------------------------------------
# Baseline 3: Diurnal Climatology
# ---------------------------------------------------------------------------
def baseline_diurnal_climatology(train_inputs, train_targets, train_manifest,
                                  val_inputs, val_manifest, norm_params):
    print("  Baseline 3: Diurnal Climatology")
    fire_mask_train = fire_active_mask(train_targets)
    train_fire_idx = np.where(fire_mask_train)[0]

    hour_frp = {h: [] for h in range(24)}

    for idx in train_fire_idx:
        t_last = train_manifest.iloc[idx]["t_last"]
        for k in range(1, 13):
            target_ts = t_last + pd.Timedelta(hours=k)
            h = target_ts.hour
            hour_frp[h].append(float(train_targets[idx, k - 1]))

    clim = {}
    all_vals = [v for vals in hour_frp.values() for v in vals]
    overall_mean = float(np.mean(all_vals)) if all_vals else 0.0
    for h in range(24):
        clim[h] = float(np.mean(hour_frp[h])) if hour_frp[h] else overall_mean

    # Apply to val
    N = val_inputs.shape[0]
    preds = np.zeros((N, 12), dtype=np.float32)
    for i in range(N):
        t_last = val_manifest.iloc[i]["t_last"]
        for k in range(1, 13):
            target_ts = t_last + pd.Timedelta(hours=k)
            h = target_ts.hour
            preds[i, k - 1] = clim[h]

    return preds


# ---------------------------------------------------------------------------
# Baseline 4: Linear Trend Extrapolation
# ---------------------------------------------------------------------------
def baseline_linear_trend(val_inputs, norm_params):
    print("  Baseline 4: Linear Trend Extrapolation")
    lag1 = get_val_feature_raw(val_inputs, "frp_lag_1h", norm_params)
    lag2 = get_val_feature_raw(val_inputs, "frp_lag_2h", norm_params)
    lag3 = get_val_feature_raw(val_inputs, "frp_lag_3h", norm_params)
    lag6 = get_val_feature_raw(val_inputs, "frp_lag_6h", norm_params)

    x = np.array([-5.0, -2.0, -1.0, 0.0])
    N = len(lag1)
    preds = np.zeros((N, 12), dtype=np.float32)

    for i in range(N):
        y = np.array([lag6[i], lag3[i], lag2[i], lag1[i]])
        if np.all(y == 0):
            continue
        y = np.clip(y, 0, None)
        coef = np.polyfit(x, y, deg=1)
        slope = coef[0]
        frp_t = lag1[i]
        for k in range(1, 13):
            pred_k = slope * k + frp_t
            preds[i, k - 1] = float(max(0.0, pred_k))

    return preds


# ---------------------------------------------------------------------------
# Baseline 5: ERA5-Only Ridge Regression
# ---------------------------------------------------------------------------
ERA5_FEAT_NAMES = ["u10", "v10", "t2m_K", "d2m_K", "blh_m",
                   "wind_speed_10m", "vpd_kPa", "precip_24h_mm", "delta_wind_3h"]
ERA5_FEAT_IDX = [FEAT[n] for n in ERA5_FEAT_NAMES]


def baseline_era5_ridge(train_inputs, train_targets, val_inputs):
    print("  Baseline 5: ERA5-Only Ridge Regression")
    X_train = train_inputs[:, LAST_TS, :][:, ERA5_FEAT_IDX]
    X_val = val_inputs[:, LAST_TS, :][:, ERA5_FEAT_IDX]

    N_val = val_inputs.shape[0]
    preds = np.zeros((N_val, 12), dtype=np.float32)

    for k in range(12):
        y_train = train_targets[:, k]
        ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5)
        ridge.fit(X_train, y_train)
        preds[:, k] = np.clip(ridge.predict(X_val), 0, None)
        print(f"    step t+{k+1}: best alpha={ridge.alpha_:.1f}")

    return preds


# ---------------------------------------------------------------------------
# Baseline 6: FRP-Only AR Ridge Regression
# ---------------------------------------------------------------------------
AR_FEAT_NAMES = ["frp_lag_1h", "frp_lag_2h", "frp_lag_3h",
                 "frp_lag_6h", "frp_lag_12h", "frp_lag_24h"]
AR_FEAT_IDX = [FEAT[n] for n in AR_FEAT_NAMES]


def baseline_ar_ridge(train_inputs, train_targets, val_inputs):
    print("  Baseline 6: FRP-Only AR Ridge Regression")
    X_train = train_inputs[:, LAST_TS, :][:, AR_FEAT_IDX]
    X_val = val_inputs[:, LAST_TS, :][:, AR_FEAT_IDX]

    N_val = val_inputs.shape[0]
    preds = np.zeros((N_val, 12), dtype=np.float32)

    for k in range(12):
        y_train = train_targets[:, k]
        ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5)
        ridge.fit(X_train, y_train)
        preds[:, k] = np.clip(ridge.predict(X_val), 0, None)
        print(f"    step t+{k+1}: best alpha={ridge.alpha_:.1f}")

    return preds


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
def run_sanity_checks(all_results, val_targets, val_manifest):
    print("\n=== SANITY CHECKS ===")
    fire_mask = fire_active_mask(val_targets)

    mae_fire_t1 = {
        name: results["aggregate"]["MAE_fire_mean"]["+1"]
        for name, results in all_results.items()
    }
    pers_t1 = mae_fire_t1["persistence"]
    print(f"\n[1] MAE_fire at t+1 (all baselines):")
    for name, v in mae_fire_t1.items():
        flag = " ← persistence" if name == "persistence" else ""
        print(f"    {name:35s}: {v:,.1f} MW{flag}")
    min_name = min(mae_fire_t1, key=mae_fire_t1.get)
    if min_name != "persistence":
        print(f"  WARNING: persistence is NOT lowest at t+1 (lowest: {min_name})")
    else:
        print("  OK: persistence is lowest at t+1")

    era5_ss = all_results["era5_ridge"]["aggregate"]["SS_fire_t6_mean"]
    ar_ss = all_results["ar_ridge"]["aggregate"]["SS_fire_t6_mean"]
    print(f"\n[2] SS_fire at t+6: ERA5={era5_ss:.4f}, AR={ar_ss:.4f}")
    if era5_ss >= ar_ss:
        print("  WARNING: ERA5 SS >= AR SS — ERA5 alone outpredicts FRP history")
    else:
        print("  OK: AR SS > ERA5 SS (FRP history is more informative than ERA5 alone)")

    fire_vals = val_targets[fire_mask]
    mean_fire_frp_all_steps = float(np.mean(fire_vals))
    print(f"\n[3] Mean frp_sum_mw (fire-active samples, all steps): {mean_fire_frp_all_steps:,.1f} MW")

    print("\n[4] Per-event sample counts:")
    for event in VAL_EVENTS:
        ev_mask = val_manifest["event_name"] == event
        n_all = ev_mask.sum()
        ev_idx = ev_mask[ev_mask].index.to_numpy()
        n_fire = int(fire_mask[ev_idx].sum())
        markers = ""
        if event == "caceres_2023":
            markers = " *** CACERES (distributional outlier)"
        elif event == "split_2017":
            markers = " *** (Bora regime)"
        print(f"    {event:30s}: n_all={n_all:4d}, n_fire={n_fire:4d}{markers}")

    print("\n=== END SANITY CHECKS ===\n")


# ---------------------------------------------------------------------------
# Build full results dict for one baseline
# ---------------------------------------------------------------------------
def build_results(name, preds, val_targets, val_manifest, norm_params,
                  persistence_per_event=None):
    per_event = compute_metrics_per_event(preds, val_targets, val_manifest, norm_params)
    agg = aggregate_metrics(per_event)

    if persistence_per_event is not None:
        ss = compute_skill_scores(per_event, persistence_per_event)
        for event in VAL_EVENTS:
            per_event[event]["SS_fire_t6"] = ss["per_event"][event]
        agg["SS_fire_t6_mean"] = ss["mean"]
        agg["SS_fire_t6_std"] = ss["std"]
        agg["SS_fire_t6_excl_caceres"] = ss["excl_caceres"]
    else:
        for event in VAL_EVENTS:
            per_event[event]["SS_fire_t6"] = 0.0
        agg["SS_fire_t6_mean"] = 0.0
        agg["SS_fire_t6_std"] = 0.0
        agg["SS_fire_t6_excl_caceres"] = 0.0

    return {"per_event": per_event, "aggregate": agg}


# ---------------------------------------------------------------------------
# Diagnostic plot
# ---------------------------------------------------------------------------
def make_plot(all_results, val_targets, val_manifest):
    print("Generating diagnostic plot ...")
    fire_mask = fire_active_mask(val_targets)
    baseline_names = list(all_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(baseline_names)))

    display_names = {
        "persistence": "B1 Persistence",
        "diurnal_persistence": "B2 Diurnal Pers.",
        "diurnal_climatology": "B3 Diurnal Clim.",
        "linear_trend": "B4 Linear Trend",
        "era5_ridge": "B5 ERA5 Ridge",
        "ar_ridge": "B6 AR Ridge",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Phase 3A (v3): Baseline Comparison — Val Split (5 events, LOOKBACK=48)",
        fontsize=12
    )

    # Panel 1: MAE_fire vs horizon (mean across val events)
    ax1 = axes[0, 0]
    for (name, results), color in zip(all_results.items(), colors):
        mae_means = [results["aggregate"]["MAE_fire_mean"].get(f"+{k}", np.nan)
                     for k in range(1, 13)]
        ax1.plot(range(1, 13), mae_means, marker="o", label=display_names.get(name, name),
                 color=color, linewidth=1.8, markersize=4)
    ax1.set_xlabel("Horizon (hours ahead)")
    ax1.set_ylabel("MAE_fire (MW)")
    ax1.set_title(f"MAE_fire vs Horizon (mean, {len(VAL_EVENTS)} val events)")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, 13))

    # Panel 2: Per-event MAE_fire at t+6 (grouped bar)
    ax2 = axes[0, 1]
    n_baselines = len(baseline_names)
    n_events = len(VAL_EVENTS)
    bar_width = 0.12
    x = np.arange(n_events)
    for b_idx, (name, results) in enumerate(all_results.items()):
        vals = [results["per_event"][ev]["MAE_fire"].get("+6", 0) for ev in VAL_EVENTS]
        offset = (b_idx - n_baselines / 2 + 0.5) * bar_width
        ax2.bar(x + offset, vals, width=bar_width,
                label=display_names.get(name, name), color=colors[b_idx], alpha=0.85)
    ax2.set_xticks(x)
    short_names = [ev.replace("_2023", "").replace("_2024", "").replace("_2017", "") for ev in VAL_EVENTS]
    ax2.set_xticklabels(short_names, rotation=15, fontsize=8)
    ax2.set_ylabel("MAE_fire at t+6 (MW)")
    ax2.set_title("Per-Event MAE_fire at t+6 (5 val events)")
    ax2.legend(fontsize=6, loc="upper right")
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: Skill score vs horizon
    ax3 = axes[1, 0]
    pers_results = all_results["persistence"]
    for (name, results), color in zip(all_results.items(), colors):
        if name == "persistence":
            continue
        ss_vals = []
        for k in range(1, 13):
            step_key = f"+{k}"
            base_maes = [results["per_event"][ev]["MAE_fire"][step_key]
                         for ev in VAL_EVENTS
                         if not np.isnan(results["per_event"][ev]["MAE_fire"][step_key])]
            pers_maes = [pers_results["per_event"][ev]["MAE_fire"][step_key]
                         for ev in VAL_EVENTS
                         if not np.isnan(pers_results["per_event"][ev]["MAE_fire"][step_key])]
            if base_maes and pers_maes:
                ss = 1.0 - np.mean(base_maes) / np.mean(pers_maes)
            else:
                ss = float("nan")
            ss_vals.append(ss)
        ax3.plot(range(1, 13), ss_vals, marker="o",
                 label=display_names.get(name, name), color=color,
                 linewidth=1.8, markersize=4)
    ax3.axhline(0, color="black", linestyle="--", linewidth=1)
    ax3.set_xlabel("Horizon (hours ahead)")
    ax3.set_ylabel("Skill Score (SS_fire)")
    ax3.set_title("Skill Score vs Persistence (SS_fire, 5 val events)")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(range(1, 13))

    # Panel 4: caceres vs mean-of-other-4, best baseline
    ax4 = axes[1, 1]
    best_name = min(
        {n: r for n, r in all_results.items()},
        key=lambda n: all_results[n]["aggregate"]["MAE_fire_mean"].get("+6", float("inf"))
    )
    best_results = all_results[best_name]
    other_events = [e for e in VAL_EVENTS if e != "caceres_2023"]

    caceres_mae = [best_results["per_event"]["caceres_2023"]["MAE_fire"].get(f"+{k}", np.nan)
                   for k in range(1, 13)]
    other_maes = []
    for k in range(1, 13):
        vals = [best_results["per_event"][e]["MAE_fire"].get(f"+{k}", np.nan)
                for e in other_events if not np.isnan(
                    best_results["per_event"][e]["MAE_fire"].get(f"+{k}", np.nan))]
        other_maes.append(np.mean(vals) if vals else np.nan)

    ax4.plot(range(1, 13), caceres_mae, marker="s", color="red",
             label="caceres_2023", linewidth=1.8, markersize=5)
    ax4.plot(range(1, 13), other_maes, marker="o", color="steelblue",
             label=f"Mean ({len(other_events)} other events)", linewidth=1.8, markersize=5)
    ax4.set_xlabel("Horizon (hours ahead)")
    ax4.set_ylabel("MAE_fire (MW)")
    ax4.set_title(f"caceres_2023 vs Others — {display_names.get(best_name, best_name)}")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(range(1, 13))

    plt.tight_layout()
    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT_PLOT}")


# ---------------------------------------------------------------------------
# Build paper-ready CSV table
# ---------------------------------------------------------------------------
def build_csv_table(all_results):
    rows = []
    for name, results in all_results.items():
        agg = results["aggregate"]
        row = {"baseline": name}
        for k in HORIZONS:
            step = f"+{k}"
            row[f"MAE_fire_t{k}_mean"] = agg["MAE_fire_mean"].get(step, float("nan"))
            row[f"MAE_fire_t{k}_std"] = agg["MAE_fire_std"].get(step, float("nan"))
        row["SS_fire_t6_mean"] = agg.get("SS_fire_t6_mean", float("nan"))
        row["SS_fire_t6_std"] = agg.get("SS_fire_t6_std", float("nan"))
        row["SS_fire_t6_excl_caceres"] = agg.get("SS_fire_t6_excl_caceres", float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Three-way comparison table: v1 (4 events) vs v2 (5 events) vs v3 (5 events)
# ---------------------------------------------------------------------------
def print_v1_v2_v3_comparison(all_results_v3):
    print("\n" + "=" * 100)
    print("  THREE-WAY COMPARISON: dataset_v1 val (4 ev) vs dataset_v2 val (5 ev) vs dataset_v3 val (5 ev)")
    print("  Metric: SS_fire_t6 (skill score at t+6, fire-active samples)")
    print("=" * 100)

    v1_results = None
    if V1_RESULTS_PATH.exists():
        with open(V1_RESULTS_PATH) as f:
            v1_results = json.load(f)
    else:
        print(f"  NOTE: {V1_RESULTS_PATH} not found — v1 columns will show n/a")

    v2_results = None
    if V2_RESULTS_PATH.exists():
        with open(V2_RESULTS_PATH) as f:
            v2_results = json.load(f)
    else:
        print(f"  NOTE: {V2_RESULTS_PATH} not found — v2 columns will show n/a")

    baselines = ["persistence", "diurnal_persistence", "diurnal_climatology",
                 "linear_trend", "era5_ridge", "ar_ridge"]
    display_names = {
        "persistence":          "B1 Persistence",
        "diurnal_persistence":  "B2 Diurnal Pers.",
        "diurnal_climatology":  "B3 Diurnal Clim.",
        "linear_trend":         "B4 Linear Trend",
        "era5_ridge":           "B5 ERA5 Ridge",
        "ar_ridge":             "B6 AR Ridge",
    }

    def fmt(v):
        return f"{v:+.4f}" if not np.isnan(v) else "   n/a  "

    hdr = (f"  {'Baseline':<22} {'v1(4ev)':>10} {'v1(excl_c)':>11} "
           f"{'v2(5ev)':>10} {'v2(excl_c)':>11} "
           f"{'v3(5ev)':>10} {'v3(excl_c)':>11}  {'Dv2->v3':>9}")
    print(hdr)
    print("  " + "-" * 98)

    for name in baselines:
        dname = display_names.get(name, name)

        v1_ss = float("nan")
        v1_ss_excl = float("nan")
        if v1_results and name in v1_results:
            v1_ss = v1_results[name]["aggregate"].get("SS_fire_t6_mean", float("nan"))
            v1_ss_excl = v1_results[name]["aggregate"].get("SS_fire_t6_excl_caceres", float("nan"))

        v2_ss = float("nan")
        v2_ss_excl = float("nan")
        if v2_results and name in v2_results:
            v2_ss = v2_results[name]["aggregate"].get("SS_fire_t6_mean", float("nan"))
            v2_ss_excl = v2_results[name]["aggregate"].get("SS_fire_t6_excl_caceres", float("nan"))

        v3_ss = float("nan")
        v3_ss_excl = float("nan")
        if name in all_results_v3:
            v3_ss = all_results_v3[name]["aggregate"].get("SS_fire_t6_mean", float("nan"))
            v3_ss_excl = all_results_v3[name]["aggregate"].get("SS_fire_t6_excl_caceres", float("nan"))

        delta_v2_v3 = (v3_ss - v2_ss) if not (np.isnan(v3_ss) or np.isnan(v2_ss)) else float("nan")
        delta_str = f"{delta_v2_v3:+.4f}" if not np.isnan(delta_v2_v3) else "   n/a  "

        print(f"  {dname:<22} {fmt(v1_ss):>10} {fmt(v1_ss_excl):>11} "
              f"{fmt(v2_ss):>10} {fmt(v2_ss_excl):>11} "
              f"{fmt(v3_ss):>10} {fmt(v3_ss_excl):>11}  {delta_str:>9}")

    print()
    print("  NOTE: Dv2->v3 = v3(5ev) - v2(5ev). Same val events; larger training set (25 vs 21).")
    print("  TFT threshold commitment is SS_fire_t6 > +0.15 on test (unchanged).")
    print("=" * 100)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Phase 3A (v3) — Baseline Establishment (Val Split, 5 events)")
    print(f"  LAST_TS={LAST_TS} (48h lookback window last index)")
    print(f"  VAL_EVENTS: {VAL_EVENTS}")
    print("=" * 60)

    (train_inputs, train_targets, val_inputs, val_targets,
     train_manifest, val_manifest, norm_params) = load_data()

    print(f"\nData loaded:")
    print(f"  train: {train_inputs.shape}, targets: {train_targets.shape}")
    print(f"  val:   {val_inputs.shape}, targets: {val_targets.shape}")
    print(f"  val manifest: {len(val_manifest)} rows, events: {sorted(val_manifest['event_name'].unique())}")

    # -----------------------------------------------------------------------
    print("\n--- Computing baseline predictions ---")
    preds = {}

    preds["persistence"] = baseline_persistence(val_inputs, val_manifest, norm_params)
    preds["diurnal_persistence"] = baseline_diurnal_persistence(val_inputs, norm_params)
    preds["diurnal_climatology"] = baseline_diurnal_climatology(
        train_inputs, train_targets, train_manifest, val_inputs, val_manifest, norm_params)
    preds["linear_trend"] = baseline_linear_trend(val_inputs, norm_params)
    preds["era5_ridge"] = baseline_era5_ridge(train_inputs, train_targets, val_inputs)
    preds["ar_ridge"] = baseline_ar_ridge(train_inputs, train_targets, val_inputs)

    # -----------------------------------------------------------------------
    print("\n--- Computing metrics ---")
    all_results = {}

    all_results["persistence"] = build_results(
        "persistence", preds["persistence"], val_targets, val_manifest, norm_params,
        persistence_per_event=None
    )
    pers_per_event = all_results["persistence"]["per_event"]

    for name in ["diurnal_persistence", "diurnal_climatology", "linear_trend",
                 "era5_ridge", "ar_ridge"]:
        all_results[name] = build_results(
            name, preds[name], val_targets, val_manifest, norm_params,
            persistence_per_event=pers_per_event
        )

    # -----------------------------------------------------------------------
    run_sanity_checks(all_results, val_targets, val_manifest)

    # -----------------------------------------------------------------------
    print("--- Saving results ---")

    with open(OUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: {OUT_JSON}")

    table_df = build_csv_table(all_results)
    table_df.to_csv(OUT_CSV, index=False, float_format="%.2f")
    print(f"  Saved: {OUT_CSV}")

    print("\n=== PAPER-READY TABLE (MAE_fire, mean±std across 5 val events) ===")
    pd.set_option("display.float_format", "{:,.1f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 160)
    print(table_df.to_string(index=False))

    # -----------------------------------------------------------------------
    make_plot(all_results, val_targets, val_manifest)

    # -----------------------------------------------------------------------
    # Three-way comparison: v1 / v2 / v3
    print_v1_v2_v3_comparison(all_results)

    print("\n=== DONE ===")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
