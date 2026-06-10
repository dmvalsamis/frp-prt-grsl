"""
07_build_dataset_v3.py — Phase 2C (v3): Feature Engineering & Dataset Assembly
===============================================================================
Identical logic to 07_build_dataset_v2.py, with:
  - All outputs under dataset_v3 prefix
  - Normalization params saved to normalization_params_v3.json
  - Does NOT overwrite dataset_v1.h5, dataset_v2.h5, or their norm params

Incorporates all 34 events including Wave 2 additions:
  pedrogao_grande_2017, north_macedonia_2021, el_tarf_2022, castellon_2023
Split: 25 train / 5 val / 4 test
"""

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed"
CATALOGUE_CSV = PROCESSED / "event_catalogue.csv"
OUT_H5 = PROCESSED / "dataset_v3.h5"
OUT_MANIFEST = PROCESSED / "dataset_v3_manifest.csv"
OUT_NORM = PROCESSED / "normalization_params_v3.json"
OUT_STATS = PROCESSED / "dataset_v3_stats.json"

# ---------------------------------------------------------------------------
# Sequence parameters (unchanged from v2)
# ---------------------------------------------------------------------------
LOOKBACK = 48
HORIZON = 12

# ---------------------------------------------------------------------------
# Feature ordering (canonical — identical to v1 and v2)
# ---------------------------------------------------------------------------
TEMPORAL_FEATS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]

FRP_FEATS = [
    "log_frp",
    "frp_lag_1h", "frp_lag_2h", "frp_lag_3h",
    "frp_lag_6h", "frp_lag_12h", "frp_lag_24h",
    "frp_rolling_6h_mean", "frp_rolling_6h_std",
    "frp_trend_6h",
    "cumulative_frp",
    "hours_since_first_fire",
    "is_fire",
    "sat_fraction",
]

ERA5_FEATS = [
    "u10", "v10", "t2m_K", "d2m_K", "blh_m",
    "u850", "v850",
    "wind_speed_10m", "wind_dir_10m", "wind_speed_850",
    "vpd_kPa", "precip_1h_mm", "precip_24h_mm", "delta_wind_3h",
]

# Features that are already bounded → skip z-score normalisation
NO_NORMALISE = set(TEMPORAL_FEATS) | {"is_fire", "sat_fraction"}


# ---------------------------------------------------------------------------
# Helper: compute frp_trend_6h for a 1-D array (within one event)
# ---------------------------------------------------------------------------

def _rolling_trend(arr: np.ndarray, window: int = 6) -> np.ndarray:
    """Slope (MW/h) of a linear fit over the last `window` hours."""
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    x = np.arange(window, dtype=np.float64)
    for i in range(window - 1, n):
        y = arr[i - window + 1: i + 1]
        if np.any(np.isnan(y)):
            continue
        slope = np.polyfit(x, y, 1)[0]
        out[i] = slope
    return out


# ---------------------------------------------------------------------------
# Step 1 — Load & merge one event
# ---------------------------------------------------------------------------

def load_event(event_name: str, split: str) -> pd.DataFrame:
    frp_path = PROCESSED / f"{event_name}_hourly.csv"
    era5_path = PROCESSED / f"{event_name}_era5_hourly.csv"

    frp = pd.read_csv(frp_path, parse_dates=["slot_utc"])
    era5 = pd.read_csv(era5_path, parse_dates=["slot_utc"])

    # Drop tp_m (raw accumulation column, superseded by precip_1h_mm / precip_24h_mm)
    if "tp_m" in era5.columns:
        era5 = era5.drop(columns=["tp_m"])

    merged = frp.merge(era5, on="slot_utc", how="inner")
    merged = merged[merged["slot_type"] != "missing"].copy()
    merged = merged.sort_values("slot_utc").reset_index(drop=True)

    # sat_fraction is NaN for no_fire rows — fill with 0.0
    merged["sat_fraction"] = merged["sat_fraction"].fillna(0.0)

    merged["event_name"] = event_name
    merged["split"] = split
    return merged


# ---------------------------------------------------------------------------
# Step 2 — Feature engineering (per-event, no cross-boundary leakage)
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds all engineered columns IN-PLACE (modifies df, returns it)."""
    ts = df["slot_utc"]

    # Temporal encoding
    hour = ts.dt.hour + ts.dt.minute / 60.0
    doy = ts.dt.day_of_year.astype(float)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)

    # log_frp (before normalisation)
    df["log_frp"] = np.log1p(df["frp_sum_mw"])

    # is_fire
    df["is_fire"] = (df["slot_type"] == "fire").astype(np.float32)

    # --- per-event operations ---
    result_parts = []
    for event, grp in df.groupby("event_name", sort=False):
        grp = grp.copy().sort_values("slot_utc").reset_index(drop=True)
        frp_vals = grp["frp_sum_mw"].values.astype(np.float64)

        # Lags (shift within event)
        for lag in [1, 2, 3, 6, 12, 24]:
            grp[f"frp_lag_{lag}h"] = grp["frp_sum_mw"].shift(lag)

        # Rolling mean / std (min_periods=1 avoids full NaN)
        grp["frp_rolling_6h_mean"] = (
            grp["frp_sum_mw"].rolling(6, min_periods=1).mean()
        )
        grp["frp_rolling_6h_std"] = (
            grp["frp_sum_mw"].rolling(6, min_periods=1).std()
        )

        # Trend
        grp["frp_trend_6h"] = _rolling_trend(frp_vals, window=6)

        # Cumulative FRP
        grp["cumulative_frp"] = grp["frp_sum_mw"].cumsum()

        # hours_since_first_fire
        fire_mask = grp["slot_type"] == "fire"
        if fire_mask.any():
            first_fire_idx = fire_mask.idxmax()
            hours_since = (
                (grp["slot_utc"] - grp.loc[first_fire_idx, "slot_utc"])
                .dt.total_seconds() / 3600.0
            )
            hours_since = hours_since.clip(lower=0)
        else:
            hours_since = pd.Series(0.0, index=grp.index)
        grp["hours_since_first_fire"] = hours_since

        result_parts.append(grp)

    df = pd.concat(result_parts, ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# Step 3 — Normalisation (train-only statistics)
# ---------------------------------------------------------------------------

def compute_normalisation(df: pd.DataFrame, feature_cols: list) -> dict:
    """Returns {feature: {mean, std}} computed on training rows only."""
    train = df[df["split"] == "train"]
    params = {}
    zero_std_feats = []
    for feat in feature_cols:
        if feat in NO_NORMALISE:
            continue
        vals = train[feat].dropna()
        mu = float(vals.mean())
        sigma = float(vals.std(ddof=0))
        if sigma == 0.0:
            print(f"  WARNING: zero std for '{feat}' in training set — will be excluded.")
            zero_std_feats.append(feat)
            continue
        params[feat] = {"mean": mu, "std": sigma}
    return params, zero_std_feats


def apply_normalisation(df: pd.DataFrame, norm_params: dict) -> pd.DataFrame:
    df = df.copy()
    for feat, p in norm_params.items():
        df[feat] = (df[feat] - p["mean"]) / p["std"]
    return df


# ---------------------------------------------------------------------------
# Step 4 — Sequence construction
# ---------------------------------------------------------------------------

def build_sequences(df: pd.DataFrame, feature_cols: list):
    """
    Slides a window of LOOKBACK + HORIZON over each event.
    Returns:
        inputs   list of (LOOKBACK, F) float32 arrays
        targets  list of (HORIZON,) float32 arrays
        meta     list of dicts
    """
    inputs, targets, meta = [], [], []

    for event, grp in df.groupby("event_name", sort=False):
        grp = grp.sort_values("slot_utc").reset_index(drop=True)
        split = grp["split"].iloc[0]
        n = len(grp)

        feat_matrix = grp[feature_cols].values.astype(np.float32)  # (n, F)
        frp_raw = grp["frp_sum_mw"].values.astype(np.float32)        # (n,)
        slot_types = grp["slot_type"].values                           # (n,)
        timestamps = grp["slot_utc"].values                            # (n,)

        for t in range(LOOKBACK - 1, n - HORIZON):
            # Input window: [t-LOOKBACK+1, t]
            inp = feat_matrix[t - LOOKBACK + 1: t + 1]      # (LOOKBACK, F)
            # Target window: [t+1, t+HORIZON]
            tgt = frp_raw[t + 1: t + HORIZON + 1]            # (HORIZON,)
            tgt_types = slot_types[t + 1: t + HORIZON + 1]

            # Skip if NaN in input
            if np.any(np.isnan(inp)):
                continue

            # Skip if NaN in target
            if np.any(np.isnan(tgt)):
                continue

            # Skip if any target slot is 'missing'
            if np.any(tgt_types == "missing"):
                continue

            inputs.append(inp)
            targets.append(tgt)
            meta.append({
                "event_name": event,
                "split": split,
                "input_start_utc": str(timestamps[t - LOOKBACK + 1]),
                "target_end_utc": str(timestamps[t + HORIZON]),
                "target_mean_frp": float(tgt.mean()),
                "target_max_frp": float(tgt.max()),
            })

    return inputs, targets, meta


# ---------------------------------------------------------------------------
# Step 5 — Save HDF5 + manifest
# ---------------------------------------------------------------------------

def save_hdf5(inputs, targets, meta, feature_cols, norm_params, creation_date):
    splits = ["train", "val", "test"]
    split_data = {s: {"inputs": [], "targets": [], "meta": []} for s in splits}

    for inp, tgt, m in zip(inputs, targets, meta):
        s = m["split"]
        split_data[s]["inputs"].append(inp)
        split_data[s]["targets"].append(tgt)
        split_data[s]["meta"].append(m)

    events_per_split = {}
    for s in splits:
        evts = sorted({m["event_name"] for m in split_data[s]["meta"]})
        events_per_split[s] = evts

    metadata_payload = {
        "feature_names": feature_cols,
        "normalization_params": norm_params,
        "events_per_split": events_per_split,
        "LOOKBACK": LOOKBACK,
        "HORIZON": HORIZON,
        "creation_date": creation_date,
    }

    with h5py.File(OUT_H5, "w") as f:
        for s in splits:
            grp = f.create_group(s)
            if split_data[s]["inputs"]:
                inp_arr = np.stack(split_data[s]["inputs"], axis=0)
                tgt_arr = np.stack(split_data[s]["targets"], axis=0)
            else:
                inp_arr = np.zeros((0, LOOKBACK, len(feature_cols)), dtype=np.float32)
                tgt_arr = np.zeros((0, HORIZON), dtype=np.float32)
            grp.create_dataset("inputs", data=inp_arr, compression="gzip")
            grp.create_dataset("targets", data=tgt_arr, compression="gzip")
        f.create_dataset(
            "metadata",
            data=json.dumps(metadata_payload),
        )

    print(f"  Saved HDF5 → {OUT_H5}")

    # Manifest CSV
    rows = []
    global_idx = 0
    for s in splits:
        for m in split_data[s]["meta"]:
            rows.append({
                "sample_idx": global_idx,
                **m,
            })
            global_idx += 1
    pd.DataFrame(rows).to_csv(OUT_MANIFEST, index=False)
    print(f"  Saved manifest → {OUT_MANIFEST}")

    return split_data, events_per_split


# ---------------------------------------------------------------------------
# Step 6 — Statistics & integrity checks
# ---------------------------------------------------------------------------

def run_checks(split_data, df_norm, feature_cols, norm_params):
    print("\n=== Integrity checks ===")
    failed = []

    all_splits = ["train", "val", "test"]

    for s in all_splits:
        inp_list = split_data[s]["inputs"]
        tgt_list = split_data[s]["targets"]
        if not inp_list:
            continue
        inp_arr = np.stack(inp_list)
        tgt_arr = np.stack(tgt_list)

        if np.any(np.isnan(inp_arr)):
            failed.append(f"NaN in {s} inputs")
        if np.any(np.isnan(tgt_arr)):
            failed.append(f"NaN in {s} targets")
        if np.any(tgt_arr < 0):
            failed.append(f"Negative target in {s}")
        if inp_arr.shape[2] != len(feature_cols):
            failed.append(f"Feature count mismatch in {s}: {inp_arr.shape[2]} vs {len(feature_cols)}")

    train_df = df_norm[df_norm["split"] == "train"]
    for feat in feature_cols:
        if feat in NO_NORMALISE or feat not in norm_params:
            continue
        col_vals = train_df[feat].dropna()
        if len(col_vals) == 0:
            continue
        mu = float(col_vals.mean())
        sigma = float(col_vals.std(ddof=0))
        if abs(mu) > 0.1:
            failed.append(f"Train normalised mean out of range for '{feat}': {mu:.4f}")
        if not (0.9 <= sigma <= 1.1):
            failed.append(f"Train normalised std out of range for '{feat}': {sigma:.4f}")

    train_events = {m["event_name"] for m in split_data["train"]["meta"]}
    for s in ["val", "test"]:
        for m in split_data[s]["meta"]:
            if m["event_name"] in train_events:
                failed.append(f"Event '{m['event_name']}' appears in both train and {s}")

    if failed:
        print(f"DATASET FAILED — {failed}")
    else:
        n_train = len(split_data["train"]["inputs"])
        n_val = len(split_data["val"]["inputs"])
        n_test = len(split_data["test"]["inputs"])
        print(f"DATASET READY — {n_train} train / {n_val} val / {n_test} test samples")

    return failed


def compute_stats(split_data, feature_cols, df_norm):
    stats = {}

    for s in ["train", "val", "test"]:
        inp_list = split_data[s]["inputs"]
        tgt_list = split_data[s]["targets"]
        if not inp_list:
            stats[s] = {}
            continue
        inp_arr = np.stack(inp_list)   # (N, LOOKBACK, F)
        tgt_arr = np.stack(tgt_list)   # (N, HORIZON)
        events = sorted({m["event_name"] for m in split_data[s]["meta"]})

        stats[s] = {
            "n_samples": int(inp_arr.shape[0]),
            "n_events": len(events),
            "events": events,
            "target_mean_frp_mw": float(tgt_arr.mean()),
            "target_std_frp_mw": float(tgt_arr.std()),
            "target_median_frp_mw": float(np.median(tgt_arr)),
            "target_p90_frp_mw": float(np.percentile(tgt_arr, 90)),
            "target_p99_frp_mw": float(np.percentile(tgt_arr, 99)),
            "target_max_frp_mw": float(tgt_arr.max()),
        }

        if s == "train":
            flat = inp_arr.reshape(-1, len(feature_cols))
            feat_stats = {}
            for i, feat in enumerate(feature_cols):
                col_raw = df_norm[df_norm["split"] == "train"][feat].dropna()
                feat_stats[feat] = {
                    "normalized_mean": float(flat[:, i].mean()),
                    "normalized_std": float(flat[:, i].std()),
                    "raw_min": float(col_raw.min()) if len(col_raw) else None,
                    "raw_max": float(col_raw.max()) if len(col_raw) else None,
                }
            stats["train_feature_stats"] = feat_stats

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Phase 2C v3 — Feature Engineering & Dataset Assembly ===")
    print(f"  LOOKBACK={LOOKBACK}, HORIZON={HORIZON}")
    print(f"  Output: {OUT_H5}\n")

    # Guard: only delete v3.h5 — never touch v1 or v2
    if OUT_H5.exists():
        print(f"WARNING: {OUT_H5} already exists — deleting partial file before rebuild.")
        OUT_H5.unlink()

    v1 = PROCESSED / "dataset_v1.h5"
    v2 = PROCESSED / "dataset_v2.h5"
    if not v1.exists():
        print("WARNING: dataset_v1.h5 not found — expected to be preserved")
    if not v2.exists():
        print("WARNING: dataset_v2.h5 not found — expected to be preserved")

    # Load event catalogue
    catalogue = pd.read_csv(CATALOGUE_CSV)[["event_name", "split"]]
    event_list = list(catalogue.itertuples(index=False, name=None))
    print(f"Events in catalogue: {len(event_list)}")
    for sp in ["train", "val", "test"]:
        evs = [e for e, s in event_list if s == sp]
        print(f"  {sp}: {len(evs)} events — {evs}")

    # Step 1 — Load & merge
    print("\n[Step 1] Loading and merging FRP + ERA5 ...")
    dfs = []
    skipped = []
    for event_name, split in event_list:
        frp_path = PROCESSED / f"{event_name}_hourly.csv"
        era5_path = PROCESSED / f"{event_name}_era5_hourly.csv"
        if not frp_path.exists():
            print(f"  SKIP (no FRP CSV): {event_name}")
            skipped.append(event_name)
            continue
        if not era5_path.exists():
            print(f"  SKIP (no ERA5 CSV): {event_name}")
            skipped.append(event_name)
            continue
        df_ev = load_event(event_name, split)
        print(f"  {event_name}: {len(df_ev)} rows after merge+drop-missing")
        dfs.append(df_ev)

    if skipped:
        print(f"\n  WARNING: skipped {len(skipped)} events: {skipped}")
        print("  STOP — all events must be present. Fix missing CSVs before rebuild.")
        return 1

    df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows (all events, non-missing): {len(df)}")

    # Step 2 — Feature engineering
    print("\n[Step 2] Engineering features ...")
    df = engineer_features(df)

    # Build canonical feature list
    all_candidate_feats = TEMPORAL_FEATS + FRP_FEATS + ERA5_FEATS
    missing_cols = [f for f in all_candidate_feats if f not in df.columns]
    if missing_cols:
        print(f"  WARNING: these features are missing from data: {missing_cols}")
    feature_cols_candidate = [f for f in all_candidate_feats if f in df.columns]

    # Step 3 — Normalisation
    print("\n[Step 3] Computing normalisation on training set ...")
    norm_params, zero_std_feats = compute_normalisation(df, feature_cols_candidate)
    feature_cols = [f for f in feature_cols_candidate if f not in zero_std_feats]
    print(f"  Features retained: {len(feature_cols)}")
    if zero_std_feats:
        print(f"  Excluded (zero std): {zero_std_feats}")

    # Save norm params to v3-specific file (NOT overwriting v1 or v2 norm params)
    with open(OUT_NORM, "w") as fh:
        json.dump(norm_params, fh, indent=2)
    print(f"  Saved normalisation params → {OUT_NORM}")

    # Apply normalisation
    df_norm = apply_normalisation(df, norm_params)

    # Step 4 — Sequences
    print(f"\n[Step 4] Building sequences (LOOKBACK={LOOKBACK}, HORIZON={HORIZON}) ...")
    inputs, targets, meta = build_sequences(df_norm, feature_cols)
    print(f"  Total valid samples: {len(inputs)}")
    for s in ["train", "val", "test"]:
        n = sum(1 for m in meta if m["split"] == s)
        evts = sorted({m["event_name"] for m in meta if m["split"] == s})
        print(f"    {s}: {n} samples from {len(evts)} events — {evts}")

    if len(inputs) == 0:
        print("ERROR: zero samples generated. Aborting — check LOOKBACK vs event durations.")
        return 1

    # Step 5 — Save
    print("\n[Step 5] Saving dataset ...")
    creation_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    split_data, events_per_split = save_hdf5(
        inputs, targets, meta, feature_cols, norm_params, creation_date
    )

    # Shape confirmation
    print("\nShape confirmation:")
    for s in ["train", "val", "test"]:
        if split_data[s]["inputs"]:
            inp_arr = np.stack(split_data[s]["inputs"])
            tgt_arr = np.stack(split_data[s]["targets"])
            print(f"  /{s}/inputs  shape: {inp_arr.shape}  (N, {LOOKBACK}, {len(feature_cols)})")
            print(f"  /{s}/targets shape: {tgt_arr.shape}  dtype={tgt_arr.dtype}")
            print(f"  max target: {tgt_arr.max():.1f} MW  (must be > 1000 if fire events present)")

    # Step 6 — Stats & checks
    print("\n[Step 6] Computing statistics ...")
    stats = compute_stats(split_data, feature_cols, df_norm)

    # Print target distribution table
    print("\nTarget distribution (raw MW):")
    print(f"  {'Split':<8} {'mean':>10} {'std':>10} {'median':>10} {'p90':>10} {'p99':>10} {'max':>12}")
    for s in ["train", "val", "test"]:
        if s not in stats or not stats[s]:
            continue
        st = stats[s]
        print(
            f"  {s:<8} {st['target_mean_frp_mw']:>10.1f} "
            f"{st['target_std_frp_mw']:>10.1f} "
            f"{st['target_median_frp_mw']:>10.1f} "
            f"{st['target_p90_frp_mw']:>10.1f} "
            f"{st['target_p99_frp_mw']:>10.1f} "
            f"{st['target_max_frp_mw']:>12.1f}"
        )

    print("\nSample counts:")
    print(f"  {'Split':<8} {'Samples':>8} {'Events':>8}")
    for s in ["train", "val", "test"]:
        if s not in stats or not stats[s]:
            continue
        print(f"  {s:<8} {stats[s]['n_samples']:>8} {stats[s]['n_events']:>8}")

    # Zero-target fraction
    print("\nZero-target fraction (all steps == 0):")
    for s in ["train", "val", "test"]:
        tgt_list = split_data[s]["targets"]
        if not tgt_list:
            continue
        tgt_arr = np.stack(tgt_list)
        zero_frac = float((tgt_arr.sum(axis=1) == 0).mean())
        print(f"  {s:<8} {zero_frac:.1%}")

    failed = run_checks(split_data, df_norm, feature_cols, norm_params)

    stats["integrity_checks_failed"] = failed
    stats["feature_names"] = feature_cols
    stats["n_features"] = len(feature_cols)
    stats["LOOKBACK"] = LOOKBACK
    stats["HORIZON"] = HORIZON
    stats["creation_date"] = creation_date

    with open(OUT_STATS, "w") as fh:
        json.dump(stats, fh, indent=2, default=str)
    print(f"\nSaved stats → {OUT_STATS}")

    if failed:
        print(f"\nERROR: integrity checks failed: {failed}")
        print("Deleting incomplete dataset_v3.h5 ...")
        OUT_H5.unlink(missing_ok=True)
        return 1

    print(f"\n{'='*60}")
    print(f"dataset_v3.h5 BUILD COMPLETE")
    print(f"  LOOKBACK={LOOKBACK}  HORIZON={HORIZON}  Features={len(feature_cols)}")
    print(f"  Samples: {stats['train']['n_samples']} train / "
          f"{stats['val']['n_samples']} val / "
          f"{stats['test']['n_samples']} test")
    print(f"  Events in val: {events_per_split['val']}")
    print(f"{'='*60}")

    # Confirm v1 and v2 still intact
    if v1.exists():
        print(f"  PRESERVED: {v1}")
    if v2.exists():
        print(f"  PRESERVED: {v2}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
