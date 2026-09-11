"""
build_dataset_v6.py — Dataset v6 Assembly (Task E2.6, 2026-08-25)
=================================================================
Frozen dev pool: 84 events, split per resubmission/split_v6_2026-08-25.csv
(FINAL, seed 20260825 after declared seed+1 remediation): 71 train / 13 val.
ba_trebinje_2022 INCLUDED, pinned train (D2). NO test group — test blocks are
sequestered and never enter a training dataset.

AMEND-2 support: inputs are stored RAW (unnormalized); FOUR normalization
parameter sets are stored in metadata, one per selection cell:
    norm[T0|T1][legacy|robust]
computed from that variant's TRAIN subset only (per-variant rule R-E).
  - legacy: z-score (mean/std), identical convention to v1–v5.
  - robust: transform then median/IQR scaling. Transforms fixed HERE, before any
    training: log1p(max(x,0)) for non-negative FRP-scale features
    (frp_lag_*, frp_rolling_6h_mean/std, cumulative_frp); signed log
    sign(x)*log1p(|x|) for frp_trend_6h; no transform for log_frp,
    hours_since_first_fire, and ERA5 features. Scale = IQR/1.349 (sigma-
    equivalent); if IQR == 0, declared fallback to std of the transformed
    values; if that is 0 the feature is excluded (logged).
TEMPORAL_FEATS, is_fire, sat_fraction are never normalized (unchanged).

T0 = v6 train events ∩ legacy-53-pool (dataset_v5 membership).
T1 = all v6 train events. Both share the identical val split.
Each train sample's manifest row carries in_T0.

Outputs:
  data/processed/dataset_v6.h5
  data/processed/dataset_v6_manifest.csv
  data/processed/normalization_params_v6.json   (all four sets)
  data/processed/dataset_v6_stats.json
"""

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
sys.stdout.reconfigure(encoding="utf-8")

BASE      = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed"
SPLIT_CSV = BASE / "resubmission" / "split_v6_2026-08-25.csv"
V5_STATS  = PROCESSED / "dataset_v5_stats.json"

VERSION      = sys.argv[1] if len(sys.argv) > 1 else "v6"   # "v6_1" = W-1 patch build
OUT_H5       = PROCESSED / f"dataset_{VERSION}.h5"
OUT_MANIFEST = PROCESSED / f"dataset_{VERSION}_manifest.csv"
OUT_NORM     = PROCESSED / f"normalization_params_{VERSION}.json"
OUT_STATS    = PROCESSED / f"dataset_{VERSION}_stats.json"

if OUT_H5.exists():
    print(f"WARNING: {OUT_H5} already exists. Delete it first to rebuild.")
    sys.exit(1)

LOOKBACK = 48
HORIZON  = 12

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
NO_NORMALISE = set(TEMPORAL_FEATS) | {"is_fire", "sat_fraction"}

LOG1P_FEATS      = {"frp_lag_1h", "frp_lag_2h", "frp_lag_3h", "frp_lag_6h",
                    "frp_lag_12h", "frp_lag_24h", "frp_rolling_6h_mean",
                    "frp_rolling_6h_std", "cumulative_frp"}
SIGNED_LOG_FEATS = {"frp_trend_6h"}


def _rolling_trend(arr, window=6):
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    x = np.arange(window, dtype=np.float64)
    for i in range(window - 1, n):
        y = arr[i - window + 1: i + 1]
        if np.any(np.isnan(y)):
            continue
        out[i] = np.polyfit(x, y, 1)[0]
    return out


def load_event(event_name, split):
    frp  = pd.read_csv(PROCESSED / f"{event_name}_hourly.csv", parse_dates=["slot_utc"])
    era5 = pd.read_csv(PROCESSED / f"{event_name}_era5_hourly.csv", parse_dates=["slot_utc"])
    if "tp_m" in era5.columns:
        era5 = era5.drop(columns=["tp_m"])
    merged = frp.merge(era5, on="slot_utc", how="inner")
    merged = merged[merged["slot_type"] != "missing"].copy()
    merged = merged.sort_values("slot_utc").reset_index(drop=True)
    merged["sat_fraction"] = merged["sat_fraction"].fillna(0.0)
    merged["event_name"] = event_name
    merged["split"] = split
    return merged


def engineer_features(df):
    ts   = df["slot_utc"]
    hour = ts.dt.hour + ts.dt.minute / 60.0
    doy  = ts.dt.day_of_year.astype(float)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["doy_sin"]  = np.sin(2 * np.pi * doy / 365.0)
    df["doy_cos"]  = np.cos(2 * np.pi * doy / 365.0)
    df["log_frp"]  = np.log1p(df["frp_sum_mw"])
    df["is_fire"]  = (df["slot_type"] == "fire").astype(np.float32)
    parts = []
    for event, grp in df.groupby("event_name", sort=False):
        grp = grp.copy().sort_values("slot_utc").reset_index(drop=True)
        frp_vals = grp["frp_sum_mw"].values.astype(np.float64)
        for lag in [1, 2, 3, 6, 12, 24]:
            grp[f"frp_lag_{lag}h"] = grp["frp_sum_mw"].shift(lag)
        grp["frp_rolling_6h_mean"] = grp["frp_sum_mw"].rolling(6, min_periods=1).mean()
        grp["frp_rolling_6h_std"]  = grp["frp_sum_mw"].rolling(6, min_periods=1).std()
        grp["frp_trend_6h"]        = _rolling_trend(frp_vals, window=6)
        grp["cumulative_frp"]      = grp["frp_sum_mw"].cumsum()
        fire_mask = grp["slot_type"] == "fire"
        if fire_mask.any():
            first_idx = fire_mask.idxmax()
            hs = (grp["slot_utc"] - grp.loc[first_idx, "slot_utc"]).dt.total_seconds() / 3600.0
            hs = hs.clip(lower=0)
        else:
            hs = pd.Series(0.0, index=grp.index)
        grp["hours_since_first_fire"] = hs
        parts.append(grp)
    return pd.concat(parts, ignore_index=True)


def robust_transform(feat, vals):
    if feat in LOG1P_FEATS:
        return np.log1p(np.maximum(vals, 0.0))
    if feat in SIGNED_LOG_FEATS:
        return np.sign(vals) * np.log1p(np.abs(vals))
    return vals


def compute_norm_legacy(train_df, feats):
    params, dropped = {}, []
    for f in feats:
        if f in NO_NORMALISE:
            continue
        v = train_df[f].dropna()
        mu, sd = float(v.mean()), float(v.std(ddof=0))
        if sd == 0.0:
            dropped.append(f)
            continue
        params[f] = {"mean": mu, "std": sd}
    return params, dropped


def compute_norm_robust(train_df, feats):
    params, dropped = {}, []
    for f in feats:
        if f in NO_NORMALISE:
            continue
        v = train_df[f].dropna().to_numpy(dtype=np.float64)
        t = robust_transform(f, v)
        med = float(np.median(t))
        iqr = float(np.percentile(t, 75) - np.percentile(t, 25))
        scale = iqr / 1.349
        scale_type = "iqr"
        if scale == 0.0:
            scale = float(np.std(t))
            scale_type = "std_fallback"
        if scale == 0.0:
            dropped.append(f)
            continue
        transform = ("log1p" if f in LOG1P_FEATS else
                     "signed_log" if f in SIGNED_LOG_FEATS else "none")
        params[f] = {"transform": transform, "center": med,
                     "scale": scale, "scale_type": scale_type}
    return params, dropped


def build_sequences(df, feature_cols):
    """Sequences over RAW features; validity = no NaN in raw inputs/targets."""
    inputs, targets, meta = [], [], []
    for event, grp in df.groupby("event_name", sort=False):
        grp = grp.sort_values("slot_utc").reset_index(drop=True)
        split = grp["split"].iloc[0]
        n = len(grp)
        fm = grp[feature_cols].values.astype(np.float32)
        frp_raw = grp["frp_sum_mw"].values.astype(np.float32)
        slot_types = grp["slot_type"].values
        timestamps = grp["slot_utc"].values
        for t in range(LOOKBACK - 1, n - HORIZON):
            inp = fm[t - LOOKBACK + 1: t + 1]
            tgt = frp_raw[t + 1: t + HORIZON + 1]
            if np.any(np.isnan(inp)) or np.any(np.isnan(tgt)):
                continue
            if np.any(slot_types[t + 1: t + HORIZON + 1] == "missing"):
                continue
            inputs.append(inp)
            targets.append(tgt)
            meta.append({
                "event_name": event, "split": split,
                "input_start_utc": str(timestamps[t - LOOKBACK + 1]),
                "target_end_utc":  str(timestamps[t + HORIZON]),
                "target_mean_frp": float(tgt.mean()),
                "target_max_frp":  float(tgt.max()),
            })
    return inputs, targets, meta


def main():
    print("=" * 70)
    print(f"Dataset {VERSION} Assembly (LOOKBACK={LOOKBACK}, HORIZON={HORIZON})")
    print("=" * 70)

    split = pd.read_csv(SPLIT_CSV)
    assert len(split) == 84, f"expected 84 frozen events, got {len(split)}"
    assert set(split["split_v6"]) == {"train", "val"}
    tr = split[split["split_v6"] == "train"]["event_name"].tolist()
    va = split[split["split_v6"] == "val"]["event_name"].tolist()
    assert "ba_trebinje_2022" in tr, "trebinje must be pinned train (D2)"
    # sequestration guard: no test-block names
    cat = pd.read_csv(PROCESSED / "event_catalogue.csv")
    test_names = set(cat[cat["split"].astype(str).str.startswith("test_")]["event_name"])
    assert not (set(tr) | set(va)) & test_names, "SEQUESTRATION BREACH: test event in v6!"
    print(f"  Train: {len(tr)}  Val: {len(va)} (split FINAL, seed "
          f"{int(split['split_seed'].iloc[0])})")

    with open(V5_STATS) as fh:
        v5 = json.load(fh)
    v5_pool = set()
    for s in ["train", "val", "test"]:
        v5_pool |= set(v5.get(s, {}).get("events", []))
    t0_events = sorted(set(tr) & v5_pool)
    print(f"  T1 train events: {len(tr)}   T0 = T1 ∩ v5-pool: {len(t0_events)}")

    print("\n[Step 1] Loading FRP + ERA5 ...")
    dfs, skipped = [], []
    for ev in tr + va:
        s = "train" if ev in set(tr) else "val"
        if not (PROCESSED / f"{ev}_hourly.csv").exists() or \
           not (PROCESSED / f"{ev}_era5_hourly.csv").exists():
            skipped.append(ev)
            continue
        d = load_event(ev, s)
        dfs.append(d)
    if skipped:
        print(f"  FATAL: missing CSVs for {skipped}")
        sys.exit(1)
    df = pd.concat(dfs, ignore_index=True)
    print(f"  {len(dfs)} events, {len(df)} rows")

    print("\n[Step 2] Engineering features ...")
    df = engineer_features(df)
    feats = [f for f in TEMPORAL_FEATS + FRP_FEATS + ERA5_FEATS if f in df.columns]

    print("\n[Step 3] Normalization parameter sets (per variant x scheme) ...")
    t1_train_df = df[df["split"] == "train"]
    t0_train_df = df[(df["split"] == "train") & (df["event_name"].isin(t0_events))]
    norm, dropped_all = {}, set()
    for vname, vdf in [("T1", t1_train_df), ("T0", t0_train_df)]:
        pl, dl = compute_norm_legacy(vdf, feats)
        pr, dr = compute_norm_robust(vdf, feats)
        norm[vname] = {"legacy": pl, "robust": pr}
        dropped_all |= set(dl) | set(dr)
        print(f"  {vname}: legacy {len(pl)} feats ({len(dl)} zero-std), "
              f"robust {len(pr)} feats ({len(dr)} zero-scale)")
    if dropped_all:
        print(f"  Dropped from feature set (zero scale in ANY cell): {sorted(dropped_all)}")
    feature_cols = [f for f in feats if f not in dropped_all]
    print(f"  Features retained: {len(feature_cols)}")

    for feat in ["frp_lag_1h", "wind_speed_850"]:
        p1 = norm["T1"]["legacy"].get(feat, {})
        p0 = norm["T0"]["legacy"].get(feat, {})
        print(f"    {feat:16s}: T1 mean={p1.get('mean', float('nan')):9.3f} "
              f"std={p1.get('std', float('nan')):9.3f} | "
              f"T0 mean={p0.get('mean', float('nan')):9.3f} "
              f"std={p0.get('std', float('nan')):9.3f}")

    print(f"\n[Step 4] Building RAW sequences ...")
    inputs, targets, meta = build_sequences(df, feature_cols)
    t0_set = set(t0_events)
    for m in meta:
        m["in_T0"] = bool(m["split"] == "val" or m["event_name"] in t0_set)
    n_tr = sum(1 for m in meta if m["split"] == "train")
    n_va = sum(1 for m in meta if m["split"] == "val")
    n_t0 = sum(1 for m in meta if m["split"] == "train" and m["in_T0"])
    print(f"  Samples: train {n_tr} (T0 subset {n_t0}) / val {n_va} / total {len(meta)}")

    print("\n[Step 5] Saving ...")
    creation = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    split_data = {s: {"inputs": [], "targets": [], "meta": []} for s in ["train", "val"]}
    for i, t, m in zip(inputs, targets, meta):
        split_data[m["split"]]["inputs"].append(i)
        split_data[m["split"]]["targets"].append(t)
        split_data[m["split"]]["meta"].append(m)
    events_per_split = {s: sorted({m["event_name"] for m in split_data[s]["meta"]})
                        for s in ["train", "val"]}
    metadata = {
        "feature_names": feature_cols,
        "inputs_are_raw": True,
        "normalization_params": norm,          # norm[T0|T1][legacy|robust]
        "robust_scheme_note": "transform per-feature (log1p / signed_log / none), "
                              "then (x - center) / scale; scale = IQR/1.349, "
                              "std fallback where IQR=0",
        "events_per_split": events_per_split,
        "t0_train_events": t0_events,
        "LOOKBACK": LOOKBACK, "HORIZON": HORIZON,
        "creation_date": creation, "dataset_version": VERSION,
        "split_source": "resubmission/split_v6_2026-08-25.csv (FINAL, seed 20260825)",
    }
    with h5py.File(OUT_H5, "w") as f:
        for s in ["train", "val"]:
            g = f.create_group(s)
            g.create_dataset("inputs", data=np.stack(split_data[s]["inputs"]),
                             compression="gzip")
            g.create_dataset("targets", data=np.stack(split_data[s]["targets"]),
                             compression="gzip")
        f.create_dataset("metadata", data=json.dumps(metadata))
    print(f"  HDF5 -> {OUT_H5}")
    rows = []
    gi = 0
    for s in ["train", "val"]:
        for m in split_data[s]["meta"]:
            rows.append({"sample_idx": gi, **m})
            gi += 1
    pd.DataFrame(rows).to_csv(OUT_MANIFEST, index=False)
    with open(OUT_NORM, "w") as fh:
        json.dump(norm, fh, indent=2)
    print(f"  Manifest -> {OUT_MANIFEST}\n  Norm params -> {OUT_NORM}")

    print("\n[Step 6] Integrity checks ...")
    failed = []
    for s in ["train", "val"]:
        ia = np.stack(split_data[s]["inputs"])
        ta = np.stack(split_data[s]["targets"])
        if np.any(np.isnan(ia)): failed.append(f"NaN in {s} raw inputs")
        if np.any(np.isnan(ta)): failed.append(f"NaN in {s} targets")
        if np.any(ta < 0):       failed.append(f"negative target in {s}")
    # normalized-stat check per cell on its own train subset
    for vname, vdf in [("T1", t1_train_df), ("T0", t0_train_df)]:
        for feat in feature_cols:
            p = norm[vname]["legacy"].get(feat)
            if p is None:
                continue
            v = vdf[feat].dropna()
            z = (v - p["mean"]) / p["std"]
            if abs(float(z.mean())) > 0.1:
                failed.append(f"{vname}/legacy mean off for {feat}: {z.mean():.4f}")
            if not (0.9 <= float(z.std(ddof=0)) <= 1.1):
                failed.append(f"{vname}/legacy std off for {feat}: {z.std(ddof=0):.4f}")
            pr = norm[vname]["robust"].get(feat)
            if pr is not None:
                t = robust_transform(feat, v.to_numpy(dtype=np.float64))
                zr = (t - pr["center"]) / pr["scale"]
                if abs(float(np.median(zr))) > 0.1:
                    failed.append(f"{vname}/robust median off for {feat}")
    tr_ev = set(events_per_split["train"])
    if tr_ev & set(events_per_split["val"]):
        failed.append("train/val event overlap")
    if failed:
        print("DATASET FAILED:")
        for x in failed:
            print("  -", x)
    else:
        print(f"DATASET READY — {n_tr} train / {n_va} val samples, "
              f"{len(feature_cols)} features, 4 norm cells")

    stats = {
        "dataset_version": VERSION, "LOOKBACK": LOOKBACK, "HORIZON": HORIZON,
        "n_features": len(feature_cols), "feature_names": feature_cols,
        "n_total": len(meta), "creation_date": creation,
        "integrity_checks_failed": failed,
        "t0_train_events": t0_events,
    }
    for s in ["train", "val"]:
        ta = np.stack(split_data[s]["targets"])
        stats[s] = {
            "n_samples": len(split_data[s]["inputs"]),
            "n_events": len(events_per_split[s]),
            "events": events_per_split[s],
            "zero_fraction": float(np.mean(ta.sum(axis=1) == 0)),
            "max_frp_mw": float(ta.max()),
        }
    with open(OUT_STATS, "w") as fh:
        json.dump(stats, fh, indent=2, default=str)
    print(f"Stats -> {OUT_STATS}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
