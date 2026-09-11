"""
Single-event download + parse + audit worker used by /add-event.
Usage:
  python _add_event_worker.py <event_name> <start> <end>
      <lon_min> <lon_max> <lat_min> <lat_max> <split>
"""

import os, sys, io, time, json, traceback
import requests
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ── credentials ───────────────────────────────────────────────────────────────
USERNAME = "<<LSASAF_USERNAME_REDACTED>>"
PASSWORD = "<<LSASAF_PASSWORD_REDACTED>>"
BASE_URL = "https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/MSG/FRP-PIXEL/HDF5"
SESSION  = requests.Session()
SESSION.auth = (USERNAME, PASSWORD)

SAT_FLAG    = 2
MAX_WORKERS = 10
RETRY_DELAY = 5
RETRY_BATCH = 5

PROJECT_ROOT = Path("e:/GRSL_Wildfire/FRP_Methodology")
RAW_DIR   = PROJECT_ROOT / "data" / "raw"
PROC_DIR  = PROJECT_ROOT / "data" / "processed"
PLOT_DIR  = PROJECT_ROOT / "data" / "plots"

# ── CLI args ──────────────────────────────────────────────────────────────────
EVENT_NAME = sys.argv[1]
START_STR  = sys.argv[2]
END_STR    = sys.argv[3]
LON_MIN    = float(sys.argv[4])
LON_MAX    = float(sys.argv[5])
LAT_MIN    = float(sys.argv[6])
LAT_MAX    = float(sys.argv[7])
SPLIT      = sys.argv[8] if len(sys.argv) > 8 else "train"

START_DATE = datetime.strptime(START_STR, "%Y-%m-%d")
END_DATE   = datetime.strptime(END_STR,   "%Y-%m-%d")

BBOX = {"lon_min": LON_MIN, "lon_max": LON_MAX,
        "lat_min": LAT_MIN, "lat_max": LAT_MAX}

EVENT_RAW   = RAW_DIR  / EVENT_NAME
EVENT_PROC  = PROC_DIR / EVENT_NAME
SLOTS_DIR   = EVENT_PROC / "slots"
MISS_CSV    = EVENT_PROC / "missing_list.csv"
HOURLY_CSV  = PROC_DIR  / f"{EVENT_NAME}_hourly.csv"

for d in (EVENT_RAW, SLOTS_DIR, PLOT_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────
def slot_range(start: datetime, end: datetime):
    """All 15-min slots from start 00:00 to end 23:45 inclusive."""
    t   = start
    end_t = end + timedelta(days=1)
    slots = []
    while t < end_t:
        slots.append(t)
        t += timedelta(minutes=15)
    return slots


def list_fname(dt): return f"HDF5_LSASAF_MSG_FRP-PIXEL-ListProduct_MSG-Disk_{dt:%Y%m%d%H%M}"
def qual_fname(dt): return f"HDF5_LSASAF_MSG_FRP-PIXEL-QualityProduct_MSG-Disk_{dt:%Y%m%d%H%M}"
def file_url(dt, fname): return f"{BASE_URL}/{dt:%Y}/{dt:%m}/{dt:%d}/{fname}"


def download_bytes(url, retries=2):
    for attempt in range(retries + 1):
        try:
            r = SESSION.get(url, timeout=90)
            return r.status_code, (r.content if r.status_code == 200 else None)
        except Exception:
            if attempt == retries:
                return 0, None
            time.sleep(2)


def read_hdf5_bytes(raw_bytes):
    return h5py.File(io.BytesIO(raw_bytes), "r")


# ── Parse ListProduct ─────────────────────────────────────────────────────────
def parse_list_product(raw_bytes, bbox, dt):
    lon_min, lon_max = bbox["lon_min"], bbox["lon_max"]
    lat_min, lat_max = bbox["lat_min"], bbox["lat_max"]
    with read_hdf5_bytes(raw_bytes) as hf:
        def read_scaled(name):
            ds  = hf[name]
            raw = ds[()].astype(float)
            mv  = ds.attrs.get("MISSING_VALUE", None)
            sf  = float(ds.attrs.get("SCALING_FACTOR", 1.0))
            off = float(ds.attrs.get("OFFSET", 0.0))
            if mv is not None:
                raw = np.where(raw == float(mv), np.nan, raw)
            return (raw - off) / sf

        lat       = read_scaled("LATITUDE")
        lon       = read_scaled("LONGITUDE")
        frp       = read_scaled("FRP")
        unc       = read_scaled("FRP_UNCERTAINTY")
        conf      = read_scaled("FIRE_CONFIDENCE")
        abs_line  = hf["ABS_LINE"][()].astype(int)
        abs_pixel = hf["ABS_PIXEL"][()].astype(int)

    mask = (
        (lat >= lat_min) & (lat <= lat_max) &
        (lon >= lon_min) & (lon <= lon_max) &
        np.isfinite(lat) & np.isfinite(lon) & np.isfinite(frp)
    )
    return pd.DataFrame({
        "slot_utc":   dt,
        "lat":        lat[mask],
        "lon":        lon[mask],
        "frp_mw":     frp[mask],
        "frp_unc_mw": unc[mask],
        "confidence": conf[mask],
        "abs_line":   abs_line[mask],
        "abs_pixel":  abs_pixel[mask],
        "saturated":  False,
    })


def apply_quality_flags(df, qual_bytes):
    with read_hdf5_bytes(qual_bytes) as qhf:
        qflag = qhf["QUALITYFLAG"][()]
    rows  = df["abs_line"].values  - 1
    cols  = df["abs_pixel"].values - 1
    valid = (rows >= 0) & (rows < 3712) & (cols >= 0) & (cols < 3712)
    flags = np.zeros(len(df), dtype=int)
    flags[valid] = qflag[rows[valid], cols[valid]]
    df = df.copy()
    df["saturated"] = (flags == SAT_FLAG)
    return df


# ── Phase A ───────────────────────────────────────────────────────────────────
def download_list_slot(args):
    dt, bbox = args
    fname  = list_fname(dt)
    url    = file_url(dt, fname)
    status, data = download_bytes(url)
    if status == 200:
        out = EVENT_RAW / fname
        if not out.exists():
            out.write_bytes(data)
        try:
            df = parse_list_product(data, bbox, dt)
        except Exception:
            df = pd.DataFrame()
        return dt, status, df
    return dt, status, None


def phase_a(slots):
    print(f"\n[{EVENT_NAME}] Phase A: {len(slots)} slots")
    missing = []
    fire    = []
    args    = [(dt, BBOX) for dt in slots]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(download_list_slot, a): a[0] for a in args}
        done = 0
        for fut in as_completed(futs):
            dt, status, df = fut.result()
            done += 1
            if status == 200:
                n = len(df) if df is not None else 0
                if done % 96 == 0 or n > 0:
                    print(f"  {dt:%Y-%m-%d %H:%M}  200  n_fire={n}  ({done}/{len(slots)})")
                if df is not None and n > 0:
                    fire.append((dt, df))
            else:
                missing.append(dt.isoformat())
                if done % 96 == 0:
                    print(f"  {dt:%Y-%m-%d %H:%M}  {status}  ({done}/{len(slots)})")

    pd.DataFrame({"slot_utc": missing}).to_csv(MISS_CSV, index=False)
    print(f"\n[{EVENT_NAME}] Phase A done: {len(slots)-len(missing)} ok, "
          f"{len(missing)} missing, {len(fire)} fire slots")
    return fire, missing


# ── Phase B ───────────────────────────────────────────────────────────────────
def download_qual_slot(args):
    dt, df_fire = args
    fname  = qual_fname(dt)
    url    = file_url(dt, fname)
    status, data = download_bytes(url)
    if status == 200:
        out = EVENT_RAW / fname
        if not out.exists():
            out.write_bytes(data)
        try:
            df_fire = apply_quality_flags(df_fire, data)
        except Exception:
            pass
        return dt, status, df_fire
    return dt, status, df_fire


def phase_b(fire_slots):
    print(f"\n[{EVENT_NAME}] Phase B: {len(fire_slots)} quality files")
    final = []
    args  = [(dt, df) for dt, df in fire_slots]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(download_qual_slot, a): a[0] for a in args}
        done = 0
        for fut in as_completed(futs):
            dt, status, df = fut.result()
            done += 1
            if done % 20 == 0:
                print(f"  [{EVENT_NAME}] Quality {done}/{len(fire_slots)}")
            final.append((dt, df))

    print(f"[{EVENT_NAME}] Saving {len(final)} per-slot CSVs ...")
    for dt, df in final:
        df.drop(columns=["abs_line", "abs_pixel"], errors="ignore").to_csv(
            SLOTS_DIR / f"{dt:%Y%m%d%H%M}.csv", index=False)
    return final


# ── Hourly aggregation ────────────────────────────────────────────────────────
def build_hourly(all_slots, fire_slots, missing_slots):
    fire_map    = {dt: df for dt, df in fire_slots}
    missing_set = set(datetime.fromisoformat(s) for s in missing_slots)

    rows = []
    hour = START_DATE
    end_h = END_DATE + timedelta(hours=23)
    while hour <= end_h:
        h_slots = [hour + timedelta(minutes=m) for m in (0, 15, 30, 45)]
        n_fire    = sum(1 for s in h_slots if s in fire_map)
        n_missing = sum(1 for s in h_slots if s in missing_set)
        n_nofire  = 4 - n_fire - n_missing

        if n_missing == 4:
            slot_type = "missing"
            frp_sum   = float("nan")
            n_pix     = 0
            sat_frac  = float("nan")
        elif n_fire > 0:
            slot_type = "fire"
            dfs       = [fire_map[s] for s in h_slots if s in fire_map]
            combined  = pd.concat(dfs, ignore_index=True)
            frp_sum   = float(combined["frp_mw"].sum())
            n_pix     = len(combined)
            sat_frac  = float(combined["saturated"].mean()) if "saturated" in combined else 0.0
        else:
            slot_type = "no_fire"
            frp_sum   = 0.0
            n_pix     = 0
            sat_frac  = 0.0

        rows.append({
            "slot_utc":      hour,
            "frp_sum_mw":    frp_sum,
            "n_pixels":      n_pix,
            "sat_fraction":  sat_frac,
            "slot_type":     slot_type,
            "n_15min_slots": 4,
            "n_missing_15":  n_missing,
        })
        hour += timedelta(hours=1)

    df = pd.DataFrame(rows)
    df["slot_utc"] = pd.to_datetime(df["slot_utc"])
    df.to_csv(HOURLY_CSV, index=False)
    print(f"Hourly CSV: {HOURLY_CSV}  ({len(df)} rows)")
    return df


# ── Verify ────────────────────────────────────────────────────────────────────
def verify(df):
    errs = []
    if str(df["slot_utc"].dtype) != "datetime64[ns]":
        errs.append("slot_utc dtype")
    if df["slot_utc"].duplicated().any():
        errs.append("duplicate slot_utc")
    bad = set(df["slot_type"].unique()) - {"fire", "no_fire", "missing"}
    if bad:
        errs.append(f"bad slot_type: {bad}")
    miss_mask = df["slot_type"] == "missing"
    if not df.loc[miss_mask, "frp_sum_mw"].isna().all():
        errs.append("frp_sum_mw not NaN for missing")
    if errs:
        print("  VERIFY FAILED:", errs)
        sys.exit(2)
    print("  Verification passed")


# ── Audit metrics ─────────────────────────────────────────────────────────────
def audit(df):
    total        = len(df)
    miss_count   = (df["slot_type"] == "missing").sum()
    fire_count   = (df["slot_type"] == "fire").sum()
    gap_rate     = miss_count / total
    sat_vals     = df.loc[df["slot_type"] == "fire", "sat_fraction"]
    sat_rate     = float(sat_vals.mean()) if len(sat_vals) > 0 else 0.0
    if np.isnan(sat_rate):
        sat_rate = 0.0

    df2 = df.copy()
    df2["date"] = df2["slot_utc"].dt.date
    complete_days = int((df2.groupby("date")["n_missing_15"].sum() == 0).sum())

    peak_frp = float(df["frp_sum_mw"].max())
    if np.isnan(peak_frp):
        peak_frp = 0.0

    return {
        "fire_slots":    int(fire_count),
        "gap_rate":      round(gap_rate, 4),
        "sat_rate":      round(sat_rate, 4),
        "complete_days": complete_days,
        "peak_frp_mw":   round(peak_frp, 1),
    }


# ── Diagnostic plot ───────────────────────────────────────────────────────────
def make_plot(df, metrics):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    df2 = df.copy()
    df2["frp_plot"] = df2["frp_sum_mw"].fillna(0)
    colors = df2["slot_type"].map({"fire": "orangered", "no_fire": "steelblue", "missing": "lightgray"})

    ax = axes[0]
    ax.bar(df2["slot_utc"], df2["frp_plot"], width=1/24, color=colors, align="edge")
    ax.set_ylabel("FRP sum (MW)")
    ax.set_title(
        f"{EVENT_NAME}  |  peak={metrics['peak_frp_mw']:.0f} MW  "
        f"gap={metrics['gap_rate']:.1%}  sat={metrics['sat_rate']:.1%}"
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    ax2 = axes[1]
    ax2.bar(df2["slot_utc"], df2["n_pixels"], width=1/24, color="steelblue", align="edge")
    ax2.set_ylabel("Fire pixels")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.tight_layout()
    out = PLOT_DIR / f"{EVENT_NAME}_diagnostic.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {out}")


# ── Update catalogue + manifest ───────────────────────────────────────────────
def update_catalogue(metrics):
    cat_path = PROC_DIR / "event_catalogue.csv"
    row = {
        "event_name":    EVENT_NAME,
        "start":         START_STR,
        "end":           END_STR,
        "lon_min":       LON_MIN, "lon_max": LON_MAX,
        "lat_min":       LAT_MIN, "lat_max": LAT_MAX,
        "split":         SPLIT,
        **metrics,
        "date_added":    datetime.utcnow().strftime("%Y-%m-%d"),
    }
    if cat_path.exists():
        cat = pd.read_csv(cat_path)
        cat = cat[cat["event_name"] != EVENT_NAME]
        cat = pd.concat([cat, pd.DataFrame([row])], ignore_index=True)
    else:
        cat = pd.DataFrame([row])
    cat.to_csv(cat_path, index=False)
    print(f"  Catalogue: {cat_path}")


def update_manifest(metrics, phase2_ready):
    mpath = PROC_DIR / "data_manifest.json"
    with open(mpath) as f:
        manifest = json.load(f)
    manifest[EVENT_NAME] = {
        "start": START_STR, "end": END_STR,
        "bbox":  {"lon_min": LON_MIN, "lon_max": LON_MAX,
                  "lat_min": LAT_MIN, "lat_max": LAT_MAX},
        "split": SPLIT,
        "phase2_ready": phase2_ready,
        **metrics,
    }
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: {mpath}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*70}")
    print(f"EVENT: {EVENT_NAME}")
    print(f"Range: {START_STR} -> {END_STR}")
    print(f"BBox:  lon [{LON_MIN}, {LON_MAX}]  lat [{LAT_MIN}, {LAT_MAX}]")
    print(f"Split: {SPLIT}")
    print(f"{'='*70}")

    slots = slot_range(START_DATE, END_DATE)
    print(f"Total 15-min slots: {len(slots)}")

    # Phase A
    fire_slots, missing_list = phase_a(slots)

    # Phase B
    fire_slots = phase_b(fire_slots)

    # Build hourly
    print("\nBuilding hourly time series ...")
    hourly_df = build_hourly(slots, fire_slots, missing_list)

    # Verify
    print("\nVerification ...")
    verify(hourly_df)

    # Audit
    print("\nAudit metrics ...")
    metrics = audit(hourly_df)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    pass_gap  = metrics["gap_rate"]      < 0.40
    pass_sat  = metrics["sat_rate"]      < 0.30
    pass_days = metrics["complete_days"] >= 3
    phase2_ready = pass_gap and pass_sat and pass_days

    # Plot
    make_plot(hourly_df, metrics)

    # Update files if pass
    if phase2_ready:
        print("\nPhase2 ready: YES -- updating catalogue and manifest")
        update_catalogue(metrics)
        update_manifest(metrics, True)
    else:
        reasons = []
        if not pass_gap:  reasons.append(f"gap_rate={metrics['gap_rate']:.3f} >= 0.40")
        if not pass_sat:  reasons.append(f"sat_rate={metrics['sat_rate']:.3f} >= 0.30")
        if not pass_days: reasons.append(f"complete_days={metrics['complete_days']} < 3")
        print(f"\nPhase2 ready: NO -- {'; '.join(reasons)}")

    # Final report
    print(f"\n{'='*70}")
    print(f"SUMMARY: {EVENT_NAME}")
    print(f"  Range:          {START_STR} -> {END_STR}")
    print(f"  BBox:           lon [{LON_MIN}, {LON_MAX}]  lat [{LAT_MIN}, {LAT_MAX}]")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"  phase2_ready:   {'YES' if phase2_ready else 'NO'}")
    print(f"{'='*70}\n")

    result = {"event": EVENT_NAME, "phase2_ready": phase2_ready, **metrics}
    print(f"RESULT_JSON: {json.dumps({k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in result.items()})}")


if __name__ == "__main__":
    main()
