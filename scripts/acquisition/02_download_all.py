"""
Tasks 2+3: Parser + bulk download of all ListProduct and QualityProduct files
for all 5 wildfire events.

Strategy (to minimise QualityProduct traffic):
  Phase A - Download ALL ListProduct slots; parse each for fire pixels in bbox.
  Phase B - Download QualityProduct ONLY for slots that had fire pixels.
            For each fire pixel, look up QUALITYFLAG[ABS_LINE-1, ABS_PIXEL-1].
            saturated = (flag == 2)

Outputs:
  data/raw/{event}/  – raw HDF5 files
  data/processed/{event}/slots/{YYYYMMDDHHMM}.csv  – per-slot parsed table
  data/processed/{event}/missing_list.csv  – 404'd ListProduct slots
  data/processed/{event}/missing_qual.csv  – 404'd QualityProduct slots
"""

import os, sys, io, time, csv, traceback
import requests
import h5py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(encoding='utf-8')

# ── credentials ───────────────────────────────────────────────────────────────
USERNAME = "<<LSASAF_USERNAME_REDACTED>>"
PASSWORD = "<<LSASAF_PASSWORD_REDACTED>>"
BASE_URL = "https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/MSG/FRP-PIXEL/HDF5"
SESSION  = requests.Session()
SESSION.auth = (USERNAME, PASSWORD)

# ── events ────────────────────────────────────────────────────────────────────
EVENTS = {
    "evia_2021":            {"start": "2021-08-03", "end": "2021-08-12",
                             "lon_min": 23.0, "lon_max": 24.5,
                             "lat_min": 38.3, "lat_max": 39.0},
    "varnavas_2024":        {"start": "2024-08-11", "end": "2024-08-16",
                             "lon_min": 23.7, "lon_max": 24.2,
                             "lat_min": 38.1, "lat_max": 38.5},
    "alexandroupolis_2023": {"start": "2023-08-19", "end": "2023-08-28",
                             "lon_min": 25.8, "lon_max": 26.8,
                             "lat_min": 40.7, "lat_max": 41.4},
    "monchique_2018":       {"start": "2018-08-03", "end": "2018-08-11",
                             "lon_min": -8.8, "lon_max": -8.2,
                             "lat_min": 37.2, "lat_max": 37.6},
    "sardinia_2021":        {"start": "2021-07-24", "end": "2021-07-31",
                             "lon_min":  8.5, "lon_max":  9.5,
                             "lat_min": 40.0, "lat_max": 40.8},
}

BASE_DIR  = "e:/GRSL_Wildfire/FRP_Methodology_03062026"
RAW_DIR   = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
MAX_WORKERS = 10   # parallel download threads

# QUALITYFLAG value that means "saturated MIR channel"
SAT_FLAG = 2

# ── helpers ───────────────────────────────────────────────────────────────────

def slot_range(start_str, end_str):
    """Generate all 15-min UTC slots [start, end) inclusive of end date."""
    t = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d") + timedelta(days=1)
    slots = []
    while t < end:
        slots.append(t)
        t += timedelta(minutes=15)
    return slots


def list_fname(dt):
    return f"HDF5_LSASAF_MSG_FRP-PIXEL-ListProduct_MSG-Disk_{dt:%Y%m%d%H%M}"

def qual_fname(dt):
    return f"HDF5_LSASAF_MSG_FRP-PIXEL-QualityProduct_MSG-Disk_{dt:%Y%m%d%H%M}"

def file_url(dt, fname):
    return f"{BASE_URL}/{dt:%Y}/{dt:%m}/{dt:%d}/{fname}"


def download_to_bytes(url, retries=2):
    """Return (status_code, bytes_or_None)."""
    for attempt in range(retries + 1):
        try:
            r = SESSION.get(url, timeout=90)
            return r.status_code, r.content if r.status_code == 200 else None
        except Exception as e:
            if attempt == retries:
                return 0, None
            time.sleep(2)


def read_hdf5_bytes(raw_bytes):
    """Open HDF5 from in-memory bytes (no temp file)."""
    buf = io.BytesIO(raw_bytes)
    return h5py.File(buf, "r")


def parse_list_product(raw_bytes, bbox, dt):
    """
    Parse ListProduct bytes → DataFrame with real-valued columns.
    Returns empty DataFrame if no fire pixels fall inside bbox.
    """
    lon_min, lon_max = bbox["lon_min"], bbox["lon_max"]
    lat_min, lat_max = bbox["lat_min"], bbox["lat_max"]

    with read_hdf5_bytes(raw_bytes) as hf:
        # Read fields – scale using dataset attributes
        def read_scaled(name):
            ds   = hf[name]
            raw  = ds[()].astype(float)
            mv   = ds.attrs.get("MISSING_VALUE", None)
            sf   = float(ds.attrs.get("SCALING_FACTOR", 1.0))
            off  = float(ds.attrs.get("OFFSET", 0.0))
            if mv is not None:
                raw = np.where(raw == float(mv), np.nan, raw)
            return (raw - off) / sf

        lat  = read_scaled("LATITUDE")         # degrees
        lon  = read_scaled("LONGITUDE")        # degrees
        frp  = read_scaled("FRP")              # MW
        unc  = read_scaled("FRP_UNCERTAINTY")  # MW
        conf = read_scaled("FIRE_CONFIDENCE")  # 0-100 fraction (raw/100)

        abs_line  = hf["ABS_LINE"][()].astype(int)   # 1-based row in 3712x3712
        abs_pixel = hf["ABS_PIXEL"][()].astype(int)  # 1-based col

    # Spatial filter
    mask = (
        (lat >= lat_min) & (lat <= lat_max) &
        (lon >= lon_min) & (lon <= lon_max) &
        np.isfinite(lat) & np.isfinite(lon) & np.isfinite(frp)
    )

    df = pd.DataFrame({
        "slot_utc":   dt,
        "lat":        lat[mask],
        "lon":        lon[mask],
        "frp_mw":     frp[mask],
        "frp_unc_mw": unc[mask],
        "confidence": conf[mask],
        "abs_line":   abs_line[mask],   # for QualityProduct lookup
        "abs_pixel":  abs_pixel[mask],
        "saturated":  False,            # filled later from QualityProduct
    })
    return df


def apply_quality_flags(df, qual_bytes):
    """
    Given a fire-pixel DataFrame and QualityProduct bytes,
    set df['saturated'] = True where QUALITYFLAG == SAT_FLAG.
    """
    with read_hdf5_bytes(qual_bytes) as qhf:
        qflag = qhf["QUALITYFLAG"][()]   # (3712, 3712)

    rows = df["abs_line"].values  - 1   # 0-based
    cols = df["abs_pixel"].values - 1

    valid = (rows >= 0) & (rows < 3712) & (cols >= 0) & (cols < 3712)
    flags = np.zeros(len(df), dtype=int)
    flags[valid] = qflag[rows[valid], cols[valid]]
    df = df.copy()
    df["saturated"] = (flags == SAT_FLAG)
    return df


# ── Phase A: Download + parse ListProducts ───────────────────────────────────

def download_list_slot(args):
    """Worker: download one ListProduct slot → (dt, status, df_or_None)."""
    dt, bbox, raw_dir = args
    fname = list_fname(dt)
    url   = file_url(dt, fname)
    status, data = download_to_bytes(url)
    if status == 200:
        # Optionally save raw file
        out_path = os.path.join(raw_dir, fname)
        if not os.path.exists(out_path):
            with open(out_path, "wb") as f:
                f.write(data)
        try:
            df = parse_list_product(data, bbox, dt)
        except Exception:
            df = pd.DataFrame()
        return dt, status, df
    return dt, status, None


def phase_a(event_name, cfg):
    print(f"\n{'='*70}")
    print(f"[{event_name}] Phase A: downloading ListProduct files")
    print(f"{'='*70}")
    slots     = slot_range(cfg["start"], cfg["end"])
    raw_dir   = os.path.join(RAW_DIR, event_name)
    proc_dir  = os.path.join(PROC_DIR, event_name)
    slots_dir = os.path.join(proc_dir, "slots")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(slots_dir, exist_ok=True)

    missing_list = []
    fire_slots   = []   # (dt, df) where len(df) > 0
    total = len(slots)

    args = [(dt, cfg, raw_dir) for dt in slots]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_list_slot, a): a[0] for a in args}
        done = 0
        for fut in as_completed(futures):
            dt, status, df = fut.result()
            done += 1
            tag = f"[{event_name}] {dt:%Y-%m-%d %H:%M}"
            if status == 200:
                n = len(df) if df is not None else 0
                if done % 50 == 0 or n > 0:
                    print(f"  {tag}  HTTP 200  n_fire_bbox={n}  ({done}/{total})")
                if df is not None and n > 0:
                    fire_slots.append((dt, df))
            elif status == 404:
                missing_list.append(dt.isoformat())
                if done % 100 == 0:
                    print(f"  {tag}  HTTP 404  (missing)  ({done}/{total})")
            else:
                missing_list.append(dt.isoformat())
                print(f"  {tag}  HTTP {status}  (error)  ({done}/{total})")

    # Save missing list
    miss_path = os.path.join(proc_dir, "missing_list.csv")
    pd.DataFrame({"slot_utc": missing_list}).to_csv(miss_path, index=False)
    print(f"\n[{event_name}] Phase A done: {len(slots)} slots, "
          f"{len(missing_list)} missing, {len(fire_slots)} fire slots in bbox")
    return fire_slots, slots_dir, raw_dir, proc_dir


# ── Phase B: Download QualityProduct for fire slots only ────────────────────

def download_qual_slot(args):
    dt, df_fire, raw_dir = args
    fname = qual_fname(dt)
    url   = file_url(dt, fname)
    status, data = download_to_bytes(url)
    if status == 200:
        out_path = os.path.join(raw_dir, fname)
        if not os.path.exists(out_path):
            with open(out_path, "wb") as f:
                f.write(data)
        try:
            df_fire = apply_quality_flags(df_fire, data)
        except Exception as e:
            pass  # keep saturated=False
        return dt, status, df_fire
    return dt, status, df_fire   # keep original df if qual missing


def phase_b(event_name, fire_slots, slots_dir, raw_dir, proc_dir):
    print(f"\n[{event_name}] Phase B: downloading QualityProduct for "
          f"{len(fire_slots)} fire slots")
    missing_qual = []
    args = [(dt, df, raw_dir) for dt, df in fire_slots]

    final_slots = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_qual_slot, a): a[0] for a in args}
        done = 0
        for fut in as_completed(futures):
            dt, status, df = fut.result()
            done += 1
            if status != 200:
                missing_qual.append(dt.isoformat())
            final_slots.append((dt, df))
            if done % 20 == 0:
                print(f"  [{event_name}] Quality {done}/{len(fire_slots)} done")

    # Save per-slot CSVs (drop abs_line/abs_pixel helper cols)
    print(f"[{event_name}] Saving {len(final_slots)} fire-slot CSVs …")
    for dt, df in final_slots:
        out_path = os.path.join(slots_dir, f"{dt:%Y%m%d%H%M}.csv")
        df.drop(columns=["abs_line", "abs_pixel"], errors="ignore").to_csv(
            out_path, index=False)

    miss_path = os.path.join(proc_dir, "missing_qual.csv")
    pd.DataFrame({"slot_utc": missing_qual}).to_csv(miss_path, index=False)
    print(f"[{event_name}] Phase B done: {len(missing_qual)} quality files missing")
    return final_slots


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for event_name, cfg in EVENTS.items():
        try:
            fire_slots, slots_dir, raw_dir, proc_dir = phase_a(event_name, cfg)
            phase_b(event_name, fire_slots, slots_dir, raw_dir, proc_dir)
        except Exception:
            print(f"\n[{event_name}] FATAL ERROR:")
            traceback.print_exc()
    print("\n[download_all] All events complete.")
