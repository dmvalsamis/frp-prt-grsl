"""
Task 4: Build hourly time series per event.

Reads per-slot CSVs from data/processed/{event}/slots/ and the
missing_list.csv, then aggregates to hourly resolution:

  frp_sum_mw   – sum of pixel FRP in bbox
  n_pixels     – active fire pixel count
  sat_fraction – fraction of pixels with saturated == True
  slot_type    – 'fire' / 'no_fire' / 'missing'

Output: data/processed/{event}_hourly.csv
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = "e:/GRSL_Wildfire/FRP_Methodology_03062026"
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

EVENTS = {
    "evia_2021":            {"start": "2021-08-03", "end": "2021-08-12"},
    "varnavas_2024":        {"start": "2024-08-11", "end": "2024-08-16"},
    "alexandroupolis_2023": {"start": "2023-08-19", "end": "2023-08-28"},
    "monchique_2018":       {"start": "2018-08-03", "end": "2018-08-11"},
    "sardinia_2021":        {"start": "2021-07-24", "end": "2021-07-31"},
}


def slot_range_15min(start_str, end_str):
    t   = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str,   "%Y-%m-%d") + timedelta(days=1)
    slots = []
    while t < end:
        slots.append(t)
        t += timedelta(minutes=15)
    return slots


def build_hourly(event_name, cfg):
    slots_dir = os.path.join(PROC_DIR, event_name, "slots")
    miss_path = os.path.join(PROC_DIR, event_name, "missing_list.csv")

    # --- load missing 15-min slots ---
    missing_15min = set()
    if os.path.exists(miss_path):
        df_miss = pd.read_csv(miss_path)
        for v in df_miss["slot_utc"]:
            try:
                missing_15min.add(pd.Timestamp(v))
            except Exception:
                pass

    # --- load all fire-slot CSVs ---
    csv_files = glob.glob(os.path.join(slots_dir, "*.csv"))
    fire_dfs  = {}
    for fp in csv_files:
        slot_str = os.path.basename(fp).replace(".csv", "")
        try:
            dt = pd.Timestamp(datetime.strptime(slot_str, "%Y%m%d%H%M"))
            df = pd.read_csv(fp)
            fire_dfs[dt] = df
        except Exception:
            pass

    # --- enumerate all expected 15-min slots ---
    all_15min = slot_range_15min(cfg["start"], cfg["end"])

    # --- build per-slot records ---
    records_15 = []
    for dt_raw in all_15min:
        dt = pd.Timestamp(dt_raw)
        if dt in missing_15min:
            records_15.append({"slot_utc": dt, "frp_sum_mw": np.nan,
                                "n_pixels": 0,  "sat_fraction": np.nan,
                                "slot_type": "missing"})
        elif dt in fire_dfs:
            df = fire_dfs[dt]
            frp_sum = df["frp_mw"].sum()
            n_pix   = len(df)
            sat_frac = df["saturated"].mean() if n_pix > 0 else 0.0
            records_15.append({"slot_utc": dt, "frp_sum_mw": frp_sum,
                                "n_pixels": n_pix, "sat_fraction": sat_frac,
                                "slot_type": "fire"})
        else:
            # slot exists but no fire pixels in bbox
            records_15.append({"slot_utc": dt, "frp_sum_mw": 0.0,
                                "n_pixels": 0,  "sat_fraction": 0.0,
                                "slot_type": "no_fire"})

    df15 = pd.DataFrame(records_15)
    df15["slot_utc"] = pd.to_datetime(df15["slot_utc"])
    df15 = df15.sort_values("slot_utc").reset_index(drop=True)

    # --- aggregate to hourly ---
    df15["hour_utc"] = df15["slot_utc"].dt.floor("h")

    def agg_hour(g):
        n_total   = len(g)
        n_missing = (g["slot_type"] == "missing").sum()
        n_fire    = (g["slot_type"] == "fire").sum()
        n_nofire  = (g["slot_type"] == "no_fire").sum()

        frp_sum   = g.loc[g["slot_type"] == "fire", "frp_sum_mw"].sum()
        n_pixels  = g.loc[g["slot_type"] == "fire", "n_pixels"].sum()
        sat_frac  = (g.loc[g["slot_type"] == "fire", "sat_fraction"].mean()
                     if n_fire > 0 else np.nan)

        # Hourly slot_type: majority rule
        if n_missing > n_total / 2:
            slot_type = "missing"
        elif n_fire > 0:
            slot_type = "fire"
        else:
            slot_type = "no_fire"

        return pd.Series({
            "frp_sum_mw":   frp_sum,
            "n_pixels":     n_pixels,
            "sat_fraction": sat_frac,
            "slot_type":    slot_type,
            "n_15min_slots": n_total,
            "n_missing_15":  n_missing,
        })

    dfh = df15.groupby("hour_utc").apply(agg_hour).reset_index()
    dfh.rename(columns={"hour_utc": "slot_utc"}, inplace=True)
    dfh = dfh.sort_values("slot_utc").reset_index(drop=True)

    out_path = os.path.join(PROC_DIR, f"{event_name}_hourly.csv")
    dfh.to_csv(out_path, index=False)

    n_fire    = (dfh["slot_type"] == "fire").sum()
    n_missing = (dfh["slot_type"] == "missing").sum()
    n_nofire  = (dfh["slot_type"] == "no_fire").sum()
    print(f"[{event_name}] hourly: {len(dfh)} slots  "
          f"fire={n_fire}  no_fire={n_nofire}  missing={n_missing}  -> {out_path}")
    return dfh


if __name__ == "__main__":
    for event_name, cfg in EVENTS.items():
        try:
            build_hourly(event_name, cfg)
        except Exception as e:
            print(f"[{event_name}] ERROR: {e}")
            import traceback; traceback.print_exc()
    print("\n[build_timeseries] Done.")
