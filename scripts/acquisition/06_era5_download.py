"""
06_era5_download.py — Phase 2B: ERA5 covariate extraction for all 26 wildfire events.

Downloads hourly ERA5 single-level and pressure-level data for each event's
spatiotemporal domain, then post-processes into aligned hourly covariate CSVs.

Usage:
    python scripts/06_era5_download.py [--event EVENT_NAME] [--skip-download] [--verify-only]

Options:
    --event       Process only this event (default: all 26)
    --skip-download  Skip CDS downloads, only post-process existing NetCDF
    --verify-only    Skip download + post-process, just run verification
"""

import argparse
import json
import math
import os
import sys
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CATALOGUE = BASE_DIR / "data" / "processed" / "event_catalogue.csv"
ERA5_DIR  = BASE_DIR / "data" / "era5"
PROC_DIR  = BASE_DIR / "data" / "processed"
VERIF_OUT = PROC_DIR / "era5_verification.json"

ERA5_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# ERA5 variable lists
# ---------------------------------------------------------------------------
SINGLE_VARS = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "boundary_layer_height",
    "total_precipitation",
]

PRESSURE_VARS = [
    "u_component_of_wind",
    "v_component_of_wind",
]
PRESSURE_LEVEL = ["850"]

HOURS_ALL = [f"{h:02d}:00" for h in range(24)]


# ---------------------------------------------------------------------------
# Load event catalogue (handles both flat and bbox-nested schemas)
# ---------------------------------------------------------------------------

def load_catalogue() -> list[dict]:
    df = pd.read_csv(CATALOGUE)
    events = []
    for _, row in df.iterrows():
        ev = row.to_dict()
        # normalise: some rows might have NaN duration_days — compute from dates
        start = pd.Timestamp(ev["start"])
        end   = pd.Timestamp(ev["end"])
        ev["start_dt"] = start
        ev["end_dt"]   = end
        # flat bbox columns are present in catalogue CSV for all events
        events.append(ev)
    return events


# ---------------------------------------------------------------------------
# CDS download helpers
# ---------------------------------------------------------------------------

def _date_range_str(start_dt: pd.Timestamp, end_dt: pd.Timestamp, lead_days: int = 3):
    """Return (date_start_str, date_end_str) with lead-time prepended."""
    dl_start = start_dt - timedelta(days=lead_days)
    dl_end   = end_dt
    return dl_start.strftime("%Y-%m-%d"), dl_end.strftime("%Y-%m-%d")


def _area(lat_min, lat_max, lon_min, lon_max, buf=0.25):
    """CDS area parameter [N, W, S, E] with buffer, snapped to 0.25°."""
    N = round(lat_max + buf, 2)
    W = round(lon_min - buf, 2)
    S = round(lat_min - buf, 2)
    E = round(lon_max + buf, 2)
    return [N, W, S, E]


def _unwrap_zip_if_needed(path: Path) -> None:
    """
    The new CDS API (v2) sometimes wraps the NetCDF in a ZIP archive, and may
    split variables across multiple NC files (e.g. instant vs. accum step types).
    If `path` is a ZIP: extract all .nc members, merge with xr.merge, save as one.
    """
    if not path.exists():
        return
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic[:2] != b"PK":
        return  # not a ZIP
    print(f"  Unwrapping ZIP: {path.name}")
    tmp_dir = path.parent / (path.stem + "_tmp_zip")
    tmp_dir.mkdir(exist_ok=True)
    try:
        with zipfile.ZipFile(path) as zf:
            nc_members = [m for m in zf.namelist() if m.endswith(".nc")]
            if not nc_members:
                raise RuntimeError(f"No .nc file found inside {path}")
            extracted = []
            for m in nc_members:
                dest = tmp_dir / Path(m).name
                with zf.open(m) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                extracted.append(dest)
        if len(extracted) == 1:
            path.unlink()
            extracted[0].rename(path)
        else:
            # Merge all NC files into one, close all handles before cleanup
            print(f"  Merging {len(extracted)} NC files: {[e.name for e in extracted]}")
            datasets = [xr.open_dataset(p) for p in extracted]
            # Windows spanning the ERA5-final/ERA5T boundary carry conflicting
            # 'expver' labels (0001 vs 0005) across stream files; the label is
            # provenance only — drop it before merging (2026-08-24 fix,
            # first hit: es_san_bartolome_de_l_2026_0605).
            datasets = [ds.drop_vars("expver") if "expver" in ds.variables
                        else ds for ds in datasets]
            merged = xr.merge(datasets, compat="override")
            # Close source datasets immediately before writing
            for ds in datasets:
                ds.close()
            tmp_nc = path.with_suffix(".tmp_nc")
            merged.to_netcdf(str(tmp_nc))
            merged.close()
            path.unlink()
            tmp_nc.rename(path)
    finally:
        # Give OS a moment to release file handles on Windows
        time.sleep(0.5)
        for f in tmp_dir.glob("*"):
            try:
                f.unlink()
            except PermissionError:
                pass  # will be cleaned up next run
        try:
            tmp_dir.rmdir()
        except OSError:
            pass
    print(f"  Unwrapped -> {path.name}")


def download_single(event: dict, client: cdsapi.Client, out_path: Path) -> None:
    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists")
        return

    date_start, date_end = _date_range_str(event["start_dt"], event["end_dt"])
    area = _area(event["lat_min"], event["lat_max"], event["lon_min"], event["lon_max"])
    date_range = f"{date_start}/{date_end}"

    print(f"  Requesting single-level: {date_start} -> {date_end}, area={area}")
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": SINGLE_VARS,
            "date":   date_range,
            "time":   HOURS_ALL,
            "area":   area,
            "format": "netcdf",
        },
        str(out_path),
    )
    _unwrap_zip_if_needed(out_path)


def download_pressure(event: dict, client: cdsapi.Client, out_path: Path) -> None:
    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists")
        return

    date_start, date_end = _date_range_str(event["start_dt"], event["end_dt"])
    area = _area(event["lat_min"], event["lat_max"], event["lon_min"], event["lon_max"])
    date_range = f"{date_start}/{date_end}"

    print(f"  Requesting pressure-level: {date_start} -> {date_end}, area={area}")
    client.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "variable": PRESSURE_VARS,
            "pressure_level": PRESSURE_LEVEL,
            "date":   date_range,
            "time":   HOURS_ALL,
            "area":   area,
            "format": "netcdf",
        },
        str(out_path),
    )
    _unwrap_zip_if_needed(out_path)


# ---------------------------------------------------------------------------
# Spatial aggregation helper
# ---------------------------------------------------------------------------

def area_weighted_mean(ds: xr.Dataset, lat_min, lat_max, lon_min, lon_max, buf=0.25) -> xr.Dataset:
    """
    Select grid cells within the buffered bbox and compute area-weighted
    (cos-latitude) mean for each time step.
    Returns an xr.Dataset with dimension 'valid_time' only.
    """
    # ERA5 NetCDF may use 'latitude'/'longitude' or 'lat'/'lon'
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"

    lat_lo = lat_min - buf
    lat_hi = lat_max + buf
    lon_lo = lon_min - buf
    lon_hi = lon_max + buf

    # Handle both ascending and descending latitude axes
    lats = ds[lat_name].values
    lons = ds[lon_name].values

    lat_mask = (lats >= lat_lo) & (lats <= lat_hi)
    lon_mask = (lons >= lon_lo) & (lons <= lon_hi)

    ds_sub = ds.sel({lat_name: lats[lat_mask], lon_name: lons[lon_mask]})

    n_cells = lat_mask.sum() * lon_mask.sum()

    # Area weights: cos(lat) for each latitude in subset
    sub_lats = ds_sub[lat_name].values
    weights  = np.cos(np.deg2rad(sub_lats))
    # Build DataArray for broadcasting
    weight_da = xr.DataArray(weights, dims=[lat_name], coords={lat_name: sub_lats})

    # Weighted mean over spatial dims
    weighted = ds_sub.weighted(weight_da)
    ds_mean  = weighted.mean(dim=[lat_name, lon_name])

    ds_mean.attrs["n_cells"] = int(n_cells)
    return ds_mean


# ---------------------------------------------------------------------------
# Precipitation hourly increment (ERA5 accumulation reset at 06Z and 18Z)
# ---------------------------------------------------------------------------

def compute_precip_1h(tp_series: pd.Series) -> pd.Series:
    """
    Convert ERA5 cumulative tp (accumulated from start of each forecast window)
    into hourly increments.

    ERA5 accumulation windows reset at 06:00 and 18:00 UTC.
    For the first hour of each window (hour == 6 or hour == 18), the raw tp
    value IS the 1-hour accumulation. For all other hours, diff with previous.
    """
    result = pd.Series(index=tp_series.index, dtype=float)
    times  = tp_series.index

    for i, ts in enumerate(times):
        h = ts.hour
        if h in (6, 18) or i == 0:
            # First hour of a new accumulation window → raw value
            result.iloc[i] = tp_series.iloc[i]
        else:
            diff = tp_series.iloc[i] - tp_series.iloc[i - 1]
            # Negative diff would indicate an unexpected reset; clip to 0
            result.iloc[i] = max(diff, 0.0)

    return result


# ---------------------------------------------------------------------------
# Post-processing: produce ERA5 hourly CSV for one event
# ---------------------------------------------------------------------------

def postprocess_event(event: dict, single_nc: Path, pressure_nc: Path) -> pd.DataFrame:
    name    = event["event_name"]
    lat_min = event["lat_min"]
    lat_max = event["lat_max"]
    lon_min = event["lon_min"]
    lon_max = event["lon_max"]

    # ---- Load NetCDF ----
    ds_sl = xr.open_dataset(single_nc)
    ds_pl = xr.open_dataset(pressure_nc)

    # Drop non-physical scalar variables that CDS API sometimes adds as data vars
    _drop_vars = ["number", "expver"]
    ds_sl = ds_sl.drop_vars([v for v in _drop_vars if v in ds_sl], errors="ignore")
    ds_pl = ds_pl.drop_vars([v for v in _drop_vars if v in ds_pl], errors="ignore")

    # For pressure-level: select 850 hPa and squeeze that dimension
    if "pressure_level" in ds_pl.dims:
        ds_pl = ds_pl.sel(pressure_level=850).drop_vars("pressure_level", errors="ignore")
    elif "pressure_level" in ds_pl.coords:
        ds_pl = ds_pl.drop_vars("pressure_level", errors="ignore")

    # ---- Spatial aggregation ----
    sl_mean = area_weighted_mean(ds_sl, lat_min, lat_max, lon_min, lon_max)
    pl_mean = area_weighted_mean(ds_pl, lat_min, lat_max, lon_min, lon_max)

    n_cells_sl = sl_mean.attrs.get("n_cells", "?")
    print(f"  {name}: {n_cells_sl} ERA5 cells used for spatial average")

    # ---- Convert to DataFrames ----
    def _to_df(ds_mean: xr.Dataset) -> pd.DataFrame:
        df = ds_mean.to_dataframe().reset_index()
        time_col = "valid_time" if "valid_time" in df.columns else "time"
        df = df.rename(columns={time_col: "slot_utc"})
        df["slot_utc"] = pd.to_datetime(df["slot_utc"]).dt.tz_localize(None)
        df = df.set_index("slot_utc").sort_index()
        # Drop any lingering dimension columns
        for c in ["number", "expver", "pressure_level"]:
            if c in df.columns:
                df = df.drop(columns=c)
        return df

    sl_df = _to_df(sl_mean)
    pl_df = _to_df(pl_mean)

    # Identify actual column names in the NetCDF (CDS uses short names)
    def _col(df, *candidates):
        for c in candidates:
            if c in df.columns:
                return df[c]
        raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")

    u10  = _col(sl_df, "u10")
    v10  = _col(sl_df, "v10")
    t2m  = _col(sl_df, "t2m")
    d2m  = _col(sl_df, "d2m")
    blh  = _col(sl_df, "blh")
    tp   = _col(sl_df, "tp")

    u850 = _col(pl_df, "u", "u850")
    v850 = _col(pl_df, "v", "v850")

    # Align indices
    idx = u10.index.sort_values()
    u10  = u10.reindex(idx)
    v10  = v10.reindex(idx)
    t2m  = t2m.reindex(idx)
    d2m  = d2m.reindex(idx)
    blh  = blh.reindex(idx)
    tp   = tp.reindex(idx)
    u850 = u850.reindex(idx)
    v850 = v850.reindex(idx)

    # ---- Derived variables ----
    wind_speed_10m  = np.sqrt(u10**2 + v10**2)
    wind_dir_10m    = (np.degrees(np.arctan2(-u10, -v10)) % 360)
    wind_speed_850  = np.sqrt(u850**2 + v850**2)

    # VPD [kPa]
    t2m_C = t2m - 273.15
    d2m_C = d2m - 273.15
    es = 0.6108 * np.exp(17.27 * t2m_C / (t2m_C + 237.3))
    ea = 0.6108 * np.exp(17.27 * d2m_C / (d2m_C + 237.3))
    vpd = es - ea

    # Precipitation [m -> mm]
    precip_1h_m  = compute_precip_1h(tp)
    precip_1h_mm = precip_1h_m * 1000.0

    # Rolling 24h sum (backward)
    precip_24h_mm = precip_1h_mm.rolling(24, min_periods=1).sum()

    # Delta wind 3h (signed, m/s)
    delta_wind_3h = wind_speed_10m - wind_speed_10m.shift(3)

    # ---- Assemble output ----
    out = pd.DataFrame({
        "slot_utc":        idx,
        "u10":             u10.values,
        "v10":             v10.values,
        "t2m_K":           t2m.values,
        "d2m_K":           d2m.values,
        "blh_m":           blh.values,
        "tp_m":            tp.values,
        "u850":            u850.values,
        "v850":            v850.values,
        "wind_speed_10m":  wind_speed_10m.values,
        "wind_dir_10m":    wind_dir_10m.values,
        "wind_speed_850":  wind_speed_850.values,
        "vpd_kPa":         vpd.values,
        "precip_1h_mm":    precip_1h_mm.values,
        "precip_24h_mm":   precip_24h_mm.values,
        "delta_wind_3h":   delta_wind_3h.values,
    })
    out["slot_utc"] = pd.to_datetime(out["slot_utc"])

    ds_sl.close()
    ds_pl.close()

    return out


# ---------------------------------------------------------------------------
# Temporal alignment check
# ---------------------------------------------------------------------------

def check_alignment(event_name: str, era5_df: pd.DataFrame) -> dict:
    frp_path = PROC_DIR / f"{event_name}_hourly.csv"
    if not frp_path.exists():
        return {"status": "FRP_CSV_NOT_FOUND", "missing_hours": []}

    frp = pd.read_csv(frp_path, parse_dates=["slot_utc"])
    frp["slot_utc"] = pd.to_datetime(frp["slot_utc"])

    era5_times = set(era5_df["slot_utc"].dt.floor("h"))
    frp_times  = set(frp["slot_utc"].dt.floor("h"))

    missing = sorted(frp_times - era5_times)
    missing_str = [str(t) for t in missing]

    if missing:
        print(f"  WARNING {event_name}: {len(missing)} FRP hours have no ERA5 match")
        for m in missing_str[:5]:
            print(f"    {m}")
    else:
        print(f"  {event_name}: alignment OK — {len(frp_times)} FRP hours all matched")

    return {
        "era5_hours": len(era5_times),
        "frp_hours":  len(frp_times),
        "matched":    len(frp_times) - len(missing),
        "missing":    len(missing),
        "missing_list": missing_str,
    }


# ---------------------------------------------------------------------------
# Verification summary
# ---------------------------------------------------------------------------

def run_verification(events: list[dict]) -> dict:
    results = {}
    header  = f"{'Event':<28} {'ERA5 h':>7} {'FRP h':>6} {'Match':>6} {'Miss':>5} {'VPD min':>8} {'Wind max':>9} {'P1h<0':>6}"
    print("\n" + "=" * 80)
    print("ERA5 VERIFICATION SUMMARY")
    print("=" * 80)
    print(header)
    print("-" * 80)

    flags = {}
    for ev in events:
        name = ev["event_name"]
        csv_path = PROC_DIR / f"{name}_era5_hourly.csv"
        if not csv_path.exists():
            print(f"  {name:<28} [no ERA5 CSV]")
            flags[name] = {"status": "MISSING_CSV"}
            continue

        df  = pd.read_csv(csv_path, parse_dates=["slot_utc"])
        aln = check_alignment(name, df)

        vpd_min    = float(df["vpd_kPa"].min())
        wind_max   = float(df["wind_speed_10m"].max())
        p1h_neg    = int((df["precip_1h_mm"] < -0.001).sum())

        flag_list = []
        if aln["missing"] > 0:        flag_list.append("ERA5/FRP_MISMATCH")
        if vpd_min < 0:                flag_list.append("VPD_NEGATIVE")
        if wind_max > 30:              flag_list.append("WIND_HIGH")
        if p1h_neg > 0:                flag_list.append("PRECIP_NEGATIVE")

        row = (f"{name:<28} {aln['era5_hours']:>7} {aln['frp_hours']:>6} "
               f"{aln['matched']:>6} {aln['missing']:>5} "
               f"{vpd_min:>8.3f} {wind_max:>9.2f} {p1h_neg:>6}")
        flag_str = "  [" + ", ".join(flag_list) + "]" if flag_list else ""
        print(row + flag_str)

        results[name] = {
            **aln,
            "vpd_min":  vpd_min,
            "wind_max": wind_max,
            "precip_1h_negatives": p1h_neg,
            "flags": flag_list,
        }

    print("=" * 80)

    # Save
    with open(VERIF_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nVerification saved -> {VERIF_OUT}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ERA5 download + post-processing")
    parser.add_argument("--event",        default=None, help="Process single event")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--verify-only",   action="store_true")
    args = parser.parse_args()

    events = load_catalogue()
    if args.event:
        events = [e for e in events if e["event_name"] == args.event]
        if not events:
            sys.exit(f"Event '{args.event}' not found in catalogue")

    if args.verify_only:
        run_verification(events)
        return

    # ---- CDS client ----
    if not args.skip_download:
        client = cdsapi.Client()

    n = len(events)
    for i, ev in enumerate(events, 1):
        name = ev["event_name"]
        print(f"\n[{i}/{n}] {name}")

        single_nc   = ERA5_DIR / f"{name}_era5_single.nc"
        pressure_nc = ERA5_DIR / f"{name}_era5_pressure.nc"
        era5_csv    = PROC_DIR / f"{name}_era5_hourly.csv"

        # ---- Download ----
        if not args.skip_download:
            try:
                print(f"  Downloading {name} ({i}/{n})... single-level")
                download_single(ev, client, single_nc)
                print(f"  Downloading {name} ({i}/{n})... pressure-level")
                download_pressure(ev, client, pressure_nc)
            except Exception as exc:
                print(f"  ERROR downloading {name}: {exc}")
                print("  Skipping post-processing for this event.")
                continue

        # ---- Post-process ----
        if not single_nc.exists() or not pressure_nc.exists():
            print(f"  Skipping post-process — NetCDF not found for {name}")
            continue

        if era5_csv.exists():
            print(f"  [skip] {era5_csv.name} already exists — re-run with --skip-download to reprocess")
            continue

        try:
            print(f"  Post-processing {name}...")
            df = postprocess_event(ev, single_nc, pressure_nc)
            df.to_csv(era5_csv, index=False)
            print(f"  Saved -> {era5_csv.name}  ({len(df)} rows)")

            # Quick alignment check
            check_alignment(name, df)
        except Exception as exc:
            print(f"  ERROR post-processing {name}: {exc}")
            import traceback; traceback.print_exc()

    # ---- Final verification ----
    print("\n--- Running final verification ---")
    run_verification(events)


if __name__ == "__main__":
    main()
