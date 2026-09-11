"""
Tasks 5+6+7: Compute audit metrics, generate diagnostic plots, print summary table.

Reads data/processed/{event}_hourly.csv for each event.

Metrics per event:
  gap_rate       – fraction of hourly slots that are 'missing'
  sat_rate       – fraction of fire slots where sat_fraction > 0.3
  complete_days  – days with < 50% missing hourly slots
  peak_frp_mw    – max frp_sum_mw across all fire slots

Thresholds (pass/fail):
  gap_rate < 0.40
  sat_rate < 0.30
  complete_days >= 3

Diagnostic plot (3-panel PNG) per event:
  Panel 1: FRP time series with gap overlay
  Panel 2: Saturation fraction bar chart
  Panel 3: Pixel count time series
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR  = "e:/GRSL_Wildfire/FRP_Methodology_03062026"
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
PLOTS_DIR = os.path.join(BASE_DIR, "data", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

EVENTS = [
    "evia_2021",
    "varnavas_2024",
    "alexandroupolis_2023",
    "monchique_2018",
    "sardinia_2021",
]

THRESHOLDS = {"gap_rate": 0.40, "sat_rate": 0.30, "complete_days": 3}


def compute_metrics(dfh):
    total_slots   = len(dfh)
    n_missing     = (dfh["slot_type"] == "missing").sum()
    fire_mask     = dfh["slot_type"] == "fire"
    n_fire        = fire_mask.sum()

    gap_rate = n_missing / total_slots if total_slots > 0 else np.nan

    sat_vals = dfh.loc[fire_mask, "sat_fraction"].dropna()
    sat_rate = (sat_vals > 0.3).mean() if len(sat_vals) > 0 else np.nan

    # Complete days: days where < 50% of the 24 hourly slots are 'missing'
    dfh2 = dfh.copy()
    dfh2["date"] = pd.to_datetime(dfh2["slot_utc"]).dt.date
    def day_complete(g):
        return (g["slot_type"] == "missing").mean() < 0.5
    complete_days = dfh2.groupby("date").apply(day_complete).sum()

    peak_frp = dfh.loc[fire_mask, "frp_sum_mw"].max() if n_fire > 0 else 0.0

    return {
        "gap_rate":      round(float(gap_rate), 4),
        "sat_rate":      round(float(sat_rate) if not np.isnan(sat_rate) else np.nan, 4),
        "complete_days": int(complete_days),
        "peak_frp_mw":  round(float(peak_frp), 1),
        "total_slots":  int(total_slots),
        "n_fire_slots": int(n_fire),
        "n_missing":    int(n_missing),
    }


def make_plot(event_name, dfh, metrics):
    dfh = dfh.copy()
    dfh["slot_utc"] = pd.to_datetime(dfh["slot_utc"])
    dfh = dfh.sort_values("slot_utc")

    fire_mask    = dfh["slot_type"] == "fire"
    miss_mask    = dfh["slot_type"] == "missing"
    t            = dfh["slot_utc"].values
    frp          = np.where(fire_mask, dfh["frp_sum_mw"], np.nan)
    sat_frac     = np.where(fire_mask, dfh["sat_fraction"].fillna(0), np.nan)
    n_pix        = np.where(fire_mask, dfh["n_pixels"], 0).astype(float)
    n_pix_plot   = np.where(fire_mask, dfh["n_pixels"], np.nan)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(
        f"{event_name.replace('_', ' ').title()}\n"
        f"gap_rate={metrics['gap_rate']:.2%}  "
        f"sat_rate={metrics['sat_rate']:.2%}  "
        f"complete_days={metrics['complete_days']}  "
        f"peak_FRP={metrics['peak_frp_mw']:.0f} MW",
        fontsize=11, fontweight="bold"
    )

    # ── Panel 1: FRP time series + gap overlay ────────────────────────────────
    ax0 = axes[0]
    # Shade gap periods
    in_gap = False
    gap_start = None
    for i, row in dfh.iterrows():
        is_gap = row["slot_type"] == "missing"
        if is_gap and not in_gap:
            gap_start = row["slot_utc"]
            in_gap    = True
        elif not is_gap and in_gap:
            ax0.axvspan(gap_start, row["slot_utc"], color="red", alpha=0.25, lw=0)
            in_gap = False
    if in_gap and gap_start is not None:
        ax0.axvspan(gap_start, dfh["slot_utc"].iloc[-1], color="red", alpha=0.25, lw=0)

    ax0.plot(t, frp, color="darkorange", lw=1.2, label="FRP sum (MW)")
    ax0.set_ylabel("FRP sum (MW)", fontsize=9)
    ax0.legend(handles=[
        plt.Line2D([0],[0], color="darkorange", lw=1.2, label="FRP sum (MW)"),
        Patch(color="red", alpha=0.25, label="Missing / gap"),
    ], fontsize=8, loc="upper right")
    ax0.grid(True, alpha=0.3)

    # ── Panel 2: Saturation fraction ──────────────────────────────────────────
    ax1 = axes[1]
    ax1.bar(t, np.nan_to_num(sat_frac, nan=0), width=0.04, color="firebrick",
            alpha=0.8, label="sat fraction")
    ax1.axhline(0.3, color="k", lw=0.8, ls="--", label="threshold 0.30")
    ax1.set_ylabel("Sat. fraction", fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # ── Panel 3: Pixel count ──────────────────────────────────────────────────
    ax2 = axes[2]
    ax2.bar(t, np.nan_to_num(n_pix_plot, nan=0), width=0.04,
            color="steelblue", alpha=0.8, label="n_pixels")
    ax2.set_ylabel("Pixel count", fontsize=9)
    ax2.set_xlabel("UTC", fontsize=9)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)

    # x-axis formatting
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator())
    fig.autofmt_xdate(rotation=30, ha="right")

    out_path = os.path.join(PLOTS_DIR, f"{event_name}_diagnostic.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out_path}")
    return out_path


def flag(val, threshold, higher_is_better=False):
    """Return 'PASS' or 'FAIL' string."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A "
    if higher_is_better:
        return "PASS" if val >= threshold else "FAIL"
    else:
        return "PASS" if val <= threshold else "FAIL"


def print_summary(results):
    print("\n" + "="*90)
    print("AUDIT SUMMARY TABLE")
    print("="*90)
    hdr = (f"{'Event':<25s}  {'gap_rate':>8s}  {'sat_rate':>8s}  "
           f"{'comp_days':>9s}  {'peak_MW':>9s}  {'gap_flag':>8s}  "
           f"{'sat_flag':>8s}  {'days_flag':>9s}  {'overall':>7s}")
    print(hdr)
    print("-"*90)

    for ev, m in results.items():
        gap_f  = flag(m["gap_rate"],      THRESHOLDS["gap_rate"],      False)
        sat_f  = flag(m["sat_rate"],      THRESHOLDS["sat_rate"],      False)
        days_f = flag(m["complete_days"], THRESHOLDS["complete_days"], True)
        overall = "OK " if all(f == "PASS" for f in [gap_f, sat_f, days_f]) else "FLAG"

        sat_str  = f"{m['sat_rate']:.2%}" if not (isinstance(m['sat_rate'], float) and np.isnan(m['sat_rate'])) else "  N/A "

        print(f"{ev:<25s}  {m['gap_rate']:>8.2%}  {sat_str:>8s}  "
              f"{m['complete_days']:>9d}  {m['peak_frp_mw']:>9.1f}  "
              f"{gap_f:>8s}  {sat_f:>8s}  {days_f:>9s}  {overall:>7s}")
    print("="*90)
    print(f"Thresholds: gap_rate < {THRESHOLDS['gap_rate']:.0%}  |  "
          f"sat_rate < {THRESHOLDS['sat_rate']:.0%}  |  "
          f"complete_days >= {THRESHOLDS['complete_days']}")
    print("="*90)


if __name__ == "__main__":
    results = {}
    for event_name in EVENTS:
        csv_path = os.path.join(PROC_DIR, f"{event_name}_hourly.csv")
        if not os.path.exists(csv_path):
            print(f"[{event_name}] WARNING: {csv_path} not found — skipping")
            continue
        dfh = pd.read_csv(csv_path, parse_dates=["slot_utc"])
        m   = compute_metrics(dfh)
        print(f"\n[{event_name}] Metrics:")
        for k, v in m.items():
            print(f"  {k}: {v}")
        make_plot(event_name, dfh, m)
        results[event_name] = m

    print_summary(results)
    print("\n[audit_metrics] Done.")
