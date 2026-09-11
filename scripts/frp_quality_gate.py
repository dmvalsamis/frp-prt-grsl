"""
frp_quality_gate.py — CANONICAL five-criteria event quality gate (consolidated 2026-08-24)

The published gate (FRP_article_final.tex §II-A, verbatim):
  ">=15 fire-active hourly slots, >=5-day duration, >=3 diurnally complete days,
   gap rate <40%, saturation <30%"

Historically these criteria were enforced in pieces (_add_event_worker.py:
gap/sat/complete_days; wave scripts 26-30: fire_slots>=15 + duration>=5d as
candidate filters). This module consolidates them into ONE function for the
resubmission program (Task E1; resubmission/FINDINGS_2026-08-24.md §7.4b).

Metric definitions are copied VERBATIM from _add_event_worker.audit():
  fire_slots    = count of slot_type == "fire"
  gap_rate      = missing slots / total slots
  sat_rate      = mean sat_fraction over fire slots (0.0 if no fire slots / NaN)
  complete_days = calendar days whose summed n_missing_15 == 0
  peak_frp_mw   = max frp_sum_mw  (metadata only — NOT a gate criterion; the
                  intensity floor was empirically rejected, FINDINGS §4)

Duration operationalization (the paper text does not define it): number of
DISTINCT calendar dates in the hourly series >= 5. Validated by the regression
test below: reproduces the historical pass/fail record on all 54 catalogued
events (PASS) and the 5 permanently-excluded events with surviving hourly CSVs
(FAIL). Any change to this file invalidates its SHA-256 anchor in the
pre-registration.

CLI:  python scripts/frp_quality_gate.py            -> run regression test
      python scripts/frp_quality_gate.py <hourly.csv> -> gate one event
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLDS = {
    "min_fire_slots": 15,
    "min_duration_days": 5,
    "min_complete_days": 3,
    "max_gap_rate": 0.40,      # strict <
    "max_sat_rate": 0.30,      # strict <
}

# Permanently excluded events (RESEARCH_LOG / CLAUDE.md) — regression negatives
KNOWN_EXCLUDED = ["halkidiki_2012", "leiria_2013", "galicia_2017",
                  "achaia_ilia_2024", "marmaris_2022", "mati_2018"]


def gate_metrics(df):
    """Compute the five gate metrics from an event hourly CSV dataframe.
    Expects columns: slot_utc (datetime), frp_sum_mw, sat_fraction, slot_type,
    n_missing_15. Logic mirrors _add_event_worker.audit() exactly."""
    total = len(df)
    miss_count = (df["slot_type"] == "missing").sum()
    fire_count = (df["slot_type"] == "fire").sum()
    gap_rate = miss_count / total if total > 0 else 1.0
    sat_vals = df.loc[df["slot_type"] == "fire", "sat_fraction"]
    sat_rate = float(sat_vals.mean()) if len(sat_vals) > 0 else 0.0
    if np.isnan(sat_rate):
        sat_rate = 0.0

    df2 = df.copy()
    df2["date"] = df2["slot_utc"].dt.date
    complete_days = int((df2.groupby("date")["n_missing_15"].sum() == 0).sum())
    duration_days = int(df2["date"].nunique())

    peak_frp = float(df["frp_sum_mw"].max())
    if np.isnan(peak_frp):
        peak_frp = 0.0

    return {
        "fire_slots": int(fire_count),
        "duration_days": duration_days,
        "complete_days": complete_days,
        "gap_rate": round(gap_rate, 4),
        "sat_rate": round(sat_rate, 4),
        "peak_frp_mw": round(peak_frp, 1),   # metadata, not a criterion
    }


def gate_pass(metrics):
    """Apply the five published criteria. Returns (bool, dict of per-criterion bools)."""
    checks = {
        "fire_slots>=15": metrics["fire_slots"] >= THRESHOLDS["min_fire_slots"],
        "duration>=5d": metrics["duration_days"] >= THRESHOLDS["min_duration_days"],
        "complete_days>=3": metrics["complete_days"] >= THRESHOLDS["min_complete_days"],
        "gap_rate<0.40": metrics["gap_rate"] < THRESHOLDS["max_gap_rate"],
        "sat_rate<0.30": metrics["sat_rate"] < THRESHOLDS["max_sat_rate"],
    }
    return all(checks.values()), checks


def gate_event_csv(csv_path):
    df = pd.read_csv(csv_path)
    df["slot_utc"] = pd.to_datetime(df["slot_utc"])
    m = gate_metrics(df)
    ok, checks = gate_pass(m)
    return m, ok, checks


def regression_test(root=None):
    """All 54 catalogued events must PASS; known-excluded events must FAIL."""
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    proc = root / "data/processed"
    cat = pd.read_csv(proc / "event_catalogue.csv")

    failures = []
    print(f"{'event':28s} {'slots':>5s} {'days':>4s} {'cmpl':>4s} "
          f"{'gap':>6s} {'sat':>6s}  verdict")
    for ev in cat["event_name"]:
        p = proc / f"{ev}_hourly.csv"
        if not p.exists():
            failures.append((ev, "MISSING_CSV"))
            print(f"{ev:28s}  -- hourly CSV missing --")
            continue
        m, ok, checks = gate_event_csv(p)
        verdict = "PASS" if ok else "FAIL"
        expected = "PASS"
        mark = "" if verdict == expected else "  << REGRESSION MISMATCH"
        if mark:
            failed = [k for k, v in checks.items() if not v]
            failures.append((ev, f"expected PASS, got FAIL on {failed}"))
        print(f"{ev:28s} {m['fire_slots']:5d} {m['duration_days']:4d} "
              f"{m['complete_days']:4d} {m['gap_rate']:6.3f} {m['sat_rate']:6.3f}"
              f"  {verdict}{mark}")

    print("\n--- Known-excluded events (must FAIL) ---")
    for ev in KNOWN_EXCLUDED:
        p = proc / f"{ev}_hourly.csv"
        if not p.exists():
            print(f"{ev:28s}  -- no hourly CSV (exclusion predates archive; OK) --")
            continue
        m, ok, checks = gate_event_csv(p)
        verdict = "PASS" if ok else "FAIL"
        mark = "" if verdict == "FAIL" else "  << REGRESSION MISMATCH"
        if mark:
            failures.append((ev, "expected FAIL, got PASS"))
        failed = [k for k, v in checks.items() if not v]
        print(f"{ev:28s} {m['fire_slots']:5d} {m['duration_days']:4d} "
              f"{m['complete_days']:4d} {m['gap_rate']:6.3f} {m['sat_rate']:6.3f}"
              f"  {verdict}{mark}  {failed}")

    print(f"\nREGRESSION: {'PASS — historical record reproduced' if not failures else 'FAIL'}")
    for ev, why in failures:
        print(f"  MISMATCH {ev}: {why}")
    return not failures


if __name__ == "__main__":
    if len(sys.argv) > 1:
        m, ok, checks = gate_event_csv(sys.argv[1])
        print(m)
        print(checks)
        print("PASS" if ok else "FAIL")
    else:
        ok = regression_test()
        sys.exit(0 if ok else 1)
