"""Export the 162-event catalogue used by the GRSL letter (v4/v5 revision), one row per
event, from deposited metadata files only. No forecast, prediction or skill value is read
or written; the per-event tables are used only for the evaluability / set-membership flags
(n_windows, n_fire, evaluable, in_primary, in_sensitivity, d7_restored).

Columns: identity, pool, split/set membership, country, ignition date, box (bbox, deg^2,
approx km^2), quality-gate metrics with the five criteria recomputed from the catalogue,
and the pixel-disjointness outcome for the 2025 block (dropped / kept-in-favour-of /
overlap fraction). Provenance (input hashes) is printed and written next to the CSV.

Output: <out_dir>/event_catalogue_162_export_<date>.csv (+ _provenance.txt)
"""
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "resubmission"
CAT = ROOT / "data" / "processed" / "event_catalogue.csv"
DEV = R / "frozen_dev_pool_2026-08-25.csv"
SPLIT = R / "split_v6_2026-08-25.csv"
BLOCKA = R / "test_event_list_blockA_LOCKED.csv"
A_PE = R / "TEST_ACCESS_RESULTS" / "access_20260828T123059Z_per_event.csv"
DROPS = R / "pixel_disjointness_test_A_2025_2026-08-25_drops.json"
B_INV = R / "EXPLORATORY_BLOCKB_2026" / "blockB_inventory.csv"
B_PE = R / "EXPLORATORY_BLOCKB_2026" / "exploratory_blockB_20260908T140106Z_per_event.csv"
INPUTS = [CAT, DEV, SPLIT, BLOCKA, A_PE, DROPS, B_INV, B_PE]
META_COLS = ["event", "n_windows", "n_fire", "evaluable", "in_primary", "in_sensitivity", "d7_restored"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cat = pd.read_csv(CAT)
    cat["ignition_year"] = pd.to_datetime(cat.start).dt.year

    dev = pd.read_csv(DEV)
    split = pd.read_csv(SPLIT)[["event_name", "split_v6", "split_seed"]]
    a_names = pd.read_csv(BLOCKA).event_name
    a_pe = pd.read_csv(A_PE)[META_COLS].rename(columns={"event": "event_name"})
    drops = json.load(open(DROPS))
    b_inv = pd.read_csv(B_INV)
    b_pe = pd.read_csv(B_PE)[["event", "n_windows", "n_fire", "evaluable"]].rename(columns={"event": "event_name"})

    assert len(dev) == 84 and len(a_names) == 46
    b_names = b_inv[b_inv.included == True].event_name
    assert len(b_names) == 32

    rows = []
    for pool, names in (("development", dev.event_name), ("2025 test block", a_names), ("2026 test block", b_names)):
        sub = cat[cat.event_name.isin(names)].copy()
        assert len(sub) == len(names), (pool, len(sub), len(names))
        sub["pool"] = pool
        rows.append(sub)
    df = pd.concat(rows, ignore_index=True)
    assert len(df) == 162 and df.event_name.is_unique

    # split / set membership
    df = df.merge(split, on="event_name", how="left")
    df = df.merge(a_pe, on="event_name", how="left", suffixes=("", "_access"))
    df = df.merge(b_pe, on="event_name", how="left", suffixes=("", "_b"))
    for c in ("n_windows", "n_fire", "evaluable"):
        df[c] = df[c].where(df[c].notna(), df[f"{c}_b"])
        df = df.drop(columns=[f"{c}_b"])
    if "d7_restored_access" in df:
        df["d7_restored"] = df["d7_restored_access"].where(df["d7_restored_access"].notna(), df["d7_restored"])
        df = df.drop(columns=["d7_restored_access"])
    b_flags = b_inv.set_index("event_name")[["gate_verdict", "included"]]
    df["gate_verdict_source"] = np.where(df.pool == "2026 test block",
                                         df.event_name.map(b_flags.gate_verdict), "frozen gate (see criteria)")

    def membership(r):
        if r.pool == "development":
            return f"dev-{r.split_v6}"
        if r.pool == "2025 test block":
            if not bool(r.evaluable):
                return "2025 non-evaluable (< 5 fire-active windows)"
            if bool(r.in_primary):
                return "2025 primary"
            if bool(r.in_sensitivity):
                return "2025 sensitivity-only"
            return "2025 descriptive-only (pixel-disjointness drop)"
        return "2026 exploratory"
    df["set_membership"] = df.apply(membership, axis=1)

    # pixel-disjointness outcome (2025 block)
    dropped = {d["dropped"]: d for d in drops}
    df["pixel_disjoint_dropped"] = df.event_name.map(lambda e: e in dropped)
    df["dropped_in_favour_of"] = df.event_name.map(lambda e: dropped[e]["kept"] if e in dropped else "")
    df["overlap_fraction"] = df.event_name.map(lambda e: dropped[e]["overlap_fraction"] if e in dropped else np.nan)

    # bounding box geometry
    dlon, dlat = df.lon_max - df.lon_min, df.lat_max - df.lat_min
    latm = np.deg2rad((df.lat_max + df.lat_min) / 2)
    df["box_area_deg2"] = (dlon * dlat).round(4)
    df["box_area_km2_approx"] = (dlon * 111.32 * np.cos(latm) * dlat * 110.57).round(0)

    # quality-gate metrics: canonical gate outputs where a gate_results_*.csv row exists (108 events;
    # the last row per event = the W-1-extended window where applicable); for the 54 legacy
    # development events (onboarded before the resubmission programme) the metrics are the
    # frozen-pool values and the duration is the downloaded window length (end - start + 1),
    # which is what the gate's duration criterion measures (see EXCLUSIONS_E2: 0-slot events
    # carry duration 11 d).
    gate_files = sorted((R).glob("gate_results_*.csv"))
    g = pd.concat([pd.read_csv(f).rename(columns={"event": "event_name"}).assign(gate_source=f.name)
                   for f in gate_files], ignore_index=True)
    g = g[g.verdict == "PASS"].sort_values(["event_name", "duration_days"]).drop_duplicates("event_name", keep="last")
    g = g.set_index("event_name")
    has = df.event_name.isin(g.index)
    for c in ("fire_slots", "duration_days", "complete_days", "gap_rate", "sat_rate"):
        df.loc[has, c] = df.loc[has, "event_name"].map(g[c]).values
    win_len = (pd.to_datetime(df.end) - pd.to_datetime(df.start)).dt.days + 1
    df.loc[~has & df.duration_days.isna(), "duration_days"] = win_len[~has & df.duration_days.isna()]
    df["gate_metrics_source"] = np.where(has, df.event_name.map(g.gate_source), "frozen pool metrics; duration = window length")
    df["gate_fire_slots_ge15"] = df.fire_slots >= 15
    df["gate_duration_ge5d"] = df.duration_days >= 5
    df["gate_complete_days_ge3"] = df.complete_days >= 3
    df["gate_gap_rate_lt040"] = df.gap_rate < 0.40
    df["gate_sat_rate_lt030"] = df.sat_rate < 0.30
    crit = ["gate_fire_slots_ge15", "gate_duration_ge5d", "gate_complete_days_ge3", "gate_gap_rate_lt040", "gate_sat_rate_lt030"]
    df["gate_all_criteria_pass"] = df[crit].all(axis=1)
    fails = df[~df.gate_all_criteria_pass]
    if len(fails):
        print("NOTE: catalogue-metric recomputation fails the gate for:",
              fails[["event_name", "pool", "fire_slots", "duration_days", "complete_days", "gap_rate", "sat_rate"]].to_string())

    cols = ["event_name", "pool", "set_membership", "country", "start", "end", "ignition_year", "duration_days",
            "lon_min", "lon_max", "lat_min", "lat_max", "box_area_deg2", "box_area_km2_approx",
            "fire_slots", "complete_days", "gap_rate", "sat_rate", "peak_frp_mw",
            *crit, "gate_all_criteria_pass", "gate_metrics_source", "gate_verdict_source",
            "n_windows", "n_fire", "evaluable", "in_primary", "in_sensitivity", "d7_restored",
            "pixel_disjoint_dropped", "dropped_in_favour_of", "overlap_fraction",
            "split_v6", "split_seed", "legacy_split", "intensity_tier", "wind_regime", "date_added"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].sort_values(["pool", "start", "event_name"]).reset_index(drop=True)

    tag = date.today().isoformat()
    out = out_dir / f"event_catalogue_162_export_{tag}.csv"
    df.to_csv(out, index=False)
    prov = [f"event_catalogue_162_export_{tag}.csv -- written {tag} by scripts/export_event_catalogue_162.py",
            "metadata only; no forecast or skill value read or written", "inputs (sha256):"]
    prov += [f"  {p.relative_to(ROOT)}  {sha(p)}" for p in INPUTS]
    prov.append(f"output sha256 {sha(out)}")
    summary = df.groupby("pool").size().to_dict()
    prov.append(f"pool counts {summary}")
    prov.append("set membership counts " + str(df.set_membership.value_counts().to_dict()))
    prov.append(f"countries: union {df.country.nunique()}; per pool " +
                str(df.groupby('pool').country.nunique().to_dict()))
    prov.append(f"gate recomputation: {int(df.gate_all_criteria_pass.sum())}/162 pass all five criteria; "
                f"metrics from canonical gate files for {int(has.sum())} events")
    test = df[df.pool != "development"]
    prov.append(f"countries over the two test blocks (78 fires): union {test.country.nunique()}; "
                f"over the 77 evaluable: {test[test.evaluable == True].country.nunique()}")
    prov.append(f"pixel-disjointness drops recorded: {int(df.pixel_disjoint_dropped.sum())} "
                f"(2025 block: in_primary {int(df.in_primary.fillna(False).astype(bool).sum())}, "
                f"in_sensitivity {int(df.in_sensitivity.fillna(False).astype(bool).sum())}, d7_restored "
                f"{int(df[df.pool == '2025 test block'].d7_restored.fillna(False).astype(bool).sum())})")
    (out_dir / f"event_catalogue_162_export_{tag}_provenance.txt").write_text("\n".join(prov) + "\n", encoding="utf-8")
    print("\n".join(prov))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "resubmission" / "submission_bundles" / "claudeai_phaseA_2026-09-11" / "07_catalogue_and_gate")
