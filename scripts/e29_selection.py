"""
e29_selection.py — Task E2.9: mechanical selection over the E2.8 grid
=====================================================================
Applies §5a as amended (AMEND-2 + AGG-1), NOTHING else:

Metrics per cell (seed-mean over seeds {0,1,2}):
  M-1: event-level MEDIAN val SS_fire_t6 (AGG-1), seed-mean of per-seed medians.
  M-2: paired win fraction vs AR Ridge OF THE SAME VARIANT (T0 cells vs
       ar_ridge_T0, T1 cells vs ar_ridge_T1): fraction of evaluable val events
       where the run's MAE_fire_t6 < AR's; seed-mean.
  M-3: worst-case robustness: min per-event SS_fire_t6; seed-mean of minima.

Rules:
  R-1': winner = the cell that wins >= 2 of 3 metrics against EVERY other cell
        pairwise; if no cell dominates, precedence M-1 > M-2 > M-3 breaks the
        tie (strictly greater at each stage; proceed to next metric on exact
        equality only).
  R-2': the winner is REJECTED in favour of the reference cell (T0 x legacy)
        if winner M-3 < reference M-3 - 0.05.
  R-3': single shot; this script reports the full trace. Any outcome the rules
        do not cover -> print ESCALATE and exit 2 (do not improvise).

Output: data/models/e28/e29_selection.json (full metric table + rule trace)
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path("e:/GRSL_Wildfire/FRP_Methodology")
E28 = ROOT / "data/models/e28"
BASE = ROOT / "data/processed/baselines_v6_1_full_val_results.json"

CELLS = ["T0_legacy", "T0_robust", "T1_legacy", "T1_robust"]
SEEDS = [0, 1, 2]
REFERENCE = "T0_legacy"


def main():
    with open(BASE) as f:
        base = json.load(f)
    ar_mae = {"T0": {ev: d["MAE_fire"]["+6"]
                     for ev, d in base["ar_ridge_T0"]["per_event"].items()},
              "T1": {ev: d["MAE_fire"]["+6"]
                     for ev, d in base["ar_ridge_T1"]["per_event"].items()}}

    table = {}
    for cell in CELLS:
        variant = cell.split("_")[0]
        m1s, m2s, m3s, meds = [], [], [], []
        for s in SEEDS:
            p = E28 / f"{variant}_{cell.split('_')[1]}_s{s}_result.json"
            if not p.exists():
                print(f"ESCALATE: missing run result {p.name}")
                sys.exit(2)
            r = json.load(open(p))
            if "error" in r:
                print(f"ESCALATE: run {r['tag']} failed: {r['error']}")
                sys.exit(2)
            pe = r["per_event"]
            ss = [d["SS_fire_t6"] for d in pe.values()
                  if d["n_fire"] > 0 and not np.isnan(d["SS_fire_t6"])]
            m1s.append(float(np.median(ss)))
            m3s.append(float(np.min(ss)))
            wins, n = 0, 0
            for ev, d in pe.items():
                if d["n_fire"] == 0 or np.isnan(d["MAE_fire_t6"]):
                    continue
                a = ar_mae[variant].get(ev)
                if a is None or np.isnan(a):
                    continue
                n += 1
                if d["MAE_fire_t6"] < a:
                    wins += 1
            m2s.append(wins / n if n else float("nan"))
            meds.append(r["val_median_ss_t6"])
        table[cell] = {"M1": float(np.mean(m1s)), "M2": float(np.mean(m2s)),
                       "M3": float(np.mean(m3s)),
                       "M1_per_seed": m1s, "M2_per_seed": m2s, "M3_per_seed": m3s}

    print(f"{'cell':12s} {'M-1 (median SS)':>16s} {'M-2 (win frac)':>15s} {'M-3 (min SS)':>13s}")
    for c in CELLS:
        t = table[c]
        print(f"{c:12s} {t['M1']:+16.4f} {t['M2']:15.3f} {t['M3']:+13.3f}")

    trace = []

    def beats(a, b):
        """a beats b on >= 2 of 3 metrics (strict inequalities)."""
        w = sum([table[a]["M1"] > table[b]["M1"],
                 table[a]["M2"] > table[b]["M2"],
                 table[a]["M3"] > table[b]["M3"]])
        l = sum([table[a]["M1"] < table[b]["M1"],
                 table[a]["M2"] < table[b]["M2"],
                 table[a]["M3"] < table[b]["M3"]])
        return w >= 2, w, l

    dominators = []
    for c in CELLS:
        alls = []
        for o in CELLS:
            if o == c:
                continue
            ok, w, l = beats(c, o)
            trace.append(f"R-1' pairwise: {c} vs {o}: wins {w}/3 -> "
                         f"{'beats' if ok else 'does not beat'}")
            alls.append(ok)
        if all(alls):
            dominators.append(c)
    trace.append(f"R-1' dominators: {dominators}")

    if len(dominators) == 1:
        winner = dominators[0]
        trace.append(f"R-1' winner by pairwise dominance: {winner}")
    elif len(dominators) == 0:
        trace.append("R-1': no dominator -> precedence tie-break M-1 > M-2 > M-3")
        cand = list(CELLS)
        winner = None
        for metric in ["M1", "M2", "M3"]:
            best = max(table[c][metric] for c in cand)
            cand = [c for c in cand if table[c][metric] == best]
            trace.append(f"  tie-break on {metric}: best={best:+.4f} -> {cand}")
            if len(cand) == 1:
                winner = cand[0]
                break
        if winner is None:
            print("ESCALATE: exact tie across all three metrics — rules do not cover.")
            for t in trace:
                print("  " + t)
            sys.exit(2)
    else:
        print(f"ESCALATE: multiple pairwise dominators {dominators} — rules do not cover.")
        for t in trace:
            print("  " + t)
        sys.exit(2)

    # R-2' guard
    if winner != REFERENCE and table[winner]["M3"] < table[REFERENCE]["M3"] - 0.05:
        trace.append(f"R-2' guard FIRED: {winner} M-3 {table[winner]['M3']:+.3f} < "
                     f"reference M-3 {table[REFERENCE]['M3']:+.3f} - 0.05 -> "
                     f"reference {REFERENCE} adopted")
        winner = REFERENCE
    else:
        trace.append(f"R-2' guard: not fired (winner M-3 {table[winner]['M3']:+.3f} "
                     f"vs reference {table[REFERENCE]['M3']:+.3f})")

    print("\n--- rule trace ---")
    for t in trace:
        print(t)
    print(f"\nSELECTED CELL: {winner}")

    with open(E28 / "e29_selection.json", "w") as f:
        json.dump({"table": table, "trace": trace, "winner": winner,
                   "reference": REFERENCE}, f, indent=2)
    print(f"-> {E28 / 'e29_selection.json'}")


if __name__ == "__main__":
    main()
