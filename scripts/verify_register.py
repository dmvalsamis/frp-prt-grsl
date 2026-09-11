#!/usr/bin/env python
"""verify_register.py -- T1: recompute every block-A figure in
resubmission/number_register.json from the locked per_event.csv alone
(plus strata.json for the frozen constants) and report recomputed vs
registered vs match. Arithmetic on already-deposited outputs only:
this is NOT a test access (VETO-3 respected; scripts/test_access_v1.py
is never invoked here).

Usage: python scripts/verify_register.py [--out resubmission/REGISTER_VERIFICATION.md]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
TAR = ROOT / "resubmission" / "TEST_ACCESS_RESULTS"
FILES = {
    "results": ("access_20260828T123059Z_results.json",
                "9f3c1d8410ccb79f2e1841ed2e04f191152d818f8e216f8dc404e7e3b8262226"),
    "per_event": ("access_20260828T123059Z_per_event.csv",
                  "2c1dd44498b3fe76be7005b493eb0d2424c8347d601efcca1a505418fceb5e79"),
    "strata": ("access_20260828T123059Z_strata.json",
               "414634555287b46d1ccbd862cffd5d108d70376b4f19ca9c34a625229fb435a0"),
    "access_record": ("access_record_20260828T123059Z.json",
                      "9f1d6bf00eac0a71e95d781b34ed6e3b8423a4fc8b55dca3eb20e3bc5cb9b363"),
    "console": ("access_console_20260828.log",
                "802ba2ccbab8765a26d2ad6f70de4547b4c583c6f2f3b727021f8853ba574eee"),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sign_test(mae_a, mae_b):
    d = np.asarray(mae_a) - np.asarray(mae_b)
    d = d[d != 0]
    wins = int((d < 0).sum())
    losses = int((d > 0).sum())
    n = wins + losses
    p = binomtest(wins, n, 0.5, alternative="greater").pvalue
    return wins, losses, n, p


def wilcoxon_p(mae_a, mae_b):
    d = np.asarray(mae_a) - np.asarray(mae_b)
    return wilcoxon(d, alternative="less").pvalue  # M2 lower MAE -> negative d


def parse_reg(registered):
    """Return (float_value or None, n_decimals, is_sci)."""
    if isinstance(registered, bool):
        return None, 0, False
    if isinstance(registered, (int, float)):
        return float(registered), 0, False
    s = str(registered)
    if "times10^{" in s:
        return float(s.replace("\\times10^{", "e").replace("}", "")), 1, True
    try:
        v = float(s)
    except ValueError:
        return None, 0, False
    dec = len(s.split(".")[1]) if "." in s else 0
    return v, dec, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "resubmission" / "REGISTER_VERIFICATION.md"))
    args = ap.parse_args()

    reg = json.loads((ROOT / "resubmission" / "number_register.json").read_text(encoding="utf-8"))
    rows = []
    state = {"ok": True}

    hash_rows = []
    for k, (fn, expected) in FILES.items():
        h = sha(TAR / fn)
        ok = h == expected
        state["ok"] &= ok
        hash_rows.append((fn, h, expected, ok))

    pe = pd.read_csv(TAR / FILES["per_event"][0])
    ev = pe[pe.evaluable == True]
    prim = ev[ev.in_primary == True]
    sens = ev[ev.in_sensitivity == True]

    def add(section, key, recomputed, registered):
        if isinstance(registered, bool) or isinstance(recomputed, bool):
            ok = bool(recomputed) == bool(registered)
            shown = str(bool(recomputed)).lower()
        else:
            r, dec, is_sci = parse_reg(registered)
            if r is None:
                ok = str(recomputed) == str(registered)
                shown = str(recomputed)
            elif is_sci:
                ok = f"{float(recomputed):.1e}" == f"{r:.1e}"
                shown = f"{float(recomputed):.3e}"
            elif isinstance(recomputed, (int, np.integer)) and dec == 0:
                ok = int(recomputed) == int(r)
                shown = str(int(recomputed))
            else:
                ok = round(float(recomputed), dec) == round(r, dec)
                shown = f"{float(recomputed):.6g}"
        state["ok"] &= bool(ok)
        rows.append((section, key, shown, str(registered), bool(ok)))

    C = reg["counts"]
    add("counts", "gate_passing_blockA", len(pe), C["gate_passing_blockA"])
    add("counts", "evaluable_blockA", len(ev), C["evaluable_blockA"])
    add("counts", "primary_n", len(prim), C["primary_n"])
    add("counts", "sensitivity_n", len(sens), C["sensitivity_n"])
    add("counts", "primary_declared_n (in_primary flag over all 46 rows)", int(pe.in_primary.sum()), C["primary_declared_n"])
    add("counts", "sensitivity_declared_n (in_sensitivity flag over all 46 rows)", int(pe.in_sensitivity.sum()), C["sensitivity_declared_n"])
    add("counts", "windows_blockA", int(pe.n_windows.sum()), C["windows_blockA"])
    add("counts", "fire_windows_blockA", int(pe.n_fire.sum()), C["fire_windows_blockA"])
    add("counts", "fire_windows_calm", int(pe.n_fire_calm.sum()), C["fire_windows_calm"])
    add("counts", "fire_windows_excursion", int(pe.n_fire_exc.sum()), C["fire_windows_excursion"])
    add("counts", "d7_restored", int(pe.d7_restored.sum()), C["d7_restored"])
    add("counts", "dec3_dropped_descriptive (evaluable and not in_sensitivity)", int((ev.in_sensitivity == False).sum()), C["dec3_dropped_descriptive"])
    zero = pe[pe.n_fire == 0]
    add("counts", "zero_fire_event", zero.event.iloc[0] if len(zero) == 1 else f"{len(zero)} rows", C["zero_fire_event"])
    add("counts", "zero_fire_event_windows", int(zero.n_windows.iloc[0]) if len(zero) == 1 else -1, C["zero_fire_event_windows"])
    nonev = pe[pe.evaluable == False]
    add("counts", "N_min consistent (min n_fire evaluable >= 5; max n_fire non-evaluable < 5)",
        bool(int(ev.n_fire.min()) >= C["N_min"] and int(nonev.n_fire.max()) < C["N_min"]), True)

    P = reg["primary"]
    w, l, n, p = sign_test(prim.MAE_M2_t6, prim.MAE_B6_t6)
    add("primary M2 vs B6", "wins", w, P["wins"])
    add("primary M2 vs B6", "losses", l, P["losses"])
    add("primary M2 vs B6", "n", n, P["n"])
    add("primary M2 vs B6", "p_one_sided", p, P["p_one_sided"])
    add("primary M2 vs B6", "wilcoxon_p", wilcoxon_p(prim.MAE_M2_t6, prim.MAE_B6_t6), P["wilcoxon_p"])
    add("primary M2 vs B6", "reject_at_005", bool(p < 0.05), P["reject_at_005"])

    Sx = reg["secondary_sensitivity"]
    w, l, n, p = sign_test(sens.MAE_M2_t6, sens.MAE_B6_t6)
    add("sensitivity M2 vs B6", "wins", w, Sx["wins"])
    add("sensitivity M2 vs B6", "losses", l, Sx["losses"])
    add("sensitivity M2 vs B6", "n", n, Sx["n"])
    add("sensitivity M2 vs B6", "p_one_sided", p, Sx["p_one_sided"])
    add("sensitivity M2 vs B6", "wilcoxon_p", wilcoxon_p(sens.MAE_M2_t6, sens.MAE_B6_t6), Sx["wilcoxon_p"])

    Sp = reg["secondary_vs_persistence"]
    w, l, n, p = sign_test(prim.MAE_M2_t6, prim.MAE_B1_t6)
    add("primary-set M2 vs B1", "wins", w, Sp["wins"])
    add("primary-set M2 vs B1", "losses", l, Sp["losses"])
    add("primary-set M2 vs B1", "n", n, Sp["n"])
    add("primary-set M2 vs B1", "p_one_sided", p, Sp["p_one_sided"])
    add("primary-set M2 vs B1", "wilcoxon_p", wilcoxon_p(prim.MAE_M2_t6, prim.MAE_B1_t6), Sp["wilcoxon_p"])

    A = reg["agg1_blockA_45"]
    add("AGG-1 (45 evaluable)", "M2_median", ev.SS_M2_t6.median(), A["M2_median"])
    add("AGG-1 (45 evaluable)", "M2_mean", ev.SS_M2_t6.mean(), A["M2_mean"])
    add("AGG-1 (45 evaluable)", "M2_std (sample, ddof=1 -- the convention of the locked results.json)", ev.SS_M2_t6.std(ddof=1), A["M2_std"])
    add("AGG-1 (45 evaluable)", "B6_median", ev.SS_B6_t6.median(), A["B6_median"])
    add("AGG-1 (45 evaluable)", "B6_mean", ev.SS_B6_t6.mean(), A["B6_mean"])
    add("AGG-1 (45 evaluable)", "M1_median", ev.SS_M1_t6.median(), A["M1_median"])
    add("AGG-1 (45 evaluable)", "M1_mean", ev.SS_M1_t6.mean(), A["M1_mean"])
    add("AGG-1 (45 evaluable)", "M2_negative_events", int((ev.SS_M2_t6 < 0).sum()), A["M2_negative_events"])
    add("AGG-1 (45 evaluable)", "B6_negative_events", int((ev.SS_B6_t6 < 0).sum()), A["B6_negative_events"])
    add("AGG-1 (45 evaluable)", "M2_min", ev.SS_M2_t6.min(), A["M2_min"])

    H = reg["horizon_medians_blockA"]
    for m in ("M2", "B6"):
        for h in (1, 3, 6, 12):
            add(f"horizon medians {m}", f"t{h}", ev[f"SS_{m}_t{h}"].median(), H[m][f"t{h}"])

    St = reg["strata_amend8"]
    for h in (1, 3, 6, 12):
        d = ev[f"SS_M2_t{h}"] - ev[f"SS_B6_t{h}"]
        r = St["horizon"][f"t{h}"]
        add(f"strata horizon t{h}", "wins", int((d > 0).sum()), r["wins"])
        add(f"strata horizon t{h}", "losses", int((d < 0).sum()), r["losses"])
        add(f"strata horizon t{h}", "median_dSS", d.median(), r["median_dSS"])
    for lvl, col in (("calm", "calm"), ("excursion", "exc")):
        d = ev[f"SS_M2_t6_{col}"] - ev[f"SS_B6_t6_{col}"]
        r = St["wind_t6"][lvl]
        add(f"strata wind {lvl}", "wins", int((d > 0).sum()), r["wins"])
        add(f"strata wind {lvl}", "losses", int((d < 0).sum()), r["losses"])
        add(f"strata wind {lvl}", "median_dSS", d.median(), r["median_dSS"])
    for lvl in ("high", "mid", "low"):
        sub = ev[ev.vol_tertile == lvl]
        d = sub.SS_M2_t6 - sub.SS_B6_t6
        r = St["volatility_t6"][lvl]
        add(f"strata volatility {lvl}", "n", len(sub), r["n"])
        add(f"strata volatility {lvl}", "wins", int((d > 0).sum()), r["wins"])
        add(f"strata volatility {lvl}", "losses", int((d < 0).sum()), r["losses"])
        add(f"strata volatility {lvl}", "median_dSS", d.median(), r["median_dSS"])

    strata = json.loads((TAR / FILES["strata"][0]).read_text(encoding="utf-8"))
    fc = strata["_meta"]["frozen_constants"]
    lo, hi = fc["vol_tertile_lo"], fc["vol_tertile_hi"]
    lab = np.where(ev.volatility_V < lo, "low", np.where(ev.volatility_V >= hi, "high", "mid"))
    add("strata volatility", "vol_tertile labels consistent with frozen boundaries",
        bool((lab == ev.vol_tertile.values).all()), True)
    for k in ("delta_wind_3h_sigma_train", "vol_tertile_lo", "vol_tertile_hi"):
        add("frozen constants (strata.json)", k, fc[k], St["frozen_constants"][k])

    res = json.loads((TAR / FILES["results"][0]).read_text(encoding="utf-8"))
    rp = res["PRIMARY_sign_test_M2_vs_B6_primary_set"]
    add("results.json witness", "PRIMARY wins/losses/n",
        f"{rp['wins']}/{rp['losses']}/{rp['n']}", f"{P['wins']}/{P['losses']}/{P['n']}")
    add("results.json witness", "M2_median_SS_t6", res["AGG1_headline"]["M2_median_SS_t6"], A["M2_median"])
    add("results.json witness", "M1_median_SS_t6", res["M1_DESCRIPTIVE"]["M1_median_SS_t6"], A["M1_median"])

    n_ok = sum(1 for r in rows if r[4])
    n_all = len(rows)
    n_hash_ok = sum(1 for r in hash_rows if r[3])
    out = []
    out.append("# Register verification -- T1 (2026-09-08)\n")
    out.append("Recomputed from `resubmission/TEST_ACCESS_RESULTS/access_20260828T123059Z_per_event.csv` alone "
               "(frozen constants read from `..._strata.json`; `..._results.json` used only as a second witness). "
               "No model, baseline, or dataset was touched; `scripts/test_access_v1.py` was NOT run. "
               "Script: `scripts/verify_register.py`.\n")
    out.append("## 1. File hashes (SHA-256) vs RESEARCH_LOG 2026-08-28\n")
    out.append("| file | recomputed | logged | match |")
    out.append("|---|---|---|---|")
    for fn, h, e, ok in hash_rows:
        out.append(f"| {fn} | `{h}` | `{e}` | {'YES' if ok else '**NO**'} |")
    out.append("")
    out.append("Method notes: sign test = one-sided exact binomial (scipy `binomtest`, alternative='greater', "
               "ties dropped) on wins = events where MAE_M2 < MAE_comparator; Wilcoxon = scipy `wilcoxon` on the "
               "paired MAE differences, alternative='less'; std = sample (ddof=1, the convention used by the locked results.json); medians = `pandas.median`. "
               "Float values are matched at the register's own printed precision.\n")
    out.append("## 2. Recomputed vs registered\n")
    out.append("| section | quantity | recomputed | registered | match |")
    out.append("|---|---|---|---|---|")
    for s, k, rc, rg, ok in rows:
        out.append(f"| {s} | {k} | {rc} | {rg} | {'YES' if ok else '**NO**'} |")
    out.append("")
    verdict = ("VERDICT: PASS -- the register is the same data as the repo's locked outputs."
               if state["ok"] else "VERDICT: MISMATCH -- stop-and-report; nothing has been altered.")
    out.append(f"## 3. Verdict\n\n**{n_ok}/{n_all} register values match; {n_hash_ok}/5 file hashes match.** {verdict}")
    out.append("\nRegister file hash (as placed at `resubmission/number_register.json`): `"
               + sha(ROOT / "resubmission" / "number_register.json") + "`")
    out.append("\nNot recomputable from per_event.csv (out of T1 scope, taken as-registered): the `validation` block "
               "(source: `resubmission/access_dryrun_val/`, `data/processed/baselines_v6_1_*`, "
               "`data/models/e28/e29_selection.json`), `model`, `registration`, and the dev-pool / block-B counts.")
    Path(args.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    return 0 if state["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
