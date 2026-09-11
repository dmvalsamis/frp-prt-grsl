#!/usr/bin/env python
"""explore_arch_decisive.py -- EXPLORATORY post-peak-day diagnostic, generalized from
scripts/explore_decisive_day.py to any predictor (branch explore-arch-2026).

For each predictor (M2, A1..A6; M1 descriptive) vs B6, over the 77 evaluable block-A +
block-B events, using the window-level predictions cached by explore_arch_eval.py:
  worst_day_share     share of the predictor's total fire-active AE(t+6) on its worst UTC day
  worst_day_is_postpeak
  postpeak_bias       mean signed error at t+6 on the UTC day after the event's peak hour
  postpeak_dAE        (AE_B6 - AE_model) summed on the post-peak day
  rest_dAE            same, all other days
  decided_by_postpeak True if removing the post-peak day flips the paired outcome vs B6
Outputs: resubmission/EXPLORATORY_ARCH_2026/decisive_day_by_arch.md, decisive_day_{model}.csv
Run:  PYTHONUTF8=1 C:/Users/dvalsamis/AppData/Local/anaconda3/python.exe scripts/explore_arch_decisive.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "resubmission" / "EXPLORATORY_ARCH_2026"
CACHE = ROOT / "data" / "models" / "explore_arch" / "eval_cache"
N_MIN_FIRE = 5


def diagnose(W, p_model, p_b6, p_b1):
    fire = W["Y"].max(axis=1) > 0
    y6 = W["Y"][:, 5]
    ae_m, ae_b6, ae_b1 = np.abs(p_model[:, 5] - y6), np.abs(p_b6[:, 5] - y6), np.abs(p_b1[:, 5] - y6)
    bias_m, bias_b6 = p_model[:, 5] - y6, p_b6[:, 5] - y6
    t0 = pd.to_datetime(W["t0_ns"])
    rows = []
    for ev in sorted(set(W["ev"])):
        m = (W["ev"] == ev) & fire
        if m.sum() < N_MIN_FIRE:
            continue
        d = pd.DataFrame({"day": pd.Series(t0[m].date).values, "ae_m": ae_m[m], "ae_b6": ae_b6[m],
                          "bias_m": bias_m[m], "bias_b6": bias_b6[m]})
        g = d.groupby("day").sum(numeric_only=True)
        worst_day = g.ae_m.idxmax()
        pp_day = (pd.Timestamp(W["peak_ns"][m][0]) + pd.Timedelta(days=1)).date()
        pp = d[d.day == pp_day]
        total_dAE = float((g.ae_b6 - g.ae_m).sum())
        pp_dAE = float((pp.ae_b6 - pp.ae_m).sum()) if len(pp) else 0.0
        rest_dAE = total_dAE - pp_dAE
        rows.append({"event": ev, "block": str(W["block"][m][0]), "n_fire": int(m.sum()),
                     "SS_model": 1 - ae_m[m].mean() / ae_b1[m].mean(),
                     "SS_B6": 1 - ae_b6[m].mean() / ae_b1[m].mean(),
                     "model_wins": bool(ae_m[m].mean() < ae_b6[m].mean()),
                     "worst_day": str(worst_day), "worst_day_share": float(g.ae_m.max() / g.ae_m.sum()),
                     "worst_day_is_postpeak": bool(worst_day == pp_day), "postpeak_n": int(len(pp)),
                     "postpeak_bias_model": float(pp.bias_m.mean()) if len(pp) else np.nan,
                     "postpeak_bias_B6": float(pp.bias_b6.mean()) if len(pp) else np.nan,
                     "postpeak_share_model": float(pp.ae_m.sum() / g.ae_m.sum()) if len(pp) else np.nan,
                     "postpeak_dAE": pp_dAE, "total_dAE": total_dAE, "rest_dAE": rest_dAE,
                     "decided_by_postpeak": bool(np.sign(total_dAE) != np.sign(rest_dAE)) if rest_dAE != 0 else False})
    return pd.DataFrame(rows)


def p_greater(w, l):
    return binomtest(w, w + l, 0.5, alternative="greater").pvalue if (w + l) else float("nan")


def main():
    W = dict(np.load(CACHE / "windows_AB.npz", allow_pickle=False))
    R = dict(np.load(CACHE / "ref_preds.npz"))
    archs = sorted(p.name.split("_")[0] for p in CACHE.glob("A?_preds_AB.npy"))
    preds = {"M2": R["M2"], **{a: np.load(CACHE / f"{a}_preds_AB.npy") for a in archs}, "M1": R["M1"]}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tables = {}
    for name, p in preds.items():
        df = diagnose(W, p, R["B6"], R["B1"])
        df.to_csv(OUT / f"decisive_day_{name}.csv", index=False)
        tables[name] = df

    L = [f"# EXPLORATORY post-peak-day diagnostic by architecture, blocks A + B ({len(tables['M2'])} events) -- {stamp}\n",
         "Every predictor is compared with B6 on the same windows (predictions cached by "
         "`explore_arch_eval.py`; B6 / B1 / M2 / M1 from the locked test_access_v1 machinery). "
         "Post-peak day = the UTC day after the event's peak FRP hour. 'flips' = events whose paired "
         "outcome vs B6 changes when that one day is removed (post hoc, descriptive). M1 is descriptive.\n"]
    for blk, lab in (("A2025", "block A (45)"), ("B2026", "block B (32)"), ("all", "pooled (77)")):
        L.append(f"## {lab}\n")
        L.append("| model | wins vs B6 | worst day = post-peak | median worst-day share | median post-peak share | "
                 "flips without post-peak day | post-peak bias > 0 | post-peak bias > B6 bias | "
                 "sign test excl. post-peak day | sign test post-peak day alone |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for name, df in tables.items():
            s = df if blk == "all" else df[df.block == blk]
            w_r, l_r = int((s.rest_dAE > 0).sum()), int((s.rest_dAE < 0).sum())
            sp = s[s.postpeak_n > 0]
            w_p, l_p = int((sp.postpeak_dAE > 0).sum()), int((sp.postpeak_dAE < 0).sum())
            L.append(f"| {name} | {int(s.model_wins.sum())}/{int((~s.model_wins).sum())} | "
                     f"{int(s.worst_day_is_postpeak.sum())} | {s.worst_day_share.median():.2f} | "
                     f"{s.postpeak_share_model.median():.2f} | {int(s.decided_by_postpeak.sum())} | "
                     f"{int((s.postpeak_bias_model > 0).sum())} | "
                     f"{int((s.postpeak_bias_model > s.postpeak_bias_B6).sum())} | "
                     f"{w_r}/{l_r} p={p_greater(w_r, l_r):.3f} | {w_p}/{l_p} p={p_greater(w_p, l_p):.3f} |")
        L.append("")
    L.append("## Post-peak-day signed bias at t+6 (mean over events with a post-peak day, MW)\n")
    L.append("| model | A: mean bias | A: median bias | B: mean bias | B: median bias | B6 mean bias A / B |")
    L.append("|---|---|---|---|---|---|")
    for name, df in tables.items():
        a, b = df[df.block == "A2025"], df[df.block == "B2026"]
        L.append(f"| {name} | {a.postpeak_bias_model.mean():+,.0f} | {a.postpeak_bias_model.median():+,.0f} | "
                 f"{b.postpeak_bias_model.mean():+,.0f} | {b.postpeak_bias_model.median():+,.0f} | "
                 f"{a.postpeak_bias_B6.mean():+,.0f} / {b.postpeak_bias_B6.mean():+,.0f} |")
    L.append("\n## The three 2026 collapse events (M2 vs B6 gap > 0.3 skill), SS_fire,t6 per model\n")
    coll = ["pt_cambra_e_carvalhal_2026_0702", "fr_porge_2026_0722", "pt_cerdeira_2026_0727"]
    L.append("| event | B6 | " + " | ".join(tables) + " |")
    L.append("|---|---|" + "---|" * len(tables))
    for ev in coll:
        cells = []
        for name, df in tables.items():
            r = df[df.event == ev]
            cells.append(f"{float(r.SS_model.iloc[0]):+.3f}" if len(r) else "--")
        r0 = tables["M2"][tables["M2"].event == ev]
        L.append(f"| {ev} | {float(r0.SS_B6.iloc[0]):+.3f} | " + " | ".join(cells) + " |")
    txt = "\n".join(L) + "\n"
    (OUT / "decisive_day_by_arch.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
