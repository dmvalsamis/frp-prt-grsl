#!/usr/bin/env python
"""explore_decisive_day.py -- EXPLORATORY: is the M2-vs-B6 event outcome decided
on the post-peak transition day? Window-level inference (locked machinery,
imported unchanged) for ALL evaluable block-A + block-B events; per event:
  worst_day_share_M2    share of M2's total fire-active AE(t+6) on its worst day
  postpeak_bias_M2/B6   mean signed error at t+6 on the UTC day after the event's
                        peak hour (positive = over-forecast)
  postpeak_dAE          (AE_B6 - AE_M2) summed on the post-peak day
  total_dAE             (AE_B6 - AE_M2) summed over all fire-active windows
  decided_by_postpeak   True if removing the post-peak day flips the winner
Output: resubmission/EXPLORATORY_BLOCKB_2026/decisive_day_<stamp>.{csv,md}
NOTE: this re-runs inference on block-A windows (already-accessed data); it
adds window-level detail that the locked event-level tables do not contain.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import test_access_v1 as T  # noqa: E402

OUT = ROOT / "resubmission" / "EXPLORATORY_BLOCKB_2026"
A_CSV = ROOT / "resubmission" / "TEST_ACCESS_RESULTS" / "access_20260828T123059Z_per_event.csv"


def main():
    T.assert_environment(); T.assert_input_hashes(dry_run=False)
    with h5py.File(T.H5, "r") as f:
        meta = json.loads(f["metadata"][()]); feats = meta["feature_names"]
        norm = meta["normalization_params"]["T1"]["robust"]
    FEAT = {n: i for i, n in enumerate(feats)}
    B_CSV = sorted(OUT.glob("exploratory_blockB_*_per_event.csv"))[-1]
    A = pd.read_csv(A_CSV); B = pd.read_csv(B_CSV)
    events = [(e, "A2025") for e in A[A.evaluable == True].event] + [(e, "B2026") for e in B[B.evaluable == True].event]

    inps, tgts, metas, evs, peaks = [], [], [], [], {}
    for ev, _ in events:
        merged = T.load_event(ev)
        peaks[ev] = merged.loc[merged.frp_sum_mw.idxmax(), "slot_utc"]
        grp = T.engineer_features(merged)
        inp, tgt, wm = T.build_sequences(grp, feats)
        inps += inp; tgts += tgt; metas += wm; evs += [ev] * len(inp)
    X = np.stack(inps); Y = np.stack(tgts); evs = np.array(evs)
    print(f"windows: {len(X)} over {len(events)} events", flush=True)
    p_m2 = T.m2_predict(X, feats, norm, torch.device("cpu"))
    p_b6 = T.b6_train_predict(X, feats)
    pers = T.persistence_anchor(X, FEAT)
    t0 = pd.to_datetime([m["target_end_utc"] for m in metas]) - pd.Timedelta(hours=T.HORIZON)
    fire = Y.max(axis=1) > 0
    y6 = Y[:, 5]; ae_m2 = np.abs(p_m2[:, 5] - y6); ae_b6 = np.abs(p_b6[:, 5] - y6); ae_b1 = np.abs(pers - y6)
    bias_m2 = p_m2[:, 5] - y6; bias_b6 = p_b6[:, 5] - y6

    rows = []
    blk = dict(events)
    for ev, _ in events:
        m = (evs == ev) & fire
        if m.sum() < T.N_MIN_FIRE:
            continue
        days = pd.Series(t0[m].date)
        d = pd.DataFrame({"day": days.values, "ae_m2": ae_m2[m], "ae_b6": ae_b6[m], "ae_b1": ae_b1[m],
                          "bias_m2": bias_m2[m], "bias_b6": bias_b6[m]})
        g = d.groupby("day").sum(numeric_only=True)
        cnt = d.groupby("day").size()
        worst_day = g.ae_m2.idxmax()
        pp_day = (pd.Timestamp(peaks[ev]) + pd.Timedelta(days=1)).date()
        pp = d[d.day == pp_day]
        total_dAE = float((g.ae_b6 - g.ae_m2).sum())
        pp_dAE = float((pp.ae_b6 - pp.ae_m2).sum()) if len(pp) else 0.0
        rest_dAE = total_dAE - pp_dAE
        rows.append({"event": ev, "block": blk[ev], "n_fire": int(m.sum()), "peak_utc": str(peaks[ev]),
                     "SS_M2": 1 - ae_m2[m].mean() / ae_b1[m].mean(), "SS_B6": 1 - ae_b6[m].mean() / ae_b1[m].mean(),
                     "M2_wins": bool(ae_m2[m].mean() < ae_b6[m].mean()),
                     "worst_day_M2": str(worst_day), "worst_day_share_M2": float(g.ae_m2.max() / g.ae_m2.sum()),
                     "worst_day_is_postpeak": bool(worst_day == pp_day),
                     "postpeak_n": int(len(pp)),
                     "postpeak_bias_M2": float(pp.bias_m2.mean()) if len(pp) else np.nan,
                     "postpeak_bias_B6": float(pp.bias_b6.mean()) if len(pp) else np.nan,
                     "postpeak_dAE": pp_dAE, "total_dAE": total_dAE, "rest_dAE": rest_dAE,
                     "decided_by_postpeak": bool(np.sign(total_dAE) != np.sign(rest_dAE)) if rest_dAE != 0 else False})
    df = pd.DataFrame(rows)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    df.to_csv(OUT / f"decisive_day_{stamp}.csv", index=False)

    L = [f"# EXPLORATORY decisive-day diagnostic, blocks A + B ({len(df)} events) -- {stamp}\n"]
    L.append("| block | n | M2 wins | worst day = post-peak day | median worst-day share of M2 AE | outcome flips without post-peak day | "
             "post-peak M2 bias > 0 | post-peak M2 bias > B6 bias |\n|---|---|---|---|---|---|---|---|")
    for b in ("A2025", "B2026", "all"):
        s = df if b == "all" else df[df.block == b]
        L.append(f"| {b} | {len(s)} | {int(s.M2_wins.sum())} | {int(s.worst_day_is_postpeak.sum())} | "
                 f"{s.worst_day_share_M2.median():.2f} | {int(s.decided_by_postpeak.sum())} | "
                 f"{int((s.postpeak_bias_M2 > 0).sum())} | {int((s.postpeak_bias_M2 > s.postpeak_bias_B6).sum())} |")
    L.append("\n## Sign test M2 vs B6 restricted to windows OUTSIDE the post-peak day (post hoc)\n")
    for b in ("A2025", "B2026", "all"):
        s = df if b == "all" else df[df.block == b]
        w = int((s.rest_dAE > 0).sum()); l = int((s.rest_dAE < 0).sum())
        L.append(f"- {b}: {w}/{l}, p = {binomtest(w, w + l, 0.5, alternative='greater').pvalue:.4f}")
    L.append("\n## Sign test on the post-peak day ALONE\n")
    for b in ("A2025", "B2026", "all"):
        s = df[(df.postpeak_n > 0)] if b == "all" else df[(df.block == b) & (df.postpeak_n > 0)]
        w = int((s.postpeak_dAE > 0).sum()); l = int((s.postpeak_dAE < 0).sum())
        L.append(f"- {b}: {w}/{l}, p = {binomtest(w, w + l, 0.5, alternative='greater').pvalue:.4f}")
    L.append("\n## Events where the outcome flips without the post-peak day\n")
    L.append("| event | block | winner | post-peak dAE (B6−M2) | rest dAE | post-peak bias M2 | bias B6 |\n|---|---|---|---|---|---|---|")
    for _, r in df[df.decided_by_postpeak].sort_values("postpeak_dAE").iterrows():
        L.append(f"| {r.event} | {r.block} | {'M2' if r.M2_wins else 'B6'} | {r.postpeak_dAE:+,.0f} | {r.rest_dAE:+,.0f} | "
                 f"{r.postpeak_bias_M2:+,.0f} | {r.postpeak_bias_B6:+,.0f} |")
    txt = "\n".join(L) + "\n"
    (OUT / f"decisive_day_{stamp}.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
