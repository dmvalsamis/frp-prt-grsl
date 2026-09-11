#!/usr/bin/env python
"""explore_arch_eval.py -- EXPLORATORY evaluation of the architecture-study ensembles on
test blocks A (2025, 45 evaluable) + B (2026, 32), once per architecture, after all
training (branch explore-arch-2026).  NOT PRE-REGISTERED; blocks A and B are examined data.

Machinery imported unchanged from the locked scripts/test_access_v1.py: load_event,
engineer_features, build_sequences, event_volatility, persistence_anchor, b6_train_predict
(deterministic refit), m2_predict (hash-verified), m1_predict (descriptive), N_min = 5,
AMEND-8 constants, sign-test / Wilcoxon definitions.

Identity check (mandatory, aborts on failure): per-event MAE_B6_t6 and MAE_M2_t6 on the
45 block-A evaluable events must equal the locked
resubmission/TEST_ACCESS_RESULTS/access_20260828T123059Z_per_event.csv to 1e-3.

Per architecture: the 10 checkpoints are hash-verified against
data/models/explore_arch/{arch}_freeze_hashes.json, the log-mean ensemble predicts every
window, and a per-event table with the access-table columns (MAE / SS at t+1/3/6/12,
calm / excursion split, volatility tertile) is written to
resubmission/EXPLORATORY_ARCH_2026/{arch}_per_event.csv.  Statistics per architecture,
per block and pooled: sign tests (one-sided exact binomial + Wilcoxon) vs B6 / B1 / M2,
AGG-1, horizon medians, AMEND-8 strata (wins + median dSS only).  Master table:
resubmission/EXPLORATORY_ARCH_2026/master_table.md (+ master_results.json).
Window-level predictions are cached under data/models/explore_arch/eval_cache/ for the
post-peak-day diagnostic (scripts/explore_arch_decisive.py).

Run:  PYTHONUTF8=1 C:/Users/dvalsamis/AppData/Local/anaconda3/python.exe scripts/explore_arch_eval.py [--refs-only]
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest, wilcoxon

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import test_access_v1 as T  # noqa: E402  (locked; imported unchanged)
import explore_arch_train as EA  # noqa: E402  (ARCHS registry; asserts environment)

OUT = ROOT / "resubmission" / "EXPLORATORY_ARCH_2026"
MODELS = EA.OUT_DIR
CACHE = MODELS / "eval_cache"
A_CSV = ROOT / "resubmission" / "TEST_ACCESS_RESULTS" / "access_20260828T123059Z_per_event.csv"
B_DIR = ROOT / "resubmission" / "EXPLORATORY_BLOCKB_2026"
H = T.STRATA_HORIZONS
REFS = ["B6", "M2", "M1", "B1"]
SETS = [("A2025", "block A (45)"), ("B2026", "block B (32)"), ("pooled", "pooled A+B (77)")]


# ----------------------------------------------------------------- windows + refs
def build_windows(feats):
    f = CACHE / "windows_AB.npz"
    if f.exists():
        z = np.load(f, allow_pickle=False)
        print(f"windows loaded from cache: {len(z['X'])} windows, {len(set(z['ev']))} events", flush=True)
        return {k: z[k] for k in z.files}
    A = pd.read_csv(A_CSV)
    B_CSV = sorted(B_DIR.glob("exploratory_blockB_*_per_event.csv"))[-1]
    B = pd.read_csv(B_CSV)
    events = ([(e, "A2025") for e in A[A.evaluable == True].event]
              + [(e, "B2026") for e in B[B.evaluable == True].event])
    X, Y, ev, blk, t0, vol, peak = [], [], [], [], [], [], []
    for e, b in events:
        merged = T.load_event(e)
        v = T.event_volatility(merged)
        pk = merged.loc[merged.frp_sum_mw.idxmax(), "slot_utc"]
        grp = T.engineer_features(merged)
        inp, tgt, wm = T.build_sequences(grp, feats)
        X += inp; Y += tgt; ev += [e] * len(inp); blk += [b] * len(inp)
        t0 += [(pd.Timestamp(m["target_end_utc"]) - pd.Timedelta(hours=T.HORIZON)).value for m in wm]
        vol += [v] * len(inp); peak += [pd.Timestamp(pk).value] * len(inp)
        print(f"  {e} [{b}]: {len(inp)} windows", flush=True)
    d = {"X": np.stack(X).astype(np.float32), "Y": np.stack(Y).astype(np.float32),
         "ev": np.array(ev), "block": np.array(blk), "t0_ns": np.array(t0, dtype=np.int64),
         "vol": np.array(vol, dtype=np.float64), "peak_ns": np.array(peak, dtype=np.int64),
         "blockB_source": np.array([B_CSV.name])}
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(f, **d)
    print(f"windows built: {len(d['X'])} over {len(events)} events -> {f}", flush=True)
    return d


def reference_preds(W, feats, norm):
    f = CACHE / "ref_preds.npz"
    if f.exists():
        z = np.load(f)
        print("reference predictions loaded from cache", flush=True)
        return {k: z[k] for k in z.files}
    FEAT = {n: i for i, n in enumerate(feats)}
    dev = torch.device("cpu")
    print("B6 (gbdt_log_T1) deterministic refit + predict ...", flush=True)
    p_b6 = T.b6_train_predict(W["X"], feats)
    print("M2 ensemble (hash-verified) ...", flush=True)
    p_m2 = T.m2_predict(W["X"], feats, norm, dev)
    print("M1 deposit ensemble (descriptive) ...", flush=True)
    p_m1 = T.m1_predict(W["X"], feats, dev)
    pers = T.persistence_anchor(W["X"], FEAT)
    p_b1 = np.tile(pers[:, None], (1, T.HORIZON))
    d = {"B6": p_b6.astype(np.float32), "M2": p_m2.astype(np.float32),
         "M1": p_m1.astype(np.float32), "B1": p_b1.astype(np.float32)}
    np.savez(f, **d)
    return d


def identity_check(W, P):
    """Per-event MAE_B6_t6 / MAE_M2_t6 on block A must equal the locked access table."""
    A = pd.read_csv(A_CSV).set_index("event")
    fire = W["Y"].max(axis=1) > 0
    worst = 0.0
    rows = []
    for e in sorted(set(W["ev"][W["block"] == "A2025"])):
        fm = (W["ev"] == e) & fire
        for name in ("B6", "M2"):
            mae = float(np.mean(np.abs(P[name][fm][:, 5] - W["Y"][fm][:, 5])))
            locked = float(A.loc[e, f"MAE_{name}_t6"])
            worst = max(worst, abs(mae - locked))
            rows.append((e, name, mae, locked, abs(mae - locked)))
    n_ev = len(rows) // 2
    if worst > 1e-3:
        bad = [r for r in rows if r[4] > 1e-3]
        raise SystemExit(f"IDENTITY CHECK FAILED: max |diff| = {worst:.3e} > 1e-3 on {len(bad)} rows: {bad[:5]}")
    print(f"IDENTITY CHECK PASSED: {n_ev} block-A events, MAE_B6_t6 and MAE_M2_t6 match the locked "
          f"access table; max |diff| = {worst:.2e} (tol 1e-3)", flush=True)
    return {"passed": True, "n_events": n_ev, "max_abs_diff": worst, "tolerance": 1e-3,
            "locked_table": str(A_CSV.relative_to(ROOT))}


# ----------------------------------------------------------------- architecture inference
def arch_predict(arch, W, feats, norm):
    f = CACHE / f"{arch}_preds_AB.npy"
    if f.exists():
        print(f"  {arch}: predictions loaded from cache", flush=True)
        return np.load(f)
    frozen = json.load(open(MODELS / f"{arch}_freeze_hashes.json"))
    seeds = frozen["ensemble_seeds"]
    FEAT = {n: i for i, n in enumerate(feats)}
    x = torch.tensor(T.normalize_T1_robust(W["X"], feats, norm))
    pers = torch.tensor(T.persistence_anchor(W["X"], FEAT))
    cls = EA.ARCHS[arch][1]
    logs = []
    for s in seeds:
        ckpt = MODELS / f"{arch}_s{s}_best.pt"
        actual = T.sha256(ckpt)
        expect = frozen["sha256"][f"checkpoint_{arch}_s{s}"]
        if actual != expect:
            raise RuntimeError(f"HASH MISMATCH {ckpt.name}: {actual} != {expect}")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = cls(len(feats), T.HORIZON)
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        ps = []
        with torch.no_grad():
            for i in range(0, len(x), 256):
                ps.append(model(x[i:i + 256], pers[i:i + 256]).numpy())
        logs.append(np.log1p(np.clip(np.concatenate(ps, 0), 0, None)))
        print(f"  {arch} member s{s}: hash OK, predicted", flush=True)
    p = np.expm1(np.mean(np.stack(logs, 0), axis=0)).astype(np.float32)
    np.save(f, p)
    return p


# ----------------------------------------------------------------- per-event table + stats
def per_event_table(W, preds, feats):
    """Same columns as the access tables, for every model name in `preds` (B1 last)."""
    FEAT = {n: i for i, n in enumerate(feats)}
    names = [n for n in preds if n != "B1"]
    fire = W["Y"].max(axis=1) > 0
    excursion = np.abs(W["X"][:, T.LAST_TS, FEAT["delta_wind_3h"]]) > T.DELTA_WIND_3H_SIGMA
    rows = []
    for e in sorted(set(W["ev"])):
        m = W["ev"] == e
        fm = m & fire
        n_fire = int(fm.sum())
        v = float(W["vol"][m][0])
        r = {"event": e, "block": str(W["block"][m][0]), "n_windows": int(m.sum()),
             "n_fire": n_fire, "evaluable": n_fire >= T.N_MIN_FIRE, "volatility_V": v,
             "vol_tertile": ("low" if v <= T.VOL_TERTILE_LO else "mid" if v <= T.VOL_TERTILE_HI
                             else "high") if not np.isnan(v) else "na"}
        for name in names + ["B1"]:
            for k in H:
                r[f"MAE_{name}_t{k}"] = float(np.mean(np.abs(preds[name][fm][:, k - 1] - W["Y"][fm][:, k - 1])))
        for k in H:
            pmk = r[f"MAE_B1_t{k}"]
            for name in names:
                r[f"SS_{name}_t{k}"] = (1.0 - r[f"MAE_{name}_t{k}"] / pmk) if pmk else float("nan")
        for rname, rmask in (("calm", ~excursion), ("exc", excursion)):
            sm = fm & rmask
            r[f"n_fire_{rname}"] = int(sm.sum())
            mae_b1 = float(np.mean(np.abs(preds["B1"][sm][:, 5] - W["Y"][sm][:, 5]))) if sm.any() else 0.0
            for name in names:
                if sm.any() and mae_b1:
                    mae = float(np.mean(np.abs(preds[name][sm][:, 5] - W["Y"][sm][:, 5])))
                    r[f"SS_{name}_t6_{rname}"] = 1.0 - mae / mae_b1
                else:
                    r[f"SS_{name}_t6_{rname}"] = float("nan")
        rows.append(r)
    return pd.DataFrame(rows)


def sign_test(sub, model, ref, k=6):
    """test_access_v1 definition: d = MAE_ref - MAE_model; wins = d > 0; ties dropped;
    one-sided exact binomial + Wilcoxon (alternative='greater')."""
    a, b = f"MAE_{model}_t{k}", f"MAE_{ref}_t{k}"
    paired = sub[[a, b]].dropna()
    d = paired[b] - paired[a]
    wins, losses = int((d > 0).sum()), int((d < 0).sum())
    n = wins + losses
    if n == 0:
        return {"wins": 0, "losses": 0, "n": 0, "p_one_sided": float("nan"), "wilcoxon_p": float("nan")}
    p = binomtest(wins, n, 0.5, alternative="greater").pvalue
    try:
        w = wilcoxon(paired[b], paired[a], alternative="greater").pvalue
    except Exception:
        w = float("nan")
    return {"wins": wins, "losses": losses, "n": n, "p_one_sided": float(p), "wilcoxon_p": float(w)}


def agg1(sub, model):
    s = sub[f"SS_{model}_t6"]
    return {"n": int(s.notna().sum()), "median": float(np.nanmedian(s)), "mean": float(np.nanmean(s)),
            "std": float(np.nanstd(s, ddof=1)), "negatives": int((s < 0).sum()), "min": float(np.nanmin(s))}


def stratum(dss):
    d = [x for x in dss if not np.isnan(x)]
    return {"n": len(d), "wins": int(sum(x > 0 for x in d)), "losses": int(sum(x < 0 for x in d)),
            "median_dSS": float(np.median(d)) if d else float("nan")}


def strata(sub, model, ref="B6"):
    return {"horizon": {f"t{k}": stratum((sub[f"SS_{model}_t{k}"] - sub[f"SS_{ref}_t{k}"]).tolist()) for k in H},
            "wind_t6": {r: stratum((sub[f"SS_{model}_t6_{c}"] - sub[f"SS_{ref}_t6_{c}"]).tolist())
                        for r, c in (("calm", "calm"), ("excursion", "exc"))},
            "volatility_t6": {t: stratum((g[f"SS_{model}_t6"] - g[f"SS_{ref}_t6"]).tolist())
                              for t, g in sub.groupby("vol_tertile") if t != "na"}}


def stats_for(df, model):
    out = {}
    for key, _ in SETS:
        sub = df if key == "pooled" else df[df.block == key]
        sub = sub[sub.evaluable]
        o = {"n": int(len(sub)), "AGG1": agg1(sub, model),
             "horizon_medians": {f"t{k}": float(np.nanmedian(sub[f"SS_{model}_t{k}"])) for k in H}}
        for ref in ("B6", "B1", "M2"):
            if ref != model:
                o[f"sign_vs_{ref}"] = sign_test(sub, model, ref)
        if model != "B6":
            o["strata_vs_B6"] = strata(sub, model)
        out[key] = o
    return out


def fmt_sign(s):
    return "--" if s is None else f"{s['wins']}/{s['losses']} p={s['p_one_sided']:.3f} (W {s['wilcoxon_p']:.3f})"


def write_master(all_stats, names, idc, n_params, stamp):
    L = [f"# EXPLORATORY architecture study -- master table, blocks A (2025) + B (2026) -- {stamp}\n",
         "**Not pre-registered. Every number is exploratory.** Block A = the 45 evaluable events of the "
         "locked 2026-08-28 access; block B = the 32 gate-passing 2026 events of the 2026-09-08 "
         "exploratory run. Windows, B6 refit, M2 and M1 ensembles, B1, N_min, AGG-1, strata and sign-test "
         "definitions are those of `scripts/test_access_v1.py`, imported unchanged. Six architectures "
         "(+ M2) were scored on the same 77 fires: one may look best by chance; no winner is headlined; "
         "an architecture that beats B6 here is a hypothesis for a 2027 pre-registration, not a claim. "
         "Wins are counted on MAE_fire,t6 (model lower = win); p = one-sided exact binomial; W = Wilcoxon.\n",
         f"Identity check: **{'PASSED' if idc['passed'] else 'FAILED'}** -- {idc['n_events']} block-A events, "
         f"MAE_B6_t6 and MAE_M2_t6 reproduce the locked access table to {idc['tolerance']:g} "
         f"(max |diff| {idc['max_abs_diff']:.2e}).\n",
         "| model | params | set | n | wins/losses vs B6 | vs B1 | vs M2 | median SS | mean +- std | neg | min |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in names:
        for key, lab in SETS:
            s = all_stats[m][key]
            a = s["AGG1"]
            L.append(f"| {m} | {n_params.get(m, '--')} | {lab} | {s['n']} | {fmt_sign(s.get('sign_vs_B6'))} | "
                     f"{fmt_sign(s.get('sign_vs_B1'))} | {fmt_sign(s.get('sign_vs_M2'))} | {a['median']:+.4f} | "
                     f"{a['mean']:+.4f} +- {a['std']:.4f} | {a['negatives']} | {a['min']:+.3f} |")
    L.append("\n## Horizon profile (median SS_fire by horizon; pooled 77)\n")
    L.append("| model | t+1 | t+3 | t+6 | t+12 |\n|---|---|---|---|---|")
    for m in names:
        hm = all_stats[m]["pooled"]["horizon_medians"]
        L.append(f"| {m} | " + " | ".join(f"{hm[f't{k}']:+.4f}" for k in H) + " |")
    L.append("\n## AMEND-8 strata vs B6 (wins/losses, median dSS; descriptive only)\n")
    L.append("| model | set | t+1 | t+3 | t+6 | t+12 | calm | excursion | vol high | vol mid | vol low |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for m in names:
        if m == "B6":
            continue
        for key, lab in SETS:
            st = all_stats[m][key]["strata_vs_B6"]
            cells = [st["horizon"][f"t{k}"] for k in H] + [st["wind_t6"]["calm"], st["wind_t6"]["excursion"]] + \
                    [st["volatility_t6"].get(t) for t in ("high", "mid", "low")]
            L.append(f"| {m} | {lab} | " + " | ".join(
                "--" if c is None else f"{c['wins']}/{c['losses']} ({c['median_dSS']:+.3f})" for c in cells) + " |")
    txt = "\n".join(L) + "\n"
    (OUT / "master_table.md").write_text(txt, encoding="utf-8")
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs-only", action="store_true",
                    help="build windows, reference predictions and the identity check only")
    a = ap.parse_args()
    T.assert_environment()
    T.assert_input_hashes(dry_run=False)
    with h5py.File(T.H5, "r") as f:
        meta = json.loads(f["metadata"][()])
        feats = meta["feature_names"]
        norm = meta["normalization_params"]["T1"]["robust"]
    OUT.mkdir(parents=True, exist_ok=True)
    W = build_windows(feats)
    P = reference_preds(W, feats, norm)
    idc = identity_check(W, P)
    with open(OUT / "identity_check.json", "w") as f:
        json.dump({**idc, "utc": datetime.now(timezone.utc).isoformat()}, f, indent=2)
    if a.refs_only:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    archs = [k for k in EA.ARCHS if (MODELS / f"{k}_freeze_hashes.json").exists()]
    missing = [k for k in EA.ARCHS if k not in archs]
    if missing:
        print(f"WARNING: no freeze table for {missing}; they are reported as not evaluated", flush=True)
    all_stats, n_params = {}, {}
    ref_df = per_event_table(W, P, feats)
    for arch in archs:
        print(f"\n=== {arch} {EA.ARCHS[arch][0]} ===", flush=True)
        p = arch_predict(arch, W, feats, norm)
        df = per_event_table(W, {arch: p, **P}, feats)
        df.to_csv(OUT / f"{arch}_per_event.csv", index=False)
        all_stats[arch] = stats_for(df, arch)
        n_params[arch] = f"{json.load(open(MODELS / f'{arch}_ensemble_val_results.json'))['n_params']:,}"
        s6 = all_stats[arch]["pooled"]["sign_vs_B6"]
        print(f"  pooled vs B6: {s6['wins']}/{s6['losses']} p={s6['p_one_sided']:.3f}; "
              f"median SS {all_stats[arch]['pooled']['AGG1']['median']:+.4f}", flush=True)
    for m in ("M2", "B6", "M1"):
        all_stats[m] = stats_for(ref_df, m)
    n_params["M2"] = f"{EA.n_params(EA.e28.PRTWithPersistenceSkip(EA.INPUT_SIZE, EA.HORIZON)):,} x10"
    ref_df.to_csv(OUT / "reference_models_per_event.csv", index=False)
    names = archs + ["M2", "B6", "M1"]
    write_master(all_stats, names, idc, n_params, stamp)
    with open(OUT / "master_results.json", "w") as f:
        json.dump({"status": "EXPLORATORY -- NOT PRE-REGISTERED; blocks A and B are examined data",
                   "utc": stamp, "identity_check": idc, "archs_evaluated": archs,
                   "archs_missing": missing, "n_params": n_params,
                   "test_access_v1_sha256": T.sha256(ROOT / "scripts" / "test_access_v1.py"),
                   "blockB_source": str(W["blockB_source"][0]),
                   "counts": {"windows": int(len(W["X"])), "fire_active": int((W["Y"].max(axis=1) > 0).sum()),
                              "events_A": int(len(set(W["ev"][W["block"] == "A2025"]))),
                              "events_B": int(len(set(W["ev"][W["block"] == "B2026"])))},
                   "stats": all_stats}, f, indent=2)
    print(f"\nwrote {OUT / 'master_table.md'} and master_results.json", flush=True)


if __name__ == "__main__":
    main()
