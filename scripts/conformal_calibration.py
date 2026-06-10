"""Validation-only conformal calibration of the PRT-Full deep ensemble.

Demonstrates that calibrated predictive intervals are attainable on validation
via leave-one-validation-event-out (LOEO) split/cross-conformal on the deployed
ensemble point forecast, and contrasts against the inter-seed Q25-Q75 baseline
(the manuscript A9 calibration definition).

Reuses the existing PRT-Full 10-seed VALIDATION predictions from
results/partA/prt_full/seed_NNN_valpreds.npy (and optionally prt_base).
The point forecast is the deployed ensemble: log-space mean over 10 seeds,
expm1, non-negativity clip — identical to the locked ensembling rule.

Sample set (matches manuscript A9 / eval_test_row2.py): fire-active VAL samples
(window mask (y>0).any over the 12 h target), evaluated at t+6. Four fire-active
val events: alexandroupolis, caceres, mandra, rhodes (split_2017 has 0 fire).

Nonconformity (log space): s = |log1p(y_t6) - log1p(yhat_ens_t6)|.
Conformal quantile with finite-sample correction: q = sorted(s_cal)[k-1],
k = ceil((n_cal+1)*(1-alpha)). Intervals in log space, mapped back with expm1,
lower bound clipped at 0.

VALIDATION SPLIT ONLY. load_split('test') is never called. Outputs:
results/calibration_validation_<date>.csv and .json.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))  # repo src/ holds the shared/ package
from shared.dataset_io import load_manifest, load_split  # noqa: E402

RES = _HERE.parent / "results"
DATE = "2026-06-02"
SEEDS = list(range(10))
FIRE_EVENTS = ["alexandroupolis_2023", "caceres_2023", "mandra_2023", "rhodes_2023"]
H6 = 5                      # t+6 column index (0-based)
LEVELS = [0.50, 0.80, 0.90]


def load_perseed_and_ensemble(cfg: str):
    """Return (per_seed (S,N,12), ensemble (N,12)) for a Part A config."""
    d = RES / "partA" / cfg
    preds = []
    for s in SEEDS:
        p = d / f"seed_{s:03d}_valpreds.npy"
        if not p.exists():
            raise FileNotFoundError(p)
        preds.append(np.load(p).astype(np.float64))
    stack = np.stack(preds, axis=0)                       # (S, N, 12)
    log_mean = np.log1p(np.clip(stack, 0.0, None)).mean(axis=0)
    ens = np.clip(np.expm1(log_mean), 0.0, None)          # (N, 12)
    return stack, ens


def conformal_quantile(s_cal: np.ndarray, level: float) -> float:
    """Split-conformal quantile of nonconformity scores with finite-sample correction."""
    n = len(s_cal)
    k = int(np.ceil((n + 1) * level))
    if k > n:
        return float("inf")
    return float(np.sort(s_cal)[k - 1])


def summarize_width(width: np.ndarray, mean_obs: float) -> dict:
    wm = float(np.mean(width))
    wmed = float(np.median(width))
    return {
        "width_mean_mw": wm,
        "width_median_mw": wmed,
        "width_mean_frac": (wm / mean_obs) if mean_obs > 0 else float("nan"),
        "width_median_frac": (wmed / mean_obs) if mean_obs > 0 else float("nan"),
    }


def run_config(cfg: str, val_targets: np.ndarray, ev: np.ndarray, fire: np.ndarray):
    stack, ens = load_perseed_and_ensemble(cfg)
    y6_all = val_targets[:, H6]
    ens6_all = ens[:, H6]

    # restrict to fire-active samples
    f = fire
    yf = y6_all[f]
    ehatf = ens6_all[f]
    evf = ev[f]
    sf = np.abs(np.log1p(yf) - np.log1p(ehatf))

    # per-seed t+6 for inter-seed baseline (fire-active)
    per_seed6 = stack[:, :, H6][:, f]                     # (S, n_fire)
    q25 = np.percentile(per_seed6, 25, axis=0)
    q75 = np.percentile(per_seed6, 75, axis=0)

    n_fire = {E: int((evf == E).sum()) for E in FIRE_EVENTS}

    # ---- LOEO cross-conformal ----
    conformal = {f"{lv:.2f}": {} for lv in LEVELS}
    for E in FIRE_EVENTS:
        te = evf == E
        cal = ~te
        yE = yf[te]
        ehatE = ehatf[te]
        mean_obs = float(np.mean(yE))
        for lv in LEVELS:
            q = conformal_quantile(sf[cal], lv)
            lo = np.clip(np.expm1(np.log1p(ehatE) - q), 0.0, None)
            hi = np.expm1(np.log1p(ehatE) + q)
            cov = float(np.mean((yE >= lo) & (yE <= hi)))
            width = hi - lo
            rec = {"event": E, "n_fire": int(te.sum()), "coverage": cov,
                   "q_log": q, "mean_obs_frp": mean_obs}
            rec.update(summarize_width(width, mean_obs))
            conformal[f"{lv:.2f}"][E] = rec

    # event-level means
    conformal_mean = {}
    for lv in LEVELS:
        recs = [conformal[f"{lv:.2f}"][E] for E in FIRE_EVENTS]
        conformal_mean[f"{lv:.2f}"] = {
            "coverage": float(np.mean([r["coverage"] for r in recs])),
            "width_mean_mw": float(np.mean([r["width_mean_mw"] for r in recs])),
            "width_median_mw": float(np.mean([r["width_median_mw"] for r in recs])),
            "width_mean_frac": float(np.mean([r["width_mean_frac"] for r in recs])),
            "width_median_frac": float(np.mean([r["width_median_frac"] for r in recs])),
        }

    # ---- inter-seed Q25-Q75 baseline (nominal 0.50) ----
    interseed = {}
    for E in FIRE_EVENTS:
        te = evf == E
        yE = yf[te]
        loE, hiE = q25[te], q75[te]
        cov = float(np.mean((yE >= loE) & (yE <= hiE)))
        width = hiE - loE
        mean_obs = float(np.mean(yE))
        rec = {"event": E, "n_fire": int(te.sum()), "coverage": cov, "mean_obs_frp": mean_obs}
        rec.update(summarize_width(width, mean_obs))
        interseed[E] = rec
    interseed_mean = {
        "coverage": float(np.mean([interseed[E]["coverage"] for E in FIRE_EVENTS])),
        "width_mean_mw": float(np.mean([interseed[E]["width_mean_mw"] for E in FIRE_EVENTS])),
        "width_median_mw": float(np.mean([interseed[E]["width_median_mw"] for E in FIRE_EVENTS])),
        "width_mean_frac": float(np.mean([interseed[E]["width_mean_frac"] for E in FIRE_EVENTS])),
        "width_median_frac": float(np.mean([interseed[E]["width_median_frac"] for E in FIRE_EVENTS])),
    }

    return {
        "config": cfg, "n_fire_per_event": n_fire,
        "conformal_per_event": conformal, "conformal_mean": conformal_mean,
        "interseed_per_event": interseed, "interseed_mean": interseed_mean,
    }


def main():
    # ---- VALIDATION ONLY ----
    _, val_targets = load_split("val")          # test split never touched
    val_targets = val_targets.astype(np.float64)
    man = load_manifest("val")
    ev = man["event_name"].values
    fire = (val_targets > 0).any(axis=1)
    N = int(val_targets.shape[0])

    print("=" * 72)
    print("VALIDATION-ONLY CONFORMAL CALIBRATION — PRT-Full deep ensemble")
    print("=" * 72)
    print(f"  seeds: {SEEDS}")
    print(f"  validation N = {N} (val); TEST split NOT loaded (load_split('test') never called)")
    print(f"  fire-active per event (t+6 sample set, window fire mask):")
    f = fire
    for E in FIRE_EVENTS:
        print(f"    {E:24s}: {int(((ev == E) & f).sum())}")
    print(f"    {'TOTAL fire-active':24s}: {int(f.sum())}")

    configs = ["prt_full"]
    # optional extra (cheap, available in validation-track outputs)
    if (RES / "partA" / "prt_base").exists():
        configs.append("prt_base")

    results = {c: run_config(c, val_targets, ev, fire) for c in configs}

    # ---- console report ----
    for c in configs:
        r = results[c]
        print("\n" + "-" * 72)
        print(f"[{c}] LOEO cross-conformal coverage (nominal | mean | per-event)")
        print("-" * 72)
        for lv in LEVELS:
            key = f"{lv:.2f}"
            cm = r["conformal_mean"][key]
            pe = " ".join(f"{E[:5]}={r['conformal_per_event'][key][E]['coverage']:.4f}"
                          for E in FIRE_EVENTS)
            print(f"  nominal {lv:.2f}: mean_cov={cm['coverage']:.4f} | {pe}")
            print(f"             width mean={cm['width_mean_mw']:.1f} MW "
                  f"(frac {cm['width_mean_frac']:.4f}), median={cm['width_median_mw']:.1f} MW "
                  f"(frac {cm['width_median_frac']:.4f})")
        im = r["interseed_mean"]
        pe = " ".join(f"{E[:5]}={r['interseed_per_event'][E]['coverage']:.4f}" for E in FIRE_EVENTS)
        print(f"  inter-seed Q25-Q75 (nominal 0.50): mean_cov={im['coverage']:.4f} | {pe}")
        print(f"             band width mean={im['width_mean_mw']:.1f} MW "
              f"(frac {im['width_mean_frac']:.4f}), median={im['width_median_mw']:.1f} MW")

    # ---- write CSV (long/tidy) ----
    csv_path = RES / f"calibration_validation_{DATE}.csv"
    with open(csv_path, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["config", "method", "nominal", "event", "n_fire", "empirical_coverage",
                    "width_mean_mw", "width_median_mw", "width_mean_frac", "width_median_frac"])
        for c in configs:
            r = results[c]
            for lv in LEVELS:
                key = f"{lv:.2f}"
                for E in FIRE_EVENTS:
                    rec = r["conformal_per_event"][key][E]
                    w.writerow([c, "conformal_loeo", f"{lv:.2f}", E, rec["n_fire"],
                                f"{rec['coverage']:.4f}", f"{rec['width_mean_mw']:.4f}",
                                f"{rec['width_median_mw']:.4f}", f"{rec['width_mean_frac']:.4f}",
                                f"{rec['width_median_frac']:.4f}"])
                cm = r["conformal_mean"][key]
                w.writerow([c, "conformal_loeo", f"{lv:.2f}", "MEAN", int(f.sum()),
                            f"{cm['coverage']:.4f}", f"{cm['width_mean_mw']:.4f}",
                            f"{cm['width_median_mw']:.4f}", f"{cm['width_mean_frac']:.4f}",
                            f"{cm['width_median_frac']:.4f}"])
            for E in FIRE_EVENTS:
                rec = r["interseed_per_event"][E]
                w.writerow([c, "interseed_q25q75", "0.50", E, rec["n_fire"],
                            f"{rec['coverage']:.4f}", f"{rec['width_mean_mw']:.4f}",
                            f"{rec['width_median_mw']:.4f}", f"{rec['width_mean_frac']:.4f}",
                            f"{rec['width_median_frac']:.4f}"])
            im = r["interseed_mean"]
            w.writerow([c, "interseed_q25q75", "0.50", "MEAN", int(f.sum()),
                        f"{im['coverage']:.4f}", f"{im['width_mean_mw']:.4f}",
                        f"{im['width_median_mw']:.4f}", f"{im['width_mean_frac']:.4f}",
                        f"{im['width_median_frac']:.4f}"])
    print(f"\n  wrote {csv_path}")

    out = {
        "date": DATE, "seeds": SEEDS, "validation_N": N,
        "test_partition_loaded": False,
        "sample_set": "fire-active (window mask) val samples, evaluated at t+6",
        "fire_active_events": FIRE_EVENTS,
        "nonconformity": "abs log-space residual |log1p(y) - log1p(yhat_ens)|",
        "conformal_scheme": "leave-one-validation-event-out split/cross-conformal, "
                            "k=ceil((n_cal+1)*(1-alpha))",
        "ensemble_rule": "clip>=0 -> log1p -> mean over 10 seeds -> expm1 -> clip>=0",
        "levels": LEVELS,
        "results": results,
    }
    json_path = RES / f"calibration_validation_{DATE}.json"
    with open(json_path, "w") as fj:
        json.dump(out, fj, indent=2)
    print(f"  wrote {json_path}")

    print("\n" + "=" * 72)
    print("GUARDRAIL CONFIRMATION")
    print("=" * 72)
    print(f"  Seeds: {SEEDS}")
    print(f"  Validation N reported: {N}. TEST partition NEVER loaded "
          f"(load_split('test') not called; 558-row test split sealed).")
    print("  Cumulative test-access budget untouched (locked at 3).")
    print("[conformal_calibration] done")


if __name__ == "__main__":
    main()
