#!/usr/bin/env python
"""explore_arch_ensemble.py -- EXPLORATORY architecture study, T2/T3 deliverables.

For each architecture A1..A6 (scripts/explore_arch_train.py, branch explore-arch-2026):
  * validation table row: per-seed median SS_fire,t6 (10 seeds), seed-mean, best epochs,
    parameter count -- beside M2 (grid row +0.3348 seed-mean over seeds 0-2, 10-seed
    ensemble +0.3131) and B6 (gbdt_log_T1, val median +0.3050);
  * 10-member LOG-MEAN ensemble of the saved val predictions
    (expm1(mean(log1p(pred)))), val AGG-1 median / mean +- std / per-event table /
    paired wins vs B6 (per-event B6 values from
    data/processed/baselines_v6_1_b2_log_table.csv, column SS_gbdt_log_T1);
  * freeze hash table over the 10 checkpoints + dataset + manifest + norm params +
    training script (mirrors scripts/e210_ensemble.py).
No selection happens here: every architecture is frozen and goes to test (T4).

Outputs:
  data/models/explore_arch/{arch}_ensemble_val_results.json
  data/models/explore_arch/{arch}_freeze_hashes.json
  resubmission/EXPLORATORY_ARCH_2026/val_table.md
Run:  PYTHONUTF8=1 C:/Users/dvalsamis/AppData/Local/anaconda3/python.exe scripts/explore_arch_ensemble.py
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import explore_arch_train as EA  # noqa: E402  (loads e28 machinery, asserts environment)

e28 = EA.e28
MODELS = EA.OUT_DIR
OUT = ROOT / "resubmission" / "EXPLORATORY_ARCH_2026"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = list(range(10))
B6_TABLE = ROOT / "data" / "processed" / "baselines_v6_1_b2_log_table.csv"
M2_DIR = ROOT / "data" / "models" / "e28"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    b6 = pd.read_csv(B6_TABLE).set_index("event")
    b6 = b6[b6["n_fire"] > 0]
    b6["MAE_B6_t6"] = b6["pers_MAE_t6"] * (1.0 - b6["SS_gbdt_log_T1"])
    b6_median = float(np.median(b6["SS_gbdt_log_T1"]))
    print(f"B6 (gbdt_log_T1) val median SS_fire,t6 = {b6_median:+.4f} over {len(b6)} events")

    # M2 reference rows from its own result files (never retrained)
    m2_res = [json.load(open(M2_DIR / f"T1_robust_s{s}_result.json")) for s in SEEDS]
    m2_ens = json.load(open(M2_DIR / "m2_ensemble_val_results.json"))
    m2_row = {"arch": "M2", "name": "PRT (M2, e28 T1_robust)", "n_params": EA.n_params(
                  e28.PRTWithPersistenceSkip(EA.INPUT_SIZE, EA.HORIZON)),
              "seed_medians": [r["val_median_ss_t6"] for r in m2_res],
              "seed_means": [r["val_mean_ss_t6"] for r in m2_res],
              "best_epochs": [r["best_epoch"] for r in m2_res],
              "ens_median": m2_ens["val_median_ss_t6"], "ens_mean": m2_ens["val_mean_ss_t6"],
              "ens_std": m2_ens["val_std_ss_t6"], "wins_vs_B6": m2_ens["val_wins_vs_B6"],
              "n_runs": 10, "n_failed": 0}

    rows = []
    for arch, (name, cls) in EA.ARCHS.items():
        res, failed = [], []
        for s in SEEDS:
            p = MODELS / f"{arch}_s{s}_result.json"
            if p.exists():
                res.append(json.load(open(p)))
            elif (MODELS / f"{arch}_s{s}_FAILED.json").exists():
                failed.append(s)
        seeds_ok = [r["seed"] for r in res]
        print(f"\n{arch} {name}: {len(res)} results, failed seeds {failed}")
        if not res:
            rows.append({"arch": arch, "name": name, "n_runs": 0, "n_failed": len(failed)})
            continue
        stack = np.stack([np.load(MODELS / f"{arch}_s{s}_val_preds.npy") for s in seeds_ok], 0)
        ens = np.expm1(np.mean(np.log1p(np.clip(stack, 0, None)), axis=0)).astype(np.float32)
        med, ss_list = e28.val_median_ss(ens)
        pe = e28.per_event_val(ens)
        wins, n = 0, 0
        for ev, d in pe.items():
            if d["n_fire"] == 0 or ev not in b6.index:
                continue
            n += 1
            wins += int(d["MAE_fire_t6"] < b6.loc[ev, "MAE_B6_t6"])
            d["SS_B6_t6"] = float(b6.loc[ev, "SS_gbdt_log_T1"])
            d["beats_B6"] = bool(d["MAE_fire_t6"] < b6.loc[ev, "MAE_B6_t6"])
        row = {"arch": arch, "name": name, "n_params": res[0]["n_params"],
               "seed_medians": [r["val_median_ss_t6"] for r in res],
               "seed_means": [r["val_mean_ss_t6"] for r in res],
               "best_epochs": [r["best_epoch"] for r in res],
               "ens_median": med, "ens_mean": float(np.mean(ss_list)),
               "ens_std": float(np.std(ss_list, ddof=1)), "wins_vs_B6": [wins, n],
               "n_runs": len(res), "n_failed": len(failed), "seeds": seeds_ok}
        rows.append(row)
        print(f"  seed-mean of medians {np.mean(row['seed_medians']):+.4f}; ensemble median "
              f"{med:+.4f}, mean {row['ens_mean']:+.4f} +- {row['ens_std']:.4f}; wins vs B6 {wins}/{n}")
        result = {"status": "EXPLORATORY", "arch": arch, "arch_name": name,
                  "cell": [EA.VARIANT, EA.SCHEME], "ensemble_seeds": seeds_ok,
                  "failed_seeds": failed,
                  "combination_rule": "log-mean: expm1(mean(log1p(preds)))",
                  "n_params": res[0]["n_params"],
                  "val_median_ss_t6": med, "val_mean_ss_t6": row["ens_mean"],
                  "val_std_ss_t6": row["ens_std"], "n_events_evaluable": len(ss_list),
                  "val_wins_vs_B6": [wins, n], "B6_val_median_ss_t6": b6_median,
                  "per_seed": [{k: r[k] for k in ("seed", "best_epoch", "val_median_ss_t6",
                                                  "val_mean_ss_t6", "residual_scale",
                                                  "train_minutes")} for r in res],
                  "per_event": pe,
                  "created_utc": datetime.now(timezone.utc).isoformat()}
        with open(MODELS / f"{arch}_ensemble_val_results.json", "w") as f:
            json.dump(result, f, indent=2)
        hashes = {f"checkpoint_{arch}_s{s}": sha(MODELS / f"{arch}_s{s}_best.pt") for s in seeds_ok}
        hashes["dataset_v6_1.h5"] = sha(ROOT / "data/processed/dataset_v6_1.h5")
        hashes["dataset_v6_1_manifest.csv"] = sha(ROOT / "data/processed/dataset_v6_1_manifest.csv")
        hashes["normalization_params_v6_1.json"] = sha(ROOT / "data/processed/normalization_params_v6_1.json")
        hashes["explore_arch_train.py"] = sha(ROOT / "scripts/explore_arch_train.py")
        hashes["e28_train_grid.py"] = sha(ROOT / "scripts/e28_train_grid.py")
        hashes["explore_arch_ensemble.py"] = sha(ROOT / "scripts/explore_arch_ensemble.py")
        hashes[f"{arch}_ensemble_val_results.json"] = sha(MODELS / f"{arch}_ensemble_val_results.json")
        with open(MODELS / f"{arch}_freeze_hashes.json", "w") as f:
            json.dump({"status": "EXPLORATORY", "arch": arch, "arch_name": name,
                       "ensemble_seeds": seeds_ok,
                       "frozen_utc": datetime.now(timezone.utc).isoformat(),
                       "sha256": hashes}, f, indent=2)

    # ---- validation table ----
    L = [f"# EXPLORATORY architecture study -- validation table (12 evaluable v6.1 val events) "
         f"-- {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n",
         "All numbers exploratory. Selection grain identical to e28: event-level MEDIAN "
         "SS_fire,t6 on the 12 evaluable val events; checkpoint = best epoch by that median. "
         "No architecture is selected or dropped on this table; all go to blocks A + B.\n",
         "| id | encoder | params | runs | per-seed median SS (s0..s9) | seed-mean of medians | "
         "seed-mean of means | best epochs | ensemble median | ensemble mean +- std | wins vs B6 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows + [m2_row]:
        if r.get("n_runs", 0) == 0:
            L.append(f"| {r['arch']} | {r['name']} | -- | 0 ({r['n_failed']} failed) | -- | -- | -- | -- | -- | -- | -- |")
            continue
        L.append(f"| {r['arch']} | {r['name']} | {r['n_params']:,} | {r['n_runs']}"
                 f"{' (' + str(r['n_failed']) + ' failed)' if r['n_failed'] else ''} | "
                 + " ".join(f"{v:+.3f}" for v in r["seed_medians"]) + " | "
                 f"{np.mean(r['seed_medians']):+.4f} | {np.mean(r['seed_means']):+.4f} | "
                 f"{min(r['best_epochs'])}-{max(r['best_epochs'])} | {r['ens_median']:+.4f} | "
                 f"{r['ens_mean']:+.4f} +- {r['ens_std']:.4f} | {r['wins_vs_B6'][0]}/{r['wins_vs_B6'][1]} |")
    L.append(f"| B6 | gbdt_log_T1 (deterministic refit) | -- | 1 | -- | -- | -- | -- | {b6_median:+.4f} | "
             f"{np.mean(b6['SS_gbdt_log_T1']):+.4f} +- {np.std(b6['SS_gbdt_log_T1'], ddof=1):.4f} | -- |")
    L.append("\nM2 reference: e28 grid row (seeds 0-2) seed-mean of medians +0.3348; 10-seed "
             "log-mean ensemble val median +0.3131 (data/models/e28/m2_ensemble_val_results.json). "
             "The M2 row above lists all 10 members. B6 per-event values: "
             "data/processed/baselines_v6_1_b2_log_table.csv, column SS_gbdt_log_T1.\n")
    L.append("## Per-event ensemble SS_fire,t6 (val)\n")
    archs_done = [r["arch"] for r in rows if r.get("n_runs", 0)]
    L.append("| event | n_fire | " + " | ".join(archs_done) + " | M2 | B6 |")
    L.append("|---|---|" + "---|" * (len(archs_done) + 2))
    ens_pe = {a: json.load(open(MODELS / f"{a}_ensemble_val_results.json"))["per_event"] for a in archs_done}
    for ev in sorted(b6.index):
        cells = [f"{ens_pe[a][ev]['SS_fire_t6']:+.3f}" for a in archs_done]
        L.append(f"| {ev} | {int(b6.loc[ev, 'n_fire'])} | " + " | ".join(cells)
                 + f" | {m2_ens['per_event'][ev]['SS_fire_t6']:+.3f} | {b6.loc[ev, 'SS_gbdt_log_T1']:+.3f} |")
    txt = "\n".join(L) + "\n"
    (OUT / "val_table.md").write_text(txt, encoding="utf-8")
    print("\n" + txt)


if __name__ == "__main__":
    main()
