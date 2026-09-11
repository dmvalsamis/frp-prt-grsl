"""
e210_ensemble.py — Task E2.10: M2 = 10-seed deep ensemble of the E2.9 winner
============================================================================
- Reads the winner cell from data/models/e28/e29_selection.json.
- Trains seeds 3..9 of that cell via the SAME e28_train_grid machinery
  (identical hyperparameters; seeds 0..2 reuse the grid checkpoints).
- Ensemble = LOG-MEAN rule (pre-declared): expm1(mean_over_seeds(log1p(pred))).
- Evaluates the ensemble on the v6.1 val split (AGG-1 median primary,
  mean ± std secondary, per-event table) side by side with B6 (+0.1703)
  and naive persistence.
- Freezes: SHA-256 table over the complete deployed configuration
  (10 checkpoints, dataset, norm params, manifest, training script,
  selection record, this script), mirroring the S2.1 pattern.

Outputs:
  data/models/e28/m2_ensemble_val_results.json
  data/models/e28/m2_freeze_hashes.json
"""

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path("e:/GRSL_Wildfire/FRP_Methodology")
E28 = ROOT / "data/models/e28"

sel = json.load(open(E28 / "e29_selection.json"))
WINNER = sel["winner"]                      # e.g. "T0_robust"
VARIANT, SCHEME = WINNER.split("_")
ENSEMBLE_SEEDS = list(range(10))

# import the grid module to reuse run_one and evaluation context
# (exec_module sets __name__ = "e28", so the grid's main() guard does not fire)
spec = importlib.util.spec_from_file_location("e28", ROOT / "scripts/e28_train_grid.py")
e28 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e28)

print(f"E2.10: winner cell = {WINNER}; ensemble seeds {ENSEMBLE_SEEDS}")

# train any missing seeds (0..2 exist from the grid; 3..9 new)
for s in ENSEMBLE_SEEDS:
    e28.run_one(VARIANT, SCHEME, s)

# ---- log-mean ensemble over the 10 members' saved val predictions ----
member_preds = []
for s in ENSEMBLE_SEEDS:
    p = E28 / f"{VARIANT}_{SCHEME}_s{s}_val_preds.npy"
    member_preds.append(np.load(p))
stack = np.stack(member_preds, axis=0)
ens = np.expm1(np.mean(np.log1p(np.clip(stack, 0, None)), axis=0)).astype(np.float32)

med, ss_list = e28.val_median_ss(ens)
pe = e28.per_event_val(ens)

base = json.load(open(ROOT / "data/processed/baselines_v6_1_full_val_results.json"))
b6_key = f"ar_ridge_{VARIANT}"   # per-variant AR for context...
b6_named = base["ar_ridge_T0"]   # the NAMED strongest test baseline (DEC-5)

print(f"\nM2 ensemble ({WINNER}, 10 seeds, log-mean) on v6.1 val:")
print(f"  median SS_fire_t6 = {med:+.4f}  (mean {np.mean(ss_list):+.4f} "
      f"± {np.std(ss_list, ddof=1):.4f}, n={len(ss_list)})")
print(f"  vs B6 AR-Ridge-T0 median +0.1703 (named strongest baseline)")
print(f"\n{'event':34s} {'n_fire':>6s} {'M2 SS':>8s} {'B6 SS':>8s} {'M2 beats B6 (MAE)':>18s}")
wins, n = 0, 0
for ev in sorted(pe):
    d = pe[ev]
    if d["n_fire"] == 0:
        print(f"{ev:34s} {0:6d} {'—':>8s} {'—':>8s} {'—':>18s}")
        continue
    b = b6_named["per_event"][ev]
    beat = d["MAE_fire_t6"] < b["MAE_fire"]["+6"]
    n += 1; wins += int(beat)
    print(f"{ev:34s} {d['n_fire']:6d} {d['SS_fire_t6']:+8.3f} "
          f"{b.get('SS_fire_t6', float('nan')):+8.3f} {str(beat):>18s}")
print(f"\nval paired wins vs B6: {wins}/{n}")

result = {"winner_cell": WINNER, "ensemble_seeds": ENSEMBLE_SEEDS,
          "combination_rule": "log-mean: expm1(mean(log1p(preds)))",
          "val_median_ss_t6": med,
          "val_mean_ss_t6": float(np.mean(ss_list)),
          "val_std_ss_t6": float(np.std(ss_list, ddof=1)),
          "n_events_evaluable": len(ss_list),
          "val_wins_vs_B6": [wins, n],
          "per_event": pe,
          "created_utc": datetime.now(timezone.utc).isoformat()}
with open(E28 / "m2_ensemble_val_results.json", "w") as f:
    json.dump(result, f, indent=2)

# ---- freeze hash table ----
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

hashes = {}
for s in ENSEMBLE_SEEDS:
    hashes[f"checkpoint_{WINNER}_s{s}"] = sha(E28 / f"{VARIANT}_{SCHEME}_s{s}_best.pt")
hashes["dataset_v6_1.h5"] = sha(ROOT / "data/processed/dataset_v6_1.h5")
hashes["normalization_params_v6_1.json"] = sha(ROOT / "data/processed/normalization_params_v6_1.json")
hashes["dataset_v6_1_manifest.csv"] = sha(ROOT / "data/processed/dataset_v6_1_manifest.csv")
hashes["e28_train_grid.py"] = sha(ROOT / "scripts/e28_train_grid.py")
hashes["e29_selection.json"] = sha(E28 / "e29_selection.json")
hashes["e210_ensemble.py"] = sha(ROOT / "scripts/e210_ensemble.py")
hashes["m2_ensemble_val_results.json"] = sha(E28 / "m2_ensemble_val_results.json")
hashes["baselines_v6_1_full_val_results.json"] = sha(ROOT / "data/processed/baselines_v6_1_full_val_results.json")
with open(E28 / "m2_freeze_hashes.json", "w") as f:
    json.dump({"winner_cell": WINNER,
               "frozen_utc": datetime.now(timezone.utc).isoformat(),
               "sha256": hashes}, f, indent=2)
print(f"\nFreeze hash table -> {E28 / 'm2_freeze_hashes.json'}")
for k, v in hashes.items():
    print(f"  {k}: {v[:16]}…")
