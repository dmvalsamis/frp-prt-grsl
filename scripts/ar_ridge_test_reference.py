"""CHECK D — Linear AR-Ridge baseline test reference for Section IV-A (<X>,<Y>).

The AR-Ridge baseline is the canonical recipe from scripts/09_baselines.py
(baseline_ar_ridge): six normalized FRP lags {1,2,3,6,12,24 h} at the last
lookback step, a per-horizon RidgeCV(alphas=[0.1,1,10,100], cv=5) FIT ON TRAIN,
predictions clipped at 0. It is the same baseline whose validation mean is the
Table II AR-Ridge value (+0.1537).

This is a fixed BASELINE evaluation on the test split (fit on train, no tuning
against test). Per the task it does NOT consume the count=3 confirmatory
test-access budget and selects nothing. SS uses the shared eval_harness, the
identical event-level path used for the PRT-Full Table I numbers.
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))  # repo src/ holds the shared/ package
from shared.eval_harness import aggregate_ss, compute_per_event_ss, make_persistence_pred  # noqa: E402
from shared.dataset_io import H5_PATH as H5, MANIFEST_PATH as MAN, NORM_PATH as NORM  # noqa: E402

AR_FEAT_IDX = [5, 6, 7, 8, 9, 10]   # frp_lag_1h,2h,3h,6h,12h,24h (dataset_io indices)
TEST_EVENTS = ["volos_2023", "varnavas_2024", "valmaior_2024", "corinthos_2024"]
PRT_FULL_MEAN = 0.2998              # locked Table I PRT-Full mean (do not alter)


def main():
    np.random.seed(0)
    with h5py.File(H5, "r") as f:
        meta = json.loads(f["metadata"][()])
        LB, H = int(meta["LOOKBACK"]), int(meta["HORIZON"])
        trX = f["train/inputs"][:]
        trY = f["train/targets"][:]
        teX = f["test/inputs"][:]        # baseline eval (non-budget); fit is on train only
        teY = f["test/targets"][:]
    LAST = LB - 1
    nm = json.load(open(NORM))
    frp_mean, frp_std = float(nm["frp_lag_1h"]["mean"]), float(nm["frp_lag_1h"]["std"])
    man = pd.read_csv(MAN)
    te_man = man[man["split"] == "test"].reset_index(drop=True)

    Xtr = trX[:, LAST, :][:, AR_FEAT_IDX]
    Xte = teX[:, LAST, :][:, AR_FEAT_IDX]
    preds = np.zeros((teX.shape[0], 12), dtype=np.float32)
    alphas = []
    for k in range(12):
        r = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5)
        r.fit(Xtr, trY[:, k])
        preds[:, k] = np.clip(r.predict(Xte), 0.0, None)
        alphas.append(float(r.alpha_))

    pers = make_persistence_pred(teX, frp_mean, frp_std, 5, H)
    pe = compute_per_event_ss(preds, teY, pers, te_man, TEST_EVENTS, horizon_1based=6)
    agg = aggregate_ss(pe)
    X = agg["mean"]
    Y = PRT_FULL_MEAN - X

    print("=" * 70)
    print("CHECK D — AR-Ridge test reference (baseline eval; non-budget)")
    print("=" * 70)
    print(f"  recipe: 6 FRP lags, per-horizon RidgeCV(alphas=[0.1,1,10,100],cv=5), fit on train")
    print(f"  per-horizon best alpha: {alphas}")
    print("  per test event AR-Ridge SS_fire_t6 (n_fire):")
    for ev in TEST_EVENTS:
        print(f"    {ev:16s}: SS={pe[ev]['ss']:+.4f}  n_fire={pe[ev]['n_fire']}")
    print(f"\n  AR-Ridge test mean SS_fire_t6  <X> = {X:+.4f}")
    print(f"  PRT-Full margin over AR-Ridge  <Y> = 0.2998 - {X:.4f} = {Y:+.4f}")
    print(f"  (AR-Ridge validation reference for context: +0.1537)")

    out = {
        "recipe": "6 FRP lags {1,2,3,6,12,24h} @ last step; per-horizon RidgeCV "
                  "alphas=[0.1,1,10,100] cv=5; fit on train; clip>=0",
        "per_horizon_best_alpha": alphas,
        "per_event": {ev: {"SS_fire_t6": pe[ev]["ss"], "n_fire": pe[ev]["n_fire"]} for ev in TEST_EVENTS},
        "X_ar_ridge_test_mean_SS_fire_t6": X,
        "Y_prt_full_margin": Y,
        "prt_full_mean_locked": PRT_FULL_MEAN,
        "ar_ridge_val_reference": 0.1537,
        "note": "Fixed baseline test evaluation; fit on train only; non-budget; selects nothing.",
    }
    outp = Path(__file__).parent / "ar_ridge_test_reference.json"
    json.dump(out, open(outp, "w"), indent=2)
    print(f"\n  wrote {outp}")


if __name__ == "__main__":
    main()
