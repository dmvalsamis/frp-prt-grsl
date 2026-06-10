"""Stage 2 (Row 2) test-set deep-ensemble evaluation — ONE pass.

Loads the 10 per-seed checkpoints produced by train_stage2_row2.py, runs each
on the held-out 4-event test split, then constructs the deep ensemble using
the locked combination rule from STAGE2_DESIGN_LOCKED.md §2.7:

    log_mean = mean over seeds of log1p(pred_seed)
    pred_ens = clip(expm1(log_mean), min=0)

Reports per-event SS_fire_t6 (4 values) and unweighted mean across events.
Bootstrap CI: event-level non-parametric, n_boot=10000, rng default_rng(42).
Also reports pre-declared sensitivity (median + trimmed-mean ensemble) and
calibration sanity (Q25-Q75 inter-seed coverage on fire-active t+6).

This script touches the test split exactly once. The seal is bypassed with
unsealed=True; this is one of the three budgeted, pre-declared test accesses
(see configs/test_access_protocol.md).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))  # repo src/ holds the shared/ package
from shared.dataset_io import (  # noqa: E402
    FRP_LAG_1H_IDX, get_frp_norm, load_manifest, load_metadata,
    load_norm_params, load_split,
)
from shared.eval_harness import (  # noqa: E402
    compute_per_event_ss, make_persistence_pred, predict_split,
)
from shared.prt_model import PRT  # noqa: E402
from shared.wind_features import build_wind_features  # noqa: E402

CKPT_DIR = HERE.parent / "models" / "prt_full_ensemble"
RESULTS_DIR = HERE.parent / "results"
TEST_EVENTS = ["corinthos_2024", "valmaior_2024", "varnavas_2024", "volos_2023"]
HIDDEN_DIM, N_HEADS, DROPOUT, N_LSTM, RS_MAX = 96, 4, 0.1, 2, 0.5


def _build_model(input_size: int, horizon: int, frp_mean: float, frp_std: float) -> PRT:
    return PRT(
        input_size=input_size, horizon=horizon,
        hidden_dim=HIDDEN_DIM, n_heads=N_HEADS, dropout=DROPOUT,
        n_lstm_layers=N_LSTM, frp_mean=frp_mean, frp_std=frp_std,
        residual_scale_max=RS_MAX, frp_lag_1h_idx=FRP_LAG_1H_IDX,
    )


def main():
    meta = load_metadata()
    HORIZON = int(meta["HORIZON"])
    frp_mean, frp_std = get_frp_norm()

    # Test split — SINGLE access, unsealed per Stage 2 §2.8
    test_inputs_raw, test_targets = load_split("test", unsealed=True)
    test_manifest = load_manifest("test")

    # Locked WindFeats stats (from any seed ckpt; verified consistent across seeds)
    ckpt_paths = sorted(CKPT_DIR.glob("seed_*.pt"))
    if len(ckpt_paths) != 10:
        raise RuntimeError(f"Expected 10 checkpoints, found {len(ckpt_paths)}")
    ref_ckpt = torch.load(ckpt_paths[0], map_location="cpu", weights_only=False)
    wind_stats = ref_ckpt["wind_feat_stats"]
    norm_params = load_norm_params()
    test_inputs, _ = build_wind_features(test_inputs_raw, norm_params, wind_stats)
    INPUT_SIZE = int(test_inputs.shape[2])
    assert INPUT_SIZE == ref_ckpt["input_size"], "INPUT_SIZE mismatch across ckpts"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_seed_preds = []  # raw MW (N, H), shape per seed
    per_seed_val_SS = []
    for cp in ckpt_paths:
        ck = torch.load(cp, map_location=device, weights_only=False)
        model = _build_model(INPUT_SIZE, HORIZON, frp_mean, frp_std).to(device)
        model.load_state_dict(ck["state_dict"])
        preds = predict_split(model, test_inputs, device=device)
        per_seed_preds.append(preds.astype(np.float32))
        per_seed_val_SS.append(float(ck["best_val_SS_paper"]))
    per_seed_preds = np.stack(per_seed_preds, axis=0)  # (S, N, H)

    # Log-mean ensemble (headline)
    log_mean = np.log1p(np.clip(per_seed_preds, 0.0, None)).mean(axis=0)
    ens_pred = np.clip(np.expm1(log_mean), 0.0, None)
    # Sensitivity: median, trimmed-mean (drop top+bottom seed per element)
    sorted_log = np.sort(np.log1p(np.clip(per_seed_preds, 0.0, None)), axis=0)
    trimmed_log_mean = sorted_log[1:-1].mean(axis=0)
    median_log = np.median(np.log1p(np.clip(per_seed_preds, 0.0, None)), axis=0)
    ens_pred_trim = np.clip(np.expm1(trimmed_log_mean), 0.0, None)
    ens_pred_med = np.clip(np.expm1(median_log), 0.0, None)

    # Persistence baseline
    test_pers = make_persistence_pred(test_inputs, frp_mean, frp_std, FRP_LAG_1H_IDX, HORIZON)

    def per_event(pred):
        return compute_per_event_ss(pred, test_targets, test_pers,
                                    test_manifest, TEST_EVENTS, horizon_1based=6)

    headline = per_event(ens_pred)
    sensitivity_med = per_event(ens_pred_med)
    sensitivity_trim = per_event(ens_pred_trim)
    per_seed_ss = []
    for s in range(per_seed_preds.shape[0]):
        per_seed_ss.append({ev: v["ss"] for ev, v in
                            per_event(per_seed_preds[s]).items()})

    # Headline aggregate: unweighted mean across 4 events
    ev_vals = np.array([headline[ev]["ss"] for ev in TEST_EVENTS])
    headline_mean = float(ev_vals.mean())

    # Event-level non-parametric bootstrap, n_boot=10000, rng default_rng(42)
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(10000):
        idx = rng.integers(0, 4, size=4)
        boot.append(ev_vals[idx].mean())
    boot = np.array(boot)
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    # Calibration: Q25-Q75 coverage on fire-active t+6
    h6 = 6 - 1
    fire = (test_targets > 0).any(axis=1)
    q25 = np.percentile(per_seed_preds[:, :, h6], 25, axis=0)
    q75 = np.percentile(per_seed_preds[:, :, h6], 75, axis=0)
    truth_t6 = test_targets[:, h6]
    inside = ((truth_t6[fire] >= q25[fire]) & (truth_t6[fire] <= q75[fire]))
    cov = float(inside.mean()) if fire.any() else float("nan")

    # Cross-seed mean ± SD on TEST (descriptive)
    seed_test_means = np.array([np.mean([per_seed_ss[s][ev] for ev in TEST_EVENTS])
                                for s in range(len(per_seed_ss))])

    out = {
        "config_label": "row2_ssat_windfeats_stage2",
        "n_seeds": int(per_seed_preds.shape[0]),
        "test_events": TEST_EVENTS,
        "ensemble_rule": "log-mean across seeds, expm1, clip>=0",
        "headline_test_SS_fire_t6": headline_mean,
        "bootstrap_95_CI": [ci_lo, ci_hi],
        "ci_excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
        "per_event_SS": {ev: float(headline[ev]["ss"]) for ev in TEST_EVENTS},
        "per_event_n_fire": {ev: int(headline[ev]["n_fire"]) for ev in TEST_EVENTS},
        "sensitivity_median_SS": float(np.mean([sensitivity_med[ev]["ss"] for ev in TEST_EVENTS])),
        "sensitivity_trimmed_mean_SS": float(np.mean([sensitivity_trim[ev]["ss"] for ev in TEST_EVENTS])),
        "calibration_Q25_Q75_coverage_t6": cov,
        "cross_seed_val_SS_mean": float(np.mean(per_seed_val_SS)),
        "cross_seed_val_SS_std": float(np.std(per_seed_val_SS, ddof=1)),
        "cross_seed_test_SS_mean": float(seed_test_means.mean()),
        "cross_seed_test_SS_std": float(seed_test_means.std(ddof=1)),
        "per_seed_test_SS_per_event": per_seed_ss,
        "per_seed_test_SS_mean": [float(x) for x in seed_test_means],
        "bootstrap_n": 10000,
        "bootstrap_rng_seed": 42,
        "test_set_accesses": 1,
        "note": "Stage 2 single test-set access. Locked design: STAGE2_DESIGN_LOCKED.md.",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "stage2_test_ensemble.json", "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 70)
    print("Stage 2 (Row 2) deep-ensemble test result")
    print("=" * 70)
    print(f"  Headline test SS_fire_t6 = {headline_mean:+.4f}")
    print(f"  Bootstrap 95% CI         = [{ci_lo:+.4f}, {ci_hi:+.4f}]  excludes 0: {out['ci_excludes_zero']}")
    print(f"  Per-event SS (corinthos / valmaior / varnavas / volos):")
    for ev in TEST_EVENTS:
        print(f"    {ev:18s}: {headline[ev]['ss']:+.4f}  (n_fire={headline[ev]['n_fire']})")
    print(f"  Sensitivity (median ensemble)        : {out['sensitivity_median_SS']:+.4f}")
    print(f"  Sensitivity (trimmed-mean ensemble)  : {out['sensitivity_trimmed_mean_SS']:+.4f}")
    print(f"  Calibration Q25-Q75 coverage         : {cov:.3f}  (target ~0.5)")
    print(f"  Cross-seed val  SS mean +/- SD       : {out['cross_seed_val_SS_mean']:+.4f} +/- {out['cross_seed_val_SS_std']:.4f}")
    print(f"  Cross-seed test SS mean +/- SD       : {out['cross_seed_test_SS_mean']:+.4f} +/- {out['cross_seed_test_SS_std']:.4f}")
    print(f"  Saved: {RESULTS_DIR / 'stage2_test_ensemble.json'}")


if __name__ == "__main__":
    main()
