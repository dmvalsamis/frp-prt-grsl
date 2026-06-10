"""Stage 2 (Row 2) trainer — PRT + SSAT + WindFeats, 10 seeds.

Mirrors the val-only sweep in redesign/scripts/06_seed_sweep_windfeats.py with
two additions required for ensemble evaluation:
  - saves best-val-SS checkpoint (state_dict) per seed
  - saves val predictions at the best epoch (for diagnostic only)

Hyperparameters and shared modules are pinned by SHA-256 in
redesign/stage2/STAGE2_DESIGN_LOCKED.md (2026-05-19). Any change to the
spec below voids Stage 2; see that document's §5 lock policy.

Test split is NOT touched here. Only val is used for best-epoch selection.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))  # repo src/ holds the shared/ package
from shared.dataset_io import (  # noqa: E402
    FRP_LAG_1H_IDX, VAL_EVENTS, get_frp_norm, load_manifest,
    load_metadata, load_norm_params, load_split,
)
from shared.eval_harness import (  # noqa: E402
    aggregate_ss, compute_per_event_ss, make_persistence_pred, predict_split,
)
from shared.event_balanced_sampler import EventBalancedBatchSampler  # noqa: E402
from shared.persistence_mae_table import compute_persistence_mae_table  # noqa: E402
from shared.prt_model import PRT  # noqa: E402
from shared.seed import set_all_seeds  # noqa: E402
from shared.ssat_loss import SSATLoss  # noqa: E402
from shared.wind_features import build_wind_features  # noqa: E402

CKPT_DIR = HERE.parent / "models" / "prt_full_ensemble"
RESULTS_DIR = HERE.parent / "results" / "per_seed"
SEEDS = list(range(10))

# --- Locked Row 2 hyperparameters (must match STAGE2_DESIGN_LOCKED.md §2) ---
HIDDEN_DIM, N_HEADS, DROPOUT, N_LSTM = 96, 4, 0.1, 2
RS_MAX = 0.5
EVENTS_PER_BATCH, SAMPLES_PER_EVENT = 8, 8
LR, WD = 1e-3, 1e-4
MAX_EPOCHS = 15
W_SSAT, W_LOG = 1.0, 0.02
FIRE_WEIGHT, LOG_ALPHA = 7.0, 0.5
HORIZON_WEIGHTS = {1: 0.10, 3: 0.20, 6: 0.50, 12: 0.20}


def train_one_seed(seed: int) -> dict:
    set_all_seeds(seed)
    meta = load_metadata()
    HORIZON = int(meta["HORIZON"])
    train_inputs, train_targets = load_split("train")
    val_inputs, val_targets = load_split("val")
    train_manifest = load_manifest("train")
    val_manifest = load_manifest("val")
    frp_mean, frp_std = get_frp_norm()

    # WindFeats (Tier 1)
    norm_params = load_norm_params()
    train_inputs, wind_stats = build_wind_features(train_inputs, norm_params, None)
    val_inputs, _ = build_wind_features(val_inputs, norm_params, wind_stats)
    INPUT_SIZE = int(train_inputs.shape[2])

    # Persistence MAE table for SSAT
    pers_tbl_np = compute_persistence_mae_table(
        train_inputs, train_targets, train_manifest, horizon=HORIZON
    )
    pers_tbl_t = {ev: torch.tensor(v, dtype=torch.float32) for ev, v in pers_tbl_np.items()}

    sampler = EventBalancedBatchSampler(
        manifest=train_manifest, targets=train_targets,
        events_per_batch=EVENTS_PER_BATCH, samples_per_event=SAMPLES_PER_EVENT, seed=seed,
    )

    train_X = torch.tensor(train_inputs, dtype=torch.float32)
    train_y = torch.tensor(train_targets, dtype=torch.float32)
    train_ev = train_manifest["event_name"].values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PRT(
        input_size=INPUT_SIZE, horizon=HORIZON,
        hidden_dim=HIDDEN_DIM, n_heads=N_HEADS, dropout=DROPOUT,
        n_lstm_layers=N_LSTM, frp_mean=frp_mean, frp_std=frp_std,
        residual_scale_max=RS_MAX, frp_lag_1h_idx=FRP_LAG_1H_IDX,
    ).to(device)

    criterion = SSATLoss(
        persistence_mae=pers_tbl_t, horizon_weights=HORIZON_WEIGHTS,
        w_ssat=W_SSAT, w_log=W_LOG, fire_weight=FIRE_WEIGHT, log_alpha=LOG_ALPHA,
    )
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=LR, steps_per_epoch=len(sampler), epochs=MAX_EPOCHS, pct_start=0.3,
    )
    val_pers = make_persistence_pred(val_inputs, frp_mean, frp_std, FRP_LAG_1H_IDX, HORIZON)

    best = {"val_SS": -float("inf"), "epoch": 0, "per_event": None, "state": None}
    history = []
    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch_idx in list(iter(sampler)):
            idx_t = torch.tensor(batch_idx, dtype=torch.long)
            xb = train_X.index_select(0, idx_t).to(device)
            yb = train_y.index_select(0, idx_t).to(device)
            ev_names = [train_ev[i] for i in batch_idx]
            optim.zero_grad()
            loss, _ = criterion(model(xb), yb, ev_names)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step(); sched.step()

        val_preds = predict_split(model, val_inputs, device=device)
        per_ev = compute_per_event_ss(val_preds, val_targets, val_pers,
                                      val_manifest, VAL_EVENTS, horizon_1based=6)
        ss_paper = aggregate_ss(per_ev)["mean"]
        history.append({"epoch": epoch, "val_SS_paper": ss_paper,
                        "per_event": {ev: v["ss"] for ev, v in per_ev.items()}})
        if not np.isnan(ss_paper) and ss_paper > best["val_SS"]:
            best.update(val_SS=ss_paper, epoch=epoch, per_event=per_ev,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    wall = time.time() - t0

    # Save checkpoint
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"seed_{seed:03d}.pt"
    torch.save({
        "state_dict": best["state"], "seed": seed,
        "best_epoch": best["epoch"], "best_val_SS_paper": best["val_SS"],
        "input_size": INPUT_SIZE, "wind_feat_stats": wind_stats,
    }, ckpt_path)

    return {
        "seed": seed, "best_epoch": best["epoch"],
        "best_val_SS_paper": best["val_SS"],
        "best_per_event": {ev: v["ss"] for ev, v in best["per_event"].items()},
        "wall_s": wall, "history": history, "input_size": INPUT_SIZE,
        "wind_feat_stats": wind_stats, "ckpt_path": str(ckpt_path),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Stage 2 (Row 2): PRT + SSAT + WindFeats — 10-seed training")
    print("Locked hyperparams per STAGE2_DESIGN_LOCKED.md §2")
    print("=" * 70)
    per_seed = []
    for seed in SEEDS:
        out_p = RESULTS_DIR / f"per_seed_{seed:03d}.json"
        if out_p.exists():
            with open(out_p) as f: r = json.load(f)
            print(f"  seed={seed}: cached best_SS={r['best_val_SS_paper']:+.4f} ep{r['best_epoch']}")
            per_seed.append(r); continue
        print(f"  seed={seed}: training...")
        r = train_one_seed(seed)
        with open(out_p, "w") as f: json.dump(r, f, indent=2)
        print(f"    done. best_SS={r['best_val_SS_paper']:+.4f} ep{r['best_epoch']} wall={r['wall_s']:.1f}s")
        per_seed.append(r)

    vals = np.array([r["best_val_SS_paper"] for r in per_seed])
    print(f"\n  10-seed val SS_fire_t6: mean={vals.mean():+.4f}  std={vals.std(ddof=1):.4f}")
    summary = {
        "config_label": "row2_ssat_windfeats",
        "n_seeds": len(SEEDS),
        "val_SS_mean": float(vals.mean()),
        "val_SS_std": float(vals.std(ddof=1)),
        "val_SS_values": [float(x) for x in vals],
        "per_seed": [{k: r[k] for k in ["seed", "best_epoch", "best_val_SS_paper",
                                         "best_per_event", "ckpt_path"]} for r in per_seed],
    }
    with open(RESULTS_DIR / "stage2_per_seed_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved: {RESULTS_DIR / 'stage2_per_seed_summary.json'}")


if __name__ == "__main__":
    main()
