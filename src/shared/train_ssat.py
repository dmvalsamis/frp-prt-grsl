"""Reusable SSAT training driver.

run_ssat_training(...) runs one (seed, w_log, ...) configuration on the
PRT model with the SSAT loss, evaluates on val each epoch, and returns
the per-epoch history. Test-split is never touched.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from .seed import set_all_seeds
from .dataset_io import (
    FRP_LAG_1H_IDX,
    VAL_EVENTS,
    get_frp_norm,
    load_manifest,
    load_metadata,
    load_norm_params,
    load_split,
)
from .prt_model import PRT
from .prt_arg_model import PRT_ARG
from .persistence_mae_table import compute_persistence_mae_table
from .event_balanced_sampler import EventBalancedBatchSampler
from .ssat_loss import SSATLoss
from .eval_harness import (
    aggregate_ss,
    compute_per_event_ss,
    make_persistence_pred,
    predict_split,
)
from .wind_features import build_wind_features


def _epoch_log(epoch, avg_ssat, avg_log, agg_all, agg_exc, rs_now, per_ev):
    parts = " ".join(
        f"{ev[:6]}={(v['ss'] if not np.isnan(v['ss']) else 0):+.2f}"
        for ev, v in per_ev.items()
    )
    print(
        f"  ep{epoch:02d}  ssat={avg_ssat:.3f}  log={avg_log:.3f}  "
        f"valSS={agg_all['mean']:+.3f}  valSS_excC={agg_exc['mean']:+.3f}  "
        f"rs={rs_now:.3f}  [{parts}]"
    )


def run_ssat_training(
    *,
    seed: int,
    out_path: Path,
    run_label: str,
    hidden_dim: int = 96,
    n_heads: int = 4,
    dropout: float = 0.1,
    n_lstm_layers: int = 2,
    residual_scale_max: float = 0.5,
    events_per_batch: int = 8,
    samples_per_event: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 15,
    w_ssat: float = 1.0,
    w_log: float = 0.2,
    fire_weight: float = 7.0,
    log_alpha: float = 0.5,
    verbose: bool = True,
    save_val_preds_path: Path | None = None,
    use_wind_features: bool = False,
    use_arg: bool = False,
    arg_shrinkage_lambda: float = 0.1,
    arg_gate_feature_indices: list[int] | None = None,
    arg_gate_init_bias: float = 1.0,
) -> dict:
    """Train one PRT+SSAT configuration. Returns the result dict (also saved to out_path)."""
    set_all_seeds(seed)

    meta = load_metadata()
    LOOKBACK = meta["LOOKBACK"]
    HORIZON = meta["HORIZON"]
    INPUT_SIZE = len(meta["feature_names"])

    train_inputs, train_targets = load_split("train")
    val_inputs, val_targets = load_split("val")
    train_manifest = load_manifest("train")
    val_manifest = load_manifest("val")
    frp_mean, frp_std = get_frp_norm()

    wind_stats = None
    if use_wind_features:
        norm_params = load_norm_params()
        train_inputs, wind_stats = build_wind_features(train_inputs, norm_params, None)
        val_inputs, _ = build_wind_features(val_inputs, norm_params, wind_stats)
        INPUT_SIZE = train_inputs.shape[2]

    pers_table_np = compute_persistence_mae_table(
        train_inputs, train_targets, train_manifest, horizon=HORIZON
    )
    pers_table_t = {ev: torch.tensor(v, dtype=torch.float32) for ev, v in pers_table_np.items()}

    sampler = EventBalancedBatchSampler(
        manifest=train_manifest,
        targets=train_targets,
        events_per_batch=events_per_batch,
        samples_per_event=samples_per_event,
        seed=seed,
    )

    train_X_t = torch.tensor(train_inputs, dtype=torch.float32)
    train_y_t = torch.tensor(train_targets, dtype=torch.float32)
    train_event_arr = train_manifest["event_name"].values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_arg:
        model = PRT_ARG(
            input_size=INPUT_SIZE,
            horizon=HORIZON,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
            n_lstm_layers=n_lstm_layers,
            frp_mean=frp_mean,
            frp_std=frp_std,
            residual_scale_max=residual_scale_max,
            frp_lag_1h_idx=FRP_LAG_1H_IDX,
            gate_feature_indices=arg_gate_feature_indices,
            gate_init_bias=arg_gate_init_bias,
        ).to(device)
    else:
        model = PRT(
            input_size=INPUT_SIZE,
            horizon=HORIZON,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
            n_lstm_layers=n_lstm_layers,
            frp_mean=frp_mean,
            frp_std=frp_std,
            residual_scale_max=residual_scale_max,
            frp_lag_1h_idx=FRP_LAG_1H_IDX,
        ).to(device)

    criterion = SSATLoss(
        persistence_mae=pers_table_t,
        w_ssat=w_ssat,
        w_log=w_log,
        fire_weight=fire_weight,
        log_alpha=log_alpha,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(sampler), epochs=max_epochs, pct_start=0.3,
    )

    val_pers_pred = make_persistence_pred(val_inputs, frp_mean, frp_std, FRP_LAG_1H_IDX, HORIZON)

    history = []
    best_val_ss = -float("inf")
    best_epoch = 0
    best_per_event = None
    best_val_preds: np.ndarray | None = None
    best_gate_stats: dict | None = None

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        ssat_sum = 0.0
        log_sum = 0.0
        shrink_sum = 0.0
        gate_mean_sum = 0.0
        n_batches = 0
        for batch_idx in list(iter(sampler)):
            idx_t = torch.tensor(batch_idx, dtype=torch.long)
            xb = train_X_t.index_select(0, idx_t).to(device)
            yb = train_y_t.index_select(0, idx_t).to(device)
            ev_names = [train_event_arr[i] for i in batch_idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss, info = criterion(pred, yb, ev_names)
            if use_arg:
                r_windy = model._last_r_windy
                g = model._last_gate
                shrink = r_windy.pow(2).mean()
                loss = loss + arg_shrinkage_lambda * shrink
                shrink_sum += float(shrink.detach().cpu())
                gate_mean_sum += float(g.mean().detach().cpu())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            ssat_sum += info["ssat_term"]
            log_sum += info["log_anchor"]
            n_batches += 1
        avg_ssat = ssat_sum / max(1, n_batches)
        avg_log = log_sum / max(1, n_batches)
        avg_shrink = shrink_sum / max(1, n_batches) if use_arg else 0.0
        avg_gate = gate_mean_sum / max(1, n_batches) if use_arg else 1.0

        val_preds = predict_split(model, val_inputs, device=device)
        per_ev = compute_per_event_ss(
            val_preds, val_targets, val_pers_pred, val_manifest, VAL_EVENTS, horizon_1based=6,
        )
        agg_all = aggregate_ss(per_ev)
        agg_exc = aggregate_ss(per_ev, exclude=["caceres_2023"])
        rs_now = float(torch.clamp(model.residual_scale, 0.0, residual_scale_max).item())

        if verbose:
            _epoch_log(epoch, avg_ssat, avg_log, agg_all, agg_exc, rs_now, per_ev)

        history.append({
            "epoch": epoch,
            "train_ssat_term": avg_ssat,
            "train_log_anchor": avg_log,
            "train_shrinkage": avg_shrink,
            "train_gate_mean": avg_gate,
            "val_SS_paper": agg_all["mean"],
            "val_SS_excl_caceres": agg_exc["mean"],
            "residual_scale": rs_now,
            "per_event": {ev: v["ss"] for ev, v in per_ev.items()},
        })

        if not np.isnan(agg_all["mean"]) and agg_all["mean"] > best_val_ss:
            best_val_ss = agg_all["mean"]
            best_epoch = epoch
            best_per_event = per_ev
            if save_val_preds_path is not None:
                best_val_preds = val_preds.copy()

    if save_val_preds_path is not None and best_val_preds is not None:
        save_val_preds_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_val_preds_path, best_val_preds)

    wall = time.time() - t0

    result = {
        "run_label": run_label,
        "seed": seed,
        "config": {
            "hidden_dim": hidden_dim, "n_heads": n_heads, "dropout": dropout,
            "n_lstm_layers": n_lstm_layers, "residual_scale_max": residual_scale_max,
            "events_per_batch": events_per_batch, "samples_per_event": samples_per_event,
            "lr": lr, "weight_decay": weight_decay, "max_epochs": max_epochs,
            "w_ssat": w_ssat, "w_log": w_log,
            "fire_weight": fire_weight, "log_alpha": log_alpha,
            "use_wind_features": use_wind_features, "input_size": int(INPUT_SIZE),
            "wind_feat_stats": wind_stats,
            "use_arg": use_arg,
            "arg_shrinkage_lambda": arg_shrinkage_lambda if use_arg else None,
            "arg_gate_init_bias": arg_gate_init_bias if use_arg else None,
            "arg_gate_feature_indices": (
                list(arg_gate_feature_indices) if arg_gate_feature_indices is not None
                else ([27, 32, 34, 35] if use_arg else None)
            ),
        },
        "best_epoch": best_epoch,
        "best_val_SS_paper": best_val_ss,
        "best_per_event": {ev: v["ss"] for ev, v in (best_per_event or {}).items()},
        "final_val_SS_paper": history[-1]["val_SS_paper"] if history else float("nan"),
        "history": history,
        "wall_s": wall,
        "note": "Pre-Stage-1 exploratory diagnostic. Val-only. Not a Stage 2 result.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    if verbose:
        print(f"  saved: {out_path}")
    return result
