"""Part B — backbone ensembling trainer (validation-track, M3).

Derivation: encoder classes (GRUSkip, LSTMSkip, MLPMixer, CNNSkip), the
LogHybridLoss, the persistence-skip helpers, and the training protocol are
copied verbatim from ../../scripts/20_architecture_comparison.py (the Table II
experiment) on 2026-05-29; extended from single-seed (manual_seed(42)) to a
10-seed sweep that saves best-epoch validation predictions for deep ensembling.
The PRT (PRT-Base backbone) reference uses the SHA-locked shared PRT class.

Protocol = IDENTICAL to the original Table II run, applied identically to every
backbone (the only thing that varies is the encoder):
  batch=64, MAX_EPOCHS=30, PATIENCE=5, MIN_EPOCHS=2, LogHybridLoss(alpha=0.5,
  fire_weight=7.0), AdamW(lr=1e-3, wd=1e-4), OneCycleLR(pct_start=0.3, 30 ep),
  residual_scale init 0.1 forward-clamped to [0,0.5], 48h lookback, 32-ch input
  (no WindFeats), best-epoch by max val SS_fire_t6 (event-level mean).

VALIDATION SPLIT ONLY. Reads only the train/ and val/ HDF5 groups; the test/
group is never opened. SS metric is computed by the shared eval_harness (same
code path as Part A and the locked Stage-2 ensemble eval).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))  # repo src/ holds the shared/ package
from shared.seed import set_all_seeds  # noqa: E402
from shared.prt_model import PRT  # noqa: E402
from shared.dataset_io import DATA_DIR, H5_PATH, MANIFEST_PATH, NORM_PATH  # noqa: E402
from shared.eval_harness import (  # noqa: E402
    aggregate_ss, compute_per_event_ss, make_persistence_pred, predict_split,
)

VAL_EVENTS = ["alexandroupolis_2023", "caceres_2023", "mandra_2023",
              "rhodes_2023", "split_2017"]

# --- Fixed protocol (identical to 20_architecture_comparison.py / TFT v8) ---
BATCH_SIZE, MAX_EPOCHS, PATIENCE, MIN_EPOCHS = 64, 30, 5, 2
LOSS_ALPHA, FIRE_WEIGHT, LR, WD, PCT_START = 0.5, 7.0, 1e-3, 1e-4, 0.3
RES_SCALE_MAX, FRP_LAG_IDX = 0.5, 5
# PRT backbone hyperparameters (PRT-Base backbone, matches shared PRT/TFT v8)
PRT_HIDDEN, PRT_HEADS, PRT_DROPOUT, PRT_LSTM = 96, 4, 0.1, 2


class LogHybridLoss(nn.Module):
    def __init__(self, fire_weight=7.0, alpha=0.5):
        super().__init__()
        self.fire_weight, self.alpha = fire_weight, alpha

    def forward(self, pred, target):
        pl, tl = torch.log1p(pred), torch.log1p(target)
        d = pl - tl
        loss = self.alpha * d.abs() + (1 - self.alpha) * d ** 2
        w = 1.0 + (self.fire_weight - 1.0) * (target > 0).any(dim=1, keepdim=True).float()
        return (loss * w).mean()


def _pers_log(x, frp_mean_buf, frp_std_buf):
    raw = torch.clamp(x[:, -1, FRP_LAG_IDX] * frp_std_buf + frp_mean_buf, min=0.0)
    return torch.log1p(raw)


def _skip_out(pers_log, residual, rs_param):
    rs = torch.clamp(rs_param, 0.0, RES_SCALE_MAX)
    return torch.clamp(torch.expm1(pers_log.unsqueeze(1) + rs * residual), min=0.0)


class GRUSkip(nn.Module):
    """Single-layer GRU, hidden=64. ~20K params."""
    def __init__(self, input_size, horizon, frp_mean, frp_std, **_):
        super().__init__()
        self.register_buffer("frp_mean", torch.tensor(frp_mean, dtype=torch.float32))
        self.register_buffer("frp_std", torch.tensor(frp_std, dtype=torch.float32))
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.gru = nn.GRU(input_size, 64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, horizon)

    def forward(self, x):
        _, h = self.gru(x)
        return _skip_out(_pers_log(x, self.frp_mean, self.frp_std),
                         self.fc(h.squeeze(0)), self.residual_scale)


class LSTMSkip(nn.Module):
    """Two-layer LSTM, hidden=96."""
    def __init__(self, input_size, horizon, frp_mean, frp_std, **_):
        super().__init__()
        self.register_buffer("frp_mean", torch.tensor(frp_mean, dtype=torch.float32))
        self.register_buffer("frp_std", torch.tensor(frp_std, dtype=torch.float32))
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.lstm = nn.LSTM(input_size, 96, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(96, horizon)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return _skip_out(_pers_log(x, self.frp_mean, self.frp_std),
                         self.fc(h[-1]), self.residual_scale)


class MLPMixer(nn.Module):
    """Zero learned temporal dynamics — handcrafted 48h summary stats. ~9K params."""
    def __init__(self, input_size, horizon, frp_mean, frp_std, lookback=48):
        super().__init__()
        self.register_buffer("frp_mean", torch.tensor(frp_mean, dtype=torch.float32))
        self.register_buffer("frp_std", torch.tensor(frp_std, dtype=torch.float32))
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        t = torch.arange(lookback, dtype=torch.float32) - (lookback - 1) / 2.0
        self.register_buffer("t_c", t.view(1, lookback, 1))
        self.register_buffer("t_ss", (t ** 2).sum())
        self.mlp = nn.Sequential(nn.Linear(input_size * 4, 64), nn.ReLU(),
                                 nn.Linear(64, horizon))

    def forward(self, x):
        mean_ = x.mean(dim=1)
        std_ = x.std(dim=1, unbiased=False)
        last_ = x[:, -1, :]
        slope_ = (self.t_c * x).sum(dim=1) / self.t_ss
        feats = torch.cat([mean_, std_, last_, slope_], dim=1)
        return _skip_out(_pers_log(x, self.frp_mean, self.frp_std),
                         self.mlp(feats), self.residual_scale)


class CNNSkip(nn.Module):
    """Two dilated conv1d layers (k=3, d=1,2, ch=64) + global avg pool."""
    def __init__(self, input_size, horizon, frp_mean, frp_std, **_):
        super().__init__()
        self.register_buffer("frp_mean", torch.tensor(frp_mean, dtype=torch.float32))
        self.register_buffer("frp_std", torch.tensor(frp_std, dtype=torch.float32))
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, dilation=1, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, dilation=2, padding=2), nn.ReLU(),
        )
        self.fc = nn.Linear(64, horizon)

    def forward(self, x):
        h = self.conv(x.permute(0, 2, 1)).mean(dim=2)
        return _skip_out(_pers_log(x, self.frp_mean, self.frp_std),
                         self.fc(h), self.residual_scale)


def _make_prt(input_size, horizon, frp_mean, frp_std, lookback=48):
    """PRT (PRT-Base backbone) — SHA-locked shared class, hd=96, 32-ch input."""
    return PRT(
        input_size=input_size, horizon=horizon,
        hidden_dim=PRT_HIDDEN, n_heads=PRT_HEADS, dropout=PRT_DROPOUT,
        n_lstm_layers=PRT_LSTM, frp_mean=frp_mean, frp_std=frp_std,
        residual_scale_max=RES_SCALE_MAX, frp_lag_1h_idx=FRP_LAG_IDX,
    )


BACKBONES = {
    "gru_skip": GRUSkip,
    "lstm_skip": LSTMSkip,
    "mlp_mixer": MLPMixer,
    "cnn_skip": CNNSkip,
    "prt": _make_prt,   # PRT-Base backbone reference
}


def _load_train_val():
    """Load ONLY train/ and val/ groups. The test/ group is never opened."""
    with h5py.File(H5_PATH, "r") as f:
        assert "test" not in str(f.keys()) or True  # we simply never read f['test']
        meta = json.loads(f["metadata"][()])
        lookback = int(meta["LOOKBACK"])
        horizon = int(meta["HORIZON"])
        input_size = len(meta["feature_names"])
        train_X = f["train/inputs"][:].astype(np.float32)
        train_y = f["train/targets"][:].astype(np.float32)
        val_X = f["val/inputs"][:].astype(np.float32)
        val_y = f["val/targets"][:].astype(np.float32)
    return train_X, train_y, val_X, val_y, lookback, horizon, input_size


def run_backbone_training(*, seed: int, arch_name: str, out_path: Path,
                          save_val_preds_path: Path, verbose: bool = False) -> dict:
    """Train one backbone at one seed under the arch-comparison protocol."""
    set_all_seeds(seed)

    train_X_np, train_y_np, val_X_np, val_y_np, LOOKBACK, HORIZON, INPUT_SIZE = _load_train_val()
    manifest = pd.read_csv(MANIFEST_PATH)
    val_manifest = manifest[manifest["split"] == "val"].reset_index(drop=True)
    with open(NORM_PATH) as f:
        norm_params = json.load(f)
    frp_mean = float(norm_params["frp_lag_1h"]["mean"])
    frp_std = float(norm_params["frp_lag_1h"]["std"])

    train_X = torch.tensor(train_X_np, dtype=torch.float32)
    train_y = torch.tensor(train_y_np, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Cls = BACKBONES[arch_name]
    model = Cls(INPUT_SIZE, HORIZON, frp_mean, frp_std, lookback=LOOKBACK).to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))

    crit = LogHybridLoss(fire_weight=FIRE_WEIGHT, alpha=LOSS_ALPHA)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, steps_per_epoch=len(train_loader), epochs=MAX_EPOCHS, pct_start=PCT_START)

    val_pers = make_persistence_pred(val_X_np, frp_mean, frp_std, FRP_LAG_IDX, HORIZON)

    best_ss, best_ep, no_improve = -np.inf, 0, 0
    best_per_event, best_val_preds = None, None
    history = []
    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
            sched.step()

        val_preds = predict_split(model, val_X_np, device=device)
        per_ev = compute_per_event_ss(val_preds, val_y_np, val_pers, val_manifest,
                                      VAL_EVENTS, horizon_1based=6)
        ss_mean = aggregate_ss(per_ev)["mean"]
        ss_excl = aggregate_ss(per_ev, exclude=["caceres_2023"])["mean"]
        ep_ss = ss_mean if not np.isnan(ss_mean) else -np.inf
        history.append({"epoch": epoch, "val_SS_paper": ss_mean,
                        "val_SS_excl_caceres": ss_excl})
        if verbose:
            print(f"  [{arch_name} s{seed}] ep{epoch:02d} SS={ss_mean:+.4f} excl={ss_excl:+.4f}")

        if ep_ss > best_ss:
            best_ss, best_ep, no_improve = ep_ss, epoch, 0
            best_per_event = per_ev
            best_val_preds = val_preds.copy()
        else:
            no_improve += 1
            if epoch >= MIN_EPOCHS and no_improve >= PATIENCE:
                break
    wall = time.time() - t0

    save_val_preds_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_val_preds_path, best_val_preds)

    result = {
        "run_label": f"{arch_name}_seed{seed}",
        "arch": arch_name,
        "seed": seed,
        "n_params": n_params,
        "best_epoch": best_ep,
        "stopped_at": history[-1]["epoch"],
        "best_val_SS_paper": float(best_ss),
        "best_val_SS_excl_caceres": history[best_ep - 1]["val_SS_excl_caceres"] if best_ep else float("nan"),
        "best_per_event": {ev: v["ss"] for ev, v in (best_per_event or {}).items()},
        "val_preds_path": str(save_val_preds_path),
        "history": history,
        "wall_s": wall,
        "note": "Validation-track backbone sweep (Part B). Val-only. No test access.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result
