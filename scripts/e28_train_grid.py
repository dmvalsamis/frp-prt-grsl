"""
e28_train_grid.py — Task E2.8: 4-cell × 3-seed PRT-Full grid on dataset_v6_1
============================================================================
Cells: {T0, T1} × {legacy, robust}; seeds {0, 1, 2}; 12 runs, sequential,
resume-safe. Declarations Q-1/Q-2/Q-3 (RESEARCH_LOG 2026-08-26) govern:
  - identical PRT-Full architecture + hyperparameters in every run (v8/v11
    convention, byte-identical below);
  - cells differ ONLY in the input representation (per-cell normalization
    params from dataset_v6_1 metadata); persistence anchor from RAW
    frp_lag_1h; loss/residual pathway log-space, unchanged;
  - checkpointing/early-stop on event-level MEDIAN val SS_fire_t6
    (gated persistence denominators, identical to the baseline suite).

No test data exists in dataset_v6_1 (sequestration). A failed run is
reported, not tuned around.

Outputs per run: data/models/e28/{cell}_s{seed}_best.pt + _result.json
Grid summary:    data/models/e28/e28_grid_summary.json (rewritten as runs land)

Run: C:/Users/dvalsamis/AppData/Local/anaconda3/python.exe scripts/e28_train_grid.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("e:/GRSL_Wildfire/FRP_Methodology")
H5_PATH = ROOT / "data/processed/dataset_v6_1.h5"
MANIFEST_PATH = ROOT / "data/processed/dataset_v6_1_manifest.csv"
OUT_DIR = ROOT / "data/models/e28"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- frozen PRT-Full hyperparameters (v8/v11, byte-identical) ----
HIDDEN_DIM = 96
N_HEADS = 4
DROPOUT = 0.1
N_LSTM_LAYERS = 2
FIRE_WEIGHT = 7.0
LOSS_ALPHA = 0.5
RESIDUAL_SCALE_MAX = 0.5
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
PATIENCE = 5
MAX_EPOCHS = 30
MIN_EPOCHS = 2
PCT_START = 0.3

CELLS = [("T0", "legacy"), ("T0", "robust"), ("T1", "legacy"), ("T1", "robust")]
SEEDS = [0, 1, 2]

LOG1P_FEATS = {"frp_lag_1h", "frp_lag_2h", "frp_lag_3h", "frp_lag_6h",
               "frp_lag_12h", "frp_lag_24h", "frp_rolling_6h_mean",
               "frp_rolling_6h_std", "cumulative_frp"}
SIGNED_FEATS = {"frp_trend_6h"}

# ---- load once ----
with h5py.File(H5_PATH, "r") as f:
    meta = json.loads(f["metadata"][()])
    LOOKBACK = meta["LOOKBACK"]
    HORIZON = meta["HORIZON"]
    FEATS = meta["feature_names"]
    NORM = meta["normalization_params"]        # NORM[variant][scheme]
    tr_inp_raw = f["train/inputs"][:]
    tr_tgt = f["train/targets"][:]
    va_inp_raw = f["val/inputs"][:]
    va_tgt = f["val/targets"][:]
assert meta.get("inputs_are_raw")
FEAT = {n: i for i, n in enumerate(FEATS)}
LAST_TS = LOOKBACK - 1
INPUT_SIZE = len(FEATS)

manifest = pd.read_csv(MANIFEST_PATH)
tr_man = manifest[manifest["split"] == "train"].reset_index(drop=True)
va_man = manifest[manifest["split"] == "val"].reset_index(drop=True)
T0_MASK = tr_man["in_T0"].to_numpy(dtype=bool)
VAL_EVENTS = sorted(va_man["event_name"].unique())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}  train={tr_inp_raw.shape}  val={va_inp_raw.shape}")

# persistence anchors from RAW inputs (gated), fixed for all cells
def persistence_anchor(raw_inp):
    frp = np.clip(raw_inp[:, LAST_TS, FEAT["frp_lag_1h"]], 0, None)
    is_fire = raw_inp[:, LAST_TS, FEAT["is_fire"]]
    return np.where(is_fire > 0, frp, 0.0).astype(np.float32)

VA_PERS = persistence_anchor(va_inp_raw)
va_fire = va_tgt.max(axis=1) > 0

# per-event persistence MAE at t+6 (denominators, identical to baseline suite)
PERS_MAE_T6 = {}
VAL_EV_IDX = {}
for ev in VAL_EVENTS:
    idx = va_man.index[va_man["event_name"] == ev].to_numpy()
    VAL_EV_IDX[ev] = idx
    fm = va_fire[idx]
    PERS_MAE_T6[ev] = (float(np.mean(np.abs(VA_PERS[idx][fm] - va_tgt[idx][:, 5][fm])))
                       if fm.any() else float("nan"))


def normalize(raw, variant, scheme):
    out = raw.astype(np.float32).copy()
    params = NORM[variant][scheme]
    for j, f in enumerate(FEATS):
        p = params.get(f)
        if p is None:
            continue
        x = out[:, :, j].astype(np.float64)
        if scheme == "legacy":
            out[:, :, j] = ((x - p["mean"]) / p["std"]).astype(np.float32)
        else:
            t = (np.log1p(np.maximum(x, 0)) if f in LOG1P_FEATS else
                 np.sign(x) * np.log1p(np.abs(x)) if f in SIGNED_FEATS else x)
            out[:, :, j] = ((t - p["center"]) / p["scale"]).astype(np.float32)
    return out


class LogHybridLoss(nn.Module):
    def __init__(self, fire_weight=7.0, alpha=0.5):
        super().__init__()
        self.fire_weight = fire_weight
        self.alpha = alpha

    def forward(self, pred, target):
        diff = torch.log1p(pred) - torch.log1p(target)
        loss = self.alpha * torch.abs(diff) + (1.0 - self.alpha) * diff ** 2
        fire_active = (target > 0).any(dim=1, keepdim=True).float()
        weight = 1.0 + (self.fire_weight - 1.0) * fire_active
        return (loss * weight).mean()


class PRTWithPersistenceSkip(nn.Module):
    """Identical v8 architecture; persistence anchor passed as raw MW tensor
    (mathematically identical to v8's denormalize-the-feature construction)."""

    def __init__(self, input_size, horizon):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.input_proj = nn.Linear(input_size, HIDDEN_DIM)
        self.input_drop = nn.Dropout(DROPOUT)
        self.lstm = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, N_LSTM_LAYERS,
                            batch_first=True,
                            dropout=DROPOUT if N_LSTM_LAYERS > 1 else 0.0)
        self.attention = nn.MultiheadAttention(HIDDEN_DIM, N_HEADS,
                                               dropout=DROPOUT, batch_first=True)
        self.norm1 = nn.LayerNorm(HIDDEN_DIM)
        self.ffn = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 4), nn.GELU(),
            nn.Dropout(DROPOUT), nn.Linear(HIDDEN_DIM * 4, HIDDEN_DIM))
        self.norm2 = nn.LayerNorm(HIDDEN_DIM)
        self.out_drop = nn.Dropout(DROPOUT)
        self.output_proj = nn.Linear(HIDDEN_DIM, horizon)

    def forward(self, x, pers_raw):
        persistence_log = torch.log1p(torch.clamp(pers_raw, min=0.0))
        h = self.input_drop(self.input_proj(x))
        lstm_out, _ = self.lstm(h)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        lstm_out = self.norm1(lstm_out + attn_out)
        lstm_out = self.norm2(lstm_out + self.ffn(lstm_out))
        residual = self.output_proj(self.out_drop(lstm_out[:, -1, :]))
        rs = torch.clamp(self.residual_scale, min=0.0, max=RESIDUAL_SCALE_MAX)
        pred_log = persistence_log.unsqueeze(1) + rs * residual
        return torch.clamp(torch.expm1(pred_log), min=0.0)


def val_median_ss(preds):
    ss = []
    for ev in VAL_EVENTS:
        idx = VAL_EV_IDX[ev]
        fm = va_fire[idx]
        pm = PERS_MAE_T6[ev]
        if not fm.any() or np.isnan(pm) or pm == 0:
            continue
        mae = float(np.mean(np.abs(preds[idx][:, 5][fm] - va_tgt[idx][:, 5][fm])))
        ss.append(1.0 - mae / pm)
    return float(np.median(ss)) if ss else float("nan"), ss


def per_event_val(preds):
    out = {}
    for ev in VAL_EVENTS:
        idx = VAL_EV_IDX[ev]
        fm = va_fire[idx]
        pm = PERS_MAE_T6[ev]
        if not fm.any():
            out[ev] = {"n_fire": 0, "MAE_fire_t6": float("nan"),
                       "SS_fire_t6": float("nan")}
            continue
        mae = float(np.mean(np.abs(preds[idx][:, 5][fm] - va_tgt[idx][:, 5][fm])))
        out[ev] = {"n_fire": int(fm.sum()), "MAE_fire_t6": mae,
                   "SS_fire_t6": (1.0 - mae / pm) if pm else float("nan")}
    return out


def run_one(variant, scheme, seed):
    tag = f"{variant}_{scheme}_s{seed}"
    res_path = OUT_DIR / f"{tag}_result.json"
    if res_path.exists():
        print(f"[{tag}] SKIP — result exists", flush=True)
        return json.load(open(res_path))

    print(f"\n===== RUN {tag} =====", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    sel = T0_MASK if variant == "T0" else np.ones(len(tr_man), bool)
    tr_x = normalize(tr_inp_raw[sel], variant, scheme)
    va_x = normalize(va_inp_raw, variant, scheme)
    tr_pers = persistence_anchor(tr_inp_raw[sel])

    tr_ds = TensorDataset(torch.tensor(tr_x), torch.tensor(tr_pers),
                          torch.tensor(tr_tgt[sel]))
    va_ds = TensorDataset(torch.tensor(va_x), torch.tensor(VA_PERS),
                          torch.tensor(va_tgt))
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                           drop_last=True, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=256, shuffle=False, num_workers=0)

    model = PRTWithPersistenceSkip(INPUT_SIZE, HORIZON).to(device)
    criterion = LogHybridLoss(FIRE_WEIGHT, LOSS_ALPHA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=len(tr_loader),
        epochs=MAX_EPOCHS, pct_start=PCT_START)

    def infer():
        model.eval()
        ps = []
        with torch.no_grad():
            for xb, pb, _ in va_loader:
                ps.append(model(xb.to(device), pb.to(device)).cpu().numpy())
        return np.concatenate(ps, 0)

    best_ss, best_epoch, no_imp = float("-inf"), 0, 0
    curve = []
    ckpt = OUT_DIR / f"{tag}_best.pt"
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tot, nb = 0.0, 0
        for xb, pb, yb in tr_loader:
            xb, pb, yb = xb.to(device), pb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb, pb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            tot += loss.item(); nb += 1
        med_ss, _ = val_median_ss(infer())
        rs = model.residual_scale.item()
        improved = (not np.isnan(med_ss)) and med_ss > best_ss
        if improved:
            best_ss, best_epoch, no_imp = med_ss, epoch, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_median_ss_t6": med_ss, "residual_scale": rs,
                        "cell": [variant, scheme], "seed": seed}, ckpt)
        else:
            no_imp += 1
        curve.append({"epoch": epoch, "train_loss": tot / nb,
                      "val_median_ss_t6": med_ss, "res_scale": rs})
        print(f"  ep{epoch:>2} loss={tot/nb:.5f} medSS={med_ss:+.4f} "
              f"rs={rs:.3f} best_ep={best_epoch}{' <--' if improved else ''}",
              flush=True)
        if epoch >= MIN_EPOCHS and no_imp >= PATIENCE:
            print(f"  early stop at ep{epoch}; best ep{best_epoch}", flush=True)
            break

    # evaluate best checkpoint
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    preds = infer()
    med_ss, ss_list = val_median_ss(preds)
    pe = per_event_val(preds)
    np.save(OUT_DIR / f"{tag}_val_preds.npy", preds)
    result = {"tag": tag, "cell": [variant, scheme], "seed": seed,
              "best_epoch": int(state["epoch"]),
              "val_median_ss_t6": med_ss,
              "val_mean_ss_t6": float(np.mean(ss_list)),
              "val_std_ss_t6": float(np.std(ss_list, ddof=1)),
              "n_events_evaluable": len(ss_list),
              "residual_scale": float(state["residual_scale"]),
              "per_event": pe, "curve": curve,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    with open(res_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{tag}] DONE best_ep={state['epoch']} median SS={med_ss:+.4f}", flush=True)
    return result


def main():
    results = []
    for variant, scheme in CELLS:
        for seed in SEEDS:
            try:
                results.append(run_one(variant, scheme, seed))
            except Exception as e:
                print(f"[{variant}_{scheme}_s{seed}] FAILED: {e}", flush=True)
                results.append({"tag": f"{variant}_{scheme}_s{seed}",
                                "error": str(e)})
            with open(OUT_DIR / "e28_grid_summary.json", "w") as f:
                json.dump(results, f, indent=2)
    print("\nGRID COMPLETE", flush=True)


if __name__ == "__main__":
    main()
