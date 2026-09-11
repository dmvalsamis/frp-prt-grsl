#!/usr/bin/env python
"""explore_arch_train.py -- EXPLORATORY architecture study (branch explore-arch-2026).

Question: is the M2 ~ B6 event-grain equivalence on test blocks A + B specific to
the PRT encoder, or a property of deep encoders in general under the same
persistence-residual wrapper?  Six encoders, identical wrapper / representation /
loss / recipe / seeds, no selection step; every architecture goes to test.

EVERYTHING this script produces is EXPLORATORY (blocks A and B are examined data;
nothing is sequestered except the 2027 season).

Reuse (imported unchanged, never edited):
  scripts/e28_train_grid.py  loaded via importlib as module "e28" -> data arrays,
      normalize(), persistence_anchor(), LogHybridLoss, val_median_ss(),
      per_event_val(), and every hyperparameter constant.
  scripts/test_access_v1.py  -> assert_environment() (pinned stack).

Recipe (byte-identical to e28 run_one except the model class): cell fixed to
("T1", "robust"); AdamW lr 1e-3 wd 1e-4; OneCycle pct 0.3; batch 64; grad-clip 1.0;
LogHybridLoss fire_weight 7.0 alpha 0.5; patience 5, max 30 / min 2 epochs;
checkpoint on event-level MEDIAN val SS_fire,t6 (12 evaluable val events).

Wrapper contract (identical to PRTWithPersistenceSkip):
    persistence_log = log1p(clamp(pers_raw, 0))
    residual        = encoder(x)                      # (B, 12)
    rs              = clamp(residual_scale, 0, 0.5)   # nn.Parameter init 0.1
    pred            = clamp(expm1(persistence_log[:, None] + rs * residual), 0)

Outputs: data/models/explore_arch/{arch}_s{seed}_{best.pt,result.json,val_preds.npy}
         data/models/explore_arch/explore_arch_summary.json (rebuilt as runs land)

Run:  PYTHONUTF8=1 C:/Users/dvalsamis/AppData/Local/anaconda3/python.exe \
        scripts/explore_arch_train.py [--archs A1,A2,...] [--seeds 0,1,...] [--smoke]
"""
import argparse
import importlib.util
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import test_access_v1 as T  # noqa: E402  (locked; read-only import)

T.assert_environment()

# ---- e28 machinery, imported unchanged (module name "e28": main() guard is inert) ----
_spec = importlib.util.spec_from_file_location("e28", ROOT / "scripts" / "e28_train_grid.py")
e28 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e28)

OUT_DIR = ROOT / "data" / "models" / "explore_arch"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VARIANT, SCHEME = "T1", "robust"          # fixed cell (M2's cell)
LOOKBACK, HORIZON, INPUT_SIZE = e28.LOOKBACK, e28.HORIZON, e28.INPUT_SIZE
device = e28.device
RS_MAX = e28.RESIDUAL_SCALE_MAX


def skip_out(pers_raw, residual, rs_param):
    """The e28 persistence-residual wrapper, verbatim in semantics."""
    persistence_log = torch.log1p(torch.clamp(pers_raw, min=0.0))
    rs = torch.clamp(rs_param, min=0.0, max=RS_MAX)
    return torch.clamp(torch.expm1(persistence_log.unsqueeze(1) + rs * residual), min=0.0)


# ======================= encoders (residual head = Linear -> HORIZON) =======================
class GRUSkip(nn.Module):
    """A1: GRU, 1 layer, hidden 64, last state (port of 20_architecture_comparison.GRUSkip)."""
    def __init__(self, input_size, horizon):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.gru = nn.GRU(input_size, 64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, horizon)

    def forward(self, x, pers_raw):
        _, h = self.gru(x)
        return skip_out(pers_raw, self.fc(h.squeeze(0)), self.residual_scale)


class LSTMSkip(nn.Module):
    """A2: LSTM, 2 layers, hidden 96, dropout 0.1, last state (port of LSTMSkip)."""
    def __init__(self, input_size, horizon):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.lstm = nn.LSTM(input_size, 96, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(96, horizon)

    def forward(self, x, pers_raw):
        _, (h, _) = self.lstm(x)
        return skip_out(pers_raw, self.fc(h[-1]), self.residual_scale)


class CNNSkip(nn.Module):
    """A3: Conv1d 32->64 k3 d1, 64->64 k3 d2, ReLU, global average pool (port of CNNSkip)."""
    def __init__(self, input_size, horizon):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, dilation=1, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, dilation=2, padding=2), nn.ReLU())
        self.fc = nn.Linear(64, horizon)

    def forward(self, x, pers_raw):
        h = self.conv(x.permute(0, 2, 1)).mean(dim=2)
        return skip_out(pers_raw, self.fc(h), self.residual_scale)


class MLPMixer(nn.Module):
    """A4: per-feature mean/std/last/slope over the 48 h window -> 128 -> 64 -> 12
    (port of MLPMixer; zero learned temporal dynamics)."""
    def __init__(self, input_size, horizon):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        t = torch.arange(LOOKBACK, dtype=torch.float32) - (LOOKBACK - 1) / 2.0
        self.register_buffer("t_c", t.view(1, LOOKBACK, 1))
        self.register_buffer("t_ss", (t ** 2).sum())
        self.mlp = nn.Sequential(nn.Linear(input_size * 4, 64), nn.ReLU(),
                                 nn.Linear(64, horizon))

    def forward(self, x, pers_raw):
        feats = torch.cat([x.mean(dim=1), x.std(dim=1, unbiased=False), x[:, -1, :],
                           (self.t_c * x).sum(dim=1) / self.t_ss], dim=1)
        return skip_out(pers_raw, self.mlp(feats), self.residual_scale)


class PRTNoLSTM(nn.Module):
    """A5: PRTWithPersistenceSkip with self.lstm deleted -- Linear 32->96, dropout,
    4-head self-attention + Add&Norm, FFN 96->384->96 + Add&Norm, last step."""
    def __init__(self, input_size, horizon):
        super().__init__()
        H, NH, D = e28.HIDDEN_DIM, e28.N_HEADS, e28.DROPOUT
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.input_proj = nn.Linear(input_size, H)
        self.input_drop = nn.Dropout(D)
        self.attention = nn.MultiheadAttention(H, NH, dropout=D, batch_first=True)
        self.norm1 = nn.LayerNorm(H)
        self.ffn = nn.Sequential(nn.Linear(H, H * 4), nn.GELU(), nn.Dropout(D),
                                 nn.Linear(H * 4, H))
        self.norm2 = nn.LayerNorm(H)
        self.out_drop = nn.Dropout(D)
        self.output_proj = nn.Linear(H, horizon)

    def forward(self, x, pers_raw):
        h = self.input_drop(self.input_proj(x))
        attn_out, _ = self.attention(h, h, h)
        h = self.norm1(h + attn_out)
        h = self.norm2(h + self.ffn(h))
        residual = self.output_proj(self.out_drop(h[:, -1, :]))
        return skip_out(pers_raw, residual, self.residual_scale)


class PatchTSTSkip(nn.Module):
    """A6 (literature encoder): PatchTST-style channel-independent patch-embedding
    transformer encoder (Nie, Nguyen, Sinthong, Kalagnanam, "A Time Series is Worth
    64 Words: Long-term Forecasting with Transformers", ICLR 2023).
    Each of the 32 input channels is cut into patches (length 8, stride 4 -> 11 patches),
    embedded by a shared Linear(8->32) + learned positional embedding, encoded by a shared
    2-layer TransformerEncoder (4 heads, FFN 64, dropout 0.1, GELU), flattened per channel
    (11x32=352) -> shared Linear(352->16) + GELU; the 32 channel codes (512) -> dropout ->
    Linear(512->12).  No RevIN (inputs are already robust-normalized by the wrapper's
    representation).  Residual head plugs into the identical persistence skip."""
    PATCH, STRIDE, D_MODEL, N_HEADS, N_LAYERS, D_FF, CH_OUT, DROP = 8, 4, 32, 4, 2, 64, 16, 0.1

    def __init__(self, input_size, horizon):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.n_ch = input_size
        self.n_patches = (LOOKBACK - self.PATCH) // self.STRIDE + 1
        self.embed = nn.Linear(self.PATCH, self.D_MODEL)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, self.D_MODEL) * 0.02)
        layer = nn.TransformerEncoderLayer(self.D_MODEL, self.N_HEADS, self.D_FF,
                                           dropout=self.DROP, activation="gelu",
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, self.N_LAYERS, enable_nested_tensor=False)
        self.ch_head = nn.Sequential(nn.Linear(self.n_patches * self.D_MODEL, self.CH_OUT), nn.GELU())
        self.out_drop = nn.Dropout(self.DROP)
        self.head = nn.Linear(self.n_ch * self.CH_OUT, horizon)

    def forward(self, x, pers_raw):
        B = x.shape[0]
        p = x.permute(0, 2, 1).unfold(2, self.PATCH, self.STRIDE)        # (B, C, P, L)
        p = p.reshape(B * self.n_ch, self.n_patches, self.PATCH)
        z = self.encoder(self.embed(p) + self.pos)                        # (B*C, P, D)
        z = self.ch_head(z.reshape(B * self.n_ch, -1)).reshape(B, -1)    # (B, C*CH_OUT)
        return skip_out(pers_raw, self.head(self.out_drop(z)), self.residual_scale)


ARCHS = {
    "A1": ("GRU-Skip", GRUSkip),
    "A2": ("LSTM-Skip", LSTMSkip),
    "A3": ("CNN-Skip", CNNSkip),
    "A4": ("MLP-Mixer", MLPMixer),
    "A5": ("PRT-noLSTM", PRTNoLSTM),
    "A6": ("PatchTST-Skip", PatchTSTSkip),
}


def n_params(model):
    return int(sum(p.numel() for p in model.parameters()))


# ---- representation: fixed cell, computed once per process ----
_DATA = {}


def data():
    if not _DATA:
        _DATA["tr_x"] = e28.normalize(e28.tr_inp_raw, VARIANT, SCHEME)
        _DATA["va_x"] = e28.normalize(e28.va_inp_raw, VARIANT, SCHEME)
        _DATA["tr_pers"] = e28.persistence_anchor(e28.tr_inp_raw)
    return _DATA


def run_one(arch, seed, smoke=False, out_dir=OUT_DIR):
    """e28.run_one with the model class swapped and the cell fixed to (T1, robust)."""
    name, cls = ARCHS[arch]
    tag = f"{arch}_s{seed}"
    res_path = out_dir / f"{tag}_result.json"
    if res_path.exists():
        print(f"[{tag}] SKIP -- result exists", flush=True)
        return json.load(open(res_path))
    max_epochs = 1 if smoke else e28.MAX_EPOCHS

    print(f"\n===== RUN {tag} ({name}) =====", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = data()
    tr_ds = TensorDataset(torch.tensor(d["tr_x"]), torch.tensor(d["tr_pers"]),
                          torch.tensor(e28.tr_tgt))
    va_ds = TensorDataset(torch.tensor(d["va_x"]), torch.tensor(e28.VA_PERS),
                          torch.tensor(e28.va_tgt))
    tr_loader = DataLoader(tr_ds, batch_size=e28.BATCH_SIZE, shuffle=True,
                           drop_last=True, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=256, shuffle=False, num_workers=0)

    model = cls(INPUT_SIZE, HORIZON).to(device)
    npar = n_params(model)
    print(f"  {name}: {npar:,} parameters", flush=True)
    criterion = e28.LogHybridLoss(e28.FIRE_WEIGHT, e28.LOSS_ALPHA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=e28.LR,
                                  weight_decay=e28.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=e28.LR, steps_per_epoch=len(tr_loader),
        epochs=max_epochs, pct_start=e28.PCT_START)

    def infer():
        model.eval()
        ps = []
        with torch.no_grad():
            for xb, pb, _ in va_loader:
                ps.append(model(xb.to(device), pb.to(device)).cpu().numpy())
        return np.concatenate(ps, 0)

    best_ss, best_epoch, no_imp = float("-inf"), 0, 0
    curve = []
    ckpt = out_dir / f"{tag}_best.pt"
    t_start = datetime.now(timezone.utc)
    for epoch in range(1, max_epochs + 1):
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
        med_ss, _ = e28.val_median_ss(infer())
        rs = model.residual_scale.item()
        improved = (not np.isnan(med_ss)) and med_ss > best_ss
        if improved:
            best_ss, best_epoch, no_imp = med_ss, epoch, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_median_ss_t6": med_ss, "residual_scale": rs,
                        "arch": arch, "arch_name": name, "cell": [VARIANT, SCHEME],
                        "seed": seed, "n_params": npar}, ckpt)
        else:
            no_imp += 1
        curve.append({"epoch": epoch, "train_loss": tot / nb,
                      "val_median_ss_t6": med_ss, "res_scale": rs})
        print(f"  ep{epoch:>2} loss={tot/nb:.5f} medSS={med_ss:+.4f} rs={rs:.3f} "
              f"best_ep={best_epoch}{' <--' if improved else ''}", flush=True)
        if epoch >= e28.MIN_EPOCHS and no_imp >= e28.PATIENCE:
            print(f"  early stop at ep{epoch}; best ep{best_epoch}", flush=True)
            break

    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    preds = infer()
    med_ss, ss_list = e28.val_median_ss(preds)
    pe = e28.per_event_val(preds)
    np.save(out_dir / f"{tag}_val_preds.npy", preds)
    result = {"tag": tag, "arch": arch, "arch_name": name, "cell": [VARIANT, SCHEME],
              "seed": seed, "n_params": npar, "smoke": bool(smoke),
              "best_epoch": int(state["epoch"]),
              "val_median_ss_t6": med_ss,
              "val_mean_ss_t6": float(np.mean(ss_list)),
              "val_std_ss_t6": float(np.std(ss_list, ddof=1)),
              "n_events_evaluable": len(ss_list),
              "residual_scale": float(state["residual_scale"]),
              "per_event": pe, "curve": curve,
              "train_minutes": (datetime.now(timezone.utc) - t_start).total_seconds() / 60.0,
              "finished_utc": datetime.now(timezone.utc).isoformat(),
              "status": "EXPLORATORY"}
    with open(res_path, "w") as f:
        json.dump(result, f, indent=2)
    fail = out_dir / f"{tag}_FAILED.json"
    if fail.exists():
        fail.unlink()
    print(f"[{tag}] DONE best_ep={state['epoch']} median SS={med_ss:+.4f} "
          f"mean={np.mean(ss_list):+.4f} ({result['train_minutes']:.1f} min)", flush=True)
    return result


def write_summary(out_dir=OUT_DIR):
    """Rebuild the summary from whatever result / failure files exist on disk
    (idempotent, so several workers can call it)."""
    rows = []
    for p in sorted(out_dir.glob("A?_s*_result.json")):
        r = json.load(open(p))
        rows.append({k: r[k] for k in ("tag", "arch", "arch_name", "seed", "n_params",
                                       "best_epoch", "val_median_ss_t6", "val_mean_ss_t6",
                                       "val_std_ss_t6", "n_events_evaluable",
                                       "residual_scale", "train_minutes", "finished_utc")})
    for p in sorted(out_dir.glob("A?_s*_FAILED.json")):
        rows.append(json.load(open(p)))
    rows.sort(key=lambda r: (int(r["seed"]), r["arch"]))
    with open(out_dir / "explore_arch_summary.json", "w") as f:
        json.dump({"status": "EXPLORATORY", "cell": [VARIANT, SCHEME],
                   "archs": {k: v[0] for k, v in ARCHS.items()},
                   "updated_utc": datetime.now(timezone.utc).isoformat(),
                   "runs": rows}, f, indent=2)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default=",".join(ARCHS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in range(10)))
    ap.add_argument("--smoke", action="store_true", help="1 epoch per arch into explore_arch/smoke/")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    archs = [x.strip() for x in a.archs.split(",") if x.strip()]
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    out_dir = OUT_DIR / "smoke" if a.smoke else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("parameter counts:")
    for k in archs:
        print(f"  {k} {ARCHS[k][0]:14s} {n_params(ARCHS[k][1](INPUT_SIZE, HORIZON)):>9,}")
    # seed-major order: a partial run still gives a balanced comparison
    for seed in seeds:
        for arch in archs:
            try:
                run_one(arch, seed, smoke=a.smoke, out_dir=out_dir)
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[{arch}_s{seed}] FAILED: {e}\n{tb}", flush=True)
                with open(out_dir / f"{arch}_s{seed}_FAILED.json", "w") as f:
                    json.dump({"tag": f"{arch}_s{seed}", "arch": arch,
                               "arch_name": ARCHS[arch][0], "seed": seed,
                               "status": "FAILED", "error": str(e), "traceback": tb,
                               "utc": datetime.now(timezone.utc).isoformat()}, f, indent=2)
            if not a.smoke:
                write_summary(out_dir)
    print("\nALL REQUESTED RUNS PROCESSED", flush=True)


if __name__ == "__main__":
    main()
