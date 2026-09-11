"""
test_access_v1.py — THE single test-access script (ACC-1, memo A4, 2026-08-26)
==============================================================================
Pre-registered evaluation script for the sequestered test blocks. Written and
val-tested BEFORE the prereg lock; SHA-256 recorded in the lock's §8 register.
Nobody improvises at access time: this script is the access.

Scope (ACC-1.1): one access event covering
  - M2 (frozen T1×robust 10-seed log-mean ensemble) predictions
  - B6 = gbdt_log_T1 (final, per B-2 renaming rule 2026-08-26): histogram GBDT,
    log1p target, cfg max_leaf_nodes=63 / lr=0.1 / max_iter=300 / early
    stopping 10% / random_state=0, retrained deterministically per horizon on
    the FULL dataset_v6_1 train pool at access time (config frozen here)
  - B1 gated flat persistence (reference)
  - PRIMARY: one-sided paired sign test, H1 median MAE_fire_t6(M2) < B6,
    over the primary event set (block A n=30 + finalized block B), α=0.05
  - SECONDARY: same sign test vs B1; DEC-1 sensitivity set (d7_restored
    included); AGG-1 median SS headline + mean±std; Wilcoxon on same pairs;
    per-event table; MAE at t+1/3/6/12
  - N_min accounting: an event enters the paired statistics iff it has ≥ 5
    fire-active forecast windows; gate-passing AND evaluable counts reported.

Modes:
  --dry-run-val
      Validation-only rehearsal (ACC-1.3). Rebuilds each v6.1 VAL event's
      windows from its hourly+ERA5 CSVs (the exact code path used at access),
      asserts equality with the frozen dataset_v6_1.h5 val arrays, then runs
      the complete evaluation machinery treating val events as pseudo-test
      events. Expected reproduction: M2 median +0.3131, B6 median +0.3050,
      M2 paired wins 9/12. Touches NO test data.
  --access EVENT_LIST_CSV --confirm-single-access
      THE single access. EVENT_LIST_CSV columns: event_name, block
      (A2025|B2026), in_primary (0/1 per DEC-1 + DEC-3), d7_restored (0/1).
      Refuses to run without --confirm-single-access. Verifies every M2
      checkpoint hash against the freeze table before predicting.

Outputs: resubmission/access_dryrun_val/ or resubmission/TEST_ACCESS_RESULTS/
  (results JSON + per-event CSV; access mode also writes an access record).

REV 4 (2026-08-28, memo IA-1): IA-B1 — event lists carry in_sensitivity; the
sensitivity sign test runs over exactly that pre-registered set (result key
SECONDARY_sign_test_M2_vs_B6_sensitivity_set); the block-A subset of any
supplied access list is asserted row-identical to the frozen
test_event_list_blockA_LOCKED.csv. IA-A2 — dataset/manifest/freeze-table
hashes asserted at startup. IA-A5 — zero-window gate-passers emit table rows.
IA-N5 — Wilcoxon computed on the paired frame.

AMENDED 2026-08-27 per memo A6 (both changes re-hashed + dry-run revalidated;
prior hash 93ed7538… recorded in prereg §8 as superseded):
  RAT-9 — M1 RE-SCOPED to the Zenodo-archived ensemble of the rejected
    submission (10.5281/zenodo.20627737, zip md5 771491878286355cd9b15d418a
    a195cb, vendored at resubmission/m1_deposit/). The complete pipeline
    reconstructs from the deposit alone: its build_dataset.py canonical
    32-feature order (asserted at runtime against our feature list), its
    normalization_params_v3.json, its wind_features.build_wind_features with
    checkpoint-embedded wind stats, its PRT class, its log-mean rule from
    eval_test.py. Every M1 checkpoint is SHA-256-verified against constants
    frozen below. ALL M1 outputs are DESCRIPTIVE — no test involves M1.
  AMEND-8 (T-1..T-4) — pre-registered descriptive stratifications: (a)
    per-horizon profile {1,3,6,12}; (b) wind-regime conditional (|delta_
    wind_3h| at the last input step vs the frozen TRAIN 1-sigma); (c)
    event-volatility tertiles (frozen TRAIN boundaries). Outputs are paired
    win counts and median deltaSS ONLY (T-1: no p-values, no CIs), written
    as ONE strata table per run (T-3).
"""

import argparse
import hashlib
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import binomtest, wilcoxon
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
E28 = ROOT / "data" / "models" / "e28"
H5 = PROC / "dataset_v6_1.h5"
FREEZE = E28 / "m2_freeze_hashes.json"

LOOKBACK, HORIZON = 48, 12
LAST_TS = LOOKBACK - 1
N_MIN_FIRE = 5              # evaluability rule, prereg §6
ALPHA = 0.05
B6_CFG = {"max_leaf_nodes": 63, "learning_rate": 0.1}   # frozen per B-2
M2_SEEDS = list(range(10))

# ---- A-d environment pin (adversarial review 2026-08-28): B6 is refit at
# access time, and determinism across library versions is not guaranteed.
# The critical four are asserted; the full pin is
# resubmission/access_environment_lock.txt (hashed in prereg §8). ----
EXPECTED_VERSIONS = {"sklearn": "1.2.2", "numpy": "1.26.4",
                     "torch": "2.6.0+cpu", "h5py": "3.8.0"}

# ---- IA-A2 (memo IA-1, 2026-08-28): runtime hash assertions for the last
# unverified inputs — dataset (B6 training data + normalization source),
# its manifest, and the M2 freeze table itself. ----
H5_SHA256 = "0a1a7c8ac23d6525bac1f93e99ec6a9befd537653baaa7495e65693f6ff75724"
MANIFEST_SHA256 = "cdff3201039e1846530568839c71254241a0c19881bf33b0bfff1bd1c367e1df"
FREEZE_SHA256 = "27b947348ff4ee493bc46c88a659ad89d1372bfcb9c572cf5ec98b222bf43cfa"

# ---- IA-B1 (memo IA-1): the block-A composition is frozen pre-lock; the
# supplied access list's block-A subset must be row-identical to this file. ----
BLOCKA_LOCKED_CSV = ROOT / "resubmission" / "test_event_list_blockA_LOCKED.csv"


def assert_input_hashes(dry_run):
    for path, want, name in [(H5, H5_SHA256, "dataset_v6_1.h5"),
                             (PROC / "dataset_v6_1_manifest.csv",
                              MANIFEST_SHA256, "dataset_v6_1_manifest.csv"),
                             (FREEZE, FREEZE_SHA256, "m2_freeze_hashes.json")]:
        actual = sha256(path)
        if actual != want:
            raise RuntimeError(f"INPUT HASH VIOLATION: {name} {actual} != "
                               f"frozen {want} — refusing to run")
    print("input hashes OK: dataset, manifest, freeze table", flush=True)


def assert_environment():
    import sklearn
    actual = {"sklearn": sklearn.__version__, "numpy": np.__version__,
              "torch": torch.__version__, "h5py": h5py.__version__}
    for lib, want in EXPECTED_VERSIONS.items():
        if actual[lib] != want:
            raise RuntimeError(
                f"ENVIRONMENT PIN VIOLATION: {lib} {actual[lib]} != frozen "
                f"{want} (see resubmission/access_environment_lock.txt). "
                f"Restore the pinned environment; do not run the access on a "
                f"drifted stack.")
    print("environment pin OK: " +
          ", ".join(f"{k}={v}" for k, v in actual.items()), flush=True)


# ---- AMEND-8 T-2 frozen constants (computed from TRAIN, 2026-08-27) ----
# 1-sigma of raw delta_wind_3h at the last input step over dataset_v6_1 TRAIN:
DELTA_WIND_3H_SIGMA = 0.7994305491447449
# Event volatility descriptor V = population std of delta(log1p(frp_sum_mw))
# over consecutive (exactly 1 h apart) fire->fire slot pairs of the event's
# merged hourly frame (slot_type != 'missing' rows). TRAIN tertile boundaries
# over the 71 v6.1 train events:
VOL_TERTILE_LO = 0.8348539322108006
VOL_TERTILE_HI = 1.0851316749933935
STRATA_HORIZONS = [1, 3, 6, 12]

# ---- M1 (RAT-9 re-scope): Zenodo deposit 10.5281/zenodo.20627737 ----
M1_DIR = ROOT / "resubmission" / "m1_deposit" / "dmvalsamis-frp-prt-grsl-4a8b44d"
M1_ZIP_MD5 = "771491878286355cd9b15d418aa195cb"   # provenance anchor (Zenodo)
M1_CKPT_SHA256 = {
    "seed_000.pt": "63d2ed06bbc9affc248a596f6acab4c373fc017facfd0a0fba29d91b79ea7fc1",
    "seed_001.pt": "c254799a6c610ecbc7b9ab74654d98de07157e5ed1c947abe5c519b31e703ddd",
    "seed_002.pt": "906591cde5750e0ba44a256a79e3f03cffd41ce9014cba9b68fab5859ebece24",
    "seed_003.pt": "ee781e157d4ee1fa0b9cb7271f4e5be24b71ce771dc9795f3fb2e50362efae63",
    "seed_004.pt": "fa2061474c97bb57fa23dc0042f604ec5335ea4e2af3679e3e623dca3a9b3ea8",
    "seed_005.pt": "74f0f4c285ed01a6abdbdde2e1be7cb89c6bcb66de4599ddcdcb2dddc21b8b21",
    "seed_006.pt": "dc619e91e043b90e66a95e73106f5881de2c9cae4f0340fcfd51467ae58650de",
    "seed_007.pt": "4562bcc472ff442d481540c1fc26ff06930ed647b65b9f0d035aa005470691f0",
    "seed_008.pt": "c13557a3fbb4c18cff7504c28b7ce8e2a01514c70a3347887fec4d1744d76bc3",
    "seed_009.pt": "d8e2d165911dab55349d8985c57a09b4bbeca23b9f4939e8404dcb6306d5b76c",
}
M1_NORM_SHA256 = "3315db8327f3990d4d3cfbc72173ab694e5af3826bf456d7603494b4471784ef"
# canonical 32-feature order from the deposit's build_dataset.py (asserted
# against our dataset feature list at runtime — any mismatch aborts M1):
M1_CANONICAL_FEATS = [
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "log_frp", "frp_lag_1h", "frp_lag_2h", "frp_lag_3h",
    "frp_lag_6h", "frp_lag_12h", "frp_lag_24h",
    "frp_rolling_6h_mean", "frp_rolling_6h_std", "frp_trend_6h",
    "cumulative_frp", "hours_since_first_fire", "is_fire", "sat_fraction",
    "u10", "v10", "t2m_K", "d2m_K", "blh_m", "u850", "v850",
    "wind_speed_10m", "wind_dir_10m", "wind_speed_850",
    "vpd_kPa", "precip_1h_mm", "precip_24h_mm", "delta_wind_3h",
]

# ---- frozen PRT-Full architecture constants (v8/v11/e28, byte-identical) ----
HIDDEN_DIM, N_HEADS, DROPOUT, N_LSTM_LAYERS = 96, 4, 0.1, 2
RESIDUAL_SCALE_MAX = 0.5

TEMPORAL_FEATS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
LOG1P_FEATS = {"frp_lag_1h", "frp_lag_2h", "frp_lag_3h", "frp_lag_6h",
               "frp_lag_12h", "frp_lag_24h", "frp_rolling_6h_mean",
               "frp_rolling_6h_std", "cumulative_frp"}
SIGNED_FEATS = {"frp_trend_6h"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------- window construction (verbatim logic from build_dataset_v6.py) ----------
def _rolling_trend(arr, window=6):
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    x = np.arange(window, dtype=np.float64)
    for i in range(window - 1, n):
        y = arr[i - window + 1: i + 1]
        if np.any(np.isnan(y)):
            continue
        out[i] = np.polyfit(x, y, 1)[0]
    return out


def load_event(event_name):
    frp = pd.read_csv(PROC / f"{event_name}_hourly.csv", parse_dates=["slot_utc"])
    era5 = pd.read_csv(PROC / f"{event_name}_era5_hourly.csv", parse_dates=["slot_utc"])
    if "tp_m" in era5.columns:
        era5 = era5.drop(columns=["tp_m"])
    merged = frp.merge(era5, on="slot_utc", how="inner")
    merged = merged[merged["slot_type"] != "missing"].copy()
    merged = merged.sort_values("slot_utc").reset_index(drop=True)
    merged["sat_fraction"] = merged["sat_fraction"].fillna(0.0)
    merged["event_name"] = event_name
    return merged


def engineer_features(df):
    ts = df["slot_utc"]
    hour = ts.dt.hour + ts.dt.minute / 60.0
    doy = ts.dt.day_of_year.astype(float)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
    df["log_frp"] = np.log1p(df["frp_sum_mw"])
    df["is_fire"] = (df["slot_type"] == "fire").astype(np.float32)
    grp = df.copy().sort_values("slot_utc").reset_index(drop=True)
    frp_vals = grp["frp_sum_mw"].values.astype(np.float64)
    for lag in [1, 2, 3, 6, 12, 24]:
        grp[f"frp_lag_{lag}h"] = grp["frp_sum_mw"].shift(lag)
    grp["frp_rolling_6h_mean"] = grp["frp_sum_mw"].rolling(6, min_periods=1).mean()
    grp["frp_rolling_6h_std"] = grp["frp_sum_mw"].rolling(6, min_periods=1).std()
    grp["frp_trend_6h"] = _rolling_trend(frp_vals, window=6)
    grp["cumulative_frp"] = grp["frp_sum_mw"].cumsum()
    fire_mask = grp["slot_type"] == "fire"
    if fire_mask.any():
        first_idx = fire_mask.idxmax()
        hs = (grp["slot_utc"] - grp.loc[first_idx, "slot_utc"]).dt.total_seconds() / 3600.0
        hs = hs.clip(lower=0)
    else:
        hs = pd.Series(0.0, index=grp.index)
    grp["hours_since_first_fire"] = hs
    return grp


def build_sequences(grp, feature_cols):
    inputs, targets, meta = [], [], []
    grp = grp.sort_values("slot_utc").reset_index(drop=True)
    n = len(grp)
    fm = grp[feature_cols].values.astype(np.float32)
    frp_raw = grp["frp_sum_mw"].values.astype(np.float32)
    slot_types = grp["slot_type"].values
    timestamps = grp["slot_utc"].values
    for t in range(LOOKBACK - 1, n - HORIZON):
        inp = fm[t - LOOKBACK + 1: t + 1]
        tgt = frp_raw[t + 1: t + HORIZON + 1]
        if np.any(np.isnan(inp)) or np.any(np.isnan(tgt)):
            continue
        if np.any(slot_types[t + 1: t + HORIZON + 1] == "missing"):
            continue
        inputs.append(inp)
        targets.append(tgt)
        meta.append({"input_start_utc": str(timestamps[t - LOOKBACK + 1]),
                     "target_end_utc": str(timestamps[t + HORIZON])})
    return inputs, targets, meta


# ---------- model (verbatim from e28_train_grid.py) ----------
class PRTWithPersistenceSkip(nn.Module):
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


def normalize_T1_robust(raw, feats, params):
    out = raw.astype(np.float32).copy()
    for j, f in enumerate(feats):
        p = params.get(f)
        if p is None:
            continue
        x = out[:, :, j].astype(np.float64)
        t = (np.log1p(np.maximum(x, 0)) if f in LOG1P_FEATS else
             np.sign(x) * np.log1p(np.abs(x)) if f in SIGNED_FEATS else x)
        out[:, :, j] = ((t - p["center"]) / p["scale"]).astype(np.float32)
    return out


def persistence_anchor(raw_inp, FEAT):
    frp = np.clip(raw_inp[:, LAST_TS, FEAT["frp_lag_1h"]], 0, None)
    is_fire = raw_inp[:, LAST_TS, FEAT["is_fire"]]
    return np.where(is_fire > 0, frp, 0.0).astype(np.float32)


def m2_predict(raw_inp, feats, norm_params, device):
    """Verify checkpoint hashes against the freeze table, then log-mean ensemble."""
    frozen = json.load(open(FREEZE))["sha256"]
    FEAT = {n: i for i, n in enumerate(feats)}
    x = torch.tensor(normalize_T1_robust(raw_inp, feats, norm_params))
    pers = torch.tensor(persistence_anchor(raw_inp, FEAT))
    logs = []
    for s in M2_SEEDS:
        ckpt = E28 / f"T1_robust_s{s}_best.pt"
        expect = frozen.get(f"checkpoint_T1_robust_s{s}")
        if expect is None:
            raise RuntimeError(f"checkpoint_T1_robust_s{s} not in freeze table — refusing")
        actual = sha256(ckpt)
        if actual != expect:
            raise RuntimeError(f"HASH MISMATCH for {ckpt.name}: {actual} != {expect}")
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model = PRTWithPersistenceSkip(len(feats), HORIZON).to(device)
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(x), 256):
                preds.append(model(x[i:i + 256].to(device),
                                   pers[i:i + 256].to(device)).cpu().numpy())
        logs.append(np.log1p(np.concatenate(preds, 0)))
        print(f"  M2 member s{s}: hash OK, predicted", flush=True)
    return np.expm1(np.mean(np.stack(logs, 0), axis=0))


def m1_predict(raw_inp, feats, device):
    """M1 (RAT-9 re-scope): inference reconstructed from the Zenodo deposit
    alone — its feature order (asserted), its v3 z-stats, its wind-feature
    module with checkpoint-embedded stats, its PRT class, its log-mean rule.
    DESCRIPTIVE ONLY."""
    assert list(feats) == M1_CANONICAL_FEATS, \
        "feature order differs from deposit canonical order — M1 aborted"
    norm_path = M1_DIR / "data" / "normalization_params_v3.json"
    assert sha256(norm_path) == M1_NORM_SHA256, "M1 norm params hash mismatch"
    norm_params = json.load(open(norm_path))
    # legacy z-score with the deposit's v3 params (identity where no params)
    x = raw_inp.astype(np.float32).copy()
    for j, f in enumerate(feats):
        p = norm_params.get(f)
        if p is not None:
            x[:, :, j] = ((x[:, :, j].astype(np.float64) - p["mean"])
                          / p["std"]).astype(np.float32)
    sys.path.insert(0, str(M1_DIR / "src"))
    from shared.prt_model import PRT                    # noqa: E402
    from shared.wind_features import build_wind_features  # noqa: E402
    frp_mean = float(norm_params["frp_lag_1h"]["mean"])
    frp_std = float(norm_params["frp_lag_1h"]["std"])
    ckpts = sorted((M1_DIR / "models" / "prt_full_ensemble").glob("seed_*.pt"))
    assert len(ckpts) == 10, f"expected 10 M1 checkpoints, found {len(ckpts)}"
    for cp in ckpts:
        assert sha256(cp) == M1_CKPT_SHA256[cp.name], \
            f"M1 HASH MISMATCH for {cp.name}"
    ref = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    x_ext, _ = build_wind_features(x, norm_params, ref["wind_feat_stats"])
    assert x_ext.shape[2] == int(ref["input_size"])
    xt = torch.tensor(x_ext)
    logs = []
    for cp in ckpts:
        ck = torch.load(cp, map_location=device, weights_only=False)
        model = PRT(input_size=x_ext.shape[2], horizon=HORIZON,
                    hidden_dim=96, n_heads=4, dropout=0.1, n_lstm_layers=2,
                    frp_mean=frp_mean, frp_std=frp_std,
                    residual_scale_max=0.5, frp_lag_1h_idx=5).to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(xt), 256):
                preds.append(model(xt[i:i + 256].to(device)).cpu().numpy())
        logs.append(np.log1p(np.clip(np.concatenate(preds, 0), 0, None)))
        print(f"  M1 member {cp.name}: hash OK, predicted", flush=True)
    return np.expm1(np.mean(np.stack(logs, 0), axis=0))


def event_volatility(merged):
    """AMEND-8 frozen descriptor: population std of delta(log1p(frp)) over
    consecutive (exactly 1 h) fire->fire slot pairs of the merged frame."""
    df = merged.sort_values("slot_utc").reset_index(drop=True)
    fire = (df["slot_type"] == "fire").to_numpy()
    lf = np.log1p(df["frp_sum_mw"].to_numpy(dtype=float))
    dt = df["slot_utc"].diff().dt.total_seconds().to_numpy()
    d = [lf[i] - lf[i - 1] for i in range(1, len(df))
         if fire[i] and fire[i - 1] and dt[i] == 3600.0]
    return float(np.std(d, ddof=0)) if len(d) >= 2 else float("nan")


def b6_train_predict(eval_inp, feats):
    """FINAL B6 = gbdt_log_T1 (B-2): retrained deterministically at access."""
    FEAT = {n: i for i, n in enumerate(feats)}
    with h5py.File(H5, "r") as f:
        tr_inp = f["train/inputs"][:]
        tr_tgt = f["train/targets"][:]
    Xt = tr_inp[:, LAST_TS, :]
    yt = np.log1p(np.clip(tr_tgt, 0, None))
    Xe = eval_inp[:, LAST_TS, :]
    out = np.zeros((len(eval_inp), HORIZON), dtype=np.float32)
    for k in range(HORIZON):
        g = HistGradientBoostingRegressor(max_iter=300, early_stopping=True,
                                          validation_fraction=0.1,
                                          random_state=0, **B6_CFG)
        g.fit(Xt, yt[:, k])
        out[:, k] = np.clip(np.expm1(g.predict(Xe)), 0, None)
    return out


# ---------- evaluation core ----------
def evaluate(event_arrays, feats, norm_params, event_flags, out_dir, label,
             event_vol=None, zero_window_events=None):
    """event_arrays: {event: (inputs, targets)}; event_flags: {event:
    {'in_primary': bool, 'd7_restored': bool, 'block': str}};
    event_vol: {event: volatility descriptor V} (AMEND-8)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FEAT = {n: i for i, n in enumerate(feats)}
    events = sorted(event_arrays)
    all_inp = np.concatenate([event_arrays[e][0] for e in events], 0)
    all_tgt = np.concatenate([event_arrays[e][1] for e in events], 0)
    ev_of = np.concatenate([[e] * len(event_arrays[e][0]) for e in events])

    print(f"[{label}] events={len(events)} windows={len(all_inp)}", flush=True)
    print("M2 ensemble inference (hash-verified members):", flush=True)
    p_m2 = m2_predict(all_inp, feats, norm_params, device)
    print("B6 (gbdt_log_T1) deterministic retrain + predict ...", flush=True)
    p_b6 = b6_train_predict(all_inp, feats)
    print("M1 (Zenodo-deposit ensemble, DESCRIPTIVE) inference:", flush=True)
    p_m1 = m1_predict(all_inp, feats, device)
    pers = persistence_anchor(all_inp, FEAT)
    p_b1 = np.tile(pers[:, None], (1, HORIZON))
    # AMEND-8 wind-regime classifier: raw |delta_wind_3h| at last input step
    dw = np.abs(all_inp[:, LAST_TS, FEAT["delta_wind_3h"]])
    excursion = dw > DELTA_WIND_3H_SIGMA

    fire = all_tgt.max(axis=1) > 0
    rows = []
    for ev in events:
        m = ev_of == ev
        fm = m & fire
        n_fire = int(fm.sum())
        r = {"event": ev, "n_windows": int(m.sum()), "n_fire": n_fire,
             "evaluable": n_fire >= N_MIN_FIRE, **event_flags[ev]}
        if event_vol is not None:
            v = event_vol.get(ev, float("nan"))
            r["volatility_V"] = v
            r["vol_tertile"] = ("low" if v <= VOL_TERTILE_LO else
                                "mid" if v <= VOL_TERTILE_HI else
                                "high") if not np.isnan(v) else "na"
        for name, p in [("M2", p_m2), ("B6", p_b6), ("M1", p_m1), ("B1", p_b1)]:
            for k in STRATA_HORIZONS:
                r[f"MAE_{name}_t{k}"] = (float(np.mean(np.abs(
                    p[fm][:, k - 1] - all_tgt[fm][:, k - 1])))
                    if n_fire else float("nan"))
        for k in STRATA_HORIZONS:
            pmk = r[f"MAE_B1_t{k}"]
            for name in ["M2", "B6", "M1"]:
                r[f"SS_{name}_t{k}"] = ((1.0 - r[f"MAE_{name}_t{k}"] / pmk)
                                        if (n_fire and pmk and not np.isnan(pmk))
                                        else float("nan"))
        # wind-regime per-event stratum values (AMEND-8 b)
        for rname, rmask in [("calm", ~excursion), ("exc", excursion)]:
            sm = fm & rmask
            n_s = int(sm.sum())
            r[f"n_fire_{rname}"] = n_s
            if n_s:
                mae_b1 = float(np.mean(np.abs(p_b1[sm][:, 5] - all_tgt[sm][:, 5])))
                for name, p in [("M2", p_m2), ("B6", p_b6)]:
                    mae = float(np.mean(np.abs(p[sm][:, 5] - all_tgt[sm][:, 5])))
                    r[f"SS_{name}_t6_{rname}"] = ((1.0 - mae / mae_b1)
                                                  if mae_b1 else float("nan"))
            else:
                r[f"SS_M2_t6_{rname}"] = float("nan")
                r[f"SS_B6_t6_{rname}"] = float("nan")
        rows.append(r)
    # IA-A5: gate-passing events with zero valid windows still get table rows
    for ev in (zero_window_events or []):
        r = {"event": ev, "n_windows": 0, "n_fire": 0, "evaluable": False,
             **event_flags[ev]}
        if event_vol is not None:
            v = event_vol.get(ev, float("nan"))
            r["volatility_V"] = v
            r["vol_tertile"] = "na"
        rows.append(r)
    df = pd.DataFrame(rows)

    # ---- AMEND-8 strata table (T-1: win counts + median deltaSS ONLY; T-3:
    # one file) ----
    def stratum_agg(dss):
        dss = [d for d in dss if not np.isnan(d)]
        if not dss:
            return {"n_events": 0, "wins": 0, "losses": 0,
                    "median_dSS": float("nan")}
        return {"n_events": len(dss),
                "wins": int(sum(d > 0 for d in dss)),
                "losses": int(sum(d < 0 for d in dss)),
                "median_dSS": float(np.median(dss))}

    elig_df = df[df["evaluable"]]
    strata = {"_meta": {
        "amendment": "AMEND-8 (memo A6, T-1..T-4)",
        "note": ("DESCRIPTIVE ONLY — paired win counts and median deltaSS "
                 "(M2 - B6); no p-values, no CIs, no promotion to "
                 "confirmatory status under any outcome (AMEND-7)"),
        "frozen_constants": {
            "delta_wind_3h_sigma_train": DELTA_WIND_3H_SIGMA,
            "vol_tertile_lo": VOL_TERTILE_LO,
            "vol_tertile_hi": VOL_TERTILE_HI,
            "horizons": STRATA_HORIZONS,
            "volatility_descriptor": ("population std of delta(log1p("
                "frp_sum_mw)) over consecutive 1h fire->fire slot pairs")}}}
    strata["horizon_profile"] = {
        f"t+{k}": stratum_agg((elig_df[f"SS_M2_t{k}"]
                               - elig_df[f"SS_B6_t{k}"]).tolist())
        for k in STRATA_HORIZONS}
    strata["wind_regime_t6"] = {
        rname: dict(stratum_agg((elig_df[f"SS_M2_t6_{rname}"]
                                 - elig_df[f"SS_B6_t6_{rname}"]).tolist()),
                    n_fire_windows=int(elig_df[f"n_fire_{rname}"].sum()))
        for rname in ["calm", "exc"]}
    if event_vol is not None:
        strata["volatility_tertiles_t6"] = {
            t: stratum_agg((sub["SS_M2_t6"] - sub["SS_B6_t6"]).tolist())
            for t, sub in elig_df.groupby("vol_tertile") if t != "na"}

    def sign_test(sub, col_a="MAE_M2_t6", col_b="MAE_B6_t6"):
        paired = sub[[col_a, col_b]].dropna()          # IA-N5: paired frame
        d = paired[col_b] - paired[col_a]              # >0 = M2 wins (lower MAE)
        wins, losses = int((d > 0).sum()), int((d < 0).sum())
        n = wins + losses                      # ties dropped
        if n == 0:
            return {"wins": 0, "losses": 0, "n": 0, "p_one_sided": float("nan")}
        p = binomtest(wins, n, 0.5, alternative="greater").pvalue
        try:
            w = wilcoxon(paired[col_b], paired[col_a],
                         alternative="greater").pvalue
        except Exception:
            w = float("nan")
        return {"wins": wins, "losses": losses, "n": n,
                "p_one_sided": float(p), "reject_at_005": bool(p < ALPHA),
                "wilcoxon_p": float(w)}

    elig = df[df["evaluable"]]
    prim = elig[elig["in_primary"]]
    sens = elig[elig["in_sensitivity"]]        # IA-B1: pre-registered set only
    res = {
        "label": label,
        "counts": {"gate_passing": len(df), "evaluable": len(elig),
                   "primary_evaluable": len(prim),
                   "sensitivity_evaluable": len(sens), "N_min": N_MIN_FIRE},
        "PRIMARY_sign_test_M2_vs_B6_primary_set": sign_test(prim),
        "SECONDARY_sign_test_M2_vs_B6_sensitivity_set": sign_test(sens),
        "SECONDARY_sign_test_M2_vs_B1_primary_set": sign_test(prim, "MAE_M2_t6", "MAE_B1_t6"),
        "AGG1_headline": {
            "M2_median_SS_t6": float(np.nanmedian(elig["SS_M2_t6"])),
            "B6_median_SS_t6": float(np.nanmedian(elig["SS_B6_t6"])),
            "M2_mean_SS_t6": float(np.nanmean(elig["SS_M2_t6"])),
            "M2_std_SS_t6": float(np.nanstd(elig["SS_M2_t6"], ddof=1)),
        },
        "M1_DESCRIPTIVE": {
            "note": ("Zenodo-deposit ensemble (RAT-9 re-scope); DESCRIPTIVE "
                     "ONLY — no test involves M1"),
            "M1_median_SS_t6": float(np.nanmedian(elig["SS_M1_t6"])),
            "M1_mean_SS_t6": float(np.nanmean(elig["SS_M1_t6"])),
        },
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "b6_definition": {"model": "HistGradientBoostingRegressor log1p target",
                          "cfg": B6_CFG, "train_pool": "dataset_v6_1 full train (T1)"},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{label}_per_event.csv", index=False)
    with open(out_dir / f"{label}_results.json", "w") as f:
        json.dump(res, f, indent=2)
    with open(out_dir / f"{label}_strata.json", "w") as f:
        json.dump(strata, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "b6_definition"},
                     indent=2), flush=True)
    print("AMEND-8 strata table:", flush=True)
    print(json.dumps({k: v for k, v in strata.items() if k != "_meta"},
                     indent=2), flush=True)
    return res


# ---------- modes ----------
def dry_run_val():
    """Rebuild val windows from CSVs, assert equality with frozen H5, evaluate."""
    with h5py.File(H5, "r") as f:
        meta = json.loads(f["metadata"][()])
        feats = meta["feature_names"]
        norm = meta["normalization_params"]["T1"]["robust"]
        va_inp = f["val/inputs"][:]
        va_tgt = f["val/targets"][:]
    man = pd.read_csv(PROC / "dataset_v6_1_manifest.csv")
    va_man = man[man["split"] == "val"].reset_index(drop=True)
    events = sorted(va_man["event_name"].unique())

    arrays, flags, vols = {}, {}, {}
    print("Rebuilding val windows from CSVs (access code path) ...", flush=True)
    for ev in events:
        merged = load_event(ev)
        vols[ev] = event_volatility(merged)
        grp = engineer_features(merged)
        inp, tgt, wmeta = build_sequences(grp, feats)
        idx = va_man.index[va_man["event_name"] == ev].to_numpy()
        h5_inp, h5_tgt = va_inp[idx], va_tgt[idx]
        assert len(inp) == len(idx), \
            f"{ev}: rebuilt {len(inp)} windows vs H5 {len(idx)}"
        # align by input_start (H5 order within event is chronological)
        order = np.argsort([m["input_start_utc"] for m in wmeta])
        h5_order = np.argsort(va_man.loc[idx, "input_start_utc"].to_numpy())
        ri = np.stack(inp)[order]
        rt = np.stack(tgt)[order]
        assert np.allclose(ri, h5_inp[h5_order], atol=1e-4, equal_nan=True), \
            f"{ev}: rebuilt inputs differ from frozen H5"
        assert np.allclose(rt, h5_tgt[h5_order], atol=1e-4), \
            f"{ev}: rebuilt targets differ from frozen H5"
        arrays[ev] = (ri, rt)
        flags[ev] = {"in_primary": True, "in_sensitivity": True,
                     "d7_restored": False, "block": "VAL"}
        print(f"  {ev}: {len(inp)} windows — MATCH frozen H5", flush=True)
    print("All val events rebuilt byte-consistent with dataset_v6_1.h5.\n", flush=True)
    evaluate(arrays, feats, norm, flags, ROOT / "resubmission" / "access_dryrun_val",
             "dryrun_val", event_vol=vols)


def access(event_list_csv):
    with h5py.File(H5, "r") as f:
        meta = json.loads(f["metadata"][()])
        feats = meta["feature_names"]
        norm = meta["normalization_params"]["T1"]["robust"]
    lst = pd.read_csv(event_list_csv)
    need = {"event_name", "block", "in_primary", "in_sensitivity", "d7_restored"}
    assert need.issubset(lst.columns), f"event list must have columns {need}"

    # IA-B1.3: the supplied list's block-A subset must be row-identical to the
    # frozen composition inside the lock's hash perimeter.
    frozen = pd.read_csv(BLOCKA_LOCKED_CSV)
    cols = ["event_name", "block", "in_primary", "in_sensitivity", "d7_restored"]
    supplied_a = (lst[lst["block"] == "A2025"][cols]
                  .astype(str).sort_values("event_name").reset_index(drop=True))
    frozen_a = (frozen[cols].astype(str).sort_values("event_name")
                .reset_index(drop=True))
    if not supplied_a.equals(frozen_a):
        raise RuntimeError(
            "BLOCK-A COMPOSITION VIOLATION: the supplied event list's block-A "
            "subset is not row-identical to the frozen "
            "test_event_list_blockA_LOCKED.csv — refusing to run")
    print(f"block-A composition OK: {len(frozen_a)} rows match frozen list",
          flush=True)

    arrays, flags, vols = {}, {}, {}
    zero_window_events = []
    for _, r in lst.iterrows():
        ev = r["event_name"]
        merged = load_event(ev)
        vols[ev] = event_volatility(merged)
        grp = engineer_features(merged)
        inp, tgt, _ = build_sequences(grp, feats)
        if not inp:
            print(f"  {ev}: 0 valid windows — reported as a zero-window row "
                  f"(IA-A5)", flush=True)
            zero_window_events.append(ev)
        arrays[ev] = (np.stack(inp) if inp else np.zeros((0, LOOKBACK, len(feats)), np.float32),
                      np.stack(tgt) if tgt else np.zeros((0, HORIZON), np.float32))
        flags[ev] = {"in_primary": bool(r["in_primary"]),
                     "in_sensitivity": bool(r["in_sensitivity"]),
                     "d7_restored": bool(r["d7_restored"]), "block": r["block"]}
    arrays = {e: a for e, a in arrays.items() if len(a[0])}
    out = ROOT / "resubmission" / "TEST_ACCESS_RESULTS"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    res = evaluate(arrays, feats, norm, flags, out, f"access_{stamp}",
                   event_vol=vols, zero_window_events=zero_window_events)
    with open(out / f"access_record_{stamp}.json", "w") as f:
        json.dump({"event_list": str(event_list_csv),
                   "event_list_sha256": sha256(event_list_csv),
                   "script_declared": "test_access_v1.py",
                   "utc": stamp}, f, indent=2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run-val", action="store_true")
    ap.add_argument("--access", metavar="EVENT_LIST_CSV")
    ap.add_argument("--confirm-single-access", action="store_true")
    a = ap.parse_args()
    assert_environment()
    assert_input_hashes(dry_run=a.dry_run_val)
    if a.dry_run_val:
        dry_run_val()
    elif a.access:
        if not a.confirm_single_access:
            print("REFUSED: --access requires --confirm-single-access.\n"
                  "This is THE single pre-registered test access (ACC-1). "
                  "Run it once, after the prereg lock and season close.")
            sys.exit(2)
        access(Path(a.access))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
