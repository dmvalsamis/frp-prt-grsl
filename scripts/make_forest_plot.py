"""
Row 2 deep-ensemble trajectory grid for the GRSL letter (Fig.~\\ref{fig:forest_plot}).

Filename `forest_plot.{pdf,png}` is kept for compatibility with the existing
`\\label{fig:forest_plot}` in section_IV.tex. The figure content (previously
a three-configuration summary slope chart) is replaced with a 2x2 grid of
Row 2 deep-ensemble forecast trajectories on the four held-out test events:
(a) Volos 2023, (b) Varnavas 2024, (c) Val Maior 2024, (d) Corinth 2024.

Per panel:
  - Observed FRP (black solid + markers)
  - Naive-persistence baseline (gray dashed; constant = mean(FRP_t0) over
    fire-active sequences in the event)
  - Row 2 deep-ensemble mean prediction (navy solid + markers)
  - Inter-seed Q25-Q75 band (navy fill_between, alpha 0.20)

Per-event arrays are aggregated as the mean across fire-active sequences
in the event (same set used to compute SS_fire_t6). This matches the
event-level grain of the headline metric.

A trajectory cache is written/read next to the figure so re-runs do not
require re-running the 10 model checkpoints.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Repo-relative. Figures (and the trajectory cache) live in <repo>/figures.
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "figures"
CKPT_DIR = REPO_ROOT / "models" / "prt_full_ensemble"  # only used on a cache miss
CACHE_PATH = OUT_DIR / "forest_plot_trajectories.json"  # shipped; no test access needed

TEST_EVENTS = ["volos_2023", "varnavas_2024", "valmaior_2024", "corinthos_2024"]

# Per-event metadata locked from Table I (per_event_SS in stage2_test_ensemble.json)
EVENTS_META = {
    "volos_2023":     {"label": "(a) Volos, 2023",     "n_fire": 89, "ss_row2": 0.4986},
    "varnavas_2024":  {"label": "(b) Varnavas, 2024",  "n_fire":  7, "ss_row2": 0.2856},
    "valmaior_2024":  {"label": "(c) Val Maior, 2024", "n_fire": 69, "ss_row2": 0.1964},
    "corinthos_2024": {"label": "(d) Corinth, 2024",   "n_fire": 21, "ss_row2": 0.2184},
}

# ----------------------------------------------------------------------------
# Data assembly (re-uses the locked Row 2 checkpoints; caches output)
# ----------------------------------------------------------------------------
def _compute_trajectories():
    """Run the 10 Row 2 seeds on the test split, aggregate per event."""
    import torch
    sys.path.insert(0, str(REPO_ROOT / "src"))  # repo src/ holds the shared/ package
    from shared.dataset_io import (  # noqa: E402
        FRP_LAG_1H_IDX, get_frp_norm, load_manifest, load_metadata,
        load_norm_params, load_split,
    )
    from shared.eval_harness import (  # noqa: E402
        make_persistence_pred, predict_split,
    )
    from shared.prt_model import PRT  # noqa: E402
    from shared.wind_features import build_wind_features  # noqa: E402

    meta = load_metadata()
    HORIZON = int(meta["HORIZON"])
    frp_mean, frp_std = get_frp_norm()

    # Test seal lift authorized for the locked Stage 2 evaluation pass; this
    # script consumes the same predictions and adds no new test-set access.
    test_inputs_raw, test_targets = load_split("test", unsealed=True)
    test_manifest = load_manifest("test")

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

    per_seed_preds = []  # raw MW, shape (S, N, H)
    for cp in ckpt_paths:
        ck = torch.load(cp, map_location=device, weights_only=False)
        model = PRT(
            input_size=INPUT_SIZE, horizon=HORIZON,
            hidden_dim=96, n_heads=4, dropout=0.1, n_lstm_layers=2,
            frp_mean=frp_mean, frp_std=frp_std,
            residual_scale_max=0.5, frp_lag_1h_idx=FRP_LAG_1H_IDX,
        ).to(device)
        model.load_state_dict(ck["state_dict"])
        preds = predict_split(model, test_inputs, device=device)
        per_seed_preds.append(preds.astype(np.float32))
    per_seed_preds = np.stack(per_seed_preds, axis=0)

    pers_pred = make_persistence_pred(
        test_inputs, frp_mean, frp_std, FRP_LAG_1H_IDX, HORIZON,
    )

    ev_arr = test_manifest["event_name"].values
    fire = (test_targets > 0).any(axis=1)

    out = {}
    for ev in TEST_EVENTS:
        m = (ev_arr == ev) & fire
        n_fire = int(m.sum())
        if n_fire == 0:
            raise RuntimeError(f"No fire-active sequences for {ev}")
        observed   = test_targets[m].mean(axis=0)          # (H,)
        persistence = pers_pred[m].mean(axis=0)            # (H,)
        seeds_per_ev = per_seed_preds[:, m, :].mean(axis=1)  # (S, H)
        out[ev] = {
            "n_fire_resolved": n_fire,
            "observed":    observed.tolist(),
            "persistence": persistence.tolist(),
            "row2_seeds":  seeds_per_ev.tolist(),
        }
    return out


def _load_trajectories():
    """Load cached trajectories, or compute and cache them.

    The shipped cache ``figures/forest_plot_trajectories.json`` is the
    authoritative, verified source for Fig. 3 and regenerates the figure with
    no data or test-set access. Recomputation (which requires the licensed
    ``dataset_v3.h5`` test split) only runs if the cache is missing or the
    env var ``FRP_FORCE_RECOMPUTE`` is set.
    """
    force = bool(os.environ.get("FRP_FORCE_RECOMPUTE"))
    if CACHE_PATH.exists() and not force:
        with open(CACHE_PATH) as f:
            return json.load(f)
    print("[forest_plot] cache miss -- running 10 PRT-Full seeds on test split "
          "(requires the licensed dataset_v3.h5 test split)...")
    traj = _compute_trajectories()
    with open(CACHE_PATH, "w") as f:
        json.dump(traj, f, indent=2)
    print(f"[forest_plot] cached: {CACHE_PATH}")
    return traj


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIX", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OBSERVED_COLOR = "black"
PERSISTENCE_COLOR = "#888888"
ROW2_COLOR = "#1f3a93"
ROW2_BAND_ALPHA = 0.20

PANEL_ORDER = [
    ("volos_2023",     (0, 0)),
    ("varnavas_2024",  (0, 1)),
    ("valmaior_2024",  (1, 0)),
    ("corinthos_2024", (1, 1)),
]


def _round_up(y, candidates=None):
    """Round y up to the next clean axis value."""
    candidates = candidates or [50, 75, 100, 150, 200, 250, 300, 400, 500,
                                 600, 800, 1000, 1500, 2000, 3000, 5000]
    for c in candidates:
        if c >= y:
            return c
    # Fall back to next round 1000
    return int(np.ceil(y / 1000.0)) * 1000


def main():
    traj = _load_trajectories()

    MM2IN = 1.0 / 25.4
    fig_w = 180 * MM2IN
    fig_h = 115 * MM2IN

    fig, axes = plt.subplots(
        2, 2, figsize=(fig_w, fig_h), sharex=True,
        gridspec_kw={"hspace": 0.32, "wspace": 0.24},
    )

    leads = np.arange(1, 13)

    legend_handles = None
    for ev, (r, c) in PANEL_ORDER:
        ax = axes[r, c]
        meta = EVENTS_META[ev]
        d = traj[ev]

        observed = np.asarray(d["observed"])
        persistence = np.asarray(d["persistence"])
        seeds = np.asarray(d["row2_seeds"])  # (S, H)
        ens_mean = seeds.mean(axis=0)
        q25 = np.percentile(seeds, 25, axis=0)
        q75 = np.percentile(seeds, 75, axis=0)

        band = ax.fill_between(
            leads, q25, q75,
            color=ROW2_COLOR, alpha=ROW2_BAND_ALPHA, linewidth=0,
            label="PRT-Full inter-seed Q25-Q75", zorder=2,
        )
        line_pers, = ax.plot(
            leads, persistence,
            color=PERSISTENCE_COLOR, linestyle="--", linewidth=1.5,
            label="Persistence", zorder=3,
        )
        line_row2, = ax.plot(
            leads, ens_mean,
            color=ROW2_COLOR, linewidth=2.0,
            marker="s", markersize=4, markerfacecolor=ROW2_COLOR,
            markeredgecolor="white", markeredgewidth=0.5,
            label="PRT-Full (deep ensemble)", zorder=4,
        )
        line_obs, = ax.plot(
            leads, observed,
            color=OBSERVED_COLOR, linewidth=2.0,
            marker="o", markersize=4, markerfacecolor=OBSERVED_COLOR,
            markeredgecolor="white", markeredgewidth=0.5,
            label="Observed", zorder=5,
        )

        if legend_handles is None:
            legend_handles = [line_obs, line_pers, (line_row2, band)]

        # Y-axis: autoscale with clean upper rounding
        ymax_data = float(max(observed.max(), persistence.max(), q75.max()))
        ymax = _round_up(ymax_data * 1.10)
        ax.set_ylim(0, ymax)

        ax.set_xticks([1, 3, 6, 9, 12])
        ax.set_xlim(0.7, 12.3)

        # Faint y-grid only
        ax.grid(axis="y", color="#eeeeee", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Panel title at top-left
        ax.set_title(meta["label"], loc="left", fontsize=9,
                     fontweight="bold", pad=4)

        # Annotation: n_fire + SS in upper-right of panel
        ann = f"$n_{{\\mathrm{{fire}}}}$ = {meta['n_fire']} h\n" \
              f"$\\mathrm{{SS}}_{{\\mathrm{{fire}},t6}}$ = +{meta['ss_row2']:.3f}"
        ax.text(0.97, 0.97, ann, transform=ax.transAxes,
                fontsize=7, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", edgecolor="none", alpha=0.75))

    # Shared axis labels
    for c in (0, 1):
        axes[1, c].set_xlabel("Lead time (hours from forecast initiation)")
    for r in (0, 1):
        axes[r, 0].set_ylabel("FRP (MW)")

    # Single figure legend, below the bottom row
    obs_h, pers_h, row2_combo = legend_handles
    fig.legend(
        handles=[obs_h, pers_h, row2_combo],
        labels=["Observed", "Persistence", "PRT-Full deep-ensemble mean (Q25-Q75 band)"],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.subplots_adjust(left=0.07, right=0.985, top=0.95, bottom=0.13)

    pdf_path = OUT_DIR / "forest_plot.pdf"
    png_path = OUT_DIR / "forest_plot.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png_path, format="png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    print(f"Wrote: {pdf_path}")
    print(f"Wrote: {png_path}")


if __name__ == "__main__":
    main()
