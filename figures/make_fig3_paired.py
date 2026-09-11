"""Fig. 3 v6 -- two panels, VECTOR PDF. TEXT RELABEL of v5 to the GRSL letter v4 naming.

Same data, geometry (3.5 x 2.25 in, same axes rectangles), colours, markers, sizes and fonts
as make_fig3_paired_v5.py. Text changes only:
  (a) legend: "A primary (29)" -> "2025 primary (29)"; "A sensitivity (2)" -> "2025 sensitivity (2)";
              "A descriptive (14)" -> "2025 descriptive (14)"; "B 2026 (32)" -> "2026 (32)"
  (a) axes:   "B6 SS_fire,t6" -> "Boosting SS_fire,t6"; "M2 SS_fire,t6" -> "PRT SS_fire,t6"
  (b) series: "B6" -> "Boosting"; "M2" -> "PRT"; "A1-A6" -> "six encoders"
Kept: the three fire-name annotations, "persistence" on the dotted zero, "lead (h)",
"median SS_fire, 77 fires", the panel letters, the marker and colour encodings.

Forced by the longer strings: the three-entry legend of v5 ("Boosting / PRT / six encoders",
52 pt wide) has no position inside the 59 pt panel (b) that does not cover a line or marker
(measured over a full anchor grid), so panel (b) carries a two-entry legend (Boosting, PRT)
in the free lower-right wedge and a direct grey label "six encoders" in the free upper-left
wedge. Both positions, and the panel-(a) legend, are asserted clear of every line, marker
and point by measurement at run time.

Inputs (deposited outputs only; hashes printed):
  TEST_ACCESS_RESULTS/access_20260828T123059Z_per_event.csv (locked; hash asserted)
  EXPLORATORY_BLOCKB_2026/exploratory_blockB_20260908T140106Z_per_event.csv
  EXPLORATORY_ARCH_2026/reference_models_per_event.csv (cross-check)
  EXPLORATORY_ARCH_2026/master_results.json (pooled horizon medians, all models)
Output: manuscript/figures/fig3_paired_v6.pdf (+ PNG). v5 files are not touched.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
A_CSV = ROOT / "resubmission" / "TEST_ACCESS_RESULTS" / "access_20260828T123059Z_per_event.csv"
B_CSV = ROOT / "resubmission" / "EXPLORATORY_BLOCKB_2026" / "exploratory_blockB_20260908T140106Z_per_event.csv"
REF_CSV = ROOT / "resubmission" / "EXPLORATORY_ARCH_2026" / "reference_models_per_event.csv"
MASTER = ROOT / "resubmission" / "EXPLORATORY_ARCH_2026" / "master_results.json"
STEM = "fig3_paired_v6"
A_LOCKED_SHA = "2c1dd44498b3fe76be7005b493eb0d2424c8347d601efcca1a505418fceb5e79"

FS_LABEL, FS_TICK, FS_SMALL = 8, 8, 7
MIN_CLEAR_PT = 2.0                 # required clearance between any label/legend and the data
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "STIX", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": FS_LABEL,
    "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_SMALL,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
C_A, C_DESC, C_B, C_IDENT = "#0072B2", "#999999", "#D55E00", "#444444"
C_B6, C_ENC = "#111111", "#B0B0B0"
LABELS = {"pt_cambra_e_carvalhal_2026_0702": "cambra", "fr_porge_2026_0722": "porge",
          "pt_cerdeira_2026_0727": "cerdeira"}
LEADS = [1, 3, 6, 12]
ENCODERS = ["A1", "A2", "A3", "A4", "A5", "A6"]   # internal run keys of master_results.json; never drawn


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def clearance_pt(bb, samples, scale):
    """min distance (pt) from any sample (x, y, radius; display units) to rectangle bb; -1 if inside"""
    dx = np.maximum(np.maximum(bb.x0 - samples[:, 0], samples[:, 0] - bb.x1), 0)
    dy = np.maximum(np.maximum(bb.y0 - samples[:, 1], samples[:, 1] - bb.y1), 0)
    d = np.hypot(dx, dy) - samples[:, 2]
    inside = (samples[:, 0] > bb.x0) & (samples[:, 0] < bb.x1) & (samples[:, 1] > bb.y0) & (samples[:, 1] < bb.y1)
    d[inside] = -1.0
    return float(d.min() * scale)


def overlaps(a, b):
    return not (a.x1 < b.x0 or a.x0 > b.x1 or a.y1 < b.y0 or a.y0 > b.y1)


def main():
    assert sha(A_CSV) == A_LOCKED_SHA, "locked 2025-block table hash mismatch"
    A, B = pd.read_csv(A_CSV), pd.read_csv(B_CSV)
    evA, evB = A[A.evaluable == True].copy(), B[B.evaluable == True].copy()
    assert len(evA) == 45 and len(evB) == 32
    prim = evA[evA.in_primary == True]
    sens = evA[(evA.in_sensitivity == True) & (evA.in_primary == False)]
    desc = evA[evA.in_sensitivity == False]
    assert (len(prim), len(sens), len(desc)) == (29, 2, 14)
    ref = pd.read_csv(REF_CSV).set_index("event")
    both = pd.concat([evA, evB]).set_index("event")
    assert len(both) == 77 and set(both.index) == set(ref.index)
    for c in ("SS_M2_t6", "SS_B6_t6"):
        assert np.allclose(both[c], ref.loc[both.index, c], atol=1e-6), c
    n_above = int((both.SS_M2_t6 > both.SS_B6_t6).sum())
    assert n_above == 43, n_above
    hm = json.load(open(MASTER))["stats"]
    med = {m: [hm[m]["pooled"]["horizon_medians"][f"t{k}"] for k in LEADS] for m in ENCODERS + ["M2", "B6"]}
    assert abs(med["M2"][2] - float(np.median(both.SS_M2_t6))) < 1e-9
    assert abs(med["B6"][2] - float(np.median(both.SS_B6_t6))) < 1e-9

    FIG_W, FIG_H = 3.5, 2.25
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ah = 0.76
    aw = ah * FIG_H / FIG_W                      # square panel (a) in figure fractions
    ax = fig.add_axes([0.125, 0.16, aw, ah])
    bx = fig.add_axes([0.755, 0.16, 0.235, ah])

    # ---- (a) paired scatter
    lo = min(both.SS_B6_t6.min(), both.SS_M2_t6.min())
    hi = max(both.SS_B6_t6.max(), both.SS_M2_t6.max())
    pad = 0.06 * (hi - lo)
    lim = (lo - pad, hi + pad)
    ax.plot(lim, lim, color=C_IDENT, lw=0.7, ls="--", zorder=1)
    ax.axhline(0, color="#CCCCCC", lw=0.5, zorder=0)
    ax.axvline(0, color="#CCCCCC", lw=0.5, zorder=0)
    ax.scatter(desc.SS_B6_t6, desc.SS_M2_t6, s=14, marker="o", facecolors="none", edgecolors=C_DESC,
               linewidths=0.75, zorder=2)
    ax.scatter(evB.SS_B6_t6, evB.SS_M2_t6, s=17, marker="^", facecolors="white", edgecolors=C_B,
               linewidths=0.75, zorder=3)
    ax.scatter(sens.SS_B6_t6, sens.SS_M2_t6, s=18, marker="s", facecolors="white", edgecolors=C_A,
               linewidths=0.9, zorder=4)
    ax.scatter(prim.SS_B6_t6, prim.SS_M2_t6, s=14, marker="o", facecolors=C_A, edgecolors="white",
               linewidths=0.35, zorder=5)
    for ev, short in LABELS.items():
        r_ = evB.set_index("event").loc[ev]
        ax.annotate(short, (r_.SS_B6_t6, r_.SS_M2_t6), xytext=(3.5, -0.5), textcoords="offset points",
                    fontsize=FS_SMALL, color=C_B, ha="left", va="center", style="italic", zorder=6)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-0.4, 0, 0.4, 0.8]); ax.set_yticks([-0.4, 0, 0.4, 0.8])
    ax.set_xlabel("Boosting $\\mathrm{SS}_{\\mathrm{fire},t6}$", labelpad=1.0)
    ax.set_ylabel("PRT $\\mathrm{SS}_{\\mathrm{fire},t6}$", labelpad=1.0)
    ax.tick_params(length=2, width=0.5, pad=1.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles = [
        Line2D([], [], marker="o", ls="", color=C_A, markersize=3.8, label=f"2025 primary ({len(prim)})"),
        Line2D([], [], marker="s", ls="", markerfacecolor="white", markeredgecolor=C_A, markersize=4.2,
               label=f"2025 sensitivity ({len(sens)})"),
        Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=C_DESC, markersize=3.8,
               label=f"2025 descriptive ({len(desc)})"),
        Line2D([], [], marker="^", ls="", markerfacecolor="white", markeredgecolor=C_B, markersize=4.2,
               label=f"2026 ({len(evB)})"),
    ]
    leg_a = ax.legend(handles=handles, loc="upper left", frameon=False, handletextpad=0.25, borderaxespad=0.05,
                      labelspacing=0.15, handlelength=0.9)
    fig.text(0.012, 0.975, "(a)", fontsize=FS_LABEL, fontweight="bold", va="top", ha="left")
    fig.text(0.655, 0.975, "(b)", fontsize=FS_LABEL, fontweight="bold", va="top", ha="left")

    # ---- (b) median SS by lead
    x = np.arange(len(LEADS))
    bx.axhline(0, color="#777777", lw=0.6, ls=":", zorder=1)
    for m in ENCODERS:
        bx.plot(x, med[m], color=C_ENC, lw=0.7, zorder=2)
    bx.plot(x, med["M2"], color=C_A, lw=1.0, marker="o", markersize=3.4, zorder=4)
    bx.plot(x, med["B6"], color=C_B6, lw=1.0, marker="s", markersize=3.2, zorder=4)
    bx.set_xticks(x); bx.set_xticklabels([str(k) for k in LEADS])
    bx.set_xlim(-0.3, len(LEADS) - 0.7)
    bx.set_ylim(-0.065, 0.42)
    bx.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    bx.set_xlabel("lead (h)", labelpad=1.0)
    bx.set_ylabel("median $\\mathrm{SS}_{\\mathrm{fire}}$, 77 fires", labelpad=1.0)
    bx.tick_params(length=2, width=0.5, pad=1.2)
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    bh = [
        Line2D([], [], color=C_B6, lw=1.0, marker="s", markersize=3.2, label="Boosting"),
        Line2D([], [], color=C_A, lw=1.0, marker="o", markersize=3.4, label="PRT"),
    ]
    leg_b = bx.legend(handles=bh, loc="lower left", bbox_to_anchor=(0.34, 0.14), frameon=False,
                      handletextpad=0.35, handlelength=1.0, labelspacing=0.15, borderaxespad=0.0)
    lab_enc = bx.text(-0.2, 0.39, "six encoders", fontsize=FS_SMALL, color="#777777", ha="left", va="center")
    lab_pers = bx.annotate("persistence", (len(LEADS) - 1, 0), xytext=(0, -1.5), textcoords="offset points",
                           fontsize=FS_SMALL, color="#777777", ha="right", va="top")

    # ---- clearance asserts by measurement (display units -> pt)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    scale = 72.0 / fig.dpi
    # (a): no point marker within 1.5 pt of any legend ink (texts and handles; the padded legend
    # rectangle itself is not the test, its empty corners are not ink)
    pts = np.array([ax.transData.transform((bx_, by_)) for bx_, by_ in zip(both.SS_B6_t6, both.SS_M2_t6)])
    handles_a = getattr(leg_a, "legend_handles", None) or leg_a.legendHandles
    inks = [t.get_window_extent(rend) for t in leg_a.get_texts()] + [h.get_window_extent(rend) for h in handles_a]
    m = (1.5 + 2.1) / scale                      # clearance + largest point-marker radius (s=18 -> 2.1 pt)
    hit = np.zeros(len(pts), dtype=bool)
    for bb in inks:
        hit |= (pts[:, 0] > bb.x0 - m) & (pts[:, 0] < bb.x1 + m) & (pts[:, 1] > bb.y0 - m) & (pts[:, 1] < bb.y1 + m)
    if hit.any():
        detail = [(ev, round(float(both.SS_B6_t6[ev]), 3), round(float(both.SS_M2_t6[ev]), 3)) for ev in both.index[hit]]
        raise AssertionError(f"(a) legend ink within 1.5 pt of points {detail}")
    # (b): series samples (segments + markers) vs legend and labels
    samples = []
    for mkey in ENCODERS + ["M2", "B6"]:
        ys = med[mkey]
        for i in range(len(LEADS) - 1):
            for t in np.linspace(0, 1, 80):
                px, py = bx.transData.transform((x[i] + t, ys[i] + t * (ys[i + 1] - ys[i])))
                samples.append((px, py, (0.35 if mkey in ENCODERS else 0.5) / scale))
        if mkey in ("M2", "B6"):
            rad = (3.4 if mkey == "M2" else 3.2) / 2 / scale
            for i in range(len(LEADS)):
                px, py = bx.transData.transform((x[i], ys[i]))
                samples.append((px, py, rad))
    samples = np.array(samples)
    axbb = bx.get_window_extent(rend)
    clear = {}
    for name, art in (("(b) legend", leg_b), ("six encoders label", lab_enc)):
        bb = art.get_window_extent(rend)
        clear[name] = clearance_pt(bb, samples, scale)
        assert clear[name] >= MIN_CLEAR_PT, f"{name} too close to data: {clear[name]:.2f} pt"
        assert bb.x1 <= axbb.x1 + 2.0 / scale and bb.y1 <= axbb.y1 + 2.0 / scale, f"{name} outside panel"
        assert not overlaps(bb, lab_pers.get_window_extent(rend)), f"{name} overlaps 'persistence'"
    assert not overlaps(leg_b.get_window_extent(rend), lab_enc.get_window_extent(rend))

    pdf = HERE / f"{STEM}.pdf"
    png = HERE / f"{STEM}_preview.png"
    fig.savefig(pdf, format="pdf")
    fig.savefig(png, format="png", dpi=400)
    print(f"Wrote {pdf}\nWrote {png}")
    print("PROVENANCE:")
    for lab, p in (("2025-block per_event (locked)", A_CSV), ("2026-block per_event", B_CSV),
                   ("reference table", REF_CSV), ("master_results", MASTER), ("fig pdf", pdf)):
        print(f"  {lab:30s} sha256 {sha(p)}")
    print(f"  (a) points: primary {len(prim)}, sensitivity-only {len(sens)}, descriptive-only {len(desc)}, "
          f"2026 block {len(evB)}; total 77; PRT above identity {n_above}")
    names = {"B6": "boosting", "M2": "PRT", "A1": "GRU", "A2": "LSTM", "A3": "dilated CNN", "A4": "MLP-Mixer",
             "A5": "PRT without LSTM", "A6": "PatchTST"}
    for mkey in ["B6", "M2"] + ENCODERS:
        print(f"  (b) {names[mkey]:17s}: " + ", ".join(f"t+{k} {v:+.3f}" for k, v in zip(LEADS, med[mkey])))
    print("  clearances: " + ", ".join(f"{k} {v:.1f} pt" for k, v in clear.items()))
    print(f"  page {FIG_W} x {FIG_H} in; small text {FS_SMALL} pt, ticks {FS_TICK} pt, labels {FS_LABEL} pt")


if __name__ == "__main__":
    main()
