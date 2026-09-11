"""Fig. 1 v6 -- event catalogue (GRSL letter v4), VECTOR PDF. TEXT-ONLY RELABEL of v5.

Identical to make_fig1_catalogue_v5.py in data, seed, geometry, colours, markers, sizes and
fonts; only the legend wording follows the v4 manuscript naming:
  "block A, ignition 2025, pre-registered (n=46)"  ->  "2025 test block, pre-registered (n=46)"
  "block B, ignition 2026, examined (n=32)"        ->  "2026 test block, examined (n=32)"
The first legend entry, axes, ticks and the dashed sequestration lines are unchanged.
The page size is asserted equal to fig1_catalogue_v5.pdf.

Inputs (metadata only -- no forecast-derived quantity):
  resubmission/frozen_dev_pool_2026-08-25.csv                     (ignition_year, peak_frp_mw)
  resubmission/test_event_list_blockA_LOCKED.csv                  (the 46 2025-block names)
  resubmission/EXPLORATORY_BLOCKB_2026/blockB_inventory.csv       (the 32 included 2026-block rows)
  data/processed/event_catalogue.csv                               (start date, peak_frp_mw)

Output: manuscript/figures/fig1_catalogue_v6.pdf (+ PNG preview). v5 files are not touched.
"""
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEV = ROOT / "resubmission" / "frozen_dev_pool_2026-08-25.csv"
BLOCKA = ROOT / "resubmission" / "test_event_list_blockA_LOCKED.csv"
BLOCKB = ROOT / "resubmission" / "EXPLORATORY_BLOCKB_2026" / "blockB_inventory.csv"
CAT = ROOT / "data" / "processed" / "event_catalogue.csv"
STEM = "fig1_catalogue_v6"
PREV = HERE / "fig1_catalogue_v5.pdf"          # geometry reference

FS_LABEL, FS_TICK, FS_LEGEND = 8, 8, 7
PAD_PT = 2.5                      # white border kept on the cropped sides (as v5)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "STIX", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": FS_LABEL,
    "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
C_DEV = "#999999"    # development pool: recessive grey (background population)
C_A = "#0072B2"      # 2025 test block (same hue as in Fig. 3)
C_B = "#D55E00"      # 2026 test block (same hue as in Fig. 3)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    dev = pd.read_csv(DEV)
    assert len(dev) == 84 and dev.ignition_year.max() <= 2024, "dev pool sequestration assert"
    cat = pd.read_csv(CAT)

    names_a = pd.read_csv(BLOCKA).event_name
    assert len(names_a) == 46
    a = cat[cat.event_name.isin(names_a)].copy()
    assert len(a) == 46 and a.peak_frp_mw.notna().all()
    a["ignition_year"] = pd.to_datetime(a.start).dt.year
    assert (a.ignition_year == 2025).all(), "2025 test block must be ignition-year 2025 only"

    inv = pd.read_csv(BLOCKB)
    b = inv[inv.included == True].copy()
    assert len(b) == 32 and (b.gate_verdict == "PASS").all()
    b["ignition_year"] = pd.to_datetime(b.start).dt.year
    assert (b.ignition_year == 2026).all(), "2026 test block must be ignition-year 2026 only"
    assert b.peak_frp_mw.notna().all()

    rng = np.random.default_rng(20260910)  # deterministic within-year jitter (same seed as v3-v5)
    jd = rng.uniform(-0.28, 0.28, len(dev))
    ja = rng.uniform(-0.28, 0.28, len(a))
    jb = rng.uniform(-0.28, 0.28, len(b))
    gw = 1e-3   # MW -> GW

    fig, ax = plt.subplots(figsize=(3.5, 1.85))
    ax.scatter(dev.ignition_year + jd, dev.peak_frp_mw * gw, s=12, marker="o",
               facecolors=C_DEV, edgecolors="white", linewidths=0.3, zorder=3)
    ax.scatter(a.ignition_year + ja, a.peak_frp_mw * gw, s=16, marker="^",
               facecolors=C_A, edgecolors="white", linewidths=0.3, zorder=4)
    ax.scatter(b.ignition_year + jb, b.peak_frp_mw * gw, s=17, marker="^",
               facecolors="white", edgecolors=C_B, linewidths=0.75, zorder=4)
    for xb in (2024.5, 2025.5):
        ax.axvline(xb, color="#666666", lw=0.6, ls="--", zorder=2)
    ax.set_yscale("log")
    ax.set_xlim(2006.2, 2026.8)
    ax.set_xticks([2007, 2010, 2013, 2016, 2019, 2022, 2025])
    ax.set_yticks([1, 10, 100])
    ax.set_yticklabels(["1", "10", "100"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Ignition year", labelpad=1.5)
    ax.set_ylabel("Peak hourly FRP (GW)", labelpad=2.0)
    ax.tick_params(length=2, width=0.5, pad=1.5)
    ax.tick_params(which="minor", length=1.2, width=0.4)
    ax.grid(axis="y", color="#E6E6E6", lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ymin = min(dev.peak_frp_mw.min(), a.peak_frp_mw.min(), b.peak_frp_mw.min()) * gw
    ymax = max(dev.peak_frp_mw.max(), a.peak_frp_mw.max(), b.peak_frp_mw.max()) * gw
    assert 0.5 < ymin < 1.0 and 100 < ymax < 200, (ymin, ymax)   # 1/10/100 GW ticks span the data
    ax.set_ylim(0.55, ymax * 19.0)   # empty band above the data holds the legend
    handles = [
        Line2D([], [], marker="o", ls="", color=C_DEV, markersize=3.8,
               label=f"development pool, ignition $\\leq$ 2024 ($n={len(dev)}$)"),
        Line2D([], [], marker="^", ls="", color=C_A, markersize=4.2,
               label=f"2025 test block, pre-registered ($n={len(a)}$)"),
        Line2D([], [], marker="^", ls="", markerfacecolor="white", markeredgecolor=C_B,
               markeredgewidth=0.8, markersize=4.4,
               label=f"2026 test block, examined ($n={len(b)}$)"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False,
              handletextpad=0.3, borderaxespad=0.15, labelspacing=0.22)

    # ---- pass 1: the v4 layout; pass 2: crop the dead left and bottom border (as v5)
    fig.subplots_adjust(left=0.125, right=0.99, top=0.985, bottom=0.20)
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())       # drawn extent, inches
    W, H = fig.get_size_inches()
    pad = PAD_PT / 72
    dead_left, dead_bottom = tb.x0 - pad, tb.y0 - pad
    assert dead_left > 0 and dead_bottom > 0, (tb.x0, tb.y0)
    pos = ax.get_position()
    ax_left, ax_bottom = pos.x0 * W - dead_left, pos.y0 * H - dead_bottom
    ax_w, ax_h = pos.width * W + dead_left, pos.height * H   # right edge and top margin unchanged
    H2 = H - dead_bottom
    fig.set_size_inches(W, H2)
    ax.set_position([ax_left / W, ax_bottom / H2, ax_w / W, ax_h / H2])
    fig.canvas.draw()
    tb2 = fig.get_tightbbox(fig.canvas.get_renderer())
    assert abs(tb2.x0 - pad) < 0.5 / 72 and abs(tb2.y0 - pad) < 0.5 / 72, (tb2.x0, tb2.y0)
    assert tb2.x1 <= W + 1e-6 and tb2.y1 <= H2 + 1e-6, "content outside the page"

    pdf = HERE / f"{STEM}.pdf"
    png = HERE / f"{STEM}_preview.png"
    fig.savefig(pdf, format="pdf")
    fig.savefig(png, format="png", dpi=400)

    # geometry guard: same page size as v5 (text-only change)
    import fitz
    r5, r6 = fitz.open(PREV)[0].rect, fitz.open(pdf)[0].rect
    assert abs(r5.width - r6.width) < 0.05 and abs(r5.height - r6.height) < 0.05, (r5, r6)

    print(f"Wrote {pdf}\nWrote {png}")
    print("PROVENANCE:")
    for lab, p in (("frozen_dev_pool", DEV), ("2025-block list", BLOCKA), ("2026-block inventory", BLOCKB),
                   ("event_catalogue", CAT), ("fig pdf", pdf)):
        print(f"  {lab:20s} sha256 {sha(p)}")
    for lab, d in (("dev pool", dev), ("2025 test block", a), ("2026 test block", b)):
        print(f"  {lab}: n={len(d)}, years {d.ignition_year.min()}-{d.ignition_year.max()}, "
              f"peak FRP min/median/max = {d.peak_frp_mw.min():.1f} / {d.peak_frp_mw.median():.1f} / "
              f"{d.peak_frp_mw.max():.1f} MW" + (f", countries {d.country.nunique()}" if "country" in d else ""))
    print(f"  total quality-screened fires drawn: {len(dev) + len(a) + len(b)}")
    print(f"  page {r6.width/72:.3f} x {r6.height/72:.3f} in (v5: {r5.width/72:.3f} x {r5.height/72:.3f} in); "
          f"font floor: legend {FS_LEGEND} pt, ticks {FS_TICK} pt, labels {FS_LABEL} pt")


if __name__ == "__main__":
    main()
