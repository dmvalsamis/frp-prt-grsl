"""Fig. 2 v4 -- persistence-residual framework with an interchangeable encoder, VECTOR PDF.
TEXT RELABEL of v3 to the GRSL letter v4 naming:
  "encoder slot"                                                        -> "interchangeable encoder"
  "slot encoders: PRT (pre-registered), GRU, LSTM, CNN, MLP, PatchTST"  -> "encoders: PRT (pre-registered), GRU, LSTM, CNN, MLP-Mixer, PatchTST"
Everything else is the v3 drawing: the 48 x 32 window with its three channel bands, the
sentence "learns only the residual g(z_t)", the equation, the persistence-prior data path,
the gate alpha in [0, 0.5], the summation, and the output chain. Page 252 x 106 pt
(3.5 x 1.47 in), all text sizes and colours unchanged.

Forced by the longer title (73.2 pt at 7.5 pt against a 68 pt box in v3): the encoder box
is 80 pt wide (64-144 pt instead of 68-136); to keep the 40 pt gate arrow, the sum node
moves from 184 to 192 pt and the output box from 202-248 to 208-248 (40 pt wide); the input
window is 50 pt wide (4-54) instead of 54. No other element moves. Text-fit asserts below.

Output: manuscript/figures/fig2_wrapper_v4.pdf (+ PNG). v3 files are not touched.
"""
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
STEM = "fig2_wrapper_v4"
W_PT, H_PT = 252.0, 106.0          # 3.5 x 1.47 in (as v3)

FS, FS_MATH, FS_EQ = 7, 7.5, 8
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "STIX", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": FS,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
INK = "#222222"
GREY_FILL, GREY_EDGE = "#F2F2F2", "#555555"
BAND = ["#E6E6E6", "#D4D4D4", "#C2C2C2"]
ACCENT = "#0072B2"           # the persistence data path, the only colour


def arrow(ax, p0, p1, color=INK, lw=0.9):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", color=color, lw=lw,
                                 mutation_scale=7, shrinkA=0.3, shrinkB=0.3, zorder=2))


def main():
    fig = plt.figure(figsize=(W_PT / 72, H_PT / 72), dpi=72)   # display unit = pt = data unit
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_PT); ax.set_ylim(0, H_PT); ax.set_aspect("equal"); ax.axis("off")
    yc = 48.0                                     # main-row centreline (pt)
    checks = []                                   # (text artist, (x0, y0, x1, y1) it must lie within, margin)

    # ---- input window: three stacked bands
    x0, w, hb = 4.0, 50.0, 9.0
    labels = ["4 temporal", "14 FRP-derived", "14 ERA5"]
    ybot = yc - 1.5 * hb
    for i, (lab, col) in enumerate(zip(labels, BAND)):
        y = ybot + (2 - i) * hb
        ax.add_patch(Rectangle((x0, y), w, hb, facecolor=col, edgecolor=GREY_EDGE, lw=0.6, zorder=3))
        t = ax.text(x0 + w / 2, y + hb / 2, lab, ha="center", va="center", fontsize=FS, zorder=4)
        checks.append((t, (x0, y, x0 + w, y + hb), 1.5))
    ax.text(x0 + w / 2, ybot + 3 * hb + 1.8, "$48\\times32$ window", ha="center", va="bottom",
            fontsize=FS_MATH, color=INK)
    xin_r = x0 + w

    # ---- interchangeable encoder (dashed box) with the R1 sentence inside
    sx0, sw, sh = 64.0, 80.0, 32.0
    ax.add_patch(Rectangle((sx0, yc - sh / 2), sw, sh, fill=False, ls=(0, (2.5, 1.5)),
                           lw=0.9, edgecolor=GREY_EDGE, zorder=3))
    slot = (sx0, yc - sh / 2, sx0 + sw, yc + sh / 2)
    # vertical tolerance 0.5 pt: the v3 line positions are kept, and the subscript descender of
    # the third line sits 1 pt above the box edge there as well
    t = ax.text(sx0 + sw / 2, yc + 9.5, "interchangeable encoder", ha="center", va="center",
                fontsize=FS_MATH, color=INK)
    checks.append((t, slot, (2.5, 0.5)))
    t = ax.text(sx0 + sw / 2, yc - 0.5, "learns only the", ha="center", va="center", fontsize=FS, color=INK)
    checks.append((t, slot, (2.5, 0.5)))
    t = ax.text(sx0 + sw / 2, yc - 9.5, "residual $g(z_t)$", ha="center", va="center", fontsize=FS_MATH, color=INK)
    checks.append((t, slot, (2.5, 0.5)))
    arrow(ax, (xin_r, yc), (sx0, yc))

    # ---- gate written on the arrow encoder -> sum
    xs, rs = 192.0, 8.0                           # summation node
    xg = (sx0 + sw + xs - rs) / 2
    arrow(ax, (sx0 + sw, yc), (xs - rs, yc))
    t = ax.text(xg, yc + 3.0, "$\\alpha\\in[0,\\,0.5]$", ha="center", va="bottom", fontsize=FS_MATH, color=INK)
    checks.append((t, (sx0 + sw, yc, xs - rs, H_PT), 2.0))
    ax.text(xg, yc - 3.0, "gate", ha="center", va="top", fontsize=FS, color=INK)

    # ---- summation node
    ax.add_patch(Circle((xs, yc), rs, facecolor="white", edgecolor=INK, lw=0.9, zorder=3))
    ax.text(xs, yc - 0.3, "$+$", ha="center", va="center", fontsize=10, zorder=4)

    # ---- persistence prior: data path from the last input step, entering the sum from below
    y_low = 6.0
    xin_last = xin_r - 2.0                        # "last step" = right edge of the window
    ax.plot([xin_last, xin_last], [ybot, y_low], color=ACCENT, lw=1.0, zorder=2)
    ax.plot([xin_last, xs], [y_low, y_low], color=ACCENT, lw=1.0, zorder=2)
    arrow(ax, (xs, y_low), (xs, yc - rs), color=ACCENT, lw=1.0)
    xl = (xin_last + xs) / 2
    t = ax.text(xl, y_low + 11.0, "persistence prior $\\log(1+y_t)$", ha="center", va="bottom",
                fontsize=FS_MATH, color=ACCENT)
    checks.append((t, (xin_last, y_low, xs, yc - sh / 2), 2.0))
    t = ax.text(xl, y_low + 2.5, "data path, no learnable parameters", ha="center", va="bottom",
                fontsize=FS, color=ACCENT)
    checks.append((t, (xin_last, y_low, xs, yc - sh / 2), 2.0))
    ax.text(xin_last - 2.0, (ybot + y_low) / 2 + 1.0, "$y_t$", ha="right", va="center",
            fontsize=FS_MATH, color=ACCENT)

    # ---- output chain
    ox0, ow, oh = 208.0, 40.0, 28.0
    ax.add_patch(FancyBboxPatch((ox0, yc - oh / 2), ow, oh, boxstyle="round,pad=0.02,rounding_size=1.2",
                                facecolor=GREY_FILL, edgecolor=GREY_EDGE, lw=0.7, zorder=3))
    t = ax.text(ox0 + ow / 2, yc, "$\\exp(\\cdot)-1$\nclip $\\geq 0$\n$\\hat{y}_{t+1:t+12}$",
                ha="center", va="center", fontsize=FS_MATH, linespacing=1.3, zorder=4)
    # horizontal fit only: the three-line math block's reported extent overstates its ink height
    # (same block and box height as v3, visually verified there)
    checks.append((t, (ox0, yc - oh / 2, ox0 + ow, yc + oh / 2), (2.0, -100.0)))
    arrow(ax, (xs + rs, yc), (ox0, yc))

    # ---- strip naming the encoders, and the framework equation
    t = ax.text(W_PT / 2, 79.5, "encoders: PRT (pre-registered), GRU, LSTM, CNN, MLP-Mixer, PatchTST",
                ha="center", va="bottom", fontsize=FS, color=INK)
    checks.append((t, (0, 0, W_PT, H_PT), 4.0))
    t = ax.text(W_PT / 2, 92.5, "$\\log(1+\\hat{y}_{t+h}) = \\log(1+y_t) + \\alpha\\,g_h(z_t)$",
                ha="center", va="bottom", fontsize=FS_EQ, color=INK)
    checks.append((t, (0, 0, W_PT, H_PT), (2.0, 0.5)))     # v3 position; reported extent reaches 1 pt below the page top

    # ---- text-fit asserts (display units are points at dpi 72 with the full-page axes)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for t, (bx0, by0, bx1, by1), m in checks:
        mx, my = m if isinstance(m, tuple) else (m, m)
        bb = t.get_window_extent(r)
        ok = bb.x0 >= bx0 + mx and bb.x1 <= bx1 - mx and bb.y0 >= by0 + my and bb.y1 <= by1 - my
        assert ok, f"text does not fit with {m} pt margin: {t.get_text()[:40]!r} bbox=({bb.x0:.1f},{bb.y0:.1f},{bb.x1:.1f},{bb.y1:.1f}) box={(bx0, by0, bx1, by1)}"

    pdf = HERE / f"{STEM}.pdf"
    png = HERE / f"{STEM}_preview.png"
    fig.savefig(pdf, format="pdf")
    fig.savefig(png, format="png", dpi=400)
    print(f"Wrote {pdf}\nWrote {png}")
    print(f"PROVENANCE: fig pdf sha256 {hashlib.sha256(pdf.read_bytes()).hexdigest()}")
    print(f"  page {W_PT/72:.2f} x {H_PT/72:.3f} in; text {FS} pt plain, {FS_MATH} pt math, {FS_EQ} pt equation; "
          f"{len(checks)} text-fit checks passed")


if __name__ == "__main__":
    main()
