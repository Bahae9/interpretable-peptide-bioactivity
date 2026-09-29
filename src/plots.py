"""Headline figures for the README."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluate import load

FIG = Path("results/figures")
TAB = Path("results/tables")

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8b8a85"
S1, S2 = "#2a78d6", "#eb6834"     # validated categorical slots 1 and 2


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9, length=3, width=0.8)
    ax.grid(axis="x", color=MUTED, alpha=0.25, lw=0.7)
    ax.set_axisbelow(True)


def cterm_sar():
    """Mean pIC50 by C-terminal residue: recovers the published ACE SAR.

    A dot plot, not bars: the interesting range is ~3.8-4.8, and zero-baselined
    bars would compress every difference into invisibility. Dots encode position,
    so the axis can span the data. Whiskers are +/- 1 standard error.
    """
    df, y, _ = load()
    d = df.assign(ct=[s[-1] for s in df.sequence])
    g = (d.groupby("ct").pIC50.agg(["mean", "count", "std"])
         .query("count >= 15").sort_values("mean"))
    g["sem"] = g["std"] / np.sqrt(g["count"])

    fig, ax = plt.subplots(figsize=(7.4, 5.2), facecolor=SURFACE)
    ypos = np.arange(len(g))
    ax.axvline(y.mean(), color=MUTED, ls="--", lw=1.2, zorder=2)
    ax.hlines(ypos, g["mean"] - g["sem"], g["mean"] + g["sem"],
              color=S1, lw=2, alpha=0.55, zorder=3)
    ax.scatter(g["mean"], ypos, s=64, color=S1, zorder=4,
               edgecolor=SURFACE, linewidth=1.5)
    ax.set_yticks(ypos, g.index)
    lo = (g["mean"] - g["sem"]).min()
    hi = (g["mean"] + g["sem"]).max()
    ax.set_xlim(lo - 0.10, hi + 0.05)
    # Value labels live outside the plotting area, so no gridline crosses them
    # and the axis range still reflects only real data.
    for i, r in enumerate(g.itertuples()):
        ax.text(1.02, i, f"{r.mean:.2f}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=8.5, color=INK2, clip_on=False)
        ax.text(1.13, i, f"n={int(r.count)}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=8.5, color=MUTED, clip_on=False)
    ax.set_ylim(-0.8, len(g) - 0.2)
    ax.text(y.mean(), -0.7, f"dataset mean {y.mean():.2f}", fontsize=8.5,
            color=INK2, ha="center", va="bottom")
    ax.set_xlabel("mean pIC50  (\u00b1 1 s.e.)", color=INK2, fontsize=9.5)
    ax.set_ylabel("C-terminal residue", color=INK2, fontsize=9.5)
    ax.set_title("Trp and Tyr at the C-terminus mark the most potent peptides",
                 color=INK, fontsize=11.5, pad=12, loc="left")
    _style(ax)
    fig.savefig(FIG / "cterm_sar.png", dpi=160, bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)


def ablation():
    """Which descriptor block carries the signal."""
    a = pd.read_csv(TAB / "ablation.csv").iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=SURFACE)
    ax.barh(a.block, a.r2_mean, xerr=a.r2_std, height=0.62, color=S1, zorder=3,
            error_kw=dict(ecolor=INK2, elinewidth=1.2, capsize=3, capthick=1.2))
    for i, r in enumerate(a.itertuples()):
        ax.text(r.r2_mean + r.r2_std + 0.012, i, f"{r.r2_mean:.3f}",
                va="center", fontsize=9, color=INK2)
    ax.set_xlim(0, (a.r2_mean + a.r2_std).max() * 1.25)
    ax.set_xlabel("R²  (nested CV, mean ± std over outer folds)",
                  color=INK2, fontsize=9.5)
    ax.set_title("Positional descriptors roughly double R² over the classic panel",
                 color=INK, fontsize=11.5, pad=12, loc="left")
    _style(ax)
    fig.savefig(FIG / "ablation.png", dpi=160, bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)


def mixture():
    """Recall of true actives from fraction-level labels only."""
    m = pd.read_csv(TAB / "mixture_recovery.csv")
    rules = list(dict.fromkeys(m.rule))
    ceil = float(m.individual_label_ceiling.iloc[0])
    rand = float(m.random.iloc[0])

    fig, axes = plt.subplots(1, len(rules), figsize=(9.5, 4.1), sharey=True,
                             facecolor=SURFACE)
    for ax, rule in zip(np.atleast_1d(axes), rules):
        s = m[m.rule == rule].sort_values("bag_size")
        ax.axhline(ceil, color=MUTED, ls="-.", lw=1.2, zorder=2)
        ax.axhline(rand, color=MUTED, ls="--", lw=1.2, zorder=2)
        for col, err, c, lab in ((("propagate_mean"), "propagate_std", S1, "label propagation"),
                                 (("mil_mean"), "mil_std", S2, "MIL max-pooling")):
            ax.errorbar(s.bag_size, s[col], yerr=s[err], color=c, lw=2,
                        marker="o", ms=7, capsize=3, zorder=4, label=lab,
                        markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_xticks(s.bag_size)
        ax.set_xlabel("peptides per fraction (bag size)", color=INK2, fontsize=9.5)
        ax.set_title(f"rule: {rule}", color=INK, fontsize=10.5, loc="left", pad=8)
        _style(ax)
        ax.grid(axis="y", color=MUTED, alpha=0.25, lw=0.7)
        ax.grid(axis="x", visible=False)

    a0 = np.atleast_1d(axes)[0]
    a0.set_ylabel("recall of true actives @ top-98", color=INK2, fontsize=9.5)
    a0.set_ylim(0, max(ceil, m.mil_mean.max()) * 1.28)
    # Reference-line labels ride just inside the right edge of the last panel.
    last = np.atleast_1d(axes)[-1]
    x0, x1 = last.get_xlim()
    inset = x1 - 0.02 * (x1 - x0)
    last.text(inset, ceil + 0.006, "individual-label ceiling",
              fontsize=8.5, color=INK2, va="bottom", ha="right")
    last.text(inset, rand + 0.006, "random", fontsize=8.5,
              color=INK2, va="bottom", ha="right")
    last.set_xlim(x0, x1)
    handles, labels = a0.get_legend_handles_labels()
    handles = [h[0] for h in handles]          # drop the errorbar caps/dashes
    a0.legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK2,
              loc="lower left", handlelength=1.6)
    fig.suptitle("Actives stay recoverable when only mixture-level labels are seen",
                 color=INK, fontsize=11.5, x=0.09, ha="left", y=1.02)
    fig.savefig(FIG / "mixture_recovery.png", dpi=160, bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    cterm_sar(); ablation(); mixture()
    print(f"figures -> {FIG}/cterm_sar.png, ablation.png, mixture_recovery.png")


if __name__ == "__main__":
    main()
