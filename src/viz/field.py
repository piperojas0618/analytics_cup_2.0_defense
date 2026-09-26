"""Pitch helper + Week-1 audit figures.

Pitch: mplsoccer ``skillcorner`` type (centre-origin metres) — no rescaling.
Colours: single-hue blue sequential ramp for magnitudes; neutral inks for text.
(Swap in ``skillcornerviz.utils.constants`` colours for the final submission.)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import Pitch

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
LINE = "#b9b8b3"
BLUE = "#2a78d6"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def skillcorner_pitch(length: float = 105, width: float = 68, half: bool = False) -> Pitch:
    return Pitch(pitch_type="skillcorner", pitch_length=length, pitch_width=width,
                 half=half, line_color=LINE, pitch_color=SURFACE, linewidth=1)


def _style(fig, ax=None):
    fig.patch.set_facecolor(SURFACE)
    if ax is not None:
        ax.set_facecolor(SURFACE)


def plot_detection_heatmap(heat: pd.DataFrame, title: str, min_n: int = 200, vmin: float = 0.5):
    """Detection rate of outfield defenders by pitch location (normalised coords:
    defending goal on the right). ``heat`` = concatenated detection_by_location."""
    g = heat.groupby(["xbin", "ybin"])[["n", "n_det"]].sum().reset_index()
    g = g[g.n >= min_n]
    g["rate"] = g.n_det / g.n
    xs, ys = np.sort(g.xbin.unique()), np.sort(g.ybin.unique())
    step = float(np.diff(xs).min()) if len(xs) > 1 else 5.0
    Z = g.pivot(index="ybin", columns="xbin", values="rate").reindex(index=ys, columns=xs)

    pitch = skillcorner_pitch()
    fig, ax = pitch.draw(figsize=(10, 7))
    _style(fig, ax)
    im = ax.pcolormesh(np.append(xs - step / 2, xs[-1] + step / 2),
                       np.append(ys - step / 2, ys[-1] + step / 2),
                       Z.to_numpy(), cmap=SEQ_BLUE, vmin=vmin, vmax=1, alpha=.9, zorder=0.5,
                       edgecolors=SURFACE, linewidth=0.5)
    ax.annotate("", xy=(30, -38.5), xytext=(-30, -38.5), annotation_clip=False,
                arrowprops=dict(arrowstyle="->", color=INK_2, lw=1))
    ax.text(0, -40.5, "attack direction  →  defending goal", ha="center", va="top",
            color=INK_2, fontsize=9)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.06)
    cb.set_label(f"share of defender-frames detected on camera (scale starts at {vmin:.0%}; "
                 f"bins with < {min_n} frames hidden)", color=INK_2, fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK_2, labelsize=8)
    cb.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(title, color=INK, fontsize=13, loc="left", pad=10)
    return fig


def plot_phase_survival(phases: pd.DataFrame, gate: float, title: str):
    """Share of eligible final-third defensive phases that survive, by phase type."""
    s = phases.groupby("team_out_of_possession_phase_type").survives.agg(["size", "mean"])
    tot = pd.DataFrame({"size": [len(phases)], "mean": [phases.survives.mean()]}, index=["ALL"])
    s = pd.concat([s.sort_values("mean"), tot])
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(s) + 1.5))
    _style(fig, ax)
    y = np.arange(len(s))
    colors = [BLUE] * (len(s) - 1) + ["#184f95"]
    ax.barh(y, s["mean"], color=colors, height=0.6)
    for yi, (n, m) in zip(y, s.itertuples(index=False)):
        ax.text(m + 0.01, yi, f"{m:.0%}  (n={n})", va="center", color=INK, fontsize=9)
    ax.axvline(gate, color=INK_2, lw=1, ls="--")
    ax.text(gate + 0.01, -0.75, f"Gate 1 ({gate:.0%})", color=INK_2, fontsize=9, va="center")
    ax.set_yticks(y, [i.replace("_", " ") for i in s.index], color=INK, fontsize=10)
    ax.set_xlim(0, 1.15)
    ax.set_ylim(-1.1, len(s) - 0.5)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.tick_params(axis="x", colors=INK_2, labelsize=8)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    fig.tight_layout()
    return fig


def plot_sensitivity(grid: pd.DataFrame, gate: float, title: str):
    """Phase survival across (detection floor × visibility minimum)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    _style(fig, ax)
    ax.imshow(grid.to_numpy(), cmap=SEQ_BLUE, vmin=0, vmax=1, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.iat[i, j]
            ax.text(j, i, f"{v:.0%}" + (" ✓" if v >= gate else ""), ha="center", va="center",
                    fontsize=10, color="white" if v > 0.55 else INK)
    ax.set_xticks(range(grid.shape[1]), [f"{c:.0%}" for c in grid.columns], color=INK_2)
    ax.set_yticks(range(grid.shape[0]), [f"≥{r}/10" for r in grid.index], color=INK_2)
    ax.set_xlabel("min. share of defending third visible on camera", color=INK_2)
    ax.set_ylabel("detected outfield defenders", color=INK_2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    fig.tight_layout()
    return fig
