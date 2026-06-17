# SPDX-License-Identifier: Apache-2.0
"""Generate Sluice's result charts (matplotlib).

Measured (4xH100): V4-Pro GPU weights at load 9.49 GiB; KV 24.4 GiB @ util 0.45;
16 of 96 local experts resident. V2-Lite (BF16, 1 GPU) ~31 GiB, fits natively.
Estimated from the 805 GiB checkpoint: V4-Pro per-rank expert shard ~191 GiB,
16-slot cache ~32 GiB, ~2 GiB per expert.

    python assets/make_charts.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

TEAL, INDIGO, AMBER, GREY, INK = "#0DB8AB", "#4338CA", "#F5A926", "#C9CED6", "#1F2430"
RED, GREEN = "#B0152F", "#0B7A33"
H100 = 80


def comparison_chart(path):
    fig, ax = plt.subplots(figsize=(7.8, 4.7), dpi=200)
    x = [0, 1, 2]
    # V2-Lite: fits natively (single GPU).
    ax.bar(0, 31, color=TEAL, width=0.62)
    # V4-Pro without offload: weights + full expert shard -> OOM.
    ax.bar(1, 9.5, color=TEAL, width=0.62)
    ax.bar(1, 191, bottom=9.5, color=GREY, width=0.62)
    # V4-Pro with Sluice: weights + resident slots + KV -> fits.
    ax.bar(2, 9.5, color=TEAL, width=0.62)
    ax.bar(2, 32, bottom=9.5, color=INDIGO, width=0.62)
    ax.bar(2, 24.4, bottom=41.5, color=AMBER, width=0.62)

    ax.axhline(H100, ls="--", lw=1.6, color="#E0457B")
    ax.text(2.52, H100 + 3, "H100 = 80 GiB", color="#E0457B", ha="right", fontsize=9)
    ax.annotate("✓ fits", (0, 35), ha="center", color=GREEN, fontweight="bold", fontsize=10)
    ax.annotate("✗ OOM\n~200 GiB", (1, 150), ha="center", va="center", color="white",
                fontweight="bold", fontsize=12)
    ax.annotate("✓ fits\n~66 GiB", (2, 70), ha="center", va="bottom", color=GREEN,
                fontweight="bold", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(["V2-Lite\n(BF16, 1 GPU)", "V4-Pro\n(no offload)",
                        "V4-Pro\n+ Sluice"], fontsize=10)
    ax.set_ylabel("GPU memory needed (GiB)", fontsize=10)
    ax.set_ylim(0, 215)
    ax.set_title("Why offload: V2-Lite fits a GPU; V4-Pro's experts don't",
                 fontsize=12.5, fontweight="bold", color=INK)
    ax.legend(handles=[
        Patch(color=TEAL, label="non-expert weights"),
        Patch(color=GREY, label="experts, all resident (est.)"),
        Patch(color=INDIGO, label="experts, Sluice cache (est.)"),
        Patch(color=AMBER, label="KV cache"),
    ], fontsize=8, loc="upper left", frameon=False, ncol=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.5, -0.02, "V4 figures per rank at EP=4 · weights & KV measured · "
             "expert shard & cache estimated from the 805 GiB checkpoint",
             ha="center", fontsize=7.5, color="#7A828F")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print("wrote", path)


def residency_chart(path):
    cols, rows, resident = 16, 6, 16
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=200)
    n = 0
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle((c, rows - 1 - r), 0.86, 0.86,
                         facecolor=AMBER if n < resident else GREY,
                         edgecolor="white", lw=1.2))
            n += 1
    ax.set_xlim(-0.3, cols + 0.3)
    ax.set_ylim(-0.3, rows + 0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("V4-Pro working set: 16 of 96 experts resident per layer/rank",
                 fontsize=12, fontweight="bold", color=INK, pad=10)
    fig.text(0.5, 0.02, "amber = streamed into GPU slots by routing   ·   "
             "grey = held in host RAM", ha="center", fontsize=9, color="#7A828F")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print("wrote", path)


def tradeoff_chart(path):
    per_expert = 191 / 96  # GiB
    slots = [8, 16, 24, 32, 48, 64, 96]
    cache = [s * per_expert for s in slots]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    ax.plot(slots, cache, "-o", color=INDIGO, lw=2.2, label="resident cache (GiB)")
    ax.axhline(46, ls="--", lw=1.5, color="#E0457B")
    ax.text(95, 48, "VRAM left for experts after weights+KV (~46 GiB)",
            ha="right", color="#E0457B", fontsize=8.5)
    ax.scatter([16], [32], s=120, color=AMBER, zorder=5, edgecolor="white")
    ax.annotate("validated: 16 slots ≈ 32 GiB", (16, 32), (24, 70),
                arrowprops=dict(arrowstyle="->", color=INK), fontsize=9, color=INK)
    ax.set_xlabel("SLUICE_SLOTS (resident experts per layer/rank)", fontsize=10)
    ax.set_ylabel("GPU cache size (GiB)", fontsize=10)
    ax.set_title("Tuning the cache (V4-Pro, EP=4 — projected)",
                 fontsize=12, fontweight="bold", color=INK)
    ax.legend(fontsize=9, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.5, -0.02, "~2 GiB per expert; raise slots until a step's experts "
             "fit, lower to save VRAM", ha="center", fontsize=7.5, color="#7A828F")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print("wrote", path)


if __name__ == "__main__":
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    comparison_chart(os.path.join(here, "chart-comparison.png"))
    residency_chart(os.path.join(here, "chart-residency.png"))
    tradeoff_chart(os.path.join(here, "chart-tradeoff.png"))
