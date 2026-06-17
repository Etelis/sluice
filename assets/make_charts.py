# SPDX-License-Identifier: Apache-2.0
"""Generate Sluice's result charts (matplotlib).

Data: DeepSeek-V4-Pro (FP8), 4xH100, EP=4.
  measured  — GPU weight memory at load 9.49 GiB; KV cache 24.4 GiB @ util 0.45;
              16 of 96 local experts resident per layer.
  estimated — per-rank expert shard ~191 GiB and resident slot cache ~32 GiB,
              derived from the 805 GiB checkpoint (mostly experts) / 4 ranks.

    python assets/make_charts.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

TEAL = "#0DB8AB"
INDIGO = "#4338CA"
AMBER = "#F5A926"
GREY = "#C9CED6"
INK = "#1F2430"
H100 = 80  # GiB


def memory_chart(path):
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)

    # Without offload: non-expert weights + full per-rank expert shard.
    ax.bar(0, 9.5, color=TEAL, width=0.6)
    ax.bar(0, 191, bottom=9.5, color=GREY, width=0.6, label="expert shard (est.)")

    # With Sluice: weights + resident slot cache + KV.
    ax.bar(1, 9.5, color=TEAL, width=0.6, label="non-expert weights (meas. 9.5)")
    ax.bar(1, 32, bottom=9.5, color=INDIGO, width=0.6, label="resident experts (est. 32)")
    ax.bar(1, 24.4, bottom=41.5, color=AMBER, width=0.6, label="KV cache (meas. 24.4)")

    ax.axhline(H100, ls="--", lw=1.6, color="#E0457B")
    ax.text(1.46, H100 + 3, "H100 = 80 GiB", color="#E0457B", ha="right", fontsize=9)

    ax.annotate("✗ OOM\n~200 GiB needed", (0, 120), ha="center", va="center",
                fontsize=11, color="#B0152F", fontweight="bold")
    ax.annotate("✓ fits\n~66 GiB", (1, 70), ha="center", va="bottom",
                fontsize=11, color="#0B7A33", fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Without offload", "With Sluice"], fontsize=11)
    ax.set_ylabel("GPU memory per rank (GiB)", fontsize=10)
    ax.set_ylim(0, 215)
    ax.set_title("DeepSeek-V4-Pro (FP8), EP=4 — fitting the experts",
                 fontsize=12, fontweight="bold", color=INK)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.5, -0.01,
             "weights & KV measured on 4×H100; expert shard & slot cache "
             "estimated from the 805 GiB checkpoint",
             ha="center", fontsize=7.5, color="#7A828F")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print("wrote", path)


def residency_chart(path):
    cols, rows, resident = 16, 6, 16  # 96 local experts, 16 resident
    fig, ax = plt.subplots(figsize=(7.2, 3.1), dpi=200)
    n = 0
    for r in range(rows):
        for c in range(cols):
            on = n < resident
            ax.add_patch(Rectangle(
                (c, rows - 1 - r), 0.86, 0.86,
                facecolor=AMBER if on else GREY,
                edgecolor="white", lw=1.2))
            n += 1
    ax.set_xlim(-0.3, cols + 0.3)
    ax.set_ylim(-0.3, rows + 0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Expert working set per layer/rank: 16 of 96 resident on GPU",
                 fontsize=12, fontweight="bold", color=INK, pad=12)
    fig.text(0.5, 0.02,
             "amber = streamed into GPU slots by routing   ·   grey = held in host RAM",
             ha="center", fontsize=9, color="#7A828F")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print("wrote", path)


if __name__ == "__main__":
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    memory_chart(os.path.join(here, "chart-memory.png"))
    residency_chart(os.path.join(here, "chart-residency.png"))
