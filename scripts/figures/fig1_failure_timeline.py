"""
fig1_failure_timeline.py

Renders paper/figures/fig1_failure_timeline.png.

Figure 1 — six-run rolling route-completion (RC) plateau and diagnostic
signatures. Data sourced from PROJECT_NOTES.md §11 and DIAGNOSIS_FOR_REVIEW.md §4.
Each bar is a training run; color encodes the diagnosed failure mode;
horizontal dashed line at 5 % marks the observed plateau.

Run this script from the repo root:
    python3 scripts/figures/fig1_failure_timeline.py
"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "paper", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# (label, final_rolling_RC_pct, failure_mode_category, duration_k_steps)
runs = [
    ("PureRL v1",     1.78, "sit-still attractor",    233),
    ("PureRL v2",     5.40, "std-pin at clamp",        892),
    ("PureRL v3",     4.03, "sparse steering grad",   104),
    ("BC+PPO v1",     5.56, "KL runaway",              28),
    ("BC+PPO v2",     5.86, "KL runaway",              31),
    ("BC+PPO v3",     5.86, "actor-freeze / drift", 2407),
]

CATEGORY_COLORS = {
    "sit-still attractor":   "#d62728",  # red
    "std-pin at clamp":      "#ff7f0e",  # orange
    "sparse steering grad":  "#bcbd22",  # olive
    "KL runaway":            "#1f77b4",  # blue
    "actor-freeze / drift":  "#9467bd",  # purple
}

labels  = [r[0] for r in runs]
rc_pcts = [r[1] for r in runs]
cats    = [r[2] for r in runs]
colors  = [CATEGORY_COLORS[c] for c in cats]
durs    = [r[3] for r in runs]

fig, ax = plt.subplots(figsize=(7.2, 3.6))
xpos = np.arange(len(runs))
bars = ax.bar(xpos, rc_pcts, color=colors, edgecolor="black", linewidth=0.6)

# Annotate each bar with its duration in k-steps
for x, y, d in zip(xpos, rc_pcts, durs):
    ax.text(x, y + 0.18, f"{d} k",
            ha="center", va="bottom", fontsize=8, color="#333333")

# Plateau band
ax.axhspan(4.4, 5.9, color="gray", alpha=0.12, zorder=0)
ax.axhline(5.0, linestyle="--", color="gray", linewidth=0.8, zorder=0)
ax.text(5.55, 5.1, "5 % plateau band (4.4–5.9 %)",
        fontsize=8, color="#555555", ha="right", va="bottom")

# Final BC eval horizontal reference
ax.axhline(7.42, linestyle=":", color="#2ca02c", linewidth=1.2, zorder=0)
ax.text(-0.4, 7.52, "Frozen BC eval (post-bugfix): 7.42 %",
        fontsize=8, color="#2ca02c", ha="left", va="bottom")

ax.set_xticks(xpos)
ax.set_xticklabels(labels, rotation=0, fontsize=9)
ax.set_ylabel("Final rolling route completion (%)", fontsize=10)
ax.set_xlabel("Training run (in chronological order)", fontsize=10)
ax.set_title("Figure 1. Six training runs — final rolling route "
             "completion by configuration", fontsize=10.5)
ax.set_ylim(0, 9.5)
ax.grid(axis="y", linestyle=":", alpha=0.3)

# Legend by category
handles = [plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS[c])
           for c in CATEGORY_COLORS]
ax.legend(handles, list(CATEGORY_COLORS.keys()),
          loc="upper right", fontsize=8, ncol=1, frameon=True,
          title="Diagnosed failure mode", title_fontsize=8)

plt.tight_layout()
out = os.path.join(OUT_DIR, "fig1_failure_timeline.png")
plt.savefig(out, dpi=180)
print(f"Wrote {out}")
