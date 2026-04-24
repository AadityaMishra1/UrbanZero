"""
fig4_reward_decomposition.py

Renders paper/figures/fig4_reward_decomposition.png.

Figure 4 — per-step reward economics under the final reward profile
for four canonical trajectories. Left panel shows the max per-step
magnitude contributed by each reward term (absolute value), right
panel shows the resulting net per-step reward for four scenarios
(sit-still, creep, drive-and-crash, drive-and-succeed). Ordering is
success > crash > creep > sit-still, demonstrating that the
idle_cost term correctly inverts the sit-still local optimum.

Data: PROJECT_NOTES.md §11 "Predicted post-fix per-step economics"
      and Section 5 of the paper.

Run this script from the repo root:
    python3 scripts/figures/fig4_reward_decomposition.py
"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "paper", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Panel (a) — max |·|/step of each reward term
term_names = [
    "progress\n(0.05·Δroute)",
    "velocity carrot\n(0.005·v/TARGET)",
    "idle cost\n(−0.15·ramp)",
    "potential shaping\n(γΦ′−Φ)",
    "terminal\n(±50)",
]
term_magnitudes = [0.021, 0.005, 0.150, 0.0105, 50.0]
term_signs      = ["+",   "+",   "−",   "±",    "±"]
term_colors     = ["#2ca02c", "#2ca02c", "#d62728",
                   "#4c72b0", "#8c564b"]

# Panel (b) — net per-step reward for four canonical trajectories
# (from §5 / PROJECT_NOTES.md per-step economics table)
scenarios = [
    ("Sit still\n(0 m/s)",       -0.183, "#d62728"),
    ("Creep\n(0.5 m/s)",          -0.108, "#ff7f0e"),
    ("Drive + crash\n(300 steps)", -0.137, "#bcbd22"),
    ("Drive + success",            +0.059, "#2ca02c"),
]
scen_labels = [s[0] for s in scenarios]
scen_values = [s[1] for s in scenarios]
scen_colors = [s[2] for s in scenarios]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0),
                               gridspec_kw={"width_ratios": [1.2, 1.0]})

# --- Panel (a) log-scale bar of max-per-step magnitudes ---
xpos_a = np.arange(len(term_names))
ax1.bar(xpos_a, term_magnitudes, color=term_colors,
        edgecolor="black", linewidth=0.5)
for x, v, s in zip(xpos_a, term_magnitudes, term_signs):
    ax1.text(x, v * 1.18, f"{s} {v:g}",
             ha="center", va="bottom", fontsize=8)
ax1.set_yscale("log")
ax1.set_xticks(xpos_a)
ax1.set_xticklabels(term_names, fontsize=8)
ax1.set_ylabel("max |contribution| per step (log)", fontsize=9.5)
ax1.set_title("(a) Reward-term magnitudes (final profile)", fontsize=10)
ax1.set_ylim(1e-3, 1e2)
ax1.grid(axis="y", linestyle=":", alpha=0.3, which="both")

# --- Panel (b) net per-step reward by scenario ---
xpos_b = np.arange(len(scenarios))
bars = ax2.bar(xpos_b, scen_values, color=scen_colors,
               edgecolor="black", linewidth=0.5)
for x, v in zip(xpos_b, scen_values):
    va = "bottom" if v >= 0 else "top"
    dy = 0.008  if v >= 0 else -0.008
    ax2.text(x, v + dy, f"{v:+.3f}",
             ha="center", va=va, fontsize=9, fontweight="bold")

ax2.axhline(0, color="black", linewidth=0.6)
ax2.set_xticks(xpos_b)
ax2.set_xticklabels(scen_labels, fontsize=8)
ax2.set_ylabel("Net reward per step", fontsize=9.5)
ax2.set_title("(b) Per-step economics, canonical trajectories", fontsize=10)
ax2.set_ylim(-0.22, 0.10)
ax2.grid(axis="y", linestyle=":", alpha=0.3)

# Ordering annotation
ax2.text(0.02, 0.04,
         "Ordering:\nsuccess  >  crash  >  creep  >  sit-still",
         transform=ax2.transAxes, fontsize=8, color="#555555",
         bbox=dict(facecolor="white", edgecolor="#cccccc", pad=3))

fig.suptitle("Figure 4. Final reward profile — term magnitudes and "
             "per-step economics.", fontsize=10.5, y=1.02)

plt.tight_layout()
out = os.path.join(OUT_DIR, "fig4_reward_decomposition.png")
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"Wrote {out}")
