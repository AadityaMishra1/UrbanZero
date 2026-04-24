"""
fig2_beacon_endpoints.py

Renders paper/figures/fig2_beacon_endpoints.png.

Figure 2 — three-panel view of final-step beacon metrics across the
six training runs: (a) policy_std, (b) approx_kl, (c) rolling avg_speed.
Shaded bands mark the Andrychowicz et al. 2021 "viable band"
(std ∈ [0.3, 0.7]) in (a) and the healthy PPO KL band (kl ≤ 0.015)
in (b). Data per-run from PROJECT_NOTES.md §11 / DIAGNOSIS_FOR_REVIEW.md §4.

Run this script from the repo root:
    python3 scripts/figures/fig2_beacon_endpoints.py
"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "paper", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Raw endpoint metrics harvested from PROJECT_NOTES.md §11 and the
# six run summaries in DIAGNOSIS_FOR_REVIEW.md §4.
# Run labels are consistent with Figure 1.
runs = [
    # (label, policy_std, approx_kl, avg_speed_ms, color)
    ("PureRL v1",   0.787, 0.0143, 0.224, "#d62728"),
    ("PureRL v2",   0.999, 0.0061, 3.174, "#ff7f0e"),
    ("PureRL v3",   0.670, 0.0081, 3.858, "#bcbd22"),
    ("BC+PPO v1",   0.222, 0.0792, 7.599, "#1f77b4"),
    ("BC+PPO v2",   0.213, 0.0864, 7.941, "#1f77b4"),
    ("BC+PPO v3",   0.579, 0.0016, 0.969, "#9467bd"),
]

labels = [r[0] for r in runs]
stds   = [r[1] for r in runs]
kls    = [r[2] for r in runs]
spds   = [r[3] for r in runs]
cols   = [r[4] for r in runs]

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))

xpos = np.arange(len(runs))

# --- Panel (a): policy_std ---
ax = axes[0]
ax.axhspan(0.30, 0.70, color="green", alpha=0.10, zorder=0,
           label="Andrychowicz et al. 2021\nviable band [0.3, 0.7]")
ax.bar(xpos, stds, color=cols, edgecolor="black", linewidth=0.6)
ax.axhline(1.0, linestyle="--", color="red", linewidth=0.8, alpha=0.6)
ax.text(5.4, 1.02, "upper clamp (old)",
        fontsize=7, color="red", ha="right")
ax.set_xticks(xpos)
ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Policy action σ (final, rolling)", fontsize=9.5)
ax.set_ylim(0, 1.15)
ax.set_title("(a) Gaussian policy σ", fontsize=10)
ax.legend(loc="upper left", fontsize=7)
ax.grid(axis="y", linestyle=":", alpha=0.3)

# --- Panel (b): approx_kl ---
ax = axes[1]
ax.axhspan(0.0, 0.015, color="green", alpha=0.10, zorder=0,
           label="Healthy PPO KL  ≤ 0.015")
ax.bar(xpos, kls, color=cols, edgecolor="black", linewidth=0.6)
ax.axhline(0.015, linestyle="--", color="green", linewidth=0.8, alpha=0.6)
ax.set_xticks(xpos)
ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("PPO approx_kl (per update)", fontsize=9.5)
ax.set_title("(b) Policy-update KL divergence", fontsize=10)
ax.legend(loc="upper right", fontsize=7)
ax.grid(axis="y", linestyle=":", alpha=0.3)

# --- Panel (c): avg_speed ---
ax = axes[2]
ax.bar(xpos, spds, color=cols, edgecolor="black", linewidth=0.6)
ax.axhline(1.0, linestyle=":", color="black", linewidth=0.8, alpha=0.5)
ax.text(5.4, 1.10, "idle_cost threshold (1 m/s)",
        fontsize=7, color="black", ha="right")
ax.set_xticks(xpos)
ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Rolling avg ego speed (m/s)", fontsize=9.5)
ax.set_title("(c) Average ego speed at endpoint", fontsize=10)
ax.grid(axis="y", linestyle=":", alpha=0.3)

fig.suptitle("Figure 2. Beacon endpoints across the six training runs.",
             fontsize=10.5, y=1.02)

plt.tight_layout()
out = os.path.join(OUT_DIR, "fig2_beacon_endpoints.png")
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"Wrote {out}")
