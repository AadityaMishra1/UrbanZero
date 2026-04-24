"""
fig3_bc_eval_distribution.py

Renders paper/figures/fig3_bc_eval_distribution.png.

Figure 3 — per-episode route-completion distribution for the frozen BC
policy, evaluated deterministically on 20 episodes (seed 1001, port
2000, Town01) after removing the `tm.set_hybrid_physics_mode` bug.
Data sourced directly from eval/bc_final_20260424_0059.json.
Episodes are colored by termination reason.

Run this script from the repo root:
    python3 scripts/figures/fig3_bc_eval_distribution.py
"""

from __future__ import annotations

import json
import os
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
EVAL_PATH = os.path.join(REPO_ROOT, "eval", "bc_final_20260424_0059.json")
OUT_DIR   = os.path.join(REPO_ROOT, "paper", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

with open(EVAL_PATH, "r") as f:
    data = json.load(f)

agg = data["aggregate"]
eps = data["episodes"]

rc_vals = [e["route_completion"] * 100 for e in eps]
reasons = [e["termination_reason"] for e in eps]

REASON_COLORS = {
    "COLLISION":      "#d62728",
    "OFF_ROUTE":      "#ff7f0e",
    "REALLY_STUCK":   "#bcbd22",
    "ROUTE_COMPLETE": "#2ca02c",
    "MAX_STEPS":      "#1f77b4",
}

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(10.0, 3.8),
    gridspec_kw={"width_ratios": [1.3, 1.0]},
)

# --- Panel (a): sorted per-episode RC bar chart ---
order = np.argsort(rc_vals)
rc_sorted    = [rc_vals[i] for i in order]
reason_sorted = [reasons[i] for i in order]
colors_sorted = [REASON_COLORS.get(r, "gray") for r in reason_sorted]

xpos = np.arange(len(rc_sorted))
ax1.bar(xpos, rc_sorted, color=colors_sorted,
        edgecolor="black", linewidth=0.4)

# Aggregate reference lines
ax1.axhline(agg["rc_mean"]   * 100, linestyle="--", color="black",
            linewidth=0.9, label=f"mean = {agg['rc_mean']*100:.2f} %")
ax1.axhline(agg["rc_median"] * 100, linestyle=":",  color="black",
            linewidth=0.9, label=f"median = {agg['rc_median']*100:.2f} %")

ax1.set_xticks(xpos)
ax1.set_xticklabels([f"{i+1}" for i in range(len(rc_sorted))], fontsize=7)
ax1.set_xlabel("Episode (sorted by RC, ascending)", fontsize=9.5)
ax1.set_ylabel("Per-episode route completion (%)", fontsize=9.5)
ax1.set_title("(a) Per-episode RC distribution, N = 20", fontsize=10)
ax1.set_ylim(0, max(rc_sorted) * 1.15)
ax1.grid(axis="y", linestyle=":", alpha=0.3)

# Legend for termination reasons actually present
reasons_present = [r for r in REASON_COLORS if r in reasons]
handles = [plt.Rectangle((0, 0), 1, 1, color=REASON_COLORS[r])
           for r in reasons_present]
labels_present = [f"{r} ({reasons.count(r)}/20)" for r in reasons_present]
leg1 = ax1.legend(handles, labels_present, loc="upper left",
                  fontsize=7.5, title="Termination reason",
                  title_fontsize=8)
ax1.add_artist(leg1)
ax1.legend(loc="upper right", fontsize=7.5)

# --- Panel (b): histogram ---
ax2.hist(rc_vals, bins=np.arange(0, 36, 3), color="#4c72b0",
         edgecolor="black", linewidth=0.5)
ax2.axvline(agg["rc_mean"]   * 100, linestyle="--", color="black",
            linewidth=0.9, label=f"mean {agg['rc_mean']*100:.2f} %")
ax2.axvline(agg["rc_max"]    * 100, linestyle="-.", color="#2ca02c",
            linewidth=0.9, label=f"max  {agg['rc_max']*100:.2f} %")
ax2.set_xlabel("Route completion (%)", fontsize=9.5)
ax2.set_ylabel("Episode count", fontsize=9.5)
ax2.set_title("(b) Histogram of per-episode RC", fontsize=10)
ax2.grid(axis="y", linestyle=":", alpha=0.3)
ax2.legend(loc="upper right", fontsize=8)

fig.suptitle(
    "Figure 3. Frozen BC policy, deterministic eval "
    "(20 episodes, seed 1001, Town01, dynamic NPCs).",
    fontsize=10.5, y=1.02,
)

plt.tight_layout()
out = os.path.join(OUT_DIR, "fig3_bc_eval_distribution.png")
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"Wrote {out}")
