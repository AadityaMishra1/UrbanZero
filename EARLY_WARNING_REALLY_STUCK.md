# EARLY WARNING: REALLY_STUCK dominating — agent not learning forward motion

**Date:** 2026-04-22 ~21:45 - 22:16 UTC-4 (T+0 to T+31min of long run)
**Tip commit:** `5814854`
**Run status:** STILL RUNNING (not killed — per instructions, flagging only)

## Current beacon (T+31min, 233k steps)

```json
{
    "timesteps": 233522,
    "fps": 123.4,
    "uptime_sec": 1892,
    "rolling_ep_count": 50,
    "rolling_ep_return": -7.338,
    "rolling_ep_len": 1924.4,
    "rolling_route_completion": 0.0178,
    "rolling_collision_rate": 0.06,
    "rolling_avg_speed_ms": 0.224,
    "termination_reasons": {
        "COLLISION": 3,
        "REALLY_STUCK": 35,
        "MAX_STEPS": 12
    },
    "policy_std": 0.7874,
    "approx_kl": 0.01433,
    "entropy_loss": -2.3589,
    "explained_variance": -0.0663,
    "ent_coef": 0.01977,
    "total_episodes": 181
}
```

## The problem

The agent has learned to NOT MOVE. The trend over 31 minutes:

| Metric | T+2min (smoke) | T+10min (smoke) | T+10min (long) | T+31min (long) | Direction |
|--------|---------------|-----------------|----------------|----------------|-----------|
| avg_speed | 2.58 m/s | 1.39 m/s | ~1.0 m/s | **0.224 m/s** | COLLAPSING |
| REALLY_STUCK % | 25% | 36% | ~50% | **70%** | RISING |
| collision_rate | 0.625 | 0.40 | ~0.15 | **0.06** | DROPPING (because car doesn't move) |
| rolling RC | 3.65% | 2.77% | ~2.0% | **1.78%** | DROPPING |
| policy_std | 0.622 | 0.671 | ~0.72 | **0.787** | RISING (exploration widening, not collapsing) |

The last 18 consecutive episodes are ALL `REALLY_STUCK`. The rolling window 50 termination reasons are 70% REALLY_STUCK, 24% MAX_STEPS (which are also zero-speed stuck runs that just hit the 3000-step cap), and only 6% COLLISION (3 episodes).

## Detailed episode trajectory

Early episodes (first ~50, ts 0-50k): mixed COLLISION/OFF_ROUTE/REALLY_STUCK — agent was exploring, hitting things, occasionally moving at 1-6 m/s. Some episodes reached 15-30% route completion.

Mid episodes (~50-120, ts 50k-150k): speed dropping, REALLY_STUCK increasing. Still some COLLISION/OFF_ROUTE episodes mixed in.

Late episodes (~120-181, ts 150k-233k): REALLY_STUCK dominates completely. Last 18 episodes are 100% REALLY_STUCK. Multiple MAX_STEPS episodes with speed=0.0 m/s. The agent has effectively learned to sit still.

## Rolling-best RC peaked early and flatlined

```
best RC = 0.0379 @ t=9522
best RC = 0.0452 @ t=15800
best RC = 0.0479 @ t=17356
best RC = 0.0519 @ t=141978  <-- last improvement, 90k steps ago
```

Peak rolling RC was 5.19% at step 142k, now dropped to 1.78% at step 233k.

## Why the idle-creep bias isn't working

The `step()` function in `carla_env.py` has an idle-creep bias: `shifted = throttle_brake + 0.3`, which maps action[1]=0 to throttle=0.3. At policy init (mean=0, std=0.6), this gives ~79% probability of applying throttle.

But the policy has LEARNED to output negative action[1] values that overcome the +0.3 bias. With std=0.787 and 233k steps of training, the policy is actively choosing to brake. The reward signal is telling it that NOT moving is better than moving:

**Hypothesis:** The reward structure penalizes the agent more for moving (collisions, off-route) than for staying still. REALLY_STUCK gives a terminal penalty of -50, but so does COLLISION. If collisions happen faster than stuck timeout, the per-step expected penalty from moving is higher than from sitting. The agent has found the local optimum of "do nothing, collect small per-step penalties, eventually get -50 from REALLY_STUCK" which is better than "move, collect big collision penalties every few hundred steps."

## PPO stats at T+31min

```
approx_kl: 0.014 (healthy)
clip_fraction: 0.124 (healthy)
entropy_loss: -2.36 (healthy, not collapsing)
policy_std: 0.787 (RISING — agent is exploring wider but finding "don't move" is best)
explained_variance: oscillating -0.07 to 0.91 (value function struggling)
policy_gradient_loss: near zero (-0.001 to -0.003) — gradients are tiny
value_loss: oscillating 0.001 to 1.47 — bimodal (stuck episodes vs rare movement episodes)
```

## What needs investigation

1. **Reward balance for REALLY_STUCK vs COLLISION**: Is the per-step reward for "sitting still" (zero speed penalty? zero collision penalty? still getting small progress reward from initial spawn drift?) actually BETTER than the per-step reward for moving and risking collision?

2. **The REALLY_STUCK penalty (-50) fires at step 1501**: That's 75 seconds of zero progress. During those 1501 steps, what reward does the agent accumulate? If it's near-zero (no movement = no penalty but also no reward), then the per-step cost of being stuck is ~0, and the terminal is -50/1501 = -0.033 per step. Compare that to a COLLISION episode of 300 steps: -50/300 = -0.167 per step. **Sitting still is 5x cheaper per step than crashing.**

3. **Is there a per-step speed reward?** If not, there's no gradient signal pushing the agent to move. The only "move" signal is route progress reward, but if the agent never discovers sustained forward motion (because early random exploration led to mostly crashes), it never observes that reward.

4. **Consider:** Adding a per-step speed bonus (CaRL uses this), or reducing the REALLY_STUCK step threshold to make "sitting still" more expensive per step, or making the REALLY_STUCK penalty much larger than COLLISION (-100 vs -50).

## Full episode history

See the complete `EPISODE END` log lines in the training log:
```
~/urbanzero/logs/train_20260422_214425.log
```

Total episodes: 181
- COLLISION: 56 (mostly early)
- REALLY_STUCK: 76 (accelerating — last 18 consecutive)
- MAX_STEPS: 29 (mostly zero-speed stuck runs)
- OFF_ROUTE: 20 (mostly early, when agent was still moving)

## Status

Run is STILL LIVE at 233k steps, 123 FPS. Not killing it per instructions — but the trend is clearly diverging from expected trajectory. At this rate, RED-3 (REALLY_STUCK >30% past 1M steps) is virtually certain.
