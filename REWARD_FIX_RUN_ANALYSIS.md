# REWARD FIX RUN ANALYSIS: Agent not learning to steer — RC flat at 5%

**Date:** 2026-04-22 22:32 - 2026-04-23 ~00:45 UTC-4
**Tip commit:** `d307a66` (fix(reward): add idle_cost + un-anneal carrot)
**Seed:** 137
**Run status:** STILL RUNNING (~900k steps) but recommending intervention

## Summary

The idle_cost fix successfully broke the sit-still attractor (issue #7). The agent now moves at 3-5 m/s average. But it drives in a straight line and never learns to steer toward waypoints. Route completion is flat at ~5% across 1500+ episodes with zero upward trend. policy_std has hit the 1.0 upper clamp, meaning the policy is essentially random noise.

**Root cause: no per-step steering/heading reward.** The route progress reward is too sparse — the agent only gets rewarded when it happens to drive in the correct direction by chance. With random steering, this is too rare to provide a usable gradient signal.

## Beacon at ~T+2h (892k steps)

```json
{
    "timesteps": 892236,
    "fps": 123.3,
    "rolling_avg_speed_ms": 3.174,
    "rolling_route_completion": 0.0574,
    "rolling_collision_rate": 0.32,
    "rolling_ep_len": 898,
    "policy_std": 0.999,
    "approx_kl": 0.00614,
    "entropy_loss": -2.36,
    "ent_coef": 0.01911,
    "termination_reasons": {"OFF_ROUTE": 20, "COLLISION": 16, "REALLY_STUCK": 14}
}
```

## Evidence: RC is flat (no learning)

Mean route completion per 100-episode window across 1300+ non-stuck episodes:

```
ep    0- 100: mean_RC=5.91%  episodes>10%RC=16
ep  100- 200: mean_RC=6.27%  episodes>10%RC=23
ep  200- 300: mean_RC=5.98%  episodes>10%RC=19
ep  300- 400: mean_RC=4.56%  episodes>10%RC=10
ep  400- 500: mean_RC=5.39%  episodes>10%RC=16
ep  500- 600: mean_RC=4.84%  episodes>10%RC=21
ep  600- 700: mean_RC=6.05%  episodes>10%RC=18
ep  700- 800: mean_RC=4.33%  episodes>10%RC=12
ep  800- 900: mean_RC=5.23%  episodes>10%RC=15
ep  900-1000: mean_RC=4.81%  episodes>10%RC=11
ep 1000-1100: mean_RC=5.67%  episodes>10%RC=15
ep 1100-1200: mean_RC=5.28%  episodes>10%RC=16
ep 1200-1300: mean_RC=5.59%  episodes>10%RC=17
```

Zero trend. Oscillating between 4.3% and 6.3%. Best single episode was 42.6% (lucky fluke). Rolling-best RC peaked at 8.24% at step 64k and hasn't improved in 830k steps.

## Evidence: Speed oscillates, doesn't improve

```
ep    0- 100: mean_speed=3.25m/s  sitting(<0.5)=1
ep  100- 200: mean_speed=3.36m/s  sitting(<0.5)=3
ep  200- 300: mean_speed=6.38m/s  sitting(<0.5)=0
ep  300- 400: mean_speed=3.89m/s  sitting(<0.5)=6
ep  400- 500: mean_speed=3.33m/s  sitting(<0.5)=4
ep  500- 600: mean_speed=3.12m/s  sitting(<0.5)=2
ep  600- 700: mean_speed=4.34m/s  sitting(<0.5)=3
ep  700- 800: mean_speed=2.25m/s  sitting(<0.5)=9
ep  800- 900: mean_speed=2.47m/s  sitting(<0.5)=9
ep  900-1000: mean_speed=4.62m/s  sitting(<0.5)=5
ep 1000-1100: mean_speed=3.77m/s  sitting(<0.5)=5
ep 1100-1200: mean_speed=4.39m/s  sitting(<0.5)=0
ep 1200-1300: mean_speed=2.67m/s  sitting(<0.5)=6
```

Swings between 2.2 and 6.4 m/s with no trend. Agent alternates between "floor it off the road" and "sit and get stuck."

## Evidence: REALLY_STUCK keeps returning

```
ep    0- 100: stuck=2%   (idle_cost working)
ep  200- 300: stuck=0%   (peak effectiveness)
ep  700- 800: stuck=31%  (rediscovered sit-still)
ep 1000-1100: stuck=18%  (partial recovery)
ep 1400-1500: stuck=31%  (back again)
```

Idle_cost isn't strong enough to permanently suppress the sit-still attractor. The policy keeps oscillating back to it.

## Evidence: policy_std at 1.0 clamp

```
T+15min: std=0.666  (healthy)
T+1h:    std=0.926  (rising fast)
T+1.5h:  std=0.956  (still rising)
T+2h:    std=0.999  (HIT THE CLAMP)
```

At 900k steps the std should be DECREASING (0.5-0.7), meaning the agent is getting confident. Instead it's at maximum — the reward signal is so noisy that wider exploration scores better than exploitation. The policy is not converging.

## Speed distribution (all 1295 non-stuck episodes)

```
<0.5 m/s (sitting):    53  (4%)
0.5-2 m/s (crawling): 399  (31%)
2-5 m/s (learning):   471  (36%)
5-10 m/s (driving):   324  (25%)
>10 m/s (flooring it): 48  (4%)
Mean: 3.69  Median: 3.30
```

## RC distribution (all 1295 non-stuck episodes)

```
0% RC:     115  (9%)
<5% RC:    735  (57%)
5-10% RC:  236  (18%)
10-20% RC: 157  (12%)
>20% RC:    52  (4%)
Mean: 5.38%  Max: 42.6%
```

## Termination distribution (all 1511 episodes)

```
OFF_ROUTE:    724  (48%)
COLLISION:    568  (38%)
REALLY_STUCK: 216  (14%)
MAX_STEPS:      3  (0.2%)
```

## Diagnosis

The agent has learned ONE thing: press the throttle (idle_cost penalty). It has NOT learned: steer toward waypoints.

**Why:** The route progress reward (`progress_reward`) only fires when the car happens to drive along the route. With random steering (std=0.999), the probability of sustained correct-direction driving is very low. The reward signal is too sparse for the policy to extract a steering gradient.

**What's needed:** A dense per-step heading reward that gives the agent a gradient toward the next waypoint on EVERY step, not just when it stumbles onto the route by accident. CaRL (NVIDIA, CoRL 2025) uses exactly this — a cosine-similarity term between the car's heading vector and the direction to the next waypoint. This turns "steer toward the route" from a sparse discovery problem into a dense signal the agent sees every single step.

**Specific suggestion:** Add to `_compute_reward()`:
```python
# Dense heading reward: cosine similarity between ego heading and direction to next waypoint
ego_fwd = vehicle.get_transform().get_forward_vector()
wp_dir = next_waypoint.location - vehicle.get_location()
# normalize
cos_sim = (ego_fwd.x * wp_dir.x + ego_fwd.y * wp_dir.y) / (|ego_fwd| * |wp_dir| + 1e-8)
heading_reward = 0.1 * cos_sim  # positive when pointing toward waypoint, negative when away
```

This would provide gradient signal for steering on every step, not just when the car happens to be on-route.

## Infrastructure status

Infra is solid. No crashes, no NaN guards, 123 FPS sustained, DummyVecEnv stable. The blocker is purely reward shaping.
