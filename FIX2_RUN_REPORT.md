# FIX-2 RUN REPORT: log_std clamp 0.7 + potential-based shaping + TM hybrid physics

**Date:** 2026-04-23 ~01:26–01:55 UTC-4
**Tip commit:** `ff7a1e1` (fix: log_std clamp 0.7 + Ng-compliant shaping + TM hybrid physics)
**Seed:** 211
**Run status:** STILL RUNNING at 104k steps (~15min wall clock) but already showing failure patterns
**Recommendation:** KILL THIS RUN — same fundamental failure as runs 1 and 2

## Summary

All three fixes landed (std clamp, potential-based shaping, TM hybrid physics) but the agent is already exhibiting the same failure modes seen in every prior run:
1. **REALLY_STUCK returning** — 12/129 visible episodes (9%), sit-still attractor not broken
2. **Collision-dominated** — 64/129 episodes (50%) end in COLLISION at crawling speeds
3. **Off-route driving** — 53/129 episodes (41%) drive fast off the road
4. **RC flat at ~4.4%** — no improvement over previous runs (5.38% run-1, 5.74% run-2)
5. **NPCs FROZEN** — all spawned NPC vehicles are stationary despite TM hybrid_physics=True

## NEW ISSUE: NPC Vehicles Frozen (Issue #9)

**All NPC traffic cars spawn but DO NOT MOVE.** They sit frozen in their spawn positions. This was visually confirmed by the operator on both CARLA viewports.

The TM hybrid physics code runs without error — `tm.set_hybrid_physics_mode(True)` and `tm.set_hybrid_physics_radius(70.0)` both execute. But the NPCs remain completely stationary.

**Impact:** Without moving traffic, the agent has no dynamic obstacles to learn to avoid. The COLLISION terminations (50% of episodes) are the agent slowly bumping into parked NPC cars or static geometry. This is not teaching realistic collision avoidance.

**Possible causes:**
- TM autopilot may not be engaging properly in sync mode
- The `tm.set_synchronous_mode(True)` call may need to happen AFTER spawning vehicles
- CARLA 0.9.15 may need `vehicle.set_autopilot(True, tm.get_port())` with explicit port
- The sync world.tick() in env.step() may not be advancing TM internal state

## Beacon at 104k steps

```json
{
    "timesteps": 104272,
    "fps": 119.46,
    "rolling_ep_count": 50,
    "rolling_ep_return": -15.354,
    "rolling_ep_len": 653.7,
    "rolling_route_completion": 0.0403,
    "rolling_collision_rate": 0.52,
    "rolling_avg_speed_ms": 3.858,
    "termination_reasons": {"REALLY_STUCK": 8, "COLLISION": 26, "OFF_ROUTE": 16},
    "policy_std": 0.67,
    "approx_kl": 0.00807,
    "clip_fraction": 0.0654,
    "entropy_loss": -2.0371,
    "explained_variance": 0.3578,
    "ent_coef": 0.0199,
    "total_episodes": 231
}
```

**Notes on beacon:**
- `rolling_collision_rate=0.52` — over HALF of recent episodes end in collision
- `rolling_route_completion=0.0403` — 4% RC, same plateau as every other run
- `policy_std=0.67` — the clamp IS working (unlike run-2 where std hit 0.999), but it doesn't matter because the reward signal is still too sparse to learn steering
- `explained_variance=0.358` — value function explains only 36% of return variance, indicating noisy/sparse rewards

## Episode Analysis (129 captured episodes)

### Termination Distribution
```
COLLISION:    64  (50%)
OFF_ROUTE:    53  (41%)
REALLY_STUCK: 12  (9%)
```

Compare to run-2 at 900k steps: OFF_ROUTE 48%, COLLISION 38%, REALLY_STUCK 14%. Same pattern.

### Route Completion Stats (120 non-stuck episodes)
```
Mean RC:   4.40%
Max RC:    30.2%
>10% RC:   14 episodes (12%)
>20% RC:   3 episodes (2.5%)
<2% RC:    48 episodes (40%)
```

40% of episodes achieve less than 2% route completion. The "good" episodes (>10%) are statistical noise from random steering.

### Speed Distribution (120 non-stuck episodes)
```
<0.5 m/s (sitting):     4  (3%)
0.5-2 m/s (crawling):  53  (44%)
2-5 m/s (slow):        43  (36%)
>5 m/s (fast):         20  (17%)
Mean speed: 2.83 m/s
```

**44% of episodes are crawling at 0.5-2 m/s.** The agent is learning to creep forward slowly and bump into things (COLLISION at low speed). The fast episodes (>5 m/s) all end OFF_ROUTE — floor the throttle and drive off the road.

### Phase Analysis

**Early episodes (first 30):** High speed (5-13 m/s), almost all OFF_ROUTE. Agent floors throttle, drives straight off road. Some lucky RC hits: 19.0%, 15.8%, 10.2%.

**Mid episodes (30-90):** Speed drops to 1-3 m/s. COLLISION becomes dominant. Agent learns that going slower reduces off-route penalty but now just bumps into parked NPCs (which don't move). A few REALLY_STUCK episodes appear.

**Late episodes (90-129):** Mixed OFF_ROUTE and COLLISION. REALLY_STUCK clusters appearing (9 in last captured buffer). Speed oscillates between 0.5 and 5 m/s — no convergence.

### Behavioral Modes (same 3 failure modes as all prior runs)

1. **"Floor it off the road"** — speed 5-13 m/s, OFF_ROUTE in 7-20s, RC 0-5%
2. **"Creep and crash"** — speed 0.5-2 m/s, COLLISION in 5-20s, RC 0-5%
3. **"Sit still"** — REALLY_STUCK after 1501 steps of no progress

The agent oscillates between these three modes and never discovers a fourth mode ("steer toward waypoints").

## What the std clamp changed (and didn't change)

**Changed:** policy_std is at 0.67 instead of 0.999. The policy is not maximally random.

**Didn't change:** The agent still can't learn steering direction. The std clamp prevents the policy from becoming pure noise, but if the reward gradient for steering doesn't exist, a slightly-less-noisy policy still can't learn to steer.

## What potential-based shaping changed (and didn't change)

**Changed:** The potential function Φ(s) = -distance_to_lookahead provides a per-step shaping signal.

**Didn't change:** The potential function is based on distance to a lookahead point along the route. But the agent needs to STEER toward that point, and the distance-based potential gives the same reward whether the agent is pointing toward or away from the waypoint. A car facing 180° away from the waypoint at distance d gets the same potential as a car facing directly toward it at distance d. **There is no heading/steering component.**

## Cross-Run Comparison: RC plateau across ALL runs

| Run | Fix | Steps | Mean RC | Max RC | policy_std |
|-----|-----|-------|---------|--------|------------|
| 1 (idle_cost) | idle_cost penalty | 900k | 5.38% | 42.6% | 0.999 (clamp) |
| 2 (reward fix) | same as run 1 | 900k | 5.74% | - | 0.999 (clamp) |
| 3 (fix-2, THIS) | std clamp + shaping + TM | 104k | 4.40% | 30.2% | 0.670 |

**Three runs, three different fix sets, same ~5% RC.** The agent cannot learn to steer. This is not a hyperparameter problem or a reward-scale problem. This is a missing-reward-component problem.

## Root Cause Analysis (updated)

The fundamental issue has been the same since run 1:

**There is no per-step reward signal for steering direction.** The agent gets:
- `progress_reward`: only when car happens to move along the route (sparse)
- `idle_cost`: punishes sitting still (works — agent moves)
- Potential-based shaping: rewards getting closer to route (but not POINTING toward route)
- Terminal rewards: +50/-50 for completion/failure (very sparse)

What's MISSING: a dense per-step signal that says "your heading is X degrees off from the target direction, and that's bad." Without this, the agent must discover the correct steering angle through random exploration, which is combinatorially unlikely.

**CaRL (NVIDIA, CoRL 2025) solves this with:**
```python
# cosine similarity between ego heading and direction to next waypoint
cos_sim = dot(ego_forward, normalize(waypoint - ego_position))
heading_reward = weight * cos_sim  # +reward when facing waypoint, -reward when facing away
```

This gives gradient signal for steering on EVERY step. The current reward only gives gradient for throttle (via idle_cost and progress_reward magnitude).

## Infrastructure Notes

- CARLA stable, no crashes, 119 FPS sustained
- DummyVecEnv stable (no IPC issues)
- Beacon and watchdog working correctly
- Spectator scripts running on both ports
- NaN guard: 0 hits
- Clip hits: 0 (reward magnitudes are fine)

## Recommendations for Remote Claude

1. **MUST ADD: Dense heading reward** — cosine similarity between ego forward vector and direction to next waypoint. This is the single most impactful missing component. Without it, no amount of reward shaping or std clamping will teach steering.

2. **MUST FIX: Frozen NPCs** (Issue #9) — TrafficManager autopilot is not engaging. Possible fix: call `vehicle.set_autopilot(True, tm.get_port())` with explicit port, or restructure the TM sync setup sequence.

3. **CONSIDER: Behavioral cloning warmstart** — commit `dfbcf9e` added a BC pipeline. Given that pure RL from scratch has failed 3 times to learn steering, a BC warmstart from expert demonstrations may be necessary to seed the policy with basic lane-following behavior before RL fine-tuning.

4. **DO NOT: Increase idle_cost or adjust reward scales** — the problem is not reward magnitude, it's a missing reward component. Turning up existing knobs will not help.

## Raw Episode Data (last 60 episodes)

```
REALLY_STUCK steps_since_progress=1501
REALLY_STUCK steps_since_progress=1501
REALLY_STUCK steps_since_progress=1501
COLLISION steps=226 (11.3s) speed=0.6m/s route=4.8% progress=12.5m/258m
REALLY_STUCK steps_since_progress=1501
COLLISION steps=331 (16.6s) speed=0.1m/s route=2.4% progress=9.2m/385m
COLLISION steps=1287 (64.4s) speed=0.6m/s route=1.3% progress=4.4m/333m
REALLY_STUCK steps_since_progress=1501
REALLY_STUCK steps_since_progress=1501
COLLISION steps=129 (6.5s) speed=1.6m/s route=4.9% progress=8.7m/177m
COLLISION steps=159 (8.0s) speed=0.9m/s route=1.1% progress=6.6m/619m
COLLISION steps=163 (8.2s) speed=0.9m/s route=2.4% progress=11.3m/466m
COLLISION steps=415 (20.8s) speed=1.2m/s route=0.8% progress=3.5m/422m
REALLY_STUCK steps_since_progress=1501
COLLISION steps=489 (24.5s) speed=1.8m/s route=12.3% progress=54.2m/442m
COLLISION steps=280 (14.0s) speed=1.1m/s route=10.2% progress=31.1m/304m
COLLISION steps=361 (18.1s) speed=1.1m/s route=1.8% progress=3.9m/216m
COLLISION steps=290 (14.5s) speed=1.0m/s route=2.9% progress=11.3m/388m
COLLISION steps=284 (14.2s) speed=0.6m/s route=1.4% progress=4.7m/338m
COLLISION steps=293 (14.7s) speed=0.6m/s route=13.5% progress=60.8m/451m
COLLISION steps=228 (11.4s) speed=0.9m/s route=7.8% progress=35.6m/456m
COLLISION steps=325 (16.2s) speed=1.4m/s route=4.2% progress=6.3m/149m
COLLISION steps=212 (10.6s) speed=0.5m/s route=1.2% progress=5.9m/482m
COLLISION steps=133 (6.7s) speed=2.1m/s route=2.0% progress=5.1m/254m
OFF_ROUTE steps=272 (13.6s) speed=3.7m/s route=2.5% progress=12.5m/497m
COLLISION steps=259 (13.0s) speed=1.2m/s route=12.3% progress=54.2m/442m
OFF_ROUTE steps=327 (16.4s) speed=5.3m/s route=8.7% progress=18.5m/212m
OFF_ROUTE steps=1054 (52.7s) speed=4.4m/s route=0.0% progress=0.0m/330m
COLLISION steps=190 (9.5s) speed=2.3m/s route=2.1% progress=10.3m/498m
COLLISION steps=436 (21.8s) speed=1.7m/s route=3.1% progress=5.0m/160m
COLLISION steps=371 (18.6s) speed=0.9m/s route=5.5% progress=22.9m/419m
COLLISION steps=309 (15.5s) speed=1.1m/s route=1.3% progress=6.6m/518m
COLLISION steps=274 (13.7s) speed=1.2m/s route=3.0% progress=10.7m/360m
OFF_ROUTE steps=539 (27.0s) speed=6.9m/s route=0.0% progress=0.0m/532m
COLLISION steps=444 (22.2s) speed=0.7m/s route=7.4% progress=9.1m/123m
OFF_ROUTE steps=325 (16.2s) speed=4.5m/s route=0.7% progress=3.6m/507m
OFF_ROUTE steps=395 (19.8s) speed=7.2m/s route=1.6% progress=4.2m/254m
OFF_ROUTE steps=1090 (54.5s) speed=2.9m/s route=30.2% progress=87.9m/291m
REALLY_STUCK steps_since_progress=1501
OFF_ROUTE steps=1545 (77.2s) speed=4.1m/s route=3.2% progress=5.0m/154m
OFF_ROUTE steps=287 (14.4s) speed=3.2m/s route=3.7% progress=9.1m/249m
OFF_ROUTE steps=254 (12.7s) speed=4.3m/s route=4.4% progress=9.1m/207m
COLLISION steps=547 (27.4s) speed=0.7m/s route=1.4% progress=4.5m/317m
OFF_ROUTE steps=283 (14.2s) speed=3.9m/s route=1.2% progress=3.4m/272m
OFF_ROUTE steps=917 (45.9s) speed=2.5m/s route=1.5% progress=4.6m/303m
OFF_ROUTE steps=615 (30.8s) speed=3.8m/s route=4.7% progress=13.7m/291m
OFF_ROUTE steps=410 (20.5s) speed=4.0m/s route=3.8% progress=8.9m/233m
OFF_ROUTE steps=1186 (59.3s) speed=0.9m/s route=0.7% progress=2.1m/291m
COLLISION steps=215 (10.8s) speed=0.8m/s route=1.4% progress=10.5m/739m
OFF_ROUTE steps=314 (15.7s) speed=2.4m/s route=4.9% progress=11.3m/232m
COLLISION steps=741 (37.1s) speed=1.3m/s route=1.3% progress=6.6m/522m
OFF_ROUTE steps=783 (39.2s) speed=5.1m/s route=1.0% progress=3.7m/374m
COLLISION steps=365 (18.2s) speed=1.4m/s route=3.5% progress=10.6m/301m
COLLISION steps=307 (15.4s) speed=0.7m/s route=2.4% progress=11.3m/466m
OFF_ROUTE steps=425 (21.2s) speed=3.6m/s route=8.1% progress=10.6m/131m
OFF_ROUTE steps=510 (25.5s) speed=4.0m/s route=1.4% progress=4.2m/309m
OFF_ROUTE steps=845 (42.2s) speed=2.9m/s route=2.1% progress=10.6m/509m
COLLISION steps=82 (4.1s) speed=0.5m/s route=1.7% progress=7.9m/468m
COLLISION steps=338 (16.9s) speed=1.0m/s route=0.0% progress=0.0m/633m
COLLISION steps=126 (6.3s) speed=0.5m/s route=1.5% progress=8.6m/591m
```
