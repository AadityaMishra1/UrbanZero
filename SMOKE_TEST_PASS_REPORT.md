# v2 SMOKE TEST REPORT (10 min) — PASS

**Date:** 2026-04-22 21:27-21:37 UTC-4
**Tip commit:** `5814854` ("fix(issue-6): switch SubprocVecEnv -> DummyVecEnv")
**CARLA ports:** 2000/3000 UP: yes (fresh instances, killed prior CarlaUE4.exe before launch)
**n_envs:** 2
**VecEnv:** DummyVecEnv (serial stepping, no IPC)

## Results

| Metric | Value |
|--------|-------|
| timesteps after 10 min | 73,692 |
| aggregate FPS | 122 |
| policy_std | 0.6707 |
| approx_kl | 0.01405 |
| entropy_loss | -2.037 |
| explained_variance | 0.5528 |
| ent_coef | 0.01993 |
| cumulative_reward_clip_hits | 0 |
| NaN-GUARD fires | 0 |
| reward-guard fires | 0 |
| total episodes | 53 |
| rolling collision rate | 0.40 |
| rolling avg speed | 1.39 m/s |
| rolling route completion | 2.77% |

## Termination reasons (rolling window 50)

```json
{"COLLISION": 20, "REALLY_STUCK": 18, "OFF_ROUTE": 5, "MAX_STEPS": 7}
```

## Cumulative EPISODE END distribution

```
     22 reason=COLLISION
     21 reason=REALLY_STUCK
      7 reason=MAX_STEPS
      5 reason=OFF_ROUTE
```

## PASS gates

| Gate | Criteria | Result | Actual |
|------|----------|--------|--------|
| P1 | >=40k timesteps | PASS | 73,692 |
| P2 | >=70 FPS | PASS | 122 |
| P3 | NaN-GUARD=0 | PASS | 0 |
| P4 | reward-guard=0 | PASS | 0 |
| P5 | clip_hits=0 | PASS | 0 |
| P6 | reasons populated | PASS | 53 episodes, 4 reason types |
| P7 | policy_std 0.3-1.0 | PASS | 0.6707 |
| P8 | approx_kl present | PASS | 0.01405 |

**Overall: PASS (8/8)**

## Anomalies / observations

1. **REALLY_STUCK is 38% of terminations** (21/55) — nearly tied with COLLISION (40%). Higher than expected for a 10-min smoke window. The agent moves (avg_speed=1.4 m/s) but gets stuck frequently. The REALLY_STUCK threshold (`steps_since_progress=1501` = 75s) may fire too aggressively for a from-scratch policy that hasn't learned steering yet. Consider raising the threshold or adding a warm-up grace period.

2. **DummyVecEnv FPS (122) is surprisingly close to SubprocVecEnv (147 in the 20:03 run).** Only ~17% slower, well above the 70 FPS gate. The IPC overhead of SubprocVecEnv was not giving much benefit for just 2 envs.

3. **explained_variance oscillates between -0.13 and 0.95** — normal for early training with sparse episode boundaries. Value function is learning.

4. **CARLA windows appear to "freeze" intermittently** because DummyVecEnv steps envs serially — each window only ticks every other step. This is cosmetic, not a bug. External spectator scripts (scripts/spectator.py) were used on both ports to follow the ego vehicles.

5. **No F-signals triggered.** No worker timeouts, no watchdog fires, no hangs, no NaN/reward guards. The DummyVecEnv fix completely eliminated the IPC failure class from issues #3/#4/#6.

## Beacon at T+10min

```json
{
    "ts": "2026-04-23T01:37:13Z",
    "pid": 618362,
    "experiment": "v2_rl",
    "status": "training",
    "timesteps": 73692,
    "fps": 122.33,
    "uptime_sec": 602,
    "rolling_ep_count": 50,
    "rolling_ep_return": -8.082,
    "rolling_ep_len": 1387.1,
    "rolling_route_completion": 0.0277,
    "rolling_collision_rate": 0.4,
    "rolling_avg_speed_ms": 1.393,
    "termination_reasons": {
        "COLLISION": 20,
        "REALLY_STUCK": 18,
        "OFF_ROUTE": 5,
        "MAX_STEPS": 7
    },
    "policy_std": 0.6707,
    "approx_kl": 0.01405,
    "clip_fraction": 0.181,
    "entropy_loss": -2.037,
    "explained_variance": 0.5528,
    "ent_coef": 0.01993,
    "cumulative_reward_clip_hits": 0,
    "total_episodes": 53,
    "carla_port": 2000,
    "last_checkpoint": "/home/aadityamishra/urbanzero/checkpoints/v2_rl/autosave_73386_steps.zip"
}
```

## Log tail (last 100 lines of PPO stats)

```
iter 71 | fps=122 | timesteps=72704 | approx_kl=0.009135 | clip_frac=0.0732 | ent_coef=0.0199 | entropy_loss=-2.04 | expl_var=0.536 | std=0.67 | value_loss=0.64
[autosave @ 73386 steps, 10.0min since last save]
iter 72 | fps=122 | timesteps=73728 | approx_kl=0.014052 | clip_frac=0.181 | ent_coef=0.0199 | entropy_loss=-2.04 | expl_var=0.553 | std=0.671 | value_loss=0.385
iter 73 | fps=122 | timesteps=74752 | approx_kl=0.004596 | clip_frac=0.0247 | ent_coef=0.0199 | entropy_loss=-2.04 | expl_var=0.937 | std=0.673 | value_loss=0.00744
iter 74 | fps=122 | timesteps=75776 | approx_kl=0.008905 | clip_frac=0.115 | ent_coef=0.0199 | entropy_loss=-2.05 | expl_var=-0.128 | std=0.674 | value_loss=0.613
```
