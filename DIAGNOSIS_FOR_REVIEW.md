# UrbanZero Training Failure — Technical Briefing for External Review

**Date:** 2026-04-23 ~04:00 local (UTC-4)
**Repo:** https://github.com/AadityaMishra1/UrbanZero branch `claude/setup-av-training-VetPV`
**Deadline:** Saturday 2026-04-25 EOD (~44 hours remaining)

## 0. Request

I am an NCSU ECE 591 student. Over the last ~12 hours my AI coding
assistant and I have been trying to train a PPO agent for autonomous
driving in CARLA 0.9.15. We've had **six consecutive failed training
runs**, and the AI assistant's proposed fixes keep missing the root
cause. I need an external review from someone with real ML/RL depth
to identify what we're actually doing wrong before I burn more
compute on a seventh attempt.

This doc is self-contained. Read it top-to-bottom.

---

## 1. Task / Environment

- **Task:** tabula-rasa RL autonomous driving in CARLA 0.9.15, Town01.
  Agent spawns on a driving lane, is given a planned route (via
  `GlobalRoutePlanner`, 200-800m), and must reach the end without
  collision or leaving the route.
- **Observation space** (SB3 `Dict`):
  - `image`: (1, 128, 128) float32 in [0,1] — single-channel normalized
    semantic-segmentation camera (class labels / 27)
  - `state`: (10,) float32 in [-1.5, 1.5]:
    - [0] speed / MAX_SPEED (14.0 m/s)
    - [1-6] three future waypoints in ego-frame (dx, dy), clamped
    - [7] signed lane offset / 5m, clamped
    - [8] traffic light state (0 no light, 0.33 green, 0.67 yellow, 1.0 red/unknown)
    - [9] route completion fraction
  - `VecFrameStack(n_stack=4, channels_order={"image":"first", "state":"last"})`
    → image (4, 128, 128), state (40,)
- **Action space:** `Box(-1, 1, shape=(2,))`
  - `action[0]` = steering ∈ [-1, 1]
  - `action[1]` = throttle_brake ∈ [-1, 1], decoded in `env.step()`:
    ```python
    shifted = action[1] + 0.3          # idle-creep bias
    throttle = max(0, min(1, shifted))
    brake = max(0, min(1, -shifted))
    ```
    So `action[1]=0` → throttle=0.3, `action[1]<-0.3` → brake.
- **Episode terminals:**
  - `COLLISION`: impulse > 2500 N, reward −50
  - `OFF_ROUTE`: min 2D distance to any waypoint in the search window > 30 m, reward −50
  - `REALLY_STUCK`: no 1 m of forward route-progress for 1500 sim steps (75 s at 20 Hz), reward −50
  - `ROUTE_COMPLETE`: 2D distance to final waypoint < 5 m AND route_completion > 0.85 AND speed < 3 m/s, reward +50
  - `MAX_STEPS`: truncation at 2000 steps, no terminal reward
- **CARLA setup:** sync mode, `fixed_delta_seconds = 0.05` (20 Hz),
  traffic manager on, 30 NPC vehicles + 10 pedestrians per env
- **Hardware:** Ryzen 7 9800X3D + RTX 4080 Super (16 GB VRAM) +
  Windows host running CARLA 0.9.15 + WSL2 running the trainer
- **2 parallel envs** via `DummyVecEnv` (single process), one CARLA
  server per port (2000, 3000), ~120 FPS aggregate training throughput

## 2. Reward function (current, pure-RL profile)

```python
def _compute_reward(self, action):
    speed = self._get_speed()

    # 1. Route progress — CaRL-minimal (Jaeger et al. 2025 §3.2)
    progress_delta = self._advance_route_index()
    TARGET_PROGRESS_CAP = TARGET_SPEED * 0.05   # = 0.4165 m/step @ 20 Hz, TARGET_SPEED = 8.33 m/s
    capped_progress = min(progress_delta, TARGET_PROGRESS_CAP)
    progress_reward = 0.05 * capped_progress    # max 0.021/step

    # 2. Velocity carrot (persistent, un-annealed)
    carrot = 0.005 * min(speed, TARGET_SPEED) / TARGET_SPEED  # max 0.005/step

    # 3. Idle cost (anti-stall) — the load-bearing pure-RL term
    idle_cost = self._idle_cost_coef * max(0.0, 1.0 - speed / 1.0)  # -0.15 * ramp at speed=0

    reward = progress_reward + carrot + idle_cost

    # 4. Terminals (± 50) — checked here, replace `reward` if fired
    # (COLLISION / OFF_ROUTE / ROUTE_COMPLETE / REALLY_STUCK)

    # 5. Potential-based shaping (Ng/Harada/Russell 1999) — added AFTER terminals
    # Φ(s) = -0.015 · min(dist2D(ego, lookahead_point), 30 m)
    # lookahead is 10 m along route from current projection (continuous)
    # F(s,s') = 0.99 · Φ(s') - Φ(s) non-terminal; F = -Φ(prev) at terminal
    if terminated:
        shaping = -self._prev_potential
    else:
        cur_potential = self._potential()
        shaping = 0.99 * cur_potential - self._prev_potential
        self._prev_potential = cur_potential
    reward += shaping   # max |F|/step ≈ 0.0105

    # Defensive clip to [-100, 100]; VecNormalize(norm_reward=True) further
    # normalizes and clips z-score to ±10
    return reward, terminated
```

**Per-step reward magnitudes:**
- Max per-step (on-route at TARGET_SPEED): ~0.021 (progress) + 0.005 (carrot) + 0.0105 (shaping) = ~0.037
- Min per-step (stopped): 0 (progress) + 0 (carrot) + −0.15 (idle_cost) = −0.15
- Terminals dominate shaping sum: ±50 vs ~21 total shaping over 1000 steps

## 3. PPO configuration

- **Framework:** stable-baselines3 2.x, SB3 `PPO` class
- **Policy:** custom `ClampedStdPolicy(ActorCriticPolicy)`. Soft upper
  clamp on `log_std` at `log(0.7) ≈ -0.357` (std ≤ 0.7), no lower clamp
- **Feature extractor:** `DrivingCNN`:
  - 5-layer Conv2d (32→64→128→128→256) with LayerNorm + ReLU on image
  - Linear(state_dim, 64) → ReLU → Linear(64, 64) → ReLU on state
  - Concat + Linear(320, 256) → ReLU → 256-dim output
- **Policy head:** `net_arch=dict(pi=[256, 128], vf=[256, 128])`,
  `log_std_init=-0.5` (std ≈ 0.607 at init)
- **Env wrapper stack:** `CarlaEnv → NaNGuardWrapper → DummyVecEnv
  → VecFrameStack(n_stack=4) → VecNormalize(norm_obs=False, norm_reward=True, clip_reward=10)`
- **Hyperparameters (pure-RL defaults):**
  - `lr = 3e-4`, `n_steps = 2048`, `batch_size = 64`, `n_epochs = 3`
  - `gamma = 0.99`, `gae_lambda = 0.95`, `clip_range = 0.2`
  - `ent_coef` linearly annealed 0.02 → 0.01 (floor) over 10M steps
  - `vf_coef = 0.5`, `max_grad_norm = 0.5`

## 4. What's been tried — six runs, all failed

### Run summary table

| Run | Seed | Env | Change | Duration | Final rolling RC | Terminal dist | Root symptom |
|---|---|---|---|---|---|---|---|
| v1 pure-RL | 42 | idle_cost=−0.15 added | 233k steps / 30 min | RC 1.78%, avg_speed 0.224 m/s | 70% REALLY_STUCK, 6% COLL | Sit-still attractor |
| v2 pure-RL | 137 | + un-anneal carrot | 900k steps / 2 h | RC 5.4% (flat) | 48% OFF_ROUTE, 38% COLL | policy_std=0.999 pinned at clamp |
| v3 pure-RL | 211 | + log_std ≤ 0.7 + potential shaping + TM hybrid physics | 104k steps / 15 min | RC 4.4% (flat) | 50% COLL, 41% OFF_ROUTE | std=0.67, wrong-lane / crash |
| BC+PPO v1 | 911 | lr=3e-4, ent=0.02 | 28k steps / 4 min | RC 5.6% (flat) | 68% COLL | approx_kl=0.079, catastrophic unlearning |
| BC+PPO v2 | 912 | lr=1e-4, ent=0.005 | 31k steps / 5 min | RC 5.9% (flat) | 64% COLL | approx_kl=0.086 (worse), entropy_loss=+0.255 |
| BC+PPO v3 | 913 | + widen σ=0.5, n_ep=1, clip=0.1 | 2.4M steps / 5 h | RC 5.9% (flat) | 12% COLL, 50% OFF_ROUTE, 38% STUCK | std drifted 0.50→0.58, speed collapsed to 1.0 m/s, frozen PPO |

### Raw beacon data per run (key metrics)

**Run v1 (idle_cost added) at 233k steps:**
```json
{"timesteps":233522,"rolling_route_completion":0.0178,"rolling_avg_speed_ms":0.224,
 "rolling_collision_rate":0.06,"policy_std":0.7874,"approx_kl":0.01433,
 "entropy_loss":-2.3589,"explained_variance":-0.0663,"ent_coef":0.01977,
 "termination_reasons":{"COLLISION":3,"REALLY_STUCK":35,"MAX_STEPS":12}}
```
**Interpretation:** Per-step cost of sitting (−0.033/step from REALLY_STUCK terminal) was cheaper than crashing (−0.167/step). Policy correctly found the sit-still local optimum.

**Run v2 (+ un-anneal carrot) at 892k steps:**
```json
{"timesteps":892236,"rolling_route_completion":0.0574,"rolling_avg_speed_ms":3.174,
 "rolling_collision_rate":0.32,"rolling_ep_len":898,"policy_std":0.999,
 "approx_kl":0.00614,"entropy_loss":-2.36,"ent_coef":0.01911,
 "termination_reasons":{"OFF_ROUTE":20,"COLLISION":16,"REALLY_STUCK":14}}
```
**Interpretation:** `policy_std` pinned at the 1.0 upper clamp for 1500+ episodes. Policy was effectively random noise. `explained_variance` oscillated −0.07 to +0.91.

**Run v3 (log_std ≤ 0.7 + potential shaping) at 104k steps:**
```json
{"timesteps":104272,"fps":119.46,"rolling_ep_return":-15.354,"rolling_ep_len":653.7,
 "rolling_route_completion":0.0403,"rolling_collision_rate":0.52,
 "rolling_avg_speed_ms":3.858,"policy_std":0.67,"approx_kl":0.00807,
 "clip_fraction":0.0654,"entropy_loss":-2.0371,"explained_variance":0.3578,
 "ent_coef":0.0199,
 "termination_reasons":{"REALLY_STUCK":8,"COLLISION":26,"OFF_ROUTE":16}}
```
**Interpretation:** std clamp worked (std=0.67, not pinned). But same three failure modes: "floor it off road" (OFF_ROUTE), "creep and crash" (COLLISION), "sit still" (REALLY_STUCK). **PC operator reported: all spawned NPCs were visibly frozen despite `tm.set_hybrid_physics_mode(True)` and `tm.set_synchronous_mode(True)`.**

**Pivot point:** after 3 pure-RL failures, user relaxed the
tabula-rasa constraint. I switched to Behavior Cloning warmstart.

**BC training results (100k frames from CARLA BehaviorAgent("normal")
on Town01, Gaussian NLL loss):**
```
epoch 01/20  NLL=-0.84  MAE=0.1951  log_std=[-0.561, -0.565]  std=[0.571, 0.568]
epoch 20/20  NLL=-2.93  MAE=0.0503  log_std=[-1.503, -1.587]  std=[0.223, 0.205]
```
Converged cleanly, monotonic loss decrease. Final MAE 0.05 on
expert actions. Call this the **"frozen BC policy"** — saved as
`bc_pretrain.zip`.

**BC+PPO v1 (lr=3e-4, ent_coef=0.02) at 28k steps:**
```json
{"timesteps":28144,"rolling_route_completion":0.0556,"rolling_collision_rate":0.68,
 "rolling_avg_speed_ms":7.599,"policy_std":0.2222,"approx_kl":0.07924,
 "clip_fraction":0.3418,"entropy_loss":0.1717,"explained_variance":0.6898,
 "ent_coef":0.01997,
 "termination_reasons":{"OFF_ROUTE":15,"COLLISION":34,"REALLY_STUCK":1}}
```
Notable: `avg_speed=7.6 m/s` from step 0 (BC prior clearly loaded —
pure-RL starts at 0-3 m/s). `policy_std=0.22` (BC-tight). But
`approx_kl=0.079` ≈ **5× healthy target of 0.015**. `clip_fraction=0.34`
≈ 2× healthy target of 0.15. Several individual episodes hit 30-38%
RC in first minutes, then collapsed as PPO eroded the BC prior.

**BC+PPO v2 (lr=1e-4, ent_coef=0.005→0.001) at 31k steps:**
```json
{"timesteps":31398,"rolling_route_completion":0.0586,"rolling_collision_rate":0.64,
 "rolling_avg_speed_ms":7.941,"policy_std":0.2133,"approx_kl":0.08643,
 "clip_fraction":0.2503,"entropy_loss":0.2548,"explained_variance":0.586,
 "ent_coef":0.00499,
 "termination_reasons":{"COLLISION":32,"REALLY_STUCK":2,"OFF_ROUTE":16}}
```
3x lr reduction had ~no effect on KL. `approx_kl=0.086` slightly
**worse** than v1. `entropy_loss=+0.255` (positive = entropy
decreasing = std collapsing further). `explained_variance` dropping.

**BC+PPO v3 (lr=1e-4, widen log_std to −0.69 (std=0.5), n_epochs=1, clip=0.1) gate trajectory:**
```
T+5min   (34k):  RC 5.56%, speed 4.5, std 0.503, kl 0.0017, clip 0.105, coll% 76
T+30min (228k):  RC 4.67%, speed 1.7, std 0.510, kl 0.0006, clip 0.017, stuck% 48  ← SOFT FAIL
T+3h   (950k):   RC 5.53%, speed 3.1, std 0.534, kl 0.0033, clip 0.159, offr% 64   ← FAIL
T+5h  (2.4M):    RC 5.86%, speed 1.0, std 0.579, kl 0.0015, clip 0.084, stuck% 38  ← FAIL

Final v3 beacon JSON:
{"timesteps":2407014,"rolling_route_completion":0.0586,"rolling_collision_rate":0.12,
 "rolling_avg_speed_ms":0.969,"rolling_ep_len":1117.7,"rolling_ep_return":-16.456,
 "policy_std":0.5786,"approx_kl":0.00155,"clip_fraction":0.084,
 "entropy_loss":-1.7423,"explained_variance":0.848,"ent_coef":0.00404,
 "termination_reasons":{"OFF_ROUTE":25,"REALLY_STUCK":19,"COLLISION":6},
 "total_episodes":2394}
```
**v3 interpretation:** KL and clip_fraction became textbook healthy
(kl=0.0015, clip=0.08) — PPO updates are stable. **But policy is
frozen / mostly random.** `entropy_loss=-1.74` (negative = entropy
increasing = std widening). `policy_std` drifted 0.50→0.58 through
the run. `avg_speed` collapsed from 4.5 → 1.0 m/s. Zero
ROUTE_COMPLETE episodes in 2394 total. `explained_variance=0.848` is
the only healthy number — the critic is fine; the actor is stuck.

### Consistent observations across ALL six runs

1. **Rolling RC plateau around 5-6%.** Across 6 runs with wildly
   different hyperparameters and different algorithms (pure-RL
   from scratch vs BC+PPO finetune), final rolling RC is always
   4.4–5.9%. This is either a real compute-budget ceiling or a
   common underlying bug.
2. **NPCs are visibly frozen in CARLA viewports.** Despite
   `tm.set_synchronous_mode(True)`, `tm.set_hybrid_physics_mode(True)`,
   `tm.set_hybrid_physics_radius(70.0)`, and (latest attempt)
   `tm.global_percentage_speed_difference(-30.0) + commit world.tick()`,
   NPCs do not move. All COLLISIONs are with stationary NPCs or
   static geometry.
3. **Some individual episodes hit 20-40% RC**, across all runs —
   including pure-RL — but these are statistical flukes, not
   learning. Rolling average doesn't climb.
4. **`explained_variance` varies widely** (0.36 pure-RL, 0.85 BC+PPO
   v3), indicating the critic can learn; it's the actor / policy
   gradient that's stuck.

## 5. Paper references that guided the design

- **Jaeger, Chitta, Geiger 2025** "CaRL" (arXiv:2504.17838) — CARLA
  PPO with minimal `progress + terminal` reward, ~85% Leaderboard score
  at 16-300 envs, 100M+ steps. Our reward is modeled on this, rescaled.
  **Our compute budget is ~1/50th of CaRL's.**
- **Ng, Harada, Russell 1999** "Policy Invariance Under Reward
  Transformations" — potential-based shaping `F = γΦ(s') - Φ(s)`
  preserves optimal policy. Our shaping term cites this.
- **Andrychowicz et al. 2021** "What Matters in On-Policy RL" —
  `ent_coef ∈ [0.003, 0.03]`, `std ∈ [0.3, 0.7]` viable band.
  Our clamp and schedule cite this.
- **Schulman et al. 2017** PPO — defaults `lr=3e-4`, `n_epochs=10`,
  `clip=0.2`. We use `n_epochs=3` (timing compromise).
- **Henderson et al. 2018** "Deep RL That Matters" — don't judge
  before 3M steps. (We've violated this constraint due to deadline
  pressure.)
- **Zhang et al. 2021** "Roach" — BC + PPO finetune with
  `KL-to-BC regularization β=0.1→0.05` over 2M steps, lr=1e-5 for BC
  finetune. **We skipped KL-to-BC** because the AI estimated it as
  4h of custom SB3 subclass work that wouldn't fit the timeline. This
  may have been the wrong call.
- **Rajeswaran et al. 2017** — never anneal ent_coef to 0, keep floor.
- **Codevilla et al. 2019** "Exploring Limitations of BC" — BC with
  MSE on multimodal expert actions regresses to mean.

## 6. Current hypothesis being tested (v4)

**H_v4:** The reward function (designed for pure-RL to break the
sit-still attractor) actively **punishes BC's expert behavior**
during finetune. Specifically:

- `idle_cost = -0.15/step` fires when speed < 1 m/s
- BC's `BehaviorAgent` expert stops at red lights, slows for traffic,
  pauses at intersections — all speeds < 1 m/s
- A 10-second red-light wait (200 steps) = `-30 reward`
- Over a 3-red-light route: -60 to -90 reward just for correct stopping
- PPO sees these trajectories as low-reward, gradient-pushes policy
  away from BC's stopping behavior
- BC taught "stop at red lights"; reward says "stopping is bad" →
  irreconcilable conflict
- Layered on top: `ent_coef=0.005` bonus pushes σ upward; without a
  strong coherent task-reward signal to pull it back, σ drifts to
  random (v3's observed pattern)

**v4 changes (BC-finetune only — pure-RL reward unchanged):**

1. `idle_cost = 0` via `URBANZERO_IDLE_COST_COEF=0` env var
2. `REALLY_STUCK` threshold 1500 → 3000 steps (150s grace)
3. `ent_coef = 1e-4` constant (no schedule) — BC provides stochasticity
4. Revert v3's log_std widening — keep BC's σ=0.22 (fine once reward
   doesn't fight it)
5. Keep v3's `lr=1e-4, n_epochs=1, clip_range=0.1` as additional safety

### Pre-registered falsification criteria

At T+30min, v4 is **falsified** if ≥2 of:
- avg_speed < 2.5 m/s
- REALLY_STUCK% > 25%
- policy_std > 0.40

(Halfway between v3's observed values and v4's predictions.)

## 7. What I want your friend to evaluate

Critical questions I genuinely don't know the answer to:

### Q1. Is H_v4 plausible as the primary cause, or am I still missing something?

The reward-BC conflict story is mechanically compelling to me, but I've
had compelling stories for v1/v2/v3 that all turned out to be wrong.

Alternative hypotheses that I've considered but not tested:

- **H_A: NPCs frozen is the dominant bug.** All collisions are with
  static cars, so the collision terminal doesn't teach dynamic
  obstacle avoidance. If this is dominant, fixing NPCs first should
  change the failure pattern.
- **H_B: Compute budget is simply insufficient.** CaRL at 1/50× our
  budget → we cap at whatever RC we're seeing. No hyperparameter
  fix can close the gap.
- **H_C: Implementation bug in reward signal.** Progress projection,
  VecNormalize stats, terminal masking, something that's silently
  corrupting the advantage signal.
- **H_D: BC policy has a latent issue.** Multimodal actions
  regressed-to-mean during NLL training; tight σ hides this.
- **H_E: Missing KL-to-BC.** Skipping Roach's KL-to-BC regularization
  means PPO has no anchor to BC, so it's fundamentally unstable in
  the BC+PPO regime regardless of what the reward looks like.

### Q2. Given these six runs, what's the best single action now?

Options on the table:
- **A)** Run v4 as described (30-min falsification test)
- **B)** Skip v4, ship frozen BC eval + writeup (BC alone averages 5-30% RC in individual episodes, deterministic eval would give a clean number)
- **C)** Implement KL-to-BC regularization properly (Roach §3.2), takes ~4 hours, then rerun
- **D)** Fix the NPC bug first, then any of A/B/C
- **E)** Something I haven't considered

### Q3. If H_v4 is right, why did v3's KL look perfect (0.0015) yet the policy still froze?

v3's KL metrics were textbook healthy. If KL is healthy, PPO should
be taking small stable updates. Yet the policy drifted σ upward and
collapsed to 1 m/s over 2.4M steps. What's actually happening?

My working explanation: `ent_coef > 0` adds a `+ent_coef` gradient
on `log_std` every update. When task-reward gradient is weak/
conflicted (per H_v4), this entropy gradient wins by default and σ
drifts upward. KL stays small because each individual update is
small — but many small "widen σ" updates still compound over
millions of steps.

Is this right? Is there a simpler/different explanation?

### Q4. Is the sit-still attractor real, or an artifact of the reward?

In pure-RL v1, the agent found "sit still" as the optimal policy
because per-step cost of being stuck was cheaper than crashing
(−0.033 vs −0.167). I added `idle_cost=-0.15` to invert that. But
now I'm told to remove it for BC context. Is there a reward
formulation that works for both pure-RL AND BC+PPO? Or are these
genuinely separate optimization problems requiring separate rewards?

## 8. Code / file references

GitHub: https://github.com/AadityaMishra1/UrbanZero/tree/claude/setup-av-training-VetPV

Key files:
- `env/carla_env.py` — env + reward (`_compute_reward` line ~737)
- `models/clamped_policy.py` — policy with log_std clamp
- `models/cnn_extractor.py` — DrivingCNN architecture
- `agents/train.py` — PPO setup + BC warmstart path
- `agents/train_bc.py` — BC trainer (Gaussian NLL)
- `scripts/collect_bc_data.py` — BC data collection with BehaviorAgent
- `scripts/eval_bc.py` — deterministic eval script for frozen BC
- `PROJECT_NOTES.md` — full scientific memory, every failure logged

GitHub Issues with raw beacon data per run:
- #9 BC+PPO v3 T+30min SOFT-FAIL
- #10 BC+PPO v3 T+3h FAIL
- #11 BC+PPO v3 midpoint FAIL (2.4M steps)

## 9. What I need from your friend

Any of:

1. **"You're missing X"** — a failure mode I haven't considered.
   Especially if it's visible in the beacon data I pasted above.
2. **"H_v4 is wrong because Y"** — a reason the reward-BC conflict
   story doesn't actually explain what we're seeing.
3. **"Don't run v4, do Z instead"** — a better next step given
   time remaining.
4. **"Ship BC frozen, here's how to frame the report"** — if they
   think further experiments are wasted compute.

I'll do whatever they say. I'd rather ship a correctly-framed
frozen-BC result than burn another 5 hours on a 7th failed run.

Deadline is Saturday EOD (~44h). I can realistically try one more
experiment (~4-6h including train + eval + report write) before
having to commit to a final deliverable.
