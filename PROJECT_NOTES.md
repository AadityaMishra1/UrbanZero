# UrbanZero Project Notes — Scientific Memory

**Purpose:** Single source of truth for design decisions, paper citations,
empirical findings, and agreed-upon plans for the v2 rewrite of UrbanZero
(CARLA + PPO autonomous-driving RL). Written so a future conversation can
pick up the thread without re-deriving context from the git log. Update
this file when a design decision changes.

**Last written:** 2026-04-23, tip `0a3f114` (code) / `90f14d3` (docs).
**Branch:** `claude/setup-av-training-VetPV`.
**Author-at-time-of-writing:** the remote-side Claude working with user
AadityaMishra1.

---

## 0. Project context and deadline

- Course project: NCSU ECE 591.
- Original stated goal (`UrbanZero.pdf`): train an autonomous vehicle agent to
  navigate an urban CARLA environment entirely via PPO, no expert
  demonstrations or hand-engineered behaviors. Tabula-rasa RL.
- **Deadline: Saturday 2026-04-25**, end of day. From when these notes were
  written (Thu 2026-04-23, early AM), ~60–72 hours remain.
- **The tabula-rasa constraint has been explicitly relaxed by the user.** If
  pure RL fails to train by a certain gate, BC warmstart is allowed. See §8
  for the agreed plan.
- Hardware: single PC, Ryzen 7 9800X3D + RTX 4080 Super (16 GB VRAM) +
  Windows host running CARLA 0.9.15 + WSL2 running the trainer.

---

## 1. The 7M-run failure — what actually happened

Prior-author Claude sessions ran 7M PPO steps across two phases
(Phase 1: no traffic, 2M steps; Phase 2: 15 NPCs, 5M steps). The failure
is documented in commit `834a8e0` ("FINAL REPORT").

### 1.1 Outcome
- **Zero route completions across 81,606 episodes.**
- Rolling route completion peaked at 7.99% at Phase-2 step 1.14M,
  then regressed to 5.95% by the end. Peak was never reshipped
  because the trainer had no best-checkpoint callback.
- 88.4% of final episodes ended in `COLLISION`, 7.2% `OFF_ROUTE`,
  4.3% `STAGNATION`, 0% `ROUTE_COMPLETE`.
- Policy entropy collapsed: `std` 0.367 → 0.230,
  `entropy_loss` −0.831 → +0.115 (near-deterministic).
- Explained variance stayed at 0.98+ — the critic perfectly fit the
  policy's bad returns, which is the diagnostic for a deterministic
  local optimum with no advantage signal left.

### 1.2 Specific failure modes (each traceable to a documented RL pathology)
1. **Perpendicular-circling attractor.** `speed_reward *= cos(angle)` at
   the old `env/carla_env.py:644-676` was non-potential shaping. At 90°
   to route, `speed_reward = 0`, idle penalty = 0 (speed ≥ 1.5),
   `progress_reward` still fires on tiny forward projection. Local
   attractor. Textbook case of Ng/Harada/Russell 1999.
2. **Standing-still as global optimum.** A report at commit
   `c625ef4` showed 94.4% `STAGNATION` at 172k steps, 0.009 m/s rolling
   speed. Old reward had thresholded idle penalties the agent could dodge.
3. **Stagnation-counter twitch-game.** Commit `8791388`'s own log
   explicitly described this: "agent barely twitches to decrement counter,
   never actually drives. Terminal-only penalties (even -5.0) can always
   be gamed."
4. **1.01 m/s hover attractor.** Commit `c9cd303` noted the discontinuity
   at speed = 1.0 m/s in the idle-penalty ramp created an attractor.
5. **Overspeed hack.** Commit `6aeb21e` noted the agent converged to
   18 m/s in a 30 km/h zone because `progress_delta` clamped at 1.5 m
   meant more speed = more reward up to ~22 m/s.
6. **Peak-then-regress.** No rolling-best checkpoint → shipped the
   worse late model.
7. **Entropy collapse.** `ent_coef = 0.001` with `log_std` clamped to
   `[-2.0, -0.7]` → the clamp pinned std to the lower floor before
   the agent discovered the route.
8. **Terminal reward invisible to critic.** Shaping sum (undiscounted)
   over a 1000-step episode ≈ 2400 vs terminal ±10. 240:1 ratio.
   Critic fit shaping; terminal never surfaced.

---

## 2. Paper citations driving every v2 design decision

Organized by topic so future-me can look up "why did we do X" in one place.

### 2.1 Reward design (the experiment's core)
- **Jaeger, Chitta, Geiger 2025** — "CaRL: Learning Scalable Planning
  Policies with Simple Rewards." arXiv:2504.17838.
  CARLA PPO with minimal `progress_delta + terminal` reward, ~85 Leaderboard
  at 16–300 envs, 100M+ steps. The reward formulation we're using is
  theirs, rescaled for our terminal/shaping ratio.
- **Ng, Harada, Russell 1999** — "Policy Invariance Under Reward
  Transformations." ICML. Potential-based shaping theorem: arbitrary
  (non-potential) shaping creates new optima. **This is the load-bearing
  theorem we invoke to justify deleting the signed-cos speed term, idle
  penalty, stagnation counter, smoothness penalty, and lateral penalty.**
- **Krakovna et al. 2020** — "Specification gaming: the flip side of AI
  ingenuity." DeepMind. Catalogues reward-hacking examples including
  circling-type attractors. The 7M-run circling matched this pattern.
- **Schulman et al. 2017** — "Proximal Policy Optimization Algorithms."
  OpenAI. Source for PPO defaults (`lr=3e-4`, `n_epochs=10`, `clip=0.2`).
- **Hessel et al. 2018** — "Rainbow: Combining Improvements in DQN."
  AAAI. §3.2 warmup shaping. Inspiration for annealed velocity carrot.
- **Vinyals et al. 2019** — "Grandmaster level in StarCraft II using
  multi-agent reinforcement learning." Nature (AlphaStar). Early-phase
  progress shaping that decays — same pattern as our carrot.

### 2.2 Exploration / entropy
- **Andrychowicz et al. 2021** — "What Matters in On-Policy RL." ICLR.
  §4.5 benchmarks `ent_coef ∈ [0.003, 0.03]` as the viable band for
  continuous control from scratch. **This is why we use 0.02 → 0.01
  (floor), not 0.001 (prior) which is below the viable band.**
- **Rajeswaran et al. 2017** — "Learning Complex Dexterous Manipulation
  with Deep RL and Demonstrations." RSS. §3 — when fine-tuning from
  imitation, annealing `ent_coef` to zero collapses the policy back to
  random. **Justifies our nonzero ent_coef floor of 0.01, never 0.**
- **Berner et al. 2019** — "Dota 2 with Large Scale Deep Reinforcement
  Learning." OpenAI Five. PPO with adaptive entropy. Reference for
  maintaining exploration pressure over long training.

### 2.3 Implementation correctness
- **Engstrom et al. 2020** — "Implementation Matters in Deep Policy
  Gradient Algorithms." ICLR. Subtle implementation details (reward
  normalization, advantage normalization, etc.) dominate algorithmic
  choices. Reason we kept VecNormalize reward but added explicit
  `[−100, 100]` env-side clip.
- **Henderson et al. 2018** — "Deep Reinforcement Learning That Matters."
  AAAI. RL early-stopping is famously unreliable; any conclusion drawn
  before ~3M steps is suspect. **Reason we committed to running at
  least 3M clean steps before drawing any scientific conclusion.**
- **Pardo et al. 2018** — "Time Limits in Reinforcement Learning." ICML.
  Correct GAE bootstrap under episode truncation — important because
  `REALLY_STUCK` is a truncation but `max_episode_steps` is a time
  limit that should bootstrap.
- **Sutton & Barto**, *Reinforcement Learning: An Introduction*. §3.4
  permits episode truncation without MDP distortion; §17.3 covers
  episode cap semantics.
- **Schulman et al. 2016** — "High-Dimensional Continuous Control Using
  Generalized Advantage Estimation." ICLR. GAE formulation (we use
  `λ = 0.95`, `γ = 0.99`).

### 2.4 Imitation / BC (the fallback lever)
- **Zhang et al. 2021** — "End-to-End Urban Driving by Imitating a
  Reinforcement Learning Coach" (Roach). ICCV. BC + PPO finetune with
  KL-to-BC regularization. Documented path to ~80% LB score in CARLA.
- **Chen et al. 2019** — "Learning by Cheating" (LBC). CoRL. Privileged
  BEV teacher → camera student via DAgger.
- **Chen & Krähenbühl 2022** — "Learning from All Vehicles" (LAV). CVPR.
  Expert-supervised with privileged info.
- **Codevilla et al. 2019** — "Exploring the Limitations of Behavior
  Cloning for Autonomous Driving." ECCV. Documents BC failure modes;
  multimodal expert → MSE-BC regresses to mean (e.g., "stop or go" at
  yellow light averages to compromise action).
- **Ross & Bagnell 2010** — "A Reduction of Imitation Learning and
  Structured Prediction to No-Regret Online Learning." ICML. DAgger
  motivation — justifies LBC-style action-noise injection during data
  collection.
- **Florence et al. 2022** — "Implicit Behavioral Cloning." CoRL.
  Conditional VAE for multimodal BC. Cited for alternate BC loss design
  if MSE BC produces mean-regressed policies on `BehaviorAgent` data.
- **Toromanoff et al. 2020** — "End-to-End Model-Free Reinforcement
  Learning for Urban Driving." CVPR. Quantized action classification
  for BC to handle saturation; reference if MSE BC is too smooth.

### 2.5 Action distribution and smoothness (referenced but NOT adopted)
- **Silver et al. 2018** — "Residual Policy Learning." Motivated the
  `throttle_brake + 0.3` idle-creep offset in `step()`. Kept from v1.
- **Chou, Maturana, Scherer 2017** — "Improving Stochastic Policy
  Gradients with Beta distributions." Considered for action
  distribution; rejected for this project because SB3 has no first-class
  Beta class and the re-implementation risk is not paper-cited-justified
  at our scale. We kept Gaussian + env-side clip.
- **Raffin et al. 2021** — "Smooth Exploration for Robotic RL" (gSDE).
  Considered; not adopted (extra implementation burden, no paper
  evidence of benefit for CARLA specifically).
- **Mysore et al. 2021** — "CAPS: Conditioning Action Policy for
  Smoothness." ICRA. Inspired the old v1 smoothness penalty; deleted
  in v2 because the BC warmstart (if triggered) provides a smoother
  prior, and the smoothness term was exploitable.

### 2.6 CARLA 0.9.15 known issues (NOT paper citations, but load-bearing)
- **CARLA #9172** — `TrafficManagerLocal.cpp` lost-notify race in 0.9.15.
  Fires when tick cadence slows (e.g., PPO train phase). I initially
  misdiagnosed this as the root cause of issue #4; it may contribute
  but is not the only cause.
- **CARLA #1996** — "only one ticking client supported in sync mode."
  Multi-client sync is explicitly unsupported.
- **CARLA #2239** — documented multi-client sync deadlock stacks.
- **CARLA #2789** — `get_trafficmanager()` hang path.
- **CARLA #3288** — TM port collision / registration issues.

---

## 3. v2 design — what changed and why (with citations)

Organized into "experiment" (what we're actually testing) vs
"infrastructure" (what makes the experiment runnable). **This
distinction is load-bearing.** Conflating them caused issues #3/#4/#6.

### 3.1 Experiment changes (the scientific hypothesis under test)

**Reward function** — rewrite to CaRL-minimal (Jaeger 2025 §3.2):
```
r_t   (step)     = 0.05 * min(progress_delta_m, TARGET_SPEED * dt)
                 + 0.005 * min(speed, TARGET_SPEED)/TARGET_SPEED * anneal_coef
r_T   (collision, off_route>30m, really_stuck 1500-step)  = -50
r_T   (route_complete, rc>0.85, 2D dist<5m, speed<3 m/s)  = +50
```
`anneal_coef = max(0, 1 - worker_step / CARROT_DECAY_STEPS)`,
default `CARROT_DECAY_STEPS = 500_000` per worker. Carrot is annealed
to zero (Hessel 2018 / Vinyals 2019 warmup pattern) so final reward
is pure CaRL-style progress + terminal.

**Deleted** — each for a paper-cited reason:
- Signed-cos `speed_reward` → non-potential shaping → Ng/Harada/Russell
  1999 says it creates new optima; 7M run circled.
- `overspeed_penalty` → redundant with `progress_delta` cap.
- `lateral_penalty` → redundant with off-route terminal.
- `smoothness_penalty` (CAPS) → would bias against BC warmstart if
  used, and was exploitable via small-but-nonzero action jitter.
- `idle_penalty` with 1.5 m/s threshold → hover attractor (Krakovna
  2020 specification gaming).
- `stagnation_counter` termination → twitch-game exploit documented
  in prior-author commit `8791388`.

**Terminal / shaping ratio** — Terminals scaled to ±50, shaping
rescaled to 0.05×. Undiscounted shaping-sum over a 1000-step episode
is ~21 vs terminal ±50; ratio **inverted** from the old 240:1. Per
CaRL §3.2 this ratio is the one the critic can actually learn from.

**Reward clip widened to `[−100, 100]`** so the ±50 terminal is never
self-clipped; VecNormalize's reward clip at ±10 still operates on the
z-score, independent of this raw clip.

**Observation space**: state dim `12 → 10`. Removed `prev_steer` and
`prev_throttle`. Reason: in a planned BC→PPO pipeline `prev_action`
would come from the BC expert during data collection but from a
Gaussian sample during PPO training, creating a hidden covariate shift.
Frame-stacking on the image supplies short-term temporal context
instead. Safer for any downstream BC variant.

**Waypoint features clamped to `[−1, 1]`** (were unbounded, could reach
±5 at sharp turns and push the CNN+MLP fusion outside its trained
distribution).

**Traffic-light state encoding**: `0 = no light`, `0.33 = green`,
`0.67 = yellow`, `1.0 = red OR unknown/off`. Old encoding conflated
"no light" with "unknown state" at 0.0, which could have taught the
agent to proceed through malfunctioning signals.

**2D horizontal goal distance** (was 3D Euclidean including Z).
Bridges and slopes can inflate 3D distance and artificially block
`ROUTE_COMPLETE`.

**Policy / exploration**:
- `log_std_init = −0.5` (std ≈ 0.6). Old was −1.0 (std ≈ 0.37).
  Andrychowicz 2021 identifies 0.5–0.7 std band as viable for
  continuous control from scratch.
- `log_std` clamp is **upper-bound only** (`std ≤ 1.0`). Old clamped
  to `[−2.0, −0.7]` → forced std into the lower floor. New design
  lets `ent_coef` maintain exploration pressure via its gradient.
- `ent_coef = 0.02` annealed linearly to `0.01` (floor, not 0)
  over 10M steps. Per Andrychowicz 2021 + Rajeswaran 2017 — 0 floor
  causes policy decay even during non-finetune training.
- `lr = 3e-4` (Schulman 2017 PPO default); `n_epochs = 3` (compromise
  between 7M run's proven-safe 2 and the PPO-default 10).

**New callbacks** (`agents/train.py`):
- `EntCoefAnnealCallback` — linear anneal from start to floor over
  `anneal_steps` env-steps, also logs `train/ent_coef` to SB3's logger
  so the beacon can read it.
- `RollingBestCallback` — saves `best_by_rc.zip` +
  `best_vecnormalize.pkl` whenever rolling route-completion sets a
  new max over a 50-episode window (warmup 20 episodes). Addresses
  the 7M run's "peak-then-regress" pathology.

**Beacon telemetry additions** (`eval/beacon_callback.py`):
- `termination_reasons` — dict of `{reason: count}` over rolling window
  (keys: `COLLISION`, `ROUTE_COMPLETE`, `REACHED_NO_PARK`, `OFF_ROUTE`,
  `REALLY_STUCK`, `MAX_STEPS`, `NAN_REWARD`, `NAN_GUARD`, `UNKNOWN`).
- `policy_std` — mean `exp(log_std)`, visible floor on exploration.
- `approx_kl`, `clip_fraction`, `entropy_loss`, `explained_variance`,
  `ent_coef` — read from SB3 logger.
- `cumulative_reward_clip_hits` — env-side ±100 clip fires. Non-zero
  is a bug signal.

**Route pipeline unchanged**: `GlobalRoutePlanner`-traced route 200–800m
from random spawn; 3 future waypoints in state vec at route offsets
[0, 2, 5]; reward is projection onto route tangent; `ROUTE_COMPLETE`
requires (rc > 0.85) AND (2D dist < 5m) AND (speed < 3 m/s).

### 3.2 Infrastructure — ALL REVERTED to v1 behavior (commit `0a3f114`)

This is the hard-earned lesson from issues #3/#4/#6. Every infra
change I layered on top of the experiment changes contributed to
silent deadlocks. **For v2 we keep v1 infrastructure identically
and only change the experiment.**

Reverts in commit `0a3f114`:
1. **TrafficManager created in `_spawn_traffic()` per-reset** (v1 pattern),
   not cached in `__init__`. v1's order: `apply_settings(sync=True)` →
   ~100 world ticks in the image-wait loop → TM created in first
   `_spawn_traffic()`. I briefly moved it into `__init__`
   (between sync-mode apply and the first tick) — matches the CARLA
   #9172 anti-pattern. Reverted.
2. **No inline `world.get_spectator().set_transform()`** in `step()`.
   I added this to auto-track the ego in each CARLA window; it added
   two new RPCs per step per worker, and pushed the issue-#6 deadlock
   from iteration 3 to iteration 0. Reverted.
3. **`world.tick()` calls are bare** — no `seconds=10.0` parameter
   at any of 6 call sites. v1 ran 7M steps on bare tick() + client
   timeout 20s. Reverted.

### 3.3 CARLA launch — match the 7M run exactly

- **2 CARLA servers** on Windows, ports 2000 and 3000.
- **Default CARLA flags**: `.\CarlaUE4.exe -carla-rpc-port=<N>` only.
  No `-quality-level=Low`, no `-ResX/-ResY`, no `-windowed`. The 7M
  run used this default set; every "smart" flag I added was extra
  variation that didn't exist in the known-working baseline.
- No external `scripts/spectator.py` running during training
  (secondary client on a sync-mode server risks the CARLA #1996/#2239
  pattern).

---

## 4. What went wrong between v1 infra and v2's first smoke attempt

Chronological timeline for the record, so this never happens again:

| Issue | Tip | Symptom | My (wrong/right) call |
|---|---|---|---|
| #2 | `3e8845c` | `ENT_COEF_START` UnboundLocalError on launch | Real bug, fixed by hoisting definitions. Closed. |
| #3 | `27331b9` | 4-env deadlock at iter 1 | **Wrong diagnosis**: I blamed GPU contention at 4-env scale, fell back to 2 envs + added `world.tick(seconds=10)` timeouts. Did not actually address the cause. Closed falsely as "fixed." |
| #4 | `e69fa2b` | 2-env deadlock at iter 3, timeouts NOT firing | Proof my issue-#3 diagnosis was wrong. I then guessed at CARLA #9172 and added TM cache + inline spectator + n_epochs=3 "fixes" without evidence. |
| #5 | — | 4 CARLA instances at Low+400×300 exhaust 15.4 GB / 16 GB VRAM. Preflight blocks launch. | Real measurement from PC side. 4 envs is out on this hardware. |
| #6 | `98c14d8` | 2-env deadlock at iter 0 — **worse** than #4 | The ddf0a8a "fix" (TM cache + inline spectator) pushed the hang EARLIER. Proved I was layering infra changes that weren't improving things. |

**The lesson** (written so future-me reads it first): *when the v1 infrastructure demonstrably works, do not "improve" it while simultaneously changing the experiment. Change ONE axis at a time. If the experiment changes don't work on v1 infra, debug the experiment. If the experiment works but throughput is bad, then look at infra.*

---

## 5. Current state — tip `0a3f114`, docs `90f14d3`

### 5.1 Code state
- `env/carla_env.py`: v1 infrastructure (bare tick, TM per-reset, no
  inline spectator) + v2 experiment (CaRL-minimal reward, 10-dim state,
  clamps, termination_reason telemetry).
- `agents/train.py`: `log_std_init=-0.5`, `ent_coef=0.02→0.01 floor`,
  `lr=3e-4`, `n_epochs=3`, `EntCoefAnnealCallback`, `RollingBestCallback`.
- `models/clamped_policy.py`: upper-only `log_std` clamp (`std ≤ 1.0`).
- `eval/beacon_callback.py`: new telemetry fields populated.
- `env/safety_wrapper.py`: `NAN_GUARD` stamps `termination_reason`.
- `scripts/`: watchdog randomizes `URBANZERO_SEED` on restart;
  `preflight.py` checks all `URBANZERO_N_ENVS` ports;
  `start_training.sh` skips auto-resume in BC-phase experiments.

### 5.2 Deployment configuration (agreed)
- `URBANZERO_N_ENVS=2`
- `URBANZERO_BASE_PORT=2000`
- Kill every prior `CarlaUE4.exe` via Task Manager before launch.
- Launch CARLA with bare `.\CarlaUE4.exe -carla-rpc-port=<N>`.
- Run smoke-test via `PC_CLAUDE_SMOKE_V2.md`'s pasted-to-user block.
- **PC-side Claude has zero code-edit authority.** All code changes
  come from the remote (me) Claude. PC-side observes, reports numbers,
  captures py-spy stacks on hang. This is load-bearing.

### 5.3 Verified invariants at tip `0a3f114`
Every one of these was regex-checked at commit time:
- No `world.tick(seconds=10.0)` anywhere in code.
- No TM-cache in `__init__`.
- No inline `world.get_spectator().set_transform` in `step()`.
- State dim = 10.
- Progress reward scaled 0.05×, carrot annealed.
- Terminal ±50 present.
- REALLY_STUCK at 1500 steps.
- Waypoint features clamped.
- tl_state encoding fixed.
- 2D goal distance.
- `_at_goal_steps` explicitly initialized in both `__init__` and
  `_reset_once`.
- `termination_reason` emitted in info dict.
- Route pipeline (`GlobalRoutePlanner`, waypoint obs,
  `ROUTE_COMPLETE` terminal) intact.

---

## 6. The agreed training plan

### 6.1 Primary (pure RL, preserves original project thesis)

1. **Smoke test** (10 min) at 2 envs per `PC_CLAUDE_SMOKE_V2.md`.
   PASS gates:
   - ≥40k env-steps in 10 min
   - ≥70 FPS aggregate (7M run measured ~100 at this config)
   - No NaN-GUARD, reward-guard, or cumulative clip hits
   - No silent hang (py-spy dump if it hangs)
   - `policy_std ∈ [0.3, 1.0]`
   - `approx_kl` populated after ≥1 PPO update

2. **Full pure-RL run** (~15M env-steps target, ~41h wall-clock at
   2 envs × ~100 FPS). Run to completion OR until a decision gate fires.

3. **Decision gates during the run** (per Henderson 2018, never judge
   RL before 3M clean steps):
   - At 3M steps: rolling RC expected to climb above single digits;
     if still ~0%, consider falling back to BC warmstart (§6.2).
   - At 8M steps: rolling RC expected somewhere in the 15–40% band
     per CaRL's learning-curve extrapolation to our compute scale.
   - At 15M steps: whatever we have is what we ship. Eval on
     Town01/02/03 from `best_by_rc.zip` (NOT the final checkpoint).

### 6.2 Fallback (BC warmstart, agreed allowed)

If pure RL fails to produce non-zero rolling RC at 3M clean steps,
pivot to Roach-style BC warmstart per `Zhang 2021`:
1. Collect ~150k frames from CARLA's `BehaviorAgent("normal")` on
   Town01 (LBC §3.2 guidance on sample complexity).
2. Inject Gaussian noise σ = 0.1 rad on steering during collection
   (LBC DAgger-style).
3. Train BC with Gaussian NLL loss (not MSE, to handle multimodal
   expert actions per Codevilla 2019 / Florence 2022).
4. Bootstrap value head with MC returns on BC dataset (~100k aligned
   (s, return) samples; larger than the 80k a fresh rollout would yield).
5. PPO finetune with KL-to-BC penalty `β = 0.1` decayed linearly to
   `0.05` (not 0) over 2M env-steps per Rajeswaran 2017.

This pivot plan is pre-agreed with the user. It is NOT to be invoked
speculatively — only after pure RL has demonstrated failure at the
3M-step gate.

### 6.3 Non-negotiable deliverables for Saturday
- A policy checkpoint (`best_by_rc.zip`) that can be evaluated
  deterministically on Town01/02/03.
- A demo video of the agent driving at least one successful route.
- A report section with: (a) learning curve, (b) termination-reason
  tally over training, (c) final eval numbers, (d) honest discussion
  of failure modes if applicable.

---

## 7. Open risks, explicitly named

### 7.1 The smoke test might still hang
If it does, the cause is somewhere in the experiment changes that
code review didn't surface. The py-spy stack dumps are the first
evidence. I have NOT found a code-level reason for this in my audits;
that doesn't mean there isn't one. **Critical: I will not speculate
about causes when the smoke test hangs again. I will wait for py-spy
output and reason from stacks.**

### 7.2 Pure RL may not produce good routes in 15M env-steps
CaRL hit 85% LB at 100M steps × 16 envs. At 15M × 2 envs we're
~1/53rd of that compute budget. There is no published result at
exactly our scale claiming convergence. A 15–40% RC at end of training
would be a real and reportable scientific result. A 0% RC would
trigger the §6.2 BC pivot.

### 7.3 Reward design might still have exploits I missed
Even CaRL-minimal can be circumvented if the route projection behaves
oddly on a pathological route (self-intersecting, parking lots).
Monitored via the beacon's `termination_reasons` dict — if
`REALLY_STUCK` dominates at any point, that's a signal the agent found
a way to be reward-positive without making real progress.

### 7.4 CARLA 0.9.15 TM race (#9172) can still fire on long runs
Even on v1 infrastructure. The 7M run happened to dodge it; any long
training run has a nonzero probability of hitting it. Mitigation:
the watchdog will restart a hung trainer after 3 minutes of beacon
staleness. Expect 0–2 restarts over 40 hours; more than that is a
problem.

### 7.5 Hardware variance
4080 Super + 9800X3D at default settings with 2 CARLA instances
should fit VRAM-wise. Expect ~4–6 GB per CARLA + ~3 GB for PPO +
~1 GB overhead = ~9–15 GB. If VRAM pressure is unexpectedly high
(other processes on the desktop?), preflight catches it and refuses
to launch; good.

---

## 8. Critical do-nots (things that broke it last time)

Written in the imperative so future-me follows them.

1. **Do NOT add infrastructure "fixes" on the same branch as the
   experiment.** Change one axis at a time. v1 infra + v2 experiment
   is the test; mixing in novel infra changes amplifies debugging
   cost exponentially.
2. **Do NOT assume the cause of a hang without py-spy evidence.**
   I did this three times in a row (issues #3, #4, #6) and was wrong
   each time. The stack dumps exist for a reason — use them.
3. **Do NOT add CARLA launch flags that the 7M run didn't use** (no
   `-quality-level=Low`, no `-ResX/-ResY`, no `-windowed`) unless we
   have a measured reason. The 7M run is the known-stable reference
   and deviating from it introduces uncontrolled variables.
4. **Do NOT cache the TrafficManager in `__init__`.** The 7M run
   created TMs per-reset in `_spawn_traffic()`. CARLA #2789/#9172
   are the load-bearing references.
5. **Do NOT run the external `scripts/spectator.py` during training.**
   It's a secondary client on a sync-mode server. CARLA #1996 /
   #2239 applies. The script exists only for offline inspection.
6. **Do NOT anneal `ent_coef` to zero.** Floor at 0.01 per Rajeswaran
   2017.
7. **Do NOT ship the final checkpoint if a better rolling-best
   exists.** `RollingBestCallback` writes `best_by_rc.zip`. Eval
   from there.
8. **Do NOT claim a result before 3M clean env-steps.** Henderson
   2018 is explicit about early-stopping artifacts in RL.
9. **Do NOT edit code from the PC-side Claude session.** All edit
   authority lives with the remote Claude. PC-side observes,
   reports, captures py-spy. This separation is what the user asked
   for after prior sessions kept edit-churning the code.

---

## 9. What I explicitly committed to the user

From the conversation, in order:

- The project's original tabula-rasa thesis (§0) has been relaxed
  at the user's explicit direction. BC warmstart is allowed if pure
  RL fails.
- Every design claim must cite a paper or a measurable observation;
  no more "I think this will work" without backing.
- The experiment axis (reward/obs/policy) and the infra axis are
  separate and will never be changed simultaneously on the same run.
- The RollingBestCallback saves the best checkpoint; final eval is
  from that, not from the last weights.
- The watchdog will restart the trainer on beacon staleness; the user
  shouldn't need to babysit.
- The BC fallback in §6.2 is pre-agreed but only invoked if the 3M-step
  gate fails.

---

## 10. Files in this repo future-me should know about

- `env/carla_env.py` — the env. §3.1 of this file describes v2 changes;
  §3.2 describes what's NOT there (reverted infra).
- `agents/train.py` — PPO setup, callback list, hyperparams.
- `models/clamped_policy.py` — `ClampedStdPolicy`, upper-only log_std clamp.
- `models/cnn_extractor.py` — `DrivingCNN`, with defensive `state.flatten(1)`.
- `env/safety_wrapper.py` — `NaNGuardWrapper`, now stamps `NAN_GUARD`
  termination reason.
- `eval/beacon_callback.py` — `~/urbanzero/beacon.json` writer; new
  telemetry fields in `_write()`.
- `eval/evaluator.py` — `DrivingMetricsCallback`, metrics to TensorBoard.
- `scripts/start_training.sh` — launches tmux training session.
  Honors `URBANZERO_AUTO_RESUME=0` and phase-aware auto-resume skip
  for `*bc*` experiments.
- `scripts/watchdog.sh` — restarts trainer on staleness; randomizes
  `URBANZERO_SEED` on restart.
- `scripts/preflight.py` — checks all `URBANZERO_N_ENVS` ports.
- `scripts/spectator.py` — offline-debug spectator. Do NOT run
  during training. Honors `URBANZERO_SPECTATOR_PORT`.
- `PC_CLAUDE_SMOKE_V2.md` — paste-ready smoke-test prompt for
  PC-side Claude.
- `AGENT_RUNBOOK.md` — prior-author's runbook (pre-v2). Historical
  reference only.
- `PC_CLAUDE_PROMPT.md` — prior-author's runtime prompt (pre-v2).
  Historical reference only.
- `reports/training_run_20260421_1833.md` — prior-author's 7M-run
  failure report. Historical reference.

---

## 11. Update log

Append here when something changes materially.

- **2026-04-23**: file created at tip `0a3f114` / docs `90f14d3` after
  the infra revert addressing issues #3/#4/#6. Awaiting PC-side smoke
  test before proceeding to the 15M-step run or the BC pivot.
