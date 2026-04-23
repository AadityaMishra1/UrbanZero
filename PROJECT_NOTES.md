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

- **2026-04-23 early AM**: file created at tip `0a3f114` / docs `90f14d3` after
  the infra revert addressing issues #3/#4/#6. Awaiting PC-side smoke
  test before proceeding to the 15M-step run or the BC pivot.
- **2026-04-23 after issue-#6 PC-side report**: root cause finally
  identified from the PC-side operator's traceback, not from any of my
  prior theorizing. The crash is `BrokenPipeError` in `ForkServerProcess-2`
  inside `stable_baselines3/common/vec_env/subproc_vec_env.py:43`
  `_worker()` calling `remote.send(...)`. Main process then blocks forever
  on `remote.recv()` via `unix_stream_read_generic`. SB3's `SubprocVecEnv`
  has no worker-death recovery — this is a documented limitation, NOT
  a CARLA or reward bug. **Proof the experiment code is fine**: the
  20:03 run that survived to iteration 3 hit 147 FPS with healthy PPO
  metrics (`approx_kl=0.0055`, `entropy_loss=-1.84`, `std=0.607`).
- **Fix pushed**: `agents/train.py` now uses `DummyVecEnv` unconditionally,
  not `SubprocVecEnv`. All envs run serially in the main process: no
  pipes, no IPC, no BrokenPipeError possible. Expected throughput
  ~90-110 FPS at 2 envs (vs 147 FPS with SubprocVecEnv) — well above
  the 70 FPS PASS gate. Reliability chosen over throughput.
- **Theories that turned out wrong, recorded so I don't retread them**:
  (a) GPU contention at 4-env scale, (b) CARLA #9172 TM race, (c) my
  own TM caching / inline spectator / `world.tick(seconds=10.0)`
  additions. None caused the observed hangs. What caused them was an
  upstream SB3 IPC fragility that triggers more under some scheduling
  patterns than others. The prior 7M run happened to dodge it
  statistically, the v2 runs happened to trigger it. v1 infra + v2
  experiment + DummyVecEnv is the correct combination.

- **2026-04-22 22:16**: long run (tip `5814854`, fresh weights from
  `cfafe73`-minus-this-commit) at 233k steps / 31 min wall-clock
  diverged: `avg_speed = 0.224 m/s`, `REALLY_STUCK ≥ 70%` in rolling
  window, last 18 consecutive episodes all `REALLY_STUCK`, rolling RC
  peaked 5.19% at step 142k and regressed to 1.78% by step 233k.
  Agent learned the sit-still local optimum — full failure report
  in `EARLY_WARNING_REALLY_STUCK.md`. PPO stats were healthy
  (`approx_kl=0.014`, `entropy_loss=-2.36`, `policy_std=0.787`),
  ruling out optimizer pathology. The explanation is a **reward
  structure exploit**, not a bug.

- **Per-step cost analysis (why sit-still dominates)**:
  REALLY_STUCK fires at 1500 steps of zero net progress. With the
  previous reward, a stuck trajectory accumulated ≈0 per-step shaping
  plus `-50` terminal, i.e. `-0.033/step`. A collision trajectory
  averaged `~300` steps before crashing, so `-50/300 = -0.167/step`.
  **Sitting still was 5x cheaper per step than moving and crashing.**
  The agent's policy gradient correctly converged to that local
  optimum. Agent 2's red-team finding A8 (pre-run review) flagged this
  exact risk; I acknowledged it but did not weight it strongly enough
  in the launch reward. That was my error.

- **Reward fix pushed** (this commit):
  1. Added `idle_cost = -0.15 * max(0.0, 1.0 - speed / 1.0)` in
     `_compute_reward`. Continuous ramp: `-0.15/step` at zero speed,
     `0` at `speed ≥ 1 m/s`. New per-step cost of sit-still becomes
     `-225` shaping + `-50` terminal over 1500 steps = `-0.183/step`,
     now **more** expensive than a `-0.167/step` crash.
  2. Un-annealed the velocity carrot (removed `anneal_coef`). Carrot
     is now `0.005 * min(speed, TARGET)/TARGET` for the full run.
     Rationale: Rajeswaran et al. 2017 §3 — annealing shaping bonuses
     to zero lets the policy regress. The 233k-step run collapsed
     while the carrot was still ~53% live, so even the partial anneal
     was insufficient. Keeping it on permanently provides a constant
     gradient rewarding forward motion, which is needed to overcome
     the kink at `speed=1.0` where `idle_cost` bottoms out.

- **Continuous-ramp idle vs v1 threshold idle**: the deleted v1
  `idle_penalty` used a hard threshold at 1.5 m/s that created a
  1.01-m/s hover attractor (once above 1, no incentive to go faster;
  once below, a cliff-style penalty). The new `idle_cost` is smooth
  at 1.0 m/s (value is 0 at 1 m/s, derivative is 0 above, -0.15 below)
  and the persistent carrot supplies the continued gradient toward
  TARGET_SPEED for `speed > 1`. So the hover-at-1.0 failure mode
  is not reintroduced.

- **Predicted post-fix per-step economics**:
  | scenario | 1500-step shaping | terminal | per-step |
  |---|---|---|---|
  | sit still at 0 m/s for 1500 steps | −225.0 | −50 (REALLY_STUCK) | −0.183 |
  | creep at 0.5 m/s pointless | −112.5 | −50 (REALLY_STUCK) | −0.108 |
  | drive at TARGET, crash at 300 steps | +7.8 progress + +1.5 carrot | −50 (COLLISION) | −0.137 |
  | drive at TARGET, full success | +31.3 progress + +7.5 carrot | +50 (ROUTE_COMPLETE) | +0.059 |

  Ranking: success (+0.059) > crash (−0.137) > slow creep (−0.108) >
  sit still (−0.183). Sit-still is now strictly dominated. Crash is
  cheaper than creep, which is the desired ordering — it pushes the
  agent to explore real driving rather than safe pointless motion.

- **Action item on instructions**: the PC-side Claude is to kill the
  current run and restart from fresh weights (not resume). Continuing
  from the 233k-step checkpoint would leak the "sit still" prior into
  the fresh reward landscape. See `PC_CLAUDE_REWARD_FIX.md` for the
  exact paste block handed to PC-side.

- **2026-04-23 ~00:45**: the idle_cost fix (`d307a66`) broke the
  sit-still attractor but exposed the next failure mode. At 900k steps
  / seed 137 / ~2h wall-clock the run showed rolling RC flat at 5-6%
  across 1500 episodes with zero trend, mean_speed oscillating 2-6 m/s,
  `policy_std` pinned at the 1.0 upper clamp for 1500+ episodes, and
  termination distribution 48% OFF_ROUTE / 38% COLLISION / 14%
  REALLY_STUCK. Best single episode RC 42.6% was a statistical fluke;
  best rolling RC (8.24%) peaked at step 64k and hadn't improved in
  830k steps. PPO stats otherwise healthy (`approx_kl=0.006`,
  `entropy_loss=-2.36`). Agent learned to throttle but not to steer.

- **Two independent red-team subagents** evaluated the options:
  (a) full §6.2 BC pivot and (b) adding a shaping term. Both
  converged on two diagnoses:
  1. `policy_std=0.999` pinned at clamp is an **exploration**
     pathology, not a gradient pathology. At std=1.0 the policy is
     emitting essentially pure noise; every episode is a different
     random walk and the critic cannot credit-assign.
     Andrychowicz 2021 §4.5 viable band for continuous control from
     scratch is [0.3, 0.7]; clamp at 1.0 was too permissive.
  2. Even with a healthy exploration band, the progress reward is
     **too sparse for a steering gradient**. It fires only when ego
     projects forward on the route tangent; lateral drift gives ~0
     signal until OFF_ROUTE at 30m. With random steering the agent
     rarely observes the "aligned and advancing" state that carries
     positive reward.

- **Both agents rejected the full §6.2 BC pivot as specified** —
  implementation burden is 28-45h (no BehaviorAgent rollout script,
  no BC trainer, no KL-to-BC PPO path in `agents/train.py`), combined
  probability of shipping 0.30-0.40, zero buffer for debugging. Early
  pivot at 900k steps also violates §6.1 (don't judge before 3M per
  Henderson 2018). The reward-shaping failure at 900k is not evidence
  of the underlying RL-vs-BC question; it's evidence of *this*
  reward's sparsity.

- **PC-side Claude's proposed fix** was `r_heading = 0.1 * cos(angle)`
  between ego heading and direction to next waypoint. Explicitly
  **rejected**: that is the signed-cos shaping deleted in v2 per
  §1.1/§2.1/§3.1. Ng/Harada/Russell 1999 says it's non-potential and
  creates new optima; the 7M-run perpendicular-circling attractor is
  the documented failure mode. Can't accept PC-side's suggestion
  because PC-side doesn't have project-notes context.

- **Fix pushed (this commit) — composed, experiment-axis only**:
  1. `models/clamped_policy.py`: lowered `LOG_STD_MAX` from `0.0`
     (std ≤ 1.0) to `log(0.7) ≈ -0.357` (std ≤ 0.7). Andrychowicz
     2021 viable band. Targets the diagnosed policy_std-at-clamp
     symptom directly.
  2. `env/carla_env.py::_compute_reward`: added potential-based
     shaping (Ng/Harada/Russell 1999). `Φ(s) = -0.03 · min(dist2D
     (ego, lookahead), 30m)` where lookahead is the point 10m ahead
     of current projection along the planned route (continuous arc-
     length via `_lookahead_point`, no waypoint-transition
     discontinuity). `F(s, s') = 0.99·Φ(s') - Φ(s)` on non-terminal
     steps; `F(s, s_terminal) = -Φ(s)` on terminals (episodic
     convention Φ(terminal):=0, Grzes 2017). Max |F| ≈ 0.021/step,
     same scale as progress_reward; preserves progress as the
     dominant signal. By construction per Ng 1999 this doesn't
     create new optima — only a denser gradient.

- **Why both changes in one run (against §8.1 "one axis at a time")**:
  deadline math: BC pivot is 28-45h of dev + run, single-axis-
  serialized experiments are ~46h (two fresh 23h runs back-to-back),
  combined this run is ~24h with a T+15min sanity gate that aborts
  early if the composed fix fails. Both agents independently flagged
  both problems. §8.1 was written to separate EXPERIMENT from INFRA;
  both of these changes are experiment-axis and target different
  diagnosed failures. Beacon telemetry (`policy_std`, `rolling_RC`,
  `termination_reasons`) lets us attribute effects per-axis.

- **Alternative option considered and rejected**: increase
  `progress_reward` scale from 0.05 to 0.15. Simpler change, but
  scales the existing sparse signal rather than densifying it. Would
  make the gradient stronger WHERE it fires, not fire it more often.
  Doesn't address the "random steering can't find the signal" root
  cause.

- **Idle_cost status**: kept. `-0.15 · max(0, 1 - speed/1.0)` still
  prevents the sit-still attractor. The 900k-step run showed
  REALLY_STUCK only 14% (vs 70% pre-idle_cost fix) — idle_cost is
  working as designed; the residual REALLY_STUCK episodes happen when
  the agent drives off the road, gets stuck on terrain, and can't
  recover. That's a steering problem, not a motivation problem.

- **Infra fix bundled (TrafficManager hybrid physics)**: user + PC
  Claude report NPCs not moving. The code sets `tm.set_synchronous_
  mode(True)` and `world.apply_settings(synchronous_mode=True)`
  correctly. Known CARLA 0.9.x quirk (carla-simulator#3860): NPCs
  outside ego's physics radius fall into a dormant state in sync mode
  unless `tm.set_hybrid_physics_mode(True)` + `tm.set_hybrid_physics_
  radius(70.0)` are set. Added both in `_spawn_traffic`, plus a
  diagnostic `[TM]` print on each spawn showing requested sync mode
  and world sync read-back. Note: this is a defensive fix inside the
  v2 infra (sync mode + v1-style per-reset TM creation). Does NOT
  reintroduce any of the reverted infra changes from §3.2.

- **Predicted post-fix outcomes (T+15min, T+1h, T+3h)**:
  | time | expected signal |
  |---|---|
  | T+15m | policy_std drops below 0.7 (enforced by clamp); avg_speed > 2 m/s (idle_cost); dense shaping begins moving the mean network |
  | T+1h  | rolling RC crosses 10% if shaping is doing its job |
  | T+3h  | rolling RC crosses 15-20%; REALLY_STUCK < 10%; approx_kl stable at 0.01-0.03; std settling in [0.4, 0.6] |
  If rolling RC is still <10% at T+1h, the shaping isn't the right
  lever and we have 23h left before deadline for the minimal-BC
  fallback (the skip-value-bootstrap, skip-KL-to-BC version — per
  Agent-1 red-team this is ~12-16h of dev work, still fits).

- **PC-side instructions**: `PC_CLAUDE_REWARD_FIX_2.md`. Kill current
  run, pull this tip, archive collapsed artifacts
  (`v2_rl.flatrc-<ts>`), fresh weights, seed 211 (different from 42
  and 137), T+15min sanity check with std threshold.

- **2026-04-23 ~01:30 BC pipeline landed**. Files:
  `scripts/collect_bc_data.py` (BehaviorAgent rollout with σ=0.1 rad
  steering noise per Ross/Bagnell 2010), `agents/train_bc.py`
  (Gaussian NLL trainer matching `ClampedStdPolicy` + `DrivingCNN`
  architecture exactly, saves SB3-compatible .zip + sibling
  `_vecnormalize.pkl`), `scripts/run_bc_pipeline.sh` (sequential
  runner), `agents/train.py` patch adding `URBANZERO_BC_WEIGHTS` env
  var path via `PPO.load()` with sibling vecnorm stitch. This is
  **Agent-1 red-team's minimal-BC fallback**: skips KL-to-BC
  penalty and skips MC-return value-head bootstrap — both deferred.
  Minimal-BC is 12-16h dev (now done) + ~4-8h data collection + BC
  train + PPO finetune. Fits deadline as a fallback path triggered
  at T+1h if fix-2 rolling RC < 8%.

- **2026-04-23 ~01:30 failure-mode stress test**. Spawned a third
  red-team subagent to specifically answer "will observed failures
  recur, and can the agent learn to drive." Findings:
  - **DEAD (mechanistic, cannot recur)**: F1 perpendicular-circling
    (no signed-cos in code), F3 stagnation-counter twitch-game
    (replaced with cumulative-progress anchor), F5 overspeed hack
    (progress cap at TARGET not 1.5m), F6 peak-then-regress
    (RollingBestCallback active), F8 terminal invisible (ratio
    inverted), F11 cos-similarity heading (rejected, not in code),
    F13 SubprocVecEnv BrokenPipe (DummyVecEnv eliminates IPC).
  - **LOW risk (probabilistic, strong mitigation)**: F2 sit-still
    global optimum (idle_cost inverts per-step cost), F4 1.01 m/s
    hover attractor (continuous ramp + persistent carrot), F7
    entropy collapse (ent_coef 0.01 floor), F9 sit-still local
    optimum at 233k (idle_cost -0.15 makes stall -0.183/step),
    F12 NPCs frozen (hybrid_physics + diagnostic).
  - **MEDIUM risk (remaining concern)**: F10 std-pinned / sparse
    steering gradient. The `log_std ≤ log(0.7)` clamp guard is
    mechanistic. The steering-gradient guard is probabilistic:
    potential shaping gives ~0.005/step per m of lateral motion,
    which integrates to useful signal over many episodes but is
    marginal per-step vs std=0.6 steer noise.
  - **Mechanistic driving argument**: (L1) forward-motion gradient
    is real — `throttle_brake + 0.3` idle-creep (bias toward
    throttle>0) + idle_cost negative gradient at speed=0 +
    progress_reward positive gradient at speed>0. (L2) lateral-
    motion gradient from potential shaping is weak-but-present
    (+0.0046/step per 1m lateral). (L3) γ=0.99 horizon is 100 steps;
    dense shaping accumulates to ~4.7 reward per horizon, large
    enough for value-function learning. (L5) no new attractors from
    shaping: constant-radius orbits get 0 from Φ but hit REALLY_STUCK
    terminal; backward motion gives 0 progress; terminal-Φ exploit
    bounded at |Φ_max|=0.9 which never overcomes ±50 terminal.

- **Verdict from the stress test**: p(rolling RC > 15% by 5M steps
  under fix-2 pure RL) ≈ **0.35**. With BC fallback bringing another
  ~0.35, combined p(ship something demoable) ≈ 0.60-0.65. The
  single make-or-break reward signal is the `throttle_brake + 0.3`
  action-shift (creates motion in the first place); the highest-
  leverage reward term is idle_cost (inverts sit-still ordering).

- **Mechanical gap named by stress test (not acted on pre-launch)**:
  the reviewer suggested adding `+α · route_progress` to Φ to
  strengthen forward-along-route gradient. Rejected because that
  term telescopes to approximately `α · γ · Δroute_progress` per
  step — the same shape as progress_reward. Mathematically
  equivalent to increasing progress_reward coefficient from 0.05.
  If fix-2 T+1h RC < 8% we'll bump `progress_reward` 0.05→0.10 AND
  trigger BC fallback in parallel rather than adding redundant
  shaping terms.

- **BC collector bugfix**: reviewer caught missing episode-boundary
  markers in the collected .npz, which would have contaminated the
  first 3 frames of each episode (~3% of dataset at 100-step eps)
  via cross-episode frame-stacking. Added `episode_starts` bool
  array to the .npz and taught `train_bc.py::_stack_frames` to
  refuse crossing episode boundaries. BC trainer also falls back
  gracefully to legacy behavior if loading an older .npz without
  the marker.

- **PC-side launch remains fix-2 as pushed (`ff7a1e1`)**. BC pipeline
  is prepared but not executed. Triggering criteria: rolling RC < 8%
  at fix-2 timestep ~700k (T+1h 40min). If triggered, remote Claude
  writes the BC paste-block; PC-side runs
  `scripts/run_bc_pipeline.sh --n_frames 150000 --epochs 30`.

- **2026-04-23 ~01:55 Run-3 failed at 104k / 15min wall**. Same 5% RC
  plateau. Beacon at 104k: rolling RC 4.03%, avg_speed 3.86 m/s,
  policy_std 0.67 (clamp held — fix worked on std), COLLISION 52%,
  OFF_ROUTE 41%, REALLY_STUCK 9%. Three failure modes persist across
  all runs: "floor it off road," "creep and crash," "sit still."
  User's key observation: **NPCs are visibly frozen in both CARLA
  viewports despite `tm.set_hybrid_physics_mode(True)`.** Hybrid
  physics alone did not fix NPC motion.

- **Decision delegated to a no-prior-context subagent per user
  instruction**. The subagent's verdict:
  - DIAGNOSIS: Three issues compound. (i) Frozen NPCs contaminate the
    reward signal — ~50% of COLLISION terminations are against
    stationary NPCs/static geometry, not learnable collision
    avoidance. (ii) At POTENTIAL_K=0.03, max |F|/step ≈ 0.021 =
    max progress_reward/step → shaping can subsidize perpendicular
    approach to the lookahead, a subtle Ng-compliant echo of the
    7M-run circling attractor. (iii) Run-3 was judged at 104k steps
    (Henderson 2018: unreliable pre-3M), but deadline math makes
    that moot.
  - DECISION: kill Run-3; launch Run-4 with POTENTIAL_K halved to
    0.015 + NPC motion fix; in PARALLEL, run BC data collection on
    the second CARLA server. Stop serializing fallbacks — hedge the
    deadline by running both paths simultaneously.
  - RULED OUT: adding a cos-heading shaping term (the 7M-run trap
    holds; all documented failure modes started with "we added one
    more shaping term"). If Run-4 fails with healthy std, bump
    progress_reward 0.05→0.10 before touching shaping.

- **NPC motion fix (issue #9)** — root cause per subagent analysis +
  CARLA issues #3860/#4030/#6349:
  1. `tm.global_percentage_speed_difference(-30.0)` — default is +30
     (NPCs drive 30% BELOW limit). In zero-speed-limit zones (off-
     road spawns, hybrid-dormant radius), 30% below 0 is 0. Negative
     value forces NPCs to drive 30% ABOVE limit.
  2. Per-vehicle `tm.vehicle_percentage_speed_difference(npc, -20.0)`,
     `tm.auto_lane_change(npc, True)`, `tm.ignore_lights_percentage
     (npc, 0.0)` — ensures each NPC has non-zero desired speed and
     correct policy settings.
  3. **`self.world.tick()` commit tick** after the spawn loop — in
     sync mode, `set_autopilot()` enqueues registration asynchronously
     in the TM; without a commit tick, the first env.step() can race
     the TM's internal vehicle-registration table and leave NPCs
     unregistered for one or more ticks. For long episodes this
     manifests as "NPCs frozen the whole episode" because the race
     loses deterministically under heavy PPO tick cadence.

- **POTENTIAL_K 0.03 → 0.015**. Max |F|/step drops to ~0.0105,
  progress_reward dominates shaping 2:1. Ng-compliance preserved
  (F = γΦ'−Φ structure unchanged). Rationale: the perpendicular-
  subsidy risk identified by the subagent — Φ reduces distance to
  lookahead as ego approaches from ANY direction, so at peak
  magnitude the shaping rewarded lateral approach just as well as
  along-route motion. Halving K + keeping progress_reward at 0.05
  restores the "drive forward on route" signal as strictly
  dominant.

- **Run-4 launch plan** (PC_CLAUDE_RUN4.md):
  - Pane A: pure-RL on port 2000 ONLY, `URBANZERO_N_ENVS=1`, seed
    311. Frees port 3000 for BC collection. Single env throughput
    ~60 FPS but this is the decision-gate run, not the main
    training run — if it passes gates, we scale back up.
  - Pane B: `scripts/run_bc_pipeline.sh --port 3000 --n_frames
    100000 --epochs 20` in parallel. Reduced from 150k/30 to fit
    ~6h wall (collect 1.5h + BC train 2h + PPO-finetune option 2h).
  - Gates: T+10min (`std ∈ [0.45, 0.70]`, avg_speed > 1.5 m/s,
    user confirms NPCs moving in viewport), T+45min (RC ≥ 6%,
    COLL% < 50, OFF_ROUTE% < 35), T+90min HARD DECISION
    (RC < 8% → promote BC to primary; RC ≥ 8% → Run-4 continues).
  - Seed 311 (runs used 42, 137, 211 previously).

- **p(ship by deadline) = 0.62** per subagent. Decomposition:
  p(Run-4 pure-RL succeeds) ≈ 0.30; p(BC pipeline produces demoable
  checkpoint) ≈ 0.55; union with mild correlation ≈ 0.62. Critical
  assumption: the NPC fix ACTUALLY fixes frozen NPCs. If Run-4's
  T+10min check still shows NPCs frozen, immediately pivot to BC
  (BehaviorAgent is rules-based and doesn't need moving NPCs).

- **2026-04-23 ~02:30 — user elected to skip Run-4 and go BC-only**.
  Tabula-rasa constraint was explicitly relaxed per §0; user's
  direction: "lets just start with the BC, lets just do that, i
  want to get this done with, nothing wrong with a lil pre training
  right?" Single-path BC plan eliminates the 90-minute "is Run-4
  working" decision gate and spends the entire compute budget on
  the expert-initialized path.

- **BC-only three-phase plan** (`PC_CLAUDE_BC_ONLY.md`):
  - Phase 1: parallel BC data collection on BOTH ports
    simultaneously. 50k frames on port 2000 (seed 77) + 50k on port
    3000 (seed 78) = 100k total. Halves wall-clock vs serial.
    Expected ~45-90min wall.
  - Phase 2: `agents/train_bc.py` concatenates both .npz files
    with episode-boundary preservation (added `nargs="+"` to the
    `--data` arg). 20-epoch Gaussian NLL training on RTX 4080S
    (~30min-2h).
  - Phase 3: PPO finetune on 2 envs with `URBANZERO_BC_WEIGHTS`
    pointing to bc_pretrain.zip. Seed 911. 5M steps (~11.5h at
    122 FPS).
  - Total wall: 12-16h. Deadline margin: 48h for eval + demo
    + report.

- **Why BC-only now vs parallel Run-4+BC**: user was tired of the
  "reward-shaping lottery" with three consecutive collapses. BC
  gives a policy that already drives (BehaviorAgent follows lanes
  and routes by construction). PPO finetunes it under the same
  reward that failed pure-RL — but starting from a good prior, the
  reward's weakness at producing initial driving behavior doesn't
  matter; it only needs to refine existing behavior. Roach 2021
  reports BC+PPO reaches 80% LB with just 10M finetune steps from
  a BehaviorAgent prior.

- **train_bc.py multi-file support** (this commit): `--data` arg
  now takes multiple .npz paths via `nargs="+"`. Concatenates them
  in order and preserves episode boundaries at join points (forces
  episode_starts[0] = True on each file so the stacker refuses to
  walk back across file boundaries). Required for the parallel-
  collection plan.

- **Expected behavior at PPO finetune T+5min** (different from any
  pure-RL run): avg_speed > 3 m/s from step 0, policy_std starts
  small (~0.3), COLLISION% < 40, OFF_ROUTE% < 25, rolling_RC > 8%.
  If T+5min looks like a pure-RL run (avg_speed ~0, RC 0%), the BC
  prior did not transfer correctly — likely a VecNormalize stats
  mismatch or PPO.load path issue. That's the one untested-E2E
  risk flagged in §11 and in PC_CLAUDE_BC_ONLY.md RED-4.

- **p(ship) under BC-only plan ≈ 0.75**. Up from 0.62 because:
  (i) no time lost on Run-4 uncertainty, (ii) BC gives a stronger
  behavioral prior than any reward-shaping we've tried, (iii) the
  remaining 25% risk is concentrated in "BC pipeline first-run
  bugs" — a localized risk with ~1-3h debug budget allocated.
