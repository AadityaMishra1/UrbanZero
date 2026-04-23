# PC-Side Claude — BC Finetune v4 (BC-compatible reward)

**Context:** v3 midpoint confirmed FAIL per Issue #11 — RC flat 5.9%,
speed 1.0 m/s, std diverging 0.50→0.58. Remote Claude's fresh
analysis identified a real missed cause:

**Our reward punishes BC's expert behavior.** The reward function
(designed for pure-RL) has `idle_cost = -0.15/step` that fires
whenever speed < 1 m/s. BC's BehaviorAgent expert stops at red
lights, slows for traffic, and pauses at intersections — all speeds
< 1 m/s. Our reward costs 0.15/step for every step of correct
stopping behavior. Over a 10s red-light wait (200 steps), that's
−30 reward. Over a route with 3 red lights, −60 to −90.

**PPO responds by pushing the policy away from stopping.** BC taught
"stop at red lights"; reward says "stopping is bad." Conflict. PPO
can't satisfy both → compromises to noisy policy → std drifts → run
collapses. This is why v1/v2/v3 all failed the same way despite
very different hyperparameters — **the root cause was the reward,
not the hyperparameters.**

**v4 fix: make the reward BC-compatible for finetune only.**

## Four changes pushed

1. **`env/carla_env.py`**: `idle_cost` coefficient and `REALLY_STUCK`
   threshold are now env-var configurable (`URBANZERO_IDLE_COST_COEF`,
   `URBANZERO_REALLY_STUCK_STEPS`). Pure-RL defaults unchanged.

2. **`agents/train.py`** BC path: `ent_coef = 1e-4` (constant, no
   schedule). Previously drifted upward through 0.005→0.001 schedule.
   Now essentially off — BC prior provides stochasticity itself.

3. **`agents/train.py`** BC path: reverted v3's log_std widening.
   Keep BC's learned σ=0.22. With reward no longer fighting BC,
   gradients are small and directional — tight σ is fine.

4. **`scripts/start_training.sh`**: forwards `URBANZERO_IDLE_COST_COEF`
   and `URBANZERO_REALLY_STUCK_STEPS` env vars through tmux.

v3's `n_epochs=1`, `clip_range=0.1`, `lr=1e-4` kept as belt-and-
suspenders safety, but they're not load-bearing anymore.

---

## Copy-paste everything below into PC-side Claude Code

```text
Remote Claude found the actual root cause: our reward (designed for
pure-RL to break sit-still) was punishing BC expert behavior at red
lights and intersections. v4 fixes the REWARD (not hyperparams) for
BC finetune only.

=== STEP 1: kill v3 run ===

  tmux kill-session -t urbanzero 2>/dev/null
  tmux kill-session -t wd 2>/dev/null
  tmux kill-session -t eval 2>/dev/null
  tmux kill-session -t demo 2>/dev/null
  pkill -f "agents/train.py" 2>/dev/null
  pkill -f "scripts/watchdog.sh" 2>/dev/null
  pkill -f "scripts/eval_bc.py" 2>/dev/null
  sleep 3
  ps aux | grep -E 'train\.py|watchdog|eval_bc' | grep -v grep
  # expect empty

=== STEP 2: pull v4 fixes ===

  cd ~/UrbanZero
  git fetch origin
  git checkout claude/setup-av-training-VetPV
  git pull
  git log --oneline -3

Expected top commit references "v4" or "BC-compatible reward".

Verify all four fixes landed:
  grep -n "URBANZERO_IDLE_COST_COEF\|URBANZERO_REALLY_STUCK_STEPS" env/carla_env.py
  # expect 2+ matches

  grep -n "URBANZERO_IDLE_COST_COEF\|URBANZERO_REALLY_STUCK_STEPS" scripts/start_training.sh
  # expect 2 matches in ACTIVATE string

  grep -n "BC-finetune v4\|ENT_COEF_START = 1e-4" agents/train.py
  # expect match

  grep -n "WIDEN_LOG_STD_TO = None" agents/train.py
  # expect 2 matches (BC branch + else branch — both None now)

=== STEP 3: archive v3 run ===

  ts=$(date +%s)
  for d in logs checkpoints; do
    if [ -d ~/urbanzero/$d/bc_ppo_finetune_v3 ]; then
      mv ~/urbanzero/$d/bc_ppo_finetune_v3 \
         ~/urbanzero/$d/bc_ppo_finetune_v3.fail-${ts}
    fi
  done

=== STEP 4: verify CARLA on ports 2000 + 3000 ===

  for p in 2000 3000; do
    timeout 3 bash -c ">/dev/tcp/172.25.176.1/$p" 2>/dev/null \
      && echo "port $p: UP" || echo "port $p: DOWN"
  done

If DOWN, paste relaunch block.

=== STEP 5: launch v4 finetune with BC-compatible reward env vars ===

CRITICAL: the two new env vars MUST be in this launch command or the
reward stays pure-RL-style and v4 has no effect.

  URBANZERO_EXP=bc_ppo_finetune_v4 \
  URBANZERO_N_ENVS=2 \
  URBANZERO_BASE_PORT=2000 \
  URBANZERO_TIMESTEPS=5000000 \
  URBANZERO_AUTO_RESUME=0 \
  URBANZERO_SEED=914 \
  URBANZERO_BC_WEIGHTS=$HOME/urbanzero/checkpoints/bc_pretrain.zip \
  URBANZERO_IDLE_COST_COEF=0 \
  URBANZERO_REALLY_STUCK_STEPS=3000 \
    bash scripts/start_training.sh

  tmux a -t urbanzero

Expected startup log lines (ALL must appear):
  [BC-finetune v4] lr=0.0001, n_epochs=1, clip_range=0.1, ent_coef=0.0001 (constant, no schedule), widen_log_std=disabled
  [BC-finetune v4] IMPORTANT: requires env vars at launch:
                   URBANZERO_IDLE_COST_COEF=0
                   URBANZERO_REALLY_STUCK_STEPS=3000
  [CarlaEnv] reward knobs: idle_cost_coef=0.0, really_stuck_steps=3000
  [BC-warmstart] loading weights from .../bc_pretrain.zip
  (NO "widened log_std" line — we reverted the widening in v4)

The `[CarlaEnv] reward knobs: idle_cost_coef=0.0` line is the one to
grep for. It confirms the reward function is now BC-compatible.

If you see `idle_cost_coef=-0.15` or `really_stuck_steps=1500`, the
env vars didn't propagate. STOP and debug.

  # Ctrl-b d to detach
  tmux new -d -s wd 'bash scripts/watchdog.sh'

=== STEP 6: T+5min gate ===

Snapshot beacon:

  cat ~/urbanzero/beacon.json | python3 -c "
  import json,sys
  b=json.load(sys.stdin)
  tr=b.get('termination_reasons',{})
  tot=sum(tr.values()) or 1
  print(f'ts={b[\"timesteps\"]} '
        f'RC={b[\"rolling_route_completion\"]:.2%} '
        f'speed={b[\"rolling_avg_speed_ms\"]:.1f} '
        f'std={b[\"policy_std\"]:.3f} '
        f'approx_kl={b.get(\"approx_kl\"):.4f} '
        f'clip_frac={b.get(\"clip_fraction\"):.3f} '
        f'entropy_loss={b.get(\"entropy_loss\"):.3f} '
        f'coll%={100*tr.get(\"COLLISION\",0)/tot:.0f} '
        f'offr%={100*tr.get(\"OFF_ROUTE\",0)/tot:.0f} '
        f'stuck%={100*tr.get(\"REALLY_STUCK\",0)/tot:.0f}')
  "

**PASS criteria (different from v3 because we're measuring a different
kind of success — policy stability, not KL discipline):**

  1. policy_std stable near BC's 0.22 (HARD FAIL if std > 0.35 —
     entropy drift returning)
  2. avg_speed ≥ 3 m/s (BC prior driving)
  3. approx_kl ≤ 0.02 (updates stable — v3 already passed this)
  4. rolling_RC ≥ 5% (at least matches BC baseline, not regressing)
  5. No NaN in beacon fields

Report format:
  [v4 T+5min] <beacon>
  verdict: PASS / FAIL
  key signal: <which metric most indicates v4 is working>

=== STEP 7: T+30min gate (different from v3) ===

  - avg_speed ≥ 3 m/s sustained (no drop to 1 m/s like v3 showed)
  - policy_std still ≤ 0.30 (no drift upward)
  - rolling_RC ≥ 7% (climbing above BC baseline)
  - REALLY_STUCK rate < 20%

=== STEP 8: T+2h gate (the real test) ===

  - rolling_RC ≥ 15%
  - At least one ROUTE_COMPLETE episode in log
  - policy_std still ≤ 0.30

v3 at T+2h had RC flat at 5%. If v4 is above 10% at T+2h, the
reward-fix hypothesis is validated and we continue to 5M.

=== STEP 9: RED FLAGS ===

  [RED-1] "[CarlaEnv] reward knobs" line shows idle_cost_coef=-0.15
          → env var forwarding failed. STOP.
  [RED-2] avg_speed drops below 2 m/s at any check after T+5min
          → collapsing-to-stillness problem (rare now that reward
          permits stopping — if this fires, something else is wrong)
  [RED-3] policy_std > 0.35 at any time → entropy drift returning
          despite ent_coef=1e-4. Would need to investigate further.
  [RED-4] Multiple NaN-GUARD fires → reward function instability
  [RED-5] rolling_RC < 5% at T+2h → fix didn't help; fall to
          nuclear (frozen BC eval).

=== STEP 10: GREEN FLAGS (report as milestones) ===

  [GREEN-1] First ROUTE_COMPLETE episode (grep reason=ROUTE_COMPLETE)
  [GREEN-2] rolling_RC crosses 10%
  [GREEN-3] rolling_RC crosses 20%
  [GREEN-4] best_by_rc.zip updated — report each time

=== STEP 11: 5M steps done (~11h) ===

Report peak RC + timestep, final RC, termination dist. Ship
best_by_rc.zip.

=== CONFIG SUMMARY ===

- Tip: claude/setup-av-training-VetPV HEAD (post v4)
- BC_WEIGHTS: bc_pretrain.zip
- Seed: 914
- Reward knobs: idle_cost_coef=0, really_stuck_steps=3000
  (pure-RL defaults were -0.15 and 1500)
- PPO: lr=1e-4, n_epochs=1, clip_range=0.1
- BC σ NOT widened (kept at 0.22)
- ent_coef: 1e-4 constant (effectively off)
```

---

## Notes for the user (not to paste)

**Why this is different from v3:**

v3 fixed the σ-amplification symptom (approx_kl was healthy) but
didn't fix the underlying reward conflict. With reward-vs-BC
conflict unresolved, PPO had two bad choices: (a) aggressively
update policy away from BC — breaks the prior, or (b) barely
update — ent_coef drifts std until random.

v4 removes the reward-vs-BC conflict at its source. PPO now sees a
reward that agrees with BC's behavior. Updates become small and
directional instead of large and conflicting.

**Why the user's intuition was right:**

They said "its gotta be ur dumb ass rewards being so stupid." They
were correct. I had been treating the reward as a fixed constant
that PPO should optimize, not as a design parameter that should be
chosen to match the behavioral prior we're starting from. Rookie
mistake on my part — I should have questioned the reward design for
BC context two runs ago.

**Expected v4 outcome:**

- Policy stays near BC mean + BC std=0.22
- Reward is now net-positive for expert-style driving (no -0.15/step
  red-light tax)
- PPO gradient gently refines BC under a compatible reward signal
- RC should climb past BC baseline (~5-8% rolling) to the 15-30%
  range where the BC model peaks on individual episodes

**If v4 still doesn't work:** nuclear (frozen BC eval) per
`PC_CLAUDE_NUCLEAR.md`. But I'm significantly more confident in v4
because this is the first time we've addressed the actual root cause
identified in the failure pattern across v1/v2/v3.

**p(ship with v4 producing better-than-BC-baseline): ~0.70.** If v4
matches BC baseline (5-8% RC) without diverging, ship that plus
BC-only as the comparison ablation. Either way, we ship.
