# PC-Side Claude — v5 "Hail Mary" Run (external reviewer's plan)

**Context:** User's friend (experienced ML/RL reviewer) reviewed the
full 6-failure history in `DIAGNOSIS_FOR_REVIEW.md` and concluded:

- **Primary cause**: frozen NPCs (H_A) + observation/action sync
  issues (H_C). I had been treating NPCs as a side issue.
- **H_v4 (reward-BC conflict) is SECONDARY** — real but not the main
  problem.
- **Entropy Gradient Dominance** confirmed for v3's paradox: KL low
  because ent_coef small per-update, but policy drifts because in the
  absence of strong task gradient the only consistent signal is
  "increase entropy for the bonus." σ widens until policy goes random.

**Reviewer's final plan (executing verbatim):**

1. **Verify NPCs move in the viewport before training.** If frozen,
   STOP — fix the TM first. No amount of reward tuning matters until
   NPCs are dynamic.
2. **Hybrid reward v5+**: `idle_cost=0`, keep progress, keep
   potential shaping, add smooth collision penalty (`-impulse/100`
   capped, not flat -50).
3. **Tighter hyperparameter guardrails**:
   - `lr = 5e-5` (lower than v3/v4's 1e-4 — "nudge the BC policy,
     not overwrite it")
   - `ent_coef = 1e-3` constant (higher than v4's 1e-4 — floor it
     but don't zero it)
   - `batch_size = 128` (was 64 × n_envs — reduce gradient variance)
4. **T+60min hard gate at RC < 10%**: if not above 10% RC at
   200k steps (~60 min), kill and ship frozen BC.

## Four changes pushed in this commit

1. `env/carla_env.py`: track `_max_collision_force` in the collision
   sensor. Reward scales terminal penalty as `-impulse/100` clamped
   to [-100, -25]. Configurable via `URBANZERO_COLLISION_COEF` (0 to
   disable scaling, fall back to flat -50).

2. `env/carla_env.py::step`: new `[NPC-diagnostic]` print every 500
   sim steps showing sampled NPC avg/max speed. Status line: `MOVING`
   or `FROZEN`. Catches the frozen-NPC bug before the run wastes
   hours.

3. `agents/train.py` BC path: `lr=5e-5`, `ent_coef=1e-3` constant,
   `batch_size=128`. Identified as v5 in the startup log.

4. `scripts/start_training.sh`: forwards `URBANZERO_COLLISION_COEF`
   through tmux (same pattern as the other reward-knob env vars).

---

## Copy-paste everything below into PC-side Claude Code

```text
User's friend (external ML/RL reviewer) diagnosed the failure pattern
and wrote a specific plan. Execute v5 verbatim.

CRITICAL: at STEP 6 you must verify NPCs are moving. If they are NOT,
STOP. Do not start training with frozen NPCs. The reviewer was
explicit: "If the NPCs aren't moving when you run your script, stop.
Fix the TM port and sync settings before burning another minute of
GPU time."

=== STEP 1: kill everything ===

  tmux kill-session -t urbanzero 2>/dev/null
  tmux kill-session -t wd 2>/dev/null
  tmux kill-session -t eval 2>/dev/null
  tmux kill-session -t demo 2>/dev/null
  tmux kill-session -t bc0 2>/dev/null
  tmux kill-session -t bc1 2>/dev/null
  pkill -f "agents/train.py" 2>/dev/null
  pkill -f "scripts/watchdog.sh" 2>/dev/null
  pkill -f "scripts/eval_bc.py" 2>/dev/null
  pkill -f "scripts/collect_bc_data.py" 2>/dev/null
  sleep 3
  ps aux | grep -E 'train|watchdog|eval_bc|collect_bc' | grep -v grep
  # expect empty

=== STEP 2: pull v5 fixes ===

  cd ~/UrbanZero
  git fetch origin
  git checkout claude/setup-av-training-VetPV
  git pull
  git log --oneline -3

Expected top commit references "v5" or "reviewer's corrections".

Verify all four fixes landed:
  grep -n "URBANZERO_COLLISION_COEF" env/carla_env.py
  # expect at least 1 match in _compute_reward

  grep -n "_max_collision_force" env/carla_env.py
  # expect ≥2 matches

  grep -n "\[NPC-diagnostic\]" env/carla_env.py
  # expect 1 match in step()

  grep -n "BC-finetune v5\|LR_BC_FINETUNE = 5e-5" agents/train.py
  # expect ≥2 matches

  grep -n "URBANZERO_COLLISION_COEF" scripts/start_training.sh
  # expect 1 match in ACTIVATE

If any grep fails, STOP.

=== STEP 3: archive v3/v4 artifacts ===

  ts=$(date +%s)
  for exp in bc_ppo_finetune_v3 bc_ppo_finetune_v4; do
    for d in logs checkpoints; do
      if [ -d ~/urbanzero/$d/$exp ]; then
        mv ~/urbanzero/$d/$exp ~/urbanzero/$d/${exp}.fail-${ts}
      fi
    done
  done

=== STEP 4: verify CARLA up ===

  for p in 2000 3000; do
    timeout 3 bash -c ">/dev/tcp/172.25.176.1/$p" 2>/dev/null \
      && echo "port $p: UP" || echo "port $p: DOWN"
  done

=== STEP 5: launch v5 "Hail Mary" ===

CRITICAL env vars — all four MUST be set or v5 silently degrades
to prior-version behavior:

  URBANZERO_EXP=bc_ppo_finetune_v5 \
  URBANZERO_N_ENVS=2 \
  URBANZERO_BASE_PORT=2000 \
  URBANZERO_TIMESTEPS=5000000 \
  URBANZERO_AUTO_RESUME=0 \
  URBANZERO_SEED=915 \
  URBANZERO_BC_WEIGHTS=$HOME/urbanzero/checkpoints/bc_pretrain.zip \
  URBANZERO_IDLE_COST_COEF=0 \
  URBANZERO_REALLY_STUCK_STEPS=3000 \
  URBANZERO_COLLISION_COEF=0.01 \
    bash scripts/start_training.sh

Expected startup log lines (ALL must appear):
  [BC-finetune v5] lr=5e-05, n_epochs=1, clip_range=0.1, ent_coef=0.001 (constant, no schedule), widen_log_std=disabled
  [BC-finetune v5] IMPORTANT: requires env vars at launch:
                   URBANZERO_IDLE_COST_COEF=0
                   URBANZERO_REALLY_STUCK_STEPS=3000
                   URBANZERO_COLLISION_COEF=0.01  (smooth collision)
  [CarlaEnv] reward knobs: idle_cost_coef=0.0, really_stuck_steps=3000
  [BC-warmstart] loading weights from .../bc_pretrain.zip
  [TM] port=8XXX sync_mode=True requested, world.sync=True hybrid_physics=True radius=70m
  [TM] spawned 30 NPCs, speed_diff=-30% global, committed via tick

If `idle_cost_coef=-0.15` or `really_stuck_steps=1500` appears, env
var forwarding failed. STOP.

  # Ctrl-b d
  tmux new -d -s wd 'bash scripts/watchdog.sh'

=== STEP 6: T+2min NPC MOTION CHECK (HARD GATE) ===

Reviewer: "If NPCs aren't moving, stop." This is non-negotiable.

  tail -200 $(ls -t ~/urbanzero/logs/v2_rl/train_*.log ~/urbanzero/logs/bc_ppo_finetune_v5/train_*.log 2>/dev/null | head -1) | \
    grep '\[NPC-diagnostic\]' | head -3

Expected output (first diagnostic fires at step 500 = ~25s into first
episode, should appear within 2 min of launch across both envs):
  [NPC-diagnostic] step=500 n_sampled=10 avg_speed=3.XX m/s max_speed=X.XX m/s status=MOVING

**If status=FROZEN** → STOP immediately:
  tmux kill-session -t urbanzero
  tmux kill-session -t wd

Paste to user:
--- paste to user ---
"NPC motion diagnostic shows NPCs are FROZEN despite v5's TM
hybrid-physics fixes. Per the external reviewer, the training
environment is invalid with static NPCs. Options:
(a) Disable traffic entirely (URBANZERO_N_ENVS runs with
    enable_traffic=False) — agent learns collision-free routes only
(b) Accept static-NPC training as a known limitation, proceed with v5
    anyway (expected to see same 5% plateau pattern)
(c) Abort v5, ship frozen BC eval.
Which do you want?"
--- end paste ---

Wait for user's call.

**If status=MOVING in BOTH env samples** → proceed to STEP 7.

Also eyeball the CARLA windows (primary/port 2000 viewport) for 30s
to confirm. Diagnostic prints are from velocity query — should match
visible motion in the spectator.

=== STEP 7: T+15min BASELINE CHECK (informational, not a gate) ===

  cat ~/urbanzero/beacon.json | python3 -c "
  import json,sys
  b=json.load(sys.stdin)
  tr=b.get('termination_reasons',{})
  tot=sum(tr.values()) or 1
  print(f'ts={b[\"timesteps\"]} '
        f'RC={b[\"rolling_route_completion\"]:.2%} '
        f'speed={b[\"rolling_avg_speed_ms\"]:.1f} '
        f'std={b[\"policy_std\"]:.3f} '
        f'kl={b.get(\"approx_kl\"):.4f} '
        f'clip={b.get(\"clip_fraction\"):.3f} '
        f'ent_loss={b.get(\"entropy_loss\"):.3f} '
        f'ev={b.get(\"explained_variance\"):.3f} '
        f'coll%={100*tr.get(\"COLLISION\",0)/tot:.0f} '
        f'offr%={100*tr.get(\"OFF_ROUTE\",0)/tot:.0f} '
        f'stuck%={100*tr.get(\"REALLY_STUCK\",0)/tot:.0f}')
  "

Report as [T+15min v5]. No pass/fail decision here — just a health
check. Expect:
  - std near BC's 0.22 (not drifting up — key difference vs v3/v4)
  - avg_speed ≥ 3 m/s
  - ep_len dropping from ~1000 (if NPCs work, episodes end faster
    from real collisions, not REALLY_STUCK timeouts)

=== STEP 8: T+60min HARD GATE (reviewer's explicit threshold) ===

Reviewer: "If at T+60min you are still at 6% RC: Switch to Option B
[frozen BC eval]."

At ~T+60min / ~200k steps:

  cat ~/urbanzero/beacon.json | python3 -c "
  ...same snapshot command...
  "

**HARD GATE**: rolling_RC ≥ 10%

- RC ≥ 10% → reviewer's hypothesis confirmed. Continue to 5M steps.
- RC < 10% → kill v5. Ship frozen BC (see STEP 10).

Report as [T+60min v5] with full snapshot and verdict:
  PASS  (RC ≥ 10%) → continuing
  FAIL  (RC < 10%) → killing and pivoting to frozen BC eval

=== STEP 9: if PASS, continue monitoring ===

At T+2h check for first ROUTE_COMPLETE (grep `reason=ROUTE_COMPLETE`).
At T+6h target RC ≥ 25%. Run to 5M steps (~11h total). Ship
best_by_rc.zip.

=== STEP 10: if FAIL, nuclear option ===

  tmux kill-session -t urbanzero
  tmux kill-session -t wd

  mkdir -p ~/urbanzero/eval
  tmux new -d -s eval \
    "source ~/urbanzero_env/bin/activate && \
     cd ~/UrbanZero && \
     python3 scripts/eval_bc.py \
       --model ~/urbanzero/checkpoints/bc_pretrain.zip \
       --episodes 20 \
       --port 2000 \
       --seed 1001 \
       --output ~/urbanzero/eval/bc_frozen_final_$(date +%Y%m%d_%H%M).json \
     2>&1 | tee ~/urbanzero/logs/bc_eval_final_$(date +%Y%m%d_%H%M).log"

Wait for completion (~20-40 min). Report aggregate RC numbers. Ship
bc_pretrain.zip as final deliverable. Record demo video.

=== RED FLAGS (STOP and report regardless of gate status) ===

  [RED-1] [CarlaEnv] reward knobs shows idle_cost_coef != 0.0
  [RED-2] [NPC-diagnostic] status=FROZEN in both envs at T+2min
  [RED-3] policy_std > 0.4 at any check point (entropy drift back)
  [RED-4] avg_speed < 1.5 m/s at any check (creep-and-crash again)
  [RED-5] NaN-GUARD or reward-guard fires
  [RED-6] cumulative_reward_clip_hits > 0

=== CONFIG SUMMARY ===

- Tip: claude/setup-av-training-VetPV HEAD (post v5)
- BC_WEIGHTS: bc_pretrain.zip (50k frame checkpoint, MAE=0.05)
- Seed: 915
- Reward: idle_cost=0, REALLY_STUCK=3000 steps, COLLISION scaled by
  impulse (-25 to -100 clamped)
- PPO: lr=5e-5, n_epochs=1, clip_range=0.1, batch_size=128
- BC σ NOT widened (kept at 0.22)
- ent_coef: 1e-3 constant
- 5M-step horizon, but T+60min hard gate at 10% RC

Report STEP 6 (NPC motion) and STEP 8 (T+60min gate) as they fire.
Everything else runs automatic.
```

---

## Notes for the user (not to paste)

**What this run tests:**

The reviewer's hypothesis is that NPCs are the primary cause + H_v4
the secondary cause. Both fixes are in v5. Concrete falsifiable
prediction:

- If NPCs move (STEP 6 `status=MOVING`) AND RC crosses 10% by T+60min,
  reviewer's hypothesis is supported. Run to 5M steps.
- If NPCs are FROZEN (STEP 6 `status=FROZEN`), training environment
  is invalid. User decides whether to proceed anyway or abort.
- If NPCs move but RC still plateaus at 5-6% by T+60min, reviewer's
  hypothesis falsified — ship frozen BC, accept the limitation.

**The T+60min RC ≥ 10% gate is the reviewer's explicit number.** I'm
not inventing thresholds. It matches their quote: "If at T+60min you
are still at 6% RC: Switch to Option B."

**Deferred items from the reviewer's notes:**
- "Check your image normalization" (labels/27 may be too low-contrast
  for road vs sidewalk). Real concern, but I don't have time to test
  this axis without blowing the budget. Flagged in the report.

**If v5 FAILS the T+60min gate:**

Ship frozen BC. Eval already scripted (`scripts/eval_bc.py`). Report
narrative: "Three pure-RL runs failed at ~5% RC; three BC+PPO runs
failed at ~5-6% RC; external domain-expert review identified frozen
NPCs + reward-BC conflict as causes; v5 with all reviewer's fixes
still plateaued → ship frozen BC at X% deterministic RC."

That's publishable-grade failure analysis.
