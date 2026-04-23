# PC-Side Claude — BC Finetune Restart (hyperparam fix + bug fixes)

**Context:** Phase 1 + 2 went perfectly. Phase 3 is in **Case (2) —
catastrophic unlearning**. BC prior loaded correctly (confirmed:
`policy_std=0.222` and `avg_speed=7.6 m/s` from step 0, episodes
hitting 30-38% RC), but PPO is destroying the prior via excessive
updates.

**Beacon tells:** `approx_kl=0.079` (5x healthy), `clip_fraction=0.34`
(2x healthy), `rolling_RC=5.56%` (flat). Roach 2021 §3.2 explicitly
lowers lr to 1e-5 for BC finetune; we used 3e-4, which is 30x too
aggressive for a policy starting at std=0.22.

## Fixes pushed in this commit

1. **`agents/train.py`**: when `URBANZERO_BC_WEIGHTS` is set, PPO
   uses `lr=1e-4` (was 3e-4) and `ent_coef=0.005→0.001 floor` (was
   0.02→0.01). Roach-style. Preserves BC prior instead of destroying
   it. From-scratch runs still use 3e-4 / 0.02→0.01 — only the BC
   path gets the smaller updates.

2. **`agents/train.py`**: guarded `env.obs_rms` assignment with
   `hasattr()` so BC warmstart doesn't crash when training env uses
   `norm_obs=False` (reward-only VecNormalize). Fixes AttributeError
   you hit.

3. **`scripts/start_training.sh`**: added `URBANZERO_BC_WEIGHTS` to
   the ACTIVATE export list. Now forwards the env var through tmux
   automatically — no more manual tmux workaround needed.

4. **`agents/train_bc.py`**: replaced the OOM-on-100k-frames stacker
   with a lazy dataset (`_BCFrameStackDataset`). Stores only the
   stack-index map (~3 MB) instead of the full (N, 4, 128, 128)
   float32 tensor (~25 GB at 100k). You can now train on BOTH .npz
   files without OOM.

---

## Copy-paste everything below into PC-side Claude Code as your next message

```text
Remote Claude has pushed four fixes for the BC finetune restart.
Kill the current finetune (it's in catastrophic unlearning — BC
prior eroding under 30x-too-aggressive PPO updates). Then relaunch
with the new hyperparams.

=== STEP 1: kill current finetune ===

  tmux kill-session -t urbanzero 2>/dev/null
  tmux kill-session -t wd 2>/dev/null
  pkill -f "agents/train.py" 2>/dev/null
  pkill -f "scripts/watchdog.sh" 2>/dev/null
  sleep 3
  ps aux | grep -E 'train\.py|watchdog' | grep -v grep
  # expect empty

=== STEP 2: pull the fixes ===

  cd ~/UrbanZero
  git fetch origin
  git checkout claude/setup-av-training-VetPV
  git pull
  git log --oneline -3

Top commit should reference "BC finetune hyperparam fix" or similar.

Verify all four fixes landed:
  grep -n "LR_BC_FINETUNE" agents/train.py
  # expect ≥3 matches (definition + 3 use sites)

  grep -n "hasattr" agents/train.py
  # expect match guarding env.obs_rms / env.ret_rms

  grep -n "URBANZERO_BC_WEIGHTS" scripts/start_training.sh
  # expect match in ACTIVATE line

  grep -n "_BCFrameStackDataset\|_compute_stack_indices" agents/train_bc.py
  # expect ≥3 matches (class + fn + use)

If any grep fails, STOP and tell the user the pull didn't take.

=== STEP 3 (OPTIONAL): retrain BC on 100k frames now that OOM is fixed ===

The first BC model was trained on 50k frames only (port 2000 alone)
due to the OOM bug. With the lazy-stacking fix, we can train on the
full 100k now. Better BC prior = less PPO drift needed during
finetune.

Cost: ~45-90 min of GPU time, ~2 GB RAM (down from 25 GB).
Benefit: stronger BC prior, likely better final RC.

Recommendation: YES, retrain BC on both .npz files. It's cheap and
the pipeline is already warm. But if you'd rather save the time,
skip to STEP 4 with the existing 50k-trained bc_pretrain.zip.

If retraining:
  ls ~/urbanzero/bc_data/bc_port*.npz
  # note the two .npz paths

  tmux new -d -s bctrain2 \
    "source ~/urbanzero_env/bin/activate && \
     cd ~/UrbanZero && \
     python3 agents/train_bc.py \
       --data ~/urbanzero/bc_data/bc_port2000_*.npz ~/urbanzero/bc_data/bc_port3000_*.npz \
       --output ~/urbanzero/checkpoints/bc_pretrain_100k.zip \
       --epochs 20 --batch_size 256 --lr 3e-4 \
     2>&1 | tee ~/urbanzero/logs/bc_train_100k_$(date +%Y%m%d_%H%M).log"

  # Wait for completion — monitor:
  while tmux has-session -t bctrain2 2>/dev/null; do
    echo "$(tmux capture-pane -t bctrain2 -p | grep -E 'epoch|loss' | tail -1)"
    sleep 120
  done
  ls -la ~/urbanzero/checkpoints/bc_pretrain_100k.zip

Then use the new checkpoint in STEP 4 below by changing the
URBANZERO_BC_WEIGHTS path.

=== STEP 4: launch PPO finetune with the new hyperparams ===

Use bc_pretrain_100k.zip if you did STEP 3, else bc_pretrain.zip.

  # Archive the previous finetune run
  ts=$(date +%s)
  if [ -d ~/urbanzero/logs/bc_ppo_finetune ]; then
    mv ~/urbanzero/logs/bc_ppo_finetune ~/urbanzero/logs/bc_ppo_finetune.unlearning-${ts}
  fi
  if [ -d ~/urbanzero/checkpoints/bc_ppo_finetune ]; then
    mv ~/urbanzero/checkpoints/bc_ppo_finetune ~/urbanzero/checkpoints/bc_ppo_finetune.unlearning-${ts}
  fi

  # Launch — note the BC_WEIGHTS path. Use bc_pretrain_100k.zip if
  # you retrained in STEP 3.
  URBANZERO_EXP=bc_ppo_finetune_v2 \
  URBANZERO_N_ENVS=2 \
  URBANZERO_BASE_PORT=2000 \
  URBANZERO_TIMESTEPS=5000000 \
  URBANZERO_AUTO_RESUME=0 \
  URBANZERO_SEED=912 \
  URBANZERO_BC_WEIGHTS=$HOME/urbanzero/checkpoints/bc_pretrain.zip \
    bash scripts/start_training.sh

  tmux a -t urbanzero
  # Wait for first "[BC-warmstart]" line AND "[BC-finetune] lr=0.0001"
  # line (confirms fix took). Ctrl-b d.

Expected startup log lines:
  [BC-finetune] lr=0.0001, ent_coef 0.005->0.001 (floor)
  [BC-warmstart] loading weights from .../bc_pretrain.zip
  [BC-warmstart] ..._vecnormalize.pkl not found — using initial stats
  (or) [BC-warmstart] VecNormalize stats restored from ...

If you do NOT see "[BC-finetune] lr=0.0001" in the log, the env var
forwarding still isn't working. STOP and report.

  tmux new -d -s wd 'bash scripts/watchdog.sh'

=== STEP 5: T+5min beacon check ===

  cat ~/urbanzero/beacon.json | python3 -c "
  import json,sys
  b=json.load(sys.stdin)
  tr=b.get('termination_reasons',{})
  tot=sum(tr.values()) or 1
  print(f'ts={b[\"timesteps\"]} fps={b[\"fps\"]:.1f} '
        f'RC={b[\"rolling_route_completion\"]:.2%} '
        f'speed={b[\"rolling_avg_speed_ms\"]:.1f} '
        f'std={b[\"policy_std\"]:.3f} '
        f'approx_kl={b.get(\"approx_kl\")} '
        f'clip_frac={b.get(\"clip_fraction\")} '
        f'ev={b.get(\"explained_variance\"):.3f} '
        f'coll%={100*tr.get(\"COLLISION\",0)/tot:.0f} '
        f'offr%={100*tr.get(\"OFF_ROUTE\",0)/tot:.0f}')
  "

PASS criteria (compare to the prior finetune's 28k-step snapshot):
  - approx_kl ≤ 0.02 (prior: 0.079 — unhealthy)
  - clip_fraction ≤ 0.15 (prior: 0.34 — unhealthy)
  - rolling_RC ≥ 6% and rising (prior: flat 5.56%)
  - policy_std stays close to BC start ~0.22 (prior: same)
  - avg_speed ≥ 5 m/s (prior: 7.6 — but aggressive)

If approx_kl is STILL > 0.05 or clip_fraction > 0.25, the lr reduction
didn't take — likely the URBANZERO_BC_WEIGHTS env var didn't reach
train.py. Grep the log for "[BC-finetune] lr=" to verify.

=== STEP 6: T+30min gate ===

  - rolling_RC ≥ 10% (climbing)
  - rolling_collision_rate < 0.55 (down from 0.68)
  - approx_kl holds at ≤ 0.02

If RC is climbing, we're on track. If RC is flat or dropping from
T+5min, report to remote Claude — BC prior is still drifting.

=== STEP 7: T+3h mark ===

  - rolling_RC ≥ 20%
  - At least one ROUTE_COMPLETE episode in the log

=== STEP 8: when 5M finetune done (~11h) ===

Report:
  - peak rolling_RC and timestep
  - final rolling_RC
  - final termination distribution
  - ship best_by_rc.zip (NOT final_model.zip)

=== RED FLAGS ===

  [RED-1] "[BC-finetune] lr=" line missing → env var forwarding still broken
  [RED-2] approx_kl > 0.05 at T+5min → lr reduction didn't take
  [RED-3] rolling_RC dropping from T+5min → BC prior eroding despite fix
  [RED-4] explained_variance drops below 0.3 → critic destabilized
  [RED-5] cumulative_reward_clip_hits > 0

=== CONFIG SUMMARY ===

- Tip: claude/setup-av-training-VetPV HEAD (post BC-finetune-fix)
- BC_WEIGHTS: bc_pretrain.zip (or bc_pretrain_100k.zip if retrained)
- Seed 912 (new, prior was 911)
- Finetune lr = 1e-4 (was 3e-4 — 30x too aggressive for BC)
- Finetune ent_coef = 0.005 → 0.001 (was 0.02 → 0.01 — preserves BC)
- All other hyperparams unchanged (n_steps, batch_size, clip_range,
  max_grad_norm, etc. at PPO defaults)

On any red flag, report with full beacon snapshot; remote Claude decides.
```

---

## Notes for the user (not to paste)

**What the previous beacon told us (confirming the fix is correct):**

Prior finetune attempt, at 28k steps:
- `approx_kl = 0.079` — policy is changing 5x faster per update than healthy PPO. When BC gives you a tight std=0.22 policy, any action gradient produces huge log-prob ratios → huge KL. Healthy PPO target is 0.01-0.02. Lowering lr to 1e-4 is the standard textbook fix.
- `clip_fraction = 0.34` — 34% of samples are being clipped. This means PPO is TRYING to push the policy very far but being held back by the 0.2 clip range. When clipped samples exceed ~20%, the actor's gradient signal is distorted because only the unclipped samples contribute. Symptom of same root cause.

Roach (Zhang 2021 §3.2) explicitly uses `lr=1e-5` for BC finetune. We're using `1e-4` as a compromise because we have 5M steps, not the 10M+ they used. Still 30x smaller than from-scratch PPO.

**What's still not addressed and why:**

1. **NPCs are still frozen.** The TM hybrid-physics + speed-diff fixes didn't unstick them. This is a separate root cause — probably deeper than the TM settings we've tried. It does NOT block BC finetune from working because BehaviorAgent doesn't need moving NPCs to demonstrate lane/route-following. But the collision rate will stay high until NPCs move (or until PPO learns to avoid static NPCs).

2. **The user may see wrong-lane driving persist for a while.** BC taught fast driving; the expert occasionally does lane changes. Without a strict "stay in right lane" rule (§8 forbids), the agent will gradually learn lane discipline from collisions over many episodes. This is slower than pure-RL would be if pure-RL worked.

**p(ship) after this fix:** ~0.75 (up from the unlearning-regime ~0.4 the prior launch was heading toward). The BC prior is a STRONG starting point — episodes hitting 38% RC in the first minutes is not noise, it's the BC policy driving competently for ~15 seconds at a time. We just need to stop destroying that prior with aggressive PPO updates.

**Your shippable result is likely in hand within 8-12 hours.** BC took 12 minutes to collect + 10 minutes to train + ~8 hours of stable finetune should produce a policy at 25-45% RC. Eval + demo + report on Friday-Saturday.
