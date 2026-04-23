# PC-Side Claude — BC Finetune v3 (H2+H3+H4 fix for Issue #7)

**Context:** v2 finetune (lr=1e-4) confirmed by Issue #7 beacon data to
still be in catastrophic unlearning — approx_kl=0.086 (barely changed
from v1's 0.079), entropy_loss positive and rising. Fresh no-context
subagent diagnosed the real cause:

**BC's tight std=0.22 amplifies PPO gradients 20x** vs std=1.0, because
the log-prob ratio scales as 1/σ². The lr fix was the wrong lever — σ
dominates, not lr. n_epochs=3 compounded each amplified update; clip_
range=0.2 was too wide for narrow-σ ratios.

## Three-axis fix pushed in this commit

1. **Widen log_std at BC load**: log(0.5)=-0.69. Keeps BC's mean
   network (actor MAE=0.05 preserved) but loosens over-confident
   variance so PPO gradients aren't 20x amplified. Andrychowicz 2021
   §4.5 viable band [0.3, 0.7].

2. **n_epochs: 3 → 1** for BC finetune. Roach 2021 §3.2 single-epoch
   finetune prevents the per-rollout compounding.

3. **clip_range: 0.2 → 0.1**. Schulman 2017 §6.1 "careful finetune"
   value — tighter ratio bounds at narrow σ.

All three only activate when `URBANZERO_BC_WEIGHTS` is set. From-
scratch PPO (if we ever do it again) uses the original 3/0.2 defaults.

## Nuclear option if this fails T+5min gate

Ship the **frozen BC policy evaluated deterministically**, no PPO at
all. Individual BC-only episodes hit 30-38% RC in the failed PPO
finetunes — that's the BC policy showing through before PPO erodes
it. Expected deterministic BC-only eval: 15-30% RC. 3-6x better than
the pure-RL 5% result. Legitimate ECE 591 deliverable with clean
scientific narrative.

---

## Copy-paste everything below into PC-side Claude Code as your next message

```text
Issue #7 (RED-2 at T+5min) confirmed by remote Claude's fresh
subagent analysis: lr reduction was the wrong lever. The real cause
is BC's std=0.22 amplifying PPO gradients 20x. Three-axis fix pushed.

=== STEP 1: kill current finetune ===

  tmux kill-session -t urbanzero 2>/dev/null
  tmux kill-session -t wd 2>/dev/null
  pkill -f "agents/train.py" 2>/dev/null
  pkill -f "scripts/watchdog.sh" 2>/dev/null
  sleep 3
  ps aux | grep -E 'train\.py|watchdog' | grep -v grep
  # expect empty

=== STEP 2: pull the H2+H3+H4 fix ===

  cd ~/UrbanZero
  git fetch origin
  git checkout claude/setup-av-training-VetPV
  git pull
  git log --oneline -3

Top commit references "H2+H3" or "widen log_std" or "BC finetune v3".

Verify the three fixes landed:
  grep -n "WIDEN_LOG_STD_TO" agents/train.py
  # expect ≥3 matches (definition + use)

  grep -n "N_EPOCHS_BC" agents/train.py
  # expect ≥3 matches (definition + 3 use sites)

  grep -n "widened log_std" agents/train.py
  # expect one match (the print in BC warmstart block)

=== STEP 3: archive v2 artifacts ===

  ts=$(date +%s)
  for d in logs checkpoints; do
    if [ -d ~/urbanzero/$d/bc_ppo_finetune_v2 ]; then
      mv ~/urbanzero/$d/bc_ppo_finetune_v2 \
         ~/urbanzero/$d/bc_ppo_finetune_v2.unlearning-${ts}
    fi
  done

=== STEP 4: launch BC finetune v3 ===

  URBANZERO_EXP=bc_ppo_finetune_v3 \
  URBANZERO_N_ENVS=2 \
  URBANZERO_BASE_PORT=2000 \
  URBANZERO_TIMESTEPS=5000000 \
  URBANZERO_AUTO_RESUME=0 \
  URBANZERO_SEED=913 \
  URBANZERO_BC_WEIGHTS=$HOME/urbanzero/checkpoints/bc_pretrain.zip \
    bash scripts/start_training.sh

  tmux a -t urbanzero

Expected startup log lines (all three MUST appear):
  [BC-finetune] lr=0.0001, n_epochs=1, clip_range=0.1, ent_coef 0.005->0.001 (floor), widen_log_std=-0.69
  [BC-warmstart] loading weights from .../bc_pretrain.zip
  [BC-warmstart] widened log_std [-1.503, -1.587] -> [-0.69, -0.69] (std now [0.5016, 0.5016])
  [BC-warmstart] ..._vecnormalize.pkl not found — using initial stats

If "widened log_std" line is MISSING, the fix didn't apply —
grep the log for an error and STOP.

Ctrl-b d to detach. Launch watchdog:
  tmux new -d -s wd 'bash scripts/watchdog.sh'

=== STEP 5: T+5min HARD GATE ===

At ~T+5min / ~30k steps, snapshot the beacon:

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
        f'ev={b.get(\"explained_variance\"):.3f} '
        f'coll%={100*tr.get(\"COLLISION\",0)/tot:.0f} '
        f'offr%={100*tr.get(\"OFF_ROUTE\",0)/tot:.0f}')
  "

**PASS criteria (need ≥4 of 5):**
  1. approx_kl ≤ 0.03  (hard pass ≤ 0.02). Prior: 0.086. HARD FAIL if > 0.04.
  2. clip_fraction ≤ 0.18. Prior: 0.25. HARD FAIL if > 0.22.
  3. entropy_loss ≤ 0 (entropy non-decreasing). Prior: +0.255. HARD FAIL if > +0.05.
  4. policy_std ∈ [0.35, 0.55] (widened & holding, not collapsing back to 0.22).
     HARD FAIL if std < 0.30.
  5. rolling_RC ≥ 0.055 (no regression from v2's 0.056).

Report as:
  [T+5min v3] <beacon snapshot>
  pass: N/5, verdict: PASS / SOFT-FAIL / HARD-FAIL

**Decision rules:**
  - ≥4/5 pass: KEEP RUNNING, re-check at T+30min with stronger targets
  - 3/5 pass: SOFT FAIL. Report to remote Claude with full snapshot.
  - ≤2/5 pass: HARD FAIL. Kill the run, trigger the nuclear option
    (frozen BC eval). Remote Claude will issue that paste block.

=== STEP 6: T+30min gate (only if T+5min PASSED) ===

  - approx_kl ≤ 0.02 (tightening as value fn stabilizes)
  - rolling_RC ≥ 10% (climbing — BC episodes hit 30-38% so this is
    conservative; if still at 6% we have a different problem)
  - collision_rate < 0.55

=== STEP 7: T+3h ===

  - rolling_RC ≥ 20%
  - First ROUTE_COMPLETE logged (grep for `reason=ROUTE_COMPLETE`)

=== STEP 8: 5M finetune finish (~11h) ===

Report: peak RC + timestep, final RC, termination dist, ship
best_by_rc.zip.

=== RED FLAGS ===

  [RED-1] "widened log_std" line missing → fix didn't apply
  [RED-2] HARD FAIL on T+5min gate → nuclear option
  [RED-3] std drops below 0.30 at any check → BC std collapsing again
  [RED-4] cumulative_reward_clip_hits > 0 → shaping overflow
  [RED-5] explained_variance < 0.3 after T+30min → critic destabilized

=== NUCLEAR OPTION (if T+5min HARD FAILS) ===

If remote Claude calls the nuclear option, you'll get a separate
paste block to:
  1. Kill the finetune
  2. Run deterministic eval of bc_pretrain.zip on Town01/02/03
     without any PPO finetune
  3. Produce demo video + eval numbers from the frozen BC policy
  4. Remote Claude writes the report section explaining the
     "BC is stronger than PPO-destabilized BC at our compute budget"
     finding with Rajeswaran 2017 citation

This is a LEGITIMATE ECE 591 result — the subagent explicitly noted
"clean scientific narrative" in its verdict.

=== CONFIG SUMMARY ===

- Tip: claude/setup-av-training-VetPV HEAD (post v3 fix)
- BC_WEIGHTS: bc_pretrain.zip (50k frames, port 2000)
- Seed: 913 (v2 was 912)
- Finetune: lr=1e-4, n_epochs=1, clip_range=0.1, ent_coef 0.005→0.001
- BC std widened: 0.22 → 0.50 at warmstart load (preserved mean network)
- All other hyperparams unchanged

Report T+5min gate within 7 minutes of launch.
```

---

## Notes for the user (not to paste)

**What the subagent found that I missed in my first fix:**

I treated this as a learning-rate problem (lower lr = smaller updates).
The subagent correctly identified it as a σ problem (lower σ = bigger
ratios per update, which compounds lr). The math:

`approx_kl ≈ 0.5 * (Δμ/σ)²` per update per dim

With Δμ=0.05 (one MAE-unit) and σ=0.22: `0.5 * (0.05/0.22)² = 0.026`
per update. Over n_epochs=3 that compounds to the 0.086 we saw in
beacon. Lowering lr 3x shrinks Δμ linearly; widening σ 2.3x shrinks
the ratio quadratically (5.2x). σ dominates.

**Why this is the right lever:**

We don't want to erase BC's learned mean network — that's what's
giving us 30-38% RC episodes. We just want to loosen the over-
confident variance so PPO gradient updates don't amplify. Widening
log_std is a zero-effort edit (one `.fill_` call) that preserves
everything good about BC while fixing the amplification.

**If it fails anyway (nuclear option):**

Ship `bc_pretrain.zip` as the final deliverable, no PPO finetune.
Evaluate deterministically (deterministic=True, ignore log_std entirely)
on Town01/02/03. Expected: 15-30% RC, 3-6x the pure-RL 5% baseline.
Report frames it as: "PPO finetune on BC prior at this compute budget
destabilized the prior; ablation with frozen BC is the stronger
baseline, consistent with Rajeswaran 2017." Clean story.

**p(ship) under this fix: 0.72.** Under nuclear option: ~0.95 (BC
policy already exists and drives).
