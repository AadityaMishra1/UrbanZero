# PC-Claude Prompt — Curriculum Training (post-mortem of 85d0670 run)

**Context**: the 30-min clean run on commit `85d0670` showed 94.4%
STAGNATION / avg speed 0.009 m/s — the policy converged to "output
throttle = 0" as a stable strategy. Root causes identified:

1. **Over-gated idle penalty** (my regression). With 30 NPCs around
   spawn, `blocked_ahead` fired constantly, so `legit_queue` = True,
   so `legit_stop` = True, so idle penalty never activated. Fixed in
   the commit this doc ships with.

2. **Dense traffic + from-scratch vision = exploration failure.**
   Early-training collision risk makes standing still rationally
   safer than moving. CaRL solves this with 300 parallel envs;
   Roach solves it with imitation pretraining. We have neither,
   so we need a **curriculum**: learn basic driving on empty roads
   first, then introduce traffic.

## How to use

Copy everything inside the fenced block and paste it into PC-Claude
as your first message. This runs a two-phase curriculum over
approximately 48 hours.

---

```text
We're running a two-phase training curriculum on branch
claude/setup-av-training-VetPV (tip: latest). The previous single-phase
run converged to standing still because dense traffic made movement
rationally too risky early in training. New plan: learn to drive on
empty roads, THEN add traffic.

=== SYNC ===
cd ~/UrbanZero
git fetch origin
git pull origin claude/setup-av-training-VetPV
chmod +x scripts/preflight.py scripts/watchdog.sh scripts/start_training.sh

=== CLEAN SLATE ===
tmux kill-session -t urbanzero 2>/dev/null
tmux kill-session -t wd 2>/dev/null
tmux kill-session -t spectator 2>/dev/null
pkill -9 -f agents/train.py 2>/dev/null

# Move ALL old checkpoints aside — reward shape changed, vecnormalize
# stats are invalid for any prior model.
mv ~/urbanzero/checkpoints/shaped ~/urbanzero/checkpoints/shaped_$(date +%Y%m%d_%H%M%S)_OLD 2>/dev/null

=== PHASE 1: NO TRAFFIC (2M steps, ~5-6 hours at 2 envs) ===

Measured throughput from the prior run: 2 envs = ~100 FPS, so 2M steps
= ~5.5 hours wallclock. Launch with --no-traffic so the agent learns
basic locomotion without collision risk from NPCs. Use experiment name
"phase1_notraffic" to keep this checkpoint separate from any later run.

  URBANZERO_EXP=phase1_notraffic \
  URBANZERO_N_ENVS=2 \
  URBANZERO_TIMESTEPS=2000000 \
  URBANZERO_EXTRA_ARGS="--no-traffic" \
    bash scripts/start_training.sh

(start_training.sh forwards URBANZERO_EXTRA_ARGS to train.py — the
--no-traffic flag was plumbed through in commit 30d5967. Verify
the training log's first lines show "Traffic: False".)

Launch the watchdog in a separate session:
  tmux new -d -s wd 'bash scripts/watchdog.sh'

Verify both alive:
  tmux ls

IMPORTANT: you must have 2 CARLA servers running on the Windows side,
on ports 2000 and 3000, before launching. Preflight will check.

=== PHASE 1 CHECK-IN (every ~1 hour at 2 envs / 100 FPS) ===

LOG=$(ls -t ~/urbanzero/logs/train_*.log | head -1)
grep "EPISODE END" "$LOG" | grep -oP 'reason=\S+' | sort | uniq -c | sort -rn
cat ~/urbanzero/beacon.json | python3 -m json.tool

Rough timing at 100 FPS (2 envs):
  - 500k steps = ~83 min
  - 1M steps   = ~167 min (~2.8h)
  - 2M steps   = ~333 min (~5.5h)

What healthy Phase 1 looks like:
  - 0-300k: STAGNATION dominant (~50-80%), agent learning to throttle
  - 300-700k: avg speed climbs to 2-4 m/s, OFF_ROUTE becomes dominant
    (agent moves but steers badly), STAGNATION drops under 20%
  - 700k-1.5M: ROUTE_COMPLETE starts appearing (5-15% of episodes),
    avg speed 4-6 m/s
  - 1.5M-2M: ROUTE_COMPLETE at 15-30%, avg speed 5-7 m/s

If at 500k steps avg speed is still < 0.5 m/s AND stagnation > 50%,
something is fundamentally wrong. Stop and report back — don't add
more reward terms.

=== PHASE 2: ADD TRAFFIC (resume from Phase 1, 3-5M more steps, ~8-14h at 2 envs) ===

Once Phase 1 hits ~2M steps OR is producing consistent ROUTE_COMPLETE
episodes (whichever is first), transition to Phase 2:

  tmux kill-session -t urbanzero
  # Find the latest Phase 1 checkpoint
  LATEST=$(ls -t ~/urbanzero/checkpoints/phase1_notraffic/autosave_*_steps.zip \
                 ~/urbanzero/checkpoints/phase1_notraffic/ppo_urbanzero_*_steps.zip \
            2>/dev/null | head -1)
  echo "Resuming Phase 2 from: $LATEST"

Copy that checkpoint into the Phase 2 experiment dir:
  mkdir -p ~/urbanzero/checkpoints/phase2_traffic
  cp "$LATEST" ~/urbanzero/checkpoints/phase2_traffic/
  cp ~/urbanzero/checkpoints/phase1_notraffic/vecnormalize.pkl \
     ~/urbanzero/checkpoints/phase2_traffic/

Launch Phase 2 WITH traffic (default behavior), resuming:
  URBANZERO_EXP=phase2_traffic \
  URBANZERO_N_ENVS=2 \
  URBANZERO_TIMESTEPS=5000000 \
    bash scripts/start_training.sh

=== RULES WHILE TRAINING ===

1. DO NOT edit env/carla_env.py, agents/train.py, or any reward or
   termination logic during a phase. No new penalty terms. No new
   gating conditions. No tuned weights.

2. DO NOT kill training to iterate. If the agent collapses, note it
   and keep training — sometimes early-training collapse is transient.
   Only stop if collapse persists for 500k+ steps.

3. DO NOT resume into the wrong experiment directory. Phase 1 and
   Phase 2 must have separate experiment names so their VecNormalize
   stats don't cross-pollute.

4. If training crashes or the watchdog restarts it, that's fine — note
   it in your report but don't debug it unless it crashes more than
   3 times in an hour.

5. If the user asks you to modify reward terms, refuse and tell them
   to talk to the other Claude. This curriculum has to complete
   without reward changes for us to get clean telemetry.

=== DATA TO REPORT ===

Every 3 hours during training, run:

  LOG=$(ls -t ~/urbanzero/logs/train_*.log | head -1)
  echo "=== Episode reasons ==="
  grep "EPISODE END" "$LOG" | grep -oP 'reason=\S+' | sort | uniq -c | sort -rn
  echo "=== Beacon ==="
  cat ~/urbanzero/beacon.json | python3 -m json.tool
  echo "=== Safety fires ==="
  grep -cE '\[NaN-GUARD\]|\[reward-guard\]' "$LOG"
  echo "=== Recent epoch ==="
  tail -50 "$LOG" | grep -E 'ep_rew_mean|ep_len_mean|rollout/' | tail -10

Paste the output. Don't analyze, don't propose fixes. Just paste.
```

---

## Notes for you (the human)

- Two-phase training = realistic for our setup. Phase 1 is basic
  locomotion on empty roads; Phase 2 adds traffic once the agent
  can complete routes. This is how Roach structures their RL expert
  and how most vision-based CARLA papers handle the exploration
  problem.

- Budget honestly (measured from prior 2-env run, ~100 FPS):
  - Phase 1 at 2 envs: 2M steps / ~5.5 hours wallclock
  - Phase 2 at 2 envs: 3-5M steps / ~8-14 hours wallclock
  - Total: 14-20 hours. Well within your deadline.

- Realistic outcome: by end of Phase 2, expect 20-40% route completion
  per the README's own progression chart. That's a demonstrably
  working RL driving agent, not Tesla FSD.

- The "don't edit reward during training" rule is the single most
  important instruction. Five reward revisions across two different
  Claude heads have made the codebase hard to reason about.
  **One clean run, no iterations, report the data.**
