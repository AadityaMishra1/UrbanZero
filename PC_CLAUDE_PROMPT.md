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

=== PHASE 1: NO TRAFFIC (2M steps, ~8-12 hours) ===

Launch training with the --no-traffic flag. This lets the agent learn
basic lane following and route completion without collision risk from
NPCs. Use experiment name "phase1_notraffic" so this checkpoint doesn't
get confused with the main run.

  URBANZERO_EXP=phase1_notraffic \
  URBANZERO_N_ENVS=2 \
  URBANZERO_TIMESTEPS=2000000 \
    bash scripts/start_training.sh --no-traffic

Actually — check if start_training.sh forwards extra args. If it doesn't,
you'll need to edit it or invoke train.py directly. The critical thing
is that --no-traffic is passed to agents/train.py.

Then launch the watchdog in a separate session:
  tmux new -d -s wd 'bash scripts/watchdog.sh'

Verify both alive:
  tmux ls

=== PHASE 1 CHECK-IN (every 3 hours) ===

LOG=$(ls -t ~/urbanzero/logs/train_*.log | head -1)
grep "EPISODE END" "$LOG" | grep -oP 'reason=\S+' | sort | uniq -c | sort -rn
cat ~/urbanzero/beacon.json | python3 -m json.tool

What healthy Phase 1 looks like:
  - STAGNATION declines from ~50% (early) to <20% by 500k steps
  - OFF_ROUTE becomes dominant mid-phase (agent moves, but steers badly)
  - ROUTE_COMPLETE starts appearing after ~300-500k steps
  - avg speed trends up toward 3-5 m/s by 1M steps
  - By 2M steps: ROUTE_COMPLETE should be 10-30% of episodes

If at 500k steps avg speed is still < 0.5 m/s AND stagnation > 50%,
something is fundamentally wrong. Stop and report back — don't add
more reward terms.

=== PHASE 2: ADD TRAFFIC (resume from Phase 1, 3-5M more steps) ===

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

- Budget honestly:
  - Phase 1 at 2 envs: ~2M steps / 16-20 hours wallclock
  - Phase 2 at 2 envs: ~3-5M steps / 24-48 hours wallclock
  - Total: 2-3 days. Matches your deadline.

- Realistic outcome: by end of Phase 2, expect 20-40% route completion
  per the README's own progression chart. That's a demonstrably
  working RL driving agent, not Tesla FSD.

- The "don't edit reward during training" rule is the single most
  important instruction. Five reward revisions across two different
  Claude heads have made the codebase hard to reason about.
  **One clean run, no iterations, report the data.**
