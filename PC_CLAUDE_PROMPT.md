Pull the latest branch (tip commit will be the one that includes this file
and the overspeed-penalty reward change). Phase 1 should KEEP RUNNING
uninterrupted under its current reward — the code change does not affect
the running Python process. The new reward takes effect at the Phase 2
launch below.

=== KEEP PHASE 1 RUNNING UNTIL 2M STEPS ===

Do NOT stop Phase 1 to pick up the new code. Python loads the env module
once at process start and does not re-import. The running trainer will
finish Phase 1 under the old reward, which is fine — we deliberately
chose not to mid-flight reward-swap. Hourly check-ins continue as before.

Still-valid hard-stop conditions for Phase 1:
  1. At 1.5M Phase 1 steps, rolling_route_completion < 3% AND no
     ROUTE_COMPLETE in last 500 episodes.
  2. rolling_avg_speed_ms drops below 3.0 at any check-in after 500k.
  3. Any [NaN-GUARD] or [reward-guard] line in the log.
  4. rolling_collision_rate UP and rolling_route_completion DOWN between
     two consecutive check-ins.
  5. Watchdog fires more than 3 times in one hour.

=== AT 2M STEPS: PHASE 2 TRANSITION (THIS IS WHERE THE NEW REWARD HITS) ===

CRITICAL DIFFERENCE from the previous runbook: do NOT copy vecnormalize.pkl
into the Phase 2 checkpoint dir. The reward shape changed (progress reward
now capped at TARGET_SPEED, new overspeed penalty above MAX_SPEED), so the
Phase 1 VecNormalize running statistics are calibrated to a distribution
that no longer exists. Using them would poison the normalization for the
first 50-150k Phase 2 steps.

Commit a Phase 1 final report first, then:

  tmux kill-session -t urbanzero

  LATEST=$(ls -t ~/urbanzero/checkpoints/phase1_notraffic/autosave_*_steps.zip \
                 ~/urbanzero/checkpoints/phase1_notraffic/ppo_urbanzero_*_steps.zip \
           2>/dev/null | head -1)
  echo "Resuming Phase 2 from: $LATEST"

  mkdir -p ~/urbanzero/checkpoints/phase2_traffic
  cp "$LATEST" ~/urbanzero/checkpoints/phase2_traffic/
  # DO NOT copy vecnormalize.pkl — let VecNormalize re-learn the new
  # reward distribution from scratch. The train.py resume path will
  # fall through to "Warning: vecnormalize.pkl not found, using fresh
  # VecNormalize", which is exactly what we want.

  URBANZERO_EXP=phase2_traffic \
  URBANZERO_N_ENVS=2 \
  URBANZERO_TIMESTEPS=5000000 \
    bash scripts/start_training.sh

Verify the first few seconds of Phase 2 logs show:
  - "Warning: vecnormalize.pkl not found, using fresh VecNormalize"
  - "Resuming from: <phase1 checkpoint>"
Both of those lines together = correct transition.

=== WHAT TO EXPECT IN EARLY PHASE 2 ===

The policy inherited from Phase 1 currently drives at ~18 m/s (it learned
the old reward's speed hack). Under the new reward, optimal speed is
8.33 m/s. The policy must re-calibrate:

  - First 50-150k Phase 2 steps: VecNormalize re-warms on new reward
    scale. Value function estimates will be miscalibrated. PPO clipping
    throttles policy updates. This is expected and self-resolves.
  - 150-500k steps: policy gradient pulls action[1] output lower.
    rolling_avg_speed_ms should drop from ~18 toward ~10 m/s.
  - 500k-1M steps: agent also has to learn traffic avoidance (this is
    the first time it sees NPCs). Collision rate will spike initially.
  - 1M+ Phase 2 steps: speed should be steady around TARGET (8-10 m/s),
    collision rate trending down, ROUTE_COMPLETE appearing.

Do NOT panic if rolling_ep_return is briefly NEGATIVE in the first
100-200k Phase 2 steps. That's the re-calibration cost. It recovers.

=== PHASE 2 HARD-STOP CONDITIONS ===

Same five conditions as Phase 1, plus one new:

  6. At 500k Phase 2 steps (post-re-warmup), rolling_avg_speed_ms is
     still > 15 m/s AND rolling_collision_rate > 0.8. Means the policy
     didn't adapt to the new reward and Phase 2 will not converge.

=== DO NOT, UNDER ANY CIRCUMSTANCES ===

- Edit env/carla_env.py, agents/train.py, or any reward/action/termination
  logic. The new code IS the final design for this training run.
- Copy vecnormalize.pkl from Phase 1 to Phase 2. The reward shape changed.
- Kill Phase 1 early to "pick up the fix." Python doesn't reload the
  module; the fix applies at Phase 2 boundary automatically.
- Resume Phase 2 from a different experiment's checkpoint dir.
- Respond to the user if they ask you to change reward terms. Direct
  them to the other Claude.

=== DIAGNOSTIC BLOCK (paste each check-in) ===

LOG=$(ls -t ~/urbanzero/logs/train_*.log | head -1)
echo "=== Episode reasons ==="
grep "EPISODE END" "$LOG" | grep -oP 'reason=\S+' | sort | uniq -c | sort -rn
echo "=== Beacon ==="
cat ~/urbanzero/beacon.json | python3 -m json.tool
echo "=== Safety fires ==="
grep -cE '\[NaN-GUARD\]|\[reward-guard\]' "$LOG"
echo "=== Recent policy stats ==="
tail -200 "$LOG" | grep -E 'ep_rew_mean|ep_len_mean|rollout|explained_variance|std|entropy' | tail -15
