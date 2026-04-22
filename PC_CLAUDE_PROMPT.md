Proceed. Do NOT stop training for any reason except the hard-stop conditions
below. Keep the 1-hour check-ins, paste the telemetry, don't analyze.

=== KEEP RUNNING UNTIL ===

Phase 1 hits 2,000,000 total timesteps. That's the trigger to transition.
At current 157 FPS (2 envs) that's roughly 1.5-2 more hours from 1M.

=== AT 2M STEPS: AUTO-TRANSITION TO PHASE 2 ===

Save a check-in report first (commit to git like the last two), then:

  tmux kill-session -t urbanzero

  LATEST=$(ls -t ~/urbanzero/checkpoints/phase1_notraffic/autosave_*_steps.zip \
                 ~/urbanzero/checkpoints/phase1_notraffic/ppo_urbanzero_*_steps.zip \
           2>/dev/null | head -1)
  echo "Resuming Phase 2 from: $LATEST"

  mkdir -p ~/urbanzero/checkpoints/phase2_traffic
  cp "$LATEST" ~/urbanzero/checkpoints/phase2_traffic/
  cp ~/urbanzero/checkpoints/phase1_notraffic/vecnormalize.pkl \
     ~/urbanzero/checkpoints/phase2_traffic/

  URBANZERO_EXP=phase2_traffic \
  URBANZERO_N_ENVS=2 \
  URBANZERO_TIMESTEPS=5000000 \
    bash scripts/start_training.sh

Do NOT pass --no-traffic to Phase 2. Traffic is on by default.

Same check-in cadence for Phase 2: every hour, paste the diagnostic block
(episode reasons, beacon, safety fires, policy stats), commit report to git.

=== HARD-STOP CONDITIONS (only these, nothing else) ===

Stop training and commit a diagnostic report (don't edit code) if any of:

1. At 1.5M Phase 1 steps, rolling_route_completion is still < 3%
   AND no episode in last 500 has ended in ROUTE_COMPLETE.
   (Means agent isn't learning to steer — structural problem, not a
   reward tweak away.)

2. rolling_avg_speed_ms drops below 3.0 at any check-in after 500k
   steps. (Policy collapse back to stationary.)

3. Any [NaN-GUARD] or [reward-guard] line appears in the log.

4. rolling_collision_rate goes UP between two consecutive check-ins
   AND rolling_route_completion goes DOWN. (Policy getting worse,
   not better.)

5. The watchdog fires more than 3 times in one hour (something is
   genuinely broken; note crashes but keep training otherwise).

=== DO NOT, UNDER ANY CIRCUMSTANCES ===

- Edit env/carla_env.py, agents/train.py, or any reward/action/termination
  logic. Not even a coefficient tune. Not even a "quick experiment."
- Kill training to iterate on reward shape.
- Resume Phase 2 from a checkpoint path that doesn't have its matching
  vecnormalize.pkl alongside it.
- Respond to the user if they ask you to change reward terms — tell them
  to consult the other Claude first.

=== WHAT TO DO IF EVERYTHING'S FINE ===

Nothing. Check in every hour, paste the standard diagnostic block,
commit the report to git, and move on. A successful Phase 1 → Phase 2
transition with no code changes is the goal.

=== DIAGNOSTIC BLOCK TO PASTE EACH CHECK-IN ===

LOG=$(ls -t ~/urbanzero/logs/train_*.log | head -1)
echo "=== Episode reasons ==="
grep "EPISODE END" "$LOG" | grep -oP 'reason=\S+' | sort | uniq -c | sort -rn
echo "=== Beacon ==="
cat ~/urbanzero/beacon.json | python3 -m json.tool
echo "=== Safety fires ==="
grep -cE '\[NaN-GUARD\]|\[reward-guard\]' "$LOG"
echo "=== Recent policy stats ==="
tail -200 "$LOG" | grep -E 'ep_rew_mean|ep_len_mean|rollout|explained_variance|std|entropy' | tail -15
