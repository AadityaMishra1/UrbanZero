Pull commit b8fa666 and do a clean restart of Phase 1. The action-space
decoder was blocking all learning — a fresh PPO policy was sampling brake
roughly as often as throttle, which meant the car never moved, so no
gradient ever flowed. The fix adds an idle-creep bias so action[1]=0
maps to throttle=0.30 (automatic-transmission semantics).

DO NOT resume from any prior checkpoint. The old policy was trained to
output brake-heavy actions under the broken decoder; those weights are
actively wrong now.

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

mv ~/urbanzero/checkpoints/phase1_notraffic ~/urbanzero/checkpoints/phase1_deadzone_OLD_$(date +%Y%m%d_%H%M%S) 2>/dev/null
mv ~/urbanzero/checkpoints/shaped ~/urbanzero/checkpoints/shaped_OLD_$(date +%Y%m%d_%H%M%S) 2>/dev/null

=== LAUNCH PHASE 1 (no traffic, 2M steps, ~5.5h at 2 envs) ===

Make sure 2 CARLA servers are running on ports 2000 and 3000 first.
Preflight will check.

URBANZERO_EXP=phase1_notraffic \
URBANZERO_N_ENVS=2 \
URBANZERO_TIMESTEPS=2000000 \
URBANZERO_EXTRA_ARGS="--no-traffic" \
  bash scripts/start_training.sh

tmux new -d -s wd 'bash scripts/watchdog.sh'

tmux ls   # should show urbanzero, wd, spectator

=== CHECK-INS (every ~1 hour, or at milestones) ===

LOG=$(ls -t ~/urbanzero/logs/train_*.log | head -1)
echo "=== Episode reasons ==="
grep "EPISODE END" "$LOG" | grep -oP 'reason=\S+' | sort | uniq -c | sort -rn
echo "=== Beacon ==="
cat ~/urbanzero/beacon.json | python3 -m json.tool
echo "=== Safety fires ==="
grep -cE '\[NaN-GUARD\]|\[reward-guard\]' "$LOG"
echo "=== Recent policy stats ==="
tail -200 "$LOG" | grep -E 'ep_rew_mean|ep_len_mean|rollout|explained_variance|std|entropy' | tail -15

Paste the output verbatim. Do not analyze, do not propose fixes.

=== WHAT HEALTHY PHASE 1 LOOKS LIKE ===

At 50k steps:
  - avg_speed > 1.5 m/s (up from 0.000 under the broken decoder)
  - STAGNATION dropping below 50%
  - OFF_ROUTE becoming dominant (agent moves but steers randomly)
  - policy std starting to change from initial 0.367

At 100k steps:
  - avg_speed > 2.5 m/s
  - STAGNATION < 30%
  - COLLISION near 0 (no traffic enabled)

At 500k steps:
  - avg_speed > 4 m/s
  - ROUTE_COMPLETE starts appearing (5-15% of episodes)

At 2M steps (end of Phase 1):
  - ROUTE_COMPLETE 15-30% of episodes
  - avg_speed 5-7 m/s

=== HARD STOP CONDITION ===

If at 100k steps the beacon still shows avg_speed < 0.5 m/s AND
STAGNATION is still > 50%, STOP training and report. That would mean
the action-space fix wasn't enough and the blocker is deeper
(observation quality, network capacity, or something else). Don't add
reward terms; just report the data.

=== PHASE 2 (after Phase 1 completes 2M steps or plateaus) ===

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

=== RULES WHILE TRAINING ===

1. DO NOT edit env/carla_env.py, agents/train.py, or any reward or
   termination logic during a phase. No new penalty terms. No new
   gating conditions. No tuned weights. No exceptions.

2. DO NOT kill training to iterate. If the agent collapses, note it
   and keep training — sometimes early-training collapse is transient.
   Only stop at the hard-stop condition above (100k steps still stuck).

3. DO NOT resume into the wrong experiment directory. Phase 1 and
   Phase 2 must stay separate so their VecNormalize stats don't cross.

4. If training crashes or the watchdog restarts it, that's fine — note
   the event but don't debug unless it crashes more than 3 times in
   an hour.

5. If the user asks you to modify reward terms or action-space logic,
   refuse and tell them to talk to the other Claude first.
