# PC-Side Claude — Reward Fix Restart Prompt

**Context:** The long training run at tip `5814854` diverged at 233k
steps. Agent learned the sit-still local optimum (avg_speed 0.224 m/s,
REALLY_STUCK ≥70%, rolling RC 1.78% and dropping, last 18 consecutive
episodes all REALLY_STUCK). The early-warning report is in
`EARLY_WARNING_REALLY_STUCK.md`.

**Root cause (confirmed, not speculation):** per-step cost of
sitting still was `-50/1501 ≈ -0.033/step`, per-step cost of crashing
after ~300 steps of real driving was `-50/300 ≈ -0.167/step`. Sitting
still was ~5x cheaper per step than crashing, so the policy gradient
correctly converged to "do nothing." The critic's explained_variance
oscillations (−0.07 to +0.91) and healthy PPO stats confirm the
optimizer is fine; the reward structure was the bug.

**Fix pushed (remote side, this session):**
1. `env/carla_env.py::_compute_reward` — added
   `idle_cost = -0.15 * max(0.0, 1.0 - speed / 1.0)`. Continuous
   ramp: `-0.15/step` at zero speed, `0` at `speed ≥ 1 m/s`. Makes
   sitting still 1500 steps cost `-225 shaping + -50 terminal =
   -0.183/step`, now strictly more expensive than crashing.
2. Un-annealed the velocity carrot (removed `anneal_coef`). Carrot is
   now live for the whole run. This provides the gradient that pulls
   the agent above 1 m/s toward TARGET_SPEED (`-idle_cost` bottoms out
   at 0 above 1 m/s, carrot continues to reward faster motion up to
   TARGET).

**Tip to use:** latest commit on `claude/setup-av-training-VetPV`
(after this fix push — `git pull` first, expect a commit subject like
`fix: add idle_cost + un-anneal carrot`).

**Critical:** Restart from **fresh weights** (`URBANZERO_AUTO_RESUME=0`).
Do NOT resume from the 233k-step checkpoint. Continuing would leak the
"sit still" prior into the new reward landscape — the whole point of
the fix is that the reward surface is different now, and the policy
needs to re-explore under the new surface.

---

## Copy-paste everything below into PC-side Claude Code as your next message

```text
The remote Claude has pushed a reward fix. Kill the current run and
restart from fresh weights on the new reward structure.

=== HARD RULES ===

1. DO NOT edit any Python or shell file. All code authority is remote
   Claude. If a number looks wrong, report it.
2. DO NOT commit. DO NOT push.
3. DO NOT resume from the 233k-step checkpoint. Fresh weights only.
4. DO NOT run eval yet. Training first.

=== STEP 1: kill the current run ===

  tmux kill-session -t urbanzero 2>/dev/null
  tmux kill-session -t wd 2>/dev/null
  tmux kill-session -t spectator 2>/dev/null
  tmux kill-session -t spec0 2>/dev/null
  tmux kill-session -t spec1 2>/dev/null
  pkill -f "agents/train.py" 2>/dev/null
  pkill -f "scripts/spectator.py" 2>/dev/null
  pkill -f "scripts/watchdog.sh" 2>/dev/null
  sleep 3

Verify nothing is still running:
  ps aux | grep -E 'train\.py|watchdog\.sh|spectator\.py' | grep -v grep
(should return empty)

=== STEP 2: pull the reward fix ===

  cd ~/UrbanZero
  git fetch origin
  git checkout claude/setup-av-training-VetPV
  git pull
  git log --oneline -5

Expected top commit subject: something starting with "fix:" referencing
"idle_cost" or "un-anneal carrot". If you do not see a new commit
since `cfafe73`, STOP and re-pull after 30 seconds — the remote push
may still be in flight.

Verify the fix landed:
  grep -n "idle_cost" env/carla_env.py
  # Should match at least two lines inside _compute_reward
  grep -n "anneal_coef" env/carla_env.py
  # Should return nothing (anneal was removed)

If either check fails, STOP and tell the user the pull didn't take.

=== STEP 3: archive the failed run's artifacts ===

Preserve the sit-still-collapse run so we have the pre-fix reference
for the report:

  ts=$(date +%s)
  if [ -d ~/urbanzero/logs/v2_rl ]; then
    mv ~/urbanzero/logs/v2_rl ~/urbanzero/logs/v2_rl.sitstill-${ts}
    mkdir -p ~/urbanzero/logs/v2_rl
  fi
  if [ -d ~/urbanzero/checkpoints/v2_rl ]; then
    mv ~/urbanzero/checkpoints/v2_rl ~/urbanzero/checkpoints/v2_rl.sitstill-${ts}
    mkdir -p ~/urbanzero/checkpoints/v2_rl
  fi
  if [ -f ~/urbanzero/beacon.json ]; then
    cp ~/urbanzero/beacon.json ~/urbanzero/beacon.sitstill-${ts}.json
  fi

=== STEP 4: verify CARLA is still up ===

For port p in 2000 3000:
  for p in 2000 3000; do
    if timeout 3 bash -c ">/dev/tcp/172.25.176.1/$p" 2>/dev/null; then
      echo "port $p: UP"
    else
      echo "port $p: DOWN"
    fi
  done

If either is DOWN, paste to user:

--- paste to user ---
"Please verify (or relaunch) both CARLA instances on ports 2000 and
3000. Same commands as before:

Window 1 (port 2000):
  cd C:\\Users\\aadit\\ECE-591\\CARLA_0.9.15\\WindowsNoEditor
  .\\CarlaUE4.exe -carla-rpc-port=2000

Window 2 (port 3000):
  cd C:\\Users\\aadit\\ECE-591\\CARLA_0.9.15\\WindowsNoEditor
  .\\CarlaUE4.exe -carla-rpc-port=3000

Reply 'ready' once both are at the pre-game menu."
--- end paste ---

=== STEP 5: launch the long run (fresh weights) ===

Identical env config to the previous long run EXCEPT
URBANZERO_AUTO_RESUME=0 (fresh start) and a different seed so the
random init differs from the collapsed run's prior:

  URBANZERO_EXP=v2_rl \
  URBANZERO_N_ENVS=2 \
  URBANZERO_BASE_PORT=2000 \
  URBANZERO_TIMESTEPS=10000000 \
  URBANZERO_AUTO_RESUME=0 \
  URBANZERO_SEED=137 \
    bash scripts/start_training.sh

Attach briefly to confirm training started:
  tmux a -t urbanzero
  # Wait until you see "[rollout]" or "timesteps/sec" lines, then:
  # Ctrl-b d to detach

=== STEP 6: launch the watchdog ===

  tmux new -d -s wd 'bash scripts/watchdog.sh'
  tmux ls
  # expect: urbanzero (training), wd (watchdog)

=== STEP 7: EARLY-TEST — 15-minute sanity check ===

The whole point of the fix is to break the sit-still attractor. At
T+15 min (~110k steps at 122 FPS), the beacon should show a radically
different trajectory from the previous run's T+15 snapshot:

  Previous run T+15min (post-collapse trajectory):
    avg_speed ~1.0 m/s and dropping
    REALLY_STUCK rising past 50%

  Expected now, with idle_cost active:
    avg_speed >= 1.5 m/s (idle_cost penalty is nonzero below 1 m/s)
    REALLY_STUCK <= 40% rolling
    rolling_ep_len DOWN from 1924 (stuck episodes were padding this)

Produce this snapshot at T+15min:

  cat ~/urbanzero/beacon.json | python3 -c "
  import json,sys
  b=json.load(sys.stdin)
  tr=b.get('termination_reasons',{})
  tot=sum(tr.values()) or 1
  stuck_pct=100*tr.get('REALLY_STUCK',0)/tot
  coll_pct=100*tr.get('COLLISION',0)/tot
  offr_pct=100*tr.get('OFF_ROUTE',0)/tot
  print(f'ts={b[\"timesteps\"]} fps={b[\"fps\"]:.1f} '
        f'speed={b[\"rolling_avg_speed_ms\"]:.3f} '
        f'ep_len={b[\"rolling_ep_len\"]:.0f} '
        f'stuck%={stuck_pct:.1f} coll%={coll_pct:.1f} offr%={offr_pct:.1f} '
        f'RC={b[\"rolling_route_completion\"]:.3%} '
        f'std={b[\"policy_std\"]:.3f} '
        f'approx_kl={b.get(\"approx_kl\")} '
        f'ent_coef={b.get(\"ent_coef\")}')
  "

Report to the user as:
  [EARLY-CHECK T+15] <output above>
  verdict: <PASS if speed>=1.5 and stuck%<=40 else FAIL>

If FAIL, do NOT kill the run — report the numbers and wait for the
remote Claude's decision. The fix might need a stronger coefficient.

=== STEP 8: normal monitoring cadence (post-sanity-check) ===

After the T+15 sanity check passes, switch to the standard cadence:
every 1-2 hours for first 6-8 hours, then every 4-6 hours overnight.
Report against this expected trajectory (unchanged from the original
long-run prompt):

  Hour  |  avg_speed  | collision_rate | REALLY_STUCK % | RC      | policy_std
  ------|-------------|----------------|----------------|---------|------------
  ~0.25 |  >=1.5 m/s  |  any           |  <=40%         |  any    |  0.5-0.8
  ~1    |  2-4 m/s    |  ~0.4-0.6      |  ~20-30%       |  ~2-5%  |  0.5-0.7
  ~3    |  4-6 m/s    |  ~0.5          |  ~10-15%       |  5-10%  |  0.5-0.7
  ~6    |  5-7 m/s    |  ~0.5          |  ~5-10%        |  10-20% |  0.4-0.7
  ~12   |  6-8 m/s    |  ~0.3          |  ~5%           |  20-40% |  0.35-0.6
  ~22   |  6-8 m/s    |  ~0.2          |  ~3%           |  35-55% |  0.3-0.5

=== STEP 9: red flags (revised for the reward fix) ===

  [RED-1] policy_std < 0.3 before timesteps=3_000_000
          (entropy collapsing; ent_coef schedule not holding)

  [RED-2] explained_variance > 0.99 sustained >= 3 consecutive checks
          (critic fit a deterministic bad policy)

  [RED-3-NEW] avg_speed < 1.0 m/s at timesteps >= 300_000
              (idle_cost not breaking the sit-still attractor — the
              whole fix is the hypothesis under test here)

  [RED-4] rolling_route_completion flat at 0% at timesteps=3_000_000
          (trigger for BC-warmstart fallback per PROJECT_NOTES §6.2)

  [RED-5] NaN-GUARD or reward-guard fires in the log
            grep -cE '\[NaN-GUARD\]|\[reward-guard\]' \
              $(ls -t ~/urbanzero/logs/v2_rl/train_*.log | head -1)
          Non-zero is a problem; paste surrounding lines.

  [RED-6] watchdog restarts > 3 times in 1 hour
            tail -50 ~/urbanzero/watchdog.log

  [RED-7] cumulative_reward_clip_hits > 0 in the beacon
          (means the raw reward hit ±100 — the new idle_cost over a
          long episode should NOT do this, but check. A 1500-step
          stuck episode is -225 shaping + -50 = -275, but that's the
          episode sum, not per-step; per-step cannot exceed -0.155
          from idle_cost alone, nowhere near ±100. If this fires,
          something else is wrong.)

  [RED-8-NEW] avg_speed > 10 m/s for >= 3 consecutive checks
              (over-revving attractor — progress_cap should prevent
              this but flag if it happens; sustained speed well above
              TARGET_SPEED=8.33 is not rewarded by anything and
              suggests reward-hacking somewhere)

=== STEP 10: green flags (unchanged) ===

  [GREEN-1] First `reason=ROUTE_COMPLETE` in the log. Paste the full
            EPISODE END line with the timestep.

  [GREEN-2] rolling_route_completion crosses 10%. Report timestep.

  [GREEN-3] rolling_route_completion crosses 25%. Report timestep.

  [GREEN-4] `[rolling-best] new best rolling RC` line — report each time.

=== STEP 11: when training finishes ===

At timesteps >= 10_000_000, training exits cleanly. Report:
  - total wall-clock hours
  - final rolling_route_completion
  - peak rolling_route_completion and timestep it was reached at
  - final termination_reasons distribution
  - which checkpoint to ship (best_by_rc.zip, NOT final_model.zip)

Then kill the watchdog:
  tmux kill-session -t wd

Do NOT start eval/demo yourself. Remote Claude will write the eval prompt.

=== CONFIG SUMMARY ===

- n_envs = 2, DummyVecEnv
- tip: latest on claude/setup-av-training-VetPV (post-reward-fix)
- URBANZERO_TIMESTEPS=10_000_000
- URBANZERO_AUTO_RESUME=0  <- fresh weights, not resume
- URBANZERO_SEED=137       <- different from the collapsed run (42)
- ent_coef 0.02 -> 0.01 over 10M steps
- log_std_init=-0.5, upper-only clamp std<=1.0
- CaRL-minimal reward + persistent carrot + idle_cost + ±50 terminals
- REALLY_STUCK at 1500 steps (unchanged)

DO NOT deviate from this config. On any red flag, report with full
context; remote Claude decides the intervention.
```

---

## Notes for the user (not to paste — context only)

- **Why fresh weights, not resume**: the 233k-step policy has the
  "sit still" behavior baked into its parameters. Under the new
  reward surface, continuing from that checkpoint means the agent
  starts with a strong "don't move" prior that the idle_cost has to
  undo. Starting from fresh initialization lets the new reward shape
  policy-formation from step 1. Cheaper to re-spend 30 min than to
  fight a bad prior for the full run.

- **Why seed 137 not 42**: superstition + hygiene. The collapsed run
  started from seed 42 and found the sit-still optimum. A different
  seed starts the PPO sampler from a different RNG state, so the
  agent's early exploration trajectory differs. Doesn't affect the
  reward surface, but removes one source of accidental comparison
  confounds ("same seed means same trajectory" intuitions are wrong
  for PPO, but people still reach for them).

- **What the EARLY-CHECK at T+15 actually tests**: the hypothesis is
  that idle_cost alone is enough force to push the policy above
  speed=1 m/s. If at T+15 min we still see avg_speed < 1.5 m/s and
  REALLY_STUCK > 40%, the -0.15 coefficient is insufficient; the fix
  would be to increase it (e.g. -0.25) or widen the ramp (e.g.
  `max(0, 1 - speed/2.0)`). That's a *decision for remote Claude*
  based on T+15 data — PC-side just reports.

- **Timeline impact**: killing the 233k-step run loses ~30 min of
  compute. Restart + fresh 10M-step run takes ~23h at 122 FPS.
  Starting ~22:30 Thursday puts finish ~21:30 Friday. Saturday is
  still clear for eval/demo/report.

- **If the fix works but RC still plateaus near 0% at 3M steps**:
  the §6.2 BC-warmstart pivot is still the pre-agreed fallback. The
  reward fix is necessary (sit-still was blocking any learning at
  all) but may not be sufficient for tabula-rasa PPO at our 15M-step
  compute budget.
