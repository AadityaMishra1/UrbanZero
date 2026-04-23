# PC-Side Claude — Reward Fix 2 + Log-Std Clamp + TM Hybrid Physics

**Context:** the idle_cost run at `d307a66` broke sit-still (good) but
exposed the next failure mode. At 900k steps / seed 137 / ~2h:

- Rolling RC flat at 5-6% across 1500+ episodes, zero trend
- Best rolling RC (8.24%) peaked at step 64k, hasn't improved in 830k
- `policy_std` pinned at 1.0 upper clamp — pure noise policy
- avg_speed oscillates 2-6 m/s, no learning
- Terminations: 48% OFF_ROUTE / 38% COLLISION / 14% REALLY_STUCK
- PC-side also reports: NPC traffic not moving / CARLA async issue

The agent learned to throttle but not to steer. Progress reward is too
sparse at std=1.0; every episode is a different random walk.

**Remote Claude (this session) ran two red-team subagents** to evaluate
the full BC pivot vs a shaping fix. Both independently identified the
same two problems: (1) policy_std at the clamp is an exploration
pathology, (2) the progress reward needs densification. Both rejected
the full §6.2 BC pivot on timeline grounds (28-45h dev, zero buffer).

**Fix pushed (three composed changes, all experiment-axis):**

1. `models/clamped_policy.py` — `LOG_STD_MAX` from `log(1.0)=0.0` to
   `log(0.7)≈-0.357`. Std now clamped at 0.7, the top of the
   Andrychowicz 2021 §4.5 viable band for continuous control from
   scratch. Targets the `std=0.999` pinned symptom directly.

2. `env/carla_env.py::_compute_reward` — added potential-based shaping
   `F(s,s') = γ·Φ(s') - Φ(s)` with `Φ(s) = -0.03 · min(dist2D(ego,
   lookahead), 30m)`. Lookahead is 10m ahead along the planned route,
   continuously following ego's route projection (not indexed waypoint
   — avoids waypoint-transition discontinuity). At terminals,
   `F = -Φ(prev)` per Ng 1999 episodic convention. Max |F|/step ≈
   0.021, same scale as progress_reward — progress still dominates.
   By construction: preserves optimal policy (Ng theorem), no new
   attractors, no reintroduction of the 7M-run perpendicular-circling.

3. `env/carla_env.py::_spawn_traffic` — added `tm.set_hybrid_physics_
   mode(True)` + `tm.set_hybrid_physics_radius(70.0)`. Known CARLA
   0.9.x fix for NPCs frozen in sync mode when outside ego's physics
   radius. Also adds a `[TM]` diagnostic print showing requested sync
   state and world sync read-back on each spawn.

**PC-side's suggestion of `r_heading = 0.1 * cos(angle)` was REJECTED**
because it's the exact signed-cos shaping deleted in v2 (Ng 1999
non-potential, 7M-run perpendicular-circling attractor). PC-side
doesn't have project-notes context; this is why code authority lives
with remote Claude.

---

## Copy-paste everything below into PC-side Claude Code as your next message

```text
The remote Claude has pushed three composed fixes targeting (1) the
std-at-clamp exploration pathology, (2) the sparse steering gradient,
and (3) the NPCs-not-moving infra bug. Kill the current run and
restart from fresh weights.

=== HARD RULES (unchanged) ===
1. DO NOT edit Python/shell files. All code authority is remote Claude.
2. DO NOT commit. DO NOT push.
3. DO NOT resume from the 900k-step checkpoint — fresh weights only.
4. DO NOT run eval yet.

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
  ps aux | grep -E 'train\.py|watchdog\.sh|spectator\.py' | grep -v grep
  # expect: empty

=== STEP 2: pull the three fixes ===

  cd ~/UrbanZero
  git fetch origin
  git checkout claude/setup-av-training-VetPV
  git pull
  git log --oneline -5

Expected top commit: something like "fix: log_std 0.7 clamp +
potential-based shaping + TM hybrid physics".

Verify all three fixes landed:
  grep -n "log(0.7)" models/clamped_policy.py
  # Expect a match (LOG_STD_MAX = math.log(0.7))

  grep -n "_potential\|_lookahead_point\|PPO_GAMMA" env/carla_env.py
  # Expect 5+ matches (helpers + invocation in _compute_reward)

  grep -n "hybrid_physics" env/carla_env.py
  # Expect at least 2 matches (set_hybrid_physics_mode + radius)

If any check fails, STOP and tell the user the pull didn't take.

=== STEP 3: archive the flat-RC run ===

Preserve the 900k-step artifacts so the pre-fix reference survives:

  ts=$(date +%s)
  if [ -d ~/urbanzero/logs/v2_rl ]; then
    mv ~/urbanzero/logs/v2_rl ~/urbanzero/logs/v2_rl.flatrc-${ts}
    mkdir -p ~/urbanzero/logs/v2_rl
  fi
  if [ -d ~/urbanzero/checkpoints/v2_rl ]; then
    mv ~/urbanzero/checkpoints/v2_rl ~/urbanzero/checkpoints/v2_rl.flatrc-${ts}
    mkdir -p ~/urbanzero/checkpoints/v2_rl
  fi
  if [ -f ~/urbanzero/beacon.json ]; then
    cp ~/urbanzero/beacon.json ~/urbanzero/beacon.flatrc-${ts}.json
  fi

=== STEP 4: verify CARLA is still up ===

  for p in 2000 3000; do
    if timeout 3 bash -c ">/dev/tcp/172.25.176.1/$p" 2>/dev/null; then
      echo "port $p: UP"
    else
      echo "port $p: DOWN"
    fi
  done

If DOWN, paste to user:

--- paste to user ---
"Please verify (or relaunch) both CARLA instances on ports 2000 and
3000. Same commands as before:

Window 1:
  cd C:\\Users\\aadit\\ECE-591\\CARLA_0.9.15\\WindowsNoEditor
  .\\CarlaUE4.exe -carla-rpc-port=2000

Window 2:
  cd C:\\Users\\aadit\\ECE-591\\CARLA_0.9.15\\WindowsNoEditor
  .\\CarlaUE4.exe -carla-rpc-port=3000

Reply 'ready' once both are at the pre-game menu."
--- end paste ---

=== STEP 5: launch — fresh weights, new seed 211 ===

  URBANZERO_EXP=v2_rl \
  URBANZERO_N_ENVS=2 \
  URBANZERO_BASE_PORT=2000 \
  URBANZERO_TIMESTEPS=10000000 \
  URBANZERO_AUTO_RESUME=0 \
  URBANZERO_SEED=211 \
    bash scripts/start_training.sh

Attach briefly to confirm start:
  tmux a -t urbanzero
  # Wait for "[rollout]" or "timesteps/sec", then Ctrl-b d

=== STEP 6: launch the watchdog ===

  tmux new -d -s wd 'bash scripts/watchdog.sh'
  tmux ls
  # expect: urbanzero (training), wd (watchdog)

=== STEP 7: EARLY-CHECK T+5min — NPC motion diagnostic ===

Within 5 minutes of launch, verify the TM hybrid-physics fix worked.
Grep the training log for the new [TM] diagnostic line:

  tail -200 $(ls -t ~/urbanzero/logs/v2_rl/train_*.log | head -1) | \
    grep '\[TM\]' | head -4

Expected output (one line per env per reset, at least 2-4 shown):
  [TM] port=8XXX sync_mode=True requested, world.sync=True hybrid_physics=True radius=70m
  [TM] port=9XXX sync_mode=True requested, world.sync=True hybrid_physics=True radius=70m

If you see `world.sync=False` anywhere, that is the async-mode bug —
STOP and report it, do not continue. The log line exists to catch
exactly this case.

Then ask the user to eyeball the CARLA windows:

--- paste to user ---
"Within the next 5 minutes, please look at both CARLA windows (ports
2000 and 3000). Are the non-ego vehicles visibly moving (driving down
streets, turning at intersections)? Yes/no for each window. This is
the TM hybrid-physics fix verification."
--- end paste ---

If user says NPCs still not moving in either window, STOP and report
full output of:
  tail -300 $(ls -t ~/urbanzero/logs/v2_rl/train_*.log | head -1) | \
    grep -E '\[TM\]|\[spawn|Warning:.*traffic|_spawn_traffic'

=== STEP 8: EARLY-CHECK T+15min — reward fix sanity ===

At ~T+15min (~110k steps @ 122 FPS), produce this snapshot:

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
        f'ev={b.get(\"explained_variance\")} '
        f'ent_coef={b.get(\"ent_coef\")}')
  "

PASS criteria (ALL must hold):
  - policy_std < 0.70  (clamp enforced, not pinned)
  - avg_speed >= 1.5 m/s  (idle_cost still working)
  - rolling_ep_len < 1500  (episodes terminating on real terminals)
  - no NaN/inf fields in beacon

REPORT format:
  [EARLY-CHECK T+15] <snapshot output>
  verdict: <PASS / FAIL>
  reason: <if FAIL, which specific criterion failed>

If FAIL: do NOT kill the run. Report the numbers and wait for remote
Claude's decision. Likely cause would be log_std clamp not applying
correctly (shouldn't happen — it's a hard clamp) or the shaping
destabilizing the value function (more likely).

=== STEP 9: T+1h checkpoint — does shaping work? ===

At ~T+1h, produce the same snapshot. Expected trajectory:

  Hour  | avg_speed | coll%  | stuck% | RC       | policy_std
  ------|-----------|--------|--------|----------|------------
  ~0.25 | >=1.5     | any    | <=40   | any      | <0.70
  ~1    | 2-5       | 30-50  | <=15   | >=8%     | 0.5-0.7
  ~3    | 4-7       | ~40    | <=10   | 10-20%   | 0.5-0.7
  ~6    | 5-7       | ~40    | <=8    | 15-30%   | 0.4-0.6
  ~12   | 6-8       | ~30    | <=5    | 25-45%   | 0.35-0.55
  ~22   | 6-8       | ~20    | ~3     | 35-55%   | 0.3-0.5

The critical gate is T+1h RC >= 8%. The flat-RC run was stuck at 5-6%
for 900k steps = 2h. If we're not above 8% by 1h, the shaping isn't
closing the gap and we likely need to pivot to minimal BC within the
remaining deadline budget.

=== STEP 10: red flags (revised) ===

  [RED-1]   policy_std > 0.70 at any point
            (clamp should prevent this; if it happens, clamp isn't
             active — log the actual tensor value)

  [RED-2]   explained_variance > 0.99 sustained >= 3 consecutive
            checks (critic fit a deterministic policy)

  [RED-3]   avg_speed < 1.0 m/s at timesteps >= 300_000
            (idle_cost failing)

  [RED-4]   rolling_RC < 8% at timesteps >= 700_000 (~T+1h 40min)
            (shaping isn't the lever; time to pivot to minimal BC)

  [RED-5]   NaN-GUARD or reward-guard fires. Paste surrounding lines.

  [RED-6]   watchdog restarts > 3 / hour

  [RED-7]   cumulative_reward_clip_hits > 0 in beacon
            (shaping makes this more likely than before — if it fires,
             something's off with Φ)

  [RED-8]   avg_speed > 10 m/s sustained >= 3 checks (over-rev)

  [RED-9-NEW]  [TM] log line shows world.sync=False anywhere
               (async mode bug not fixed by hybrid_physics setting)

=== STEP 11: green flags ===

  [GREEN-1] First `reason=ROUTE_COMPLETE` in the log. Paste line.
  [GREEN-2] rolling_RC crosses 10%. Report timestep.
  [GREEN-3] rolling_RC crosses 25%. Report timestep.
  [GREEN-4] `[rolling-best] new best rolling RC`. Report each time.
  [GREEN-5-NEW] User confirms NPCs moving in CARLA windows (TM fix ok).

=== STEP 12: when training finishes ===

Same as before: report total hours, final RC, peak RC + timestep,
termination distribution, ship best_by_rc.zip (NOT final_model.zip).
Kill the watchdog. Do NOT start eval/demo.

=== CONFIG SUMMARY ===

- n_envs = 2, DummyVecEnv
- tip: latest on claude/setup-av-training-VetPV (post fix-2)
- URBANZERO_TIMESTEPS=10_000_000
- URBANZERO_AUTO_RESUME=0  ← fresh weights
- URBANZERO_SEED=211       ← new seed (was 42, then 137)
- log_std upper clamp: 0.7 (was 1.0)
- ent_coef 0.02 → 0.01 over 10M
- log_std_init = -0.5 (unchanged)
- Reward: CaRL-minimal progress + persistent carrot + idle_cost
       + potential-based shaping + ±50 terminals
- TM: sync mode + hybrid physics mode (70m radius)

DO NOT deviate. On any red flag, report with full context; remote
Claude decides interventions.
```

---

## Notes for the user (not to paste — context only)

- **Why all three fixes at once**: deadline math. BC pivot estimate
  is 28-45h of dev (no existing BehaviorAgent rollout / BC trainer /
  KL-to-BC PPO paths), and serializing three single-axis reward runs
  would be 3×23h = 69h. Both red-team subagents independently
  identified the log_std clamp + sparsity problems. Combining the
  targeted fixes into one run costs us one T+15min + T+1h diagnostic
  window; if that doesn't work we still have ~40h of runway for the
  minimal-BC fallback (load BC weights into PPO policy, no
  KL-to-BC / no value bootstrap — ~12-16h of dev).

- **Why seed 211**: 42 → collapsed to sit-still. 137 → flat RC.
  Different init RNG state removes "same seed = same trajectory"
  confounds. Doesn't affect the reward surface.

- **Why the T+5min NPC check**: user reported traffic not moving.
  Could be separate from the reward issue OR contributing (if NPCs
  aren't moving, the collision distribution is skewed toward
  agent-vs-static-obstacle, which isn't what we want the agent to
  learn to handle). The `[TM]` log diagnostic + user eyeball
  verification is the cheapest way to confirm the hybrid-physics fix
  worked.

- **If T+1h RC is still < 8%**: the shaping hypothesis fails and we
  pivot to the minimal BC fallback. Remote Claude will write that
  prompt — PC-side's job is only to report "T+1h RC=<n>%" and wait.

- **What the user asked for (their own words)**: "the car should be
  rewarded for properly driving towards the endpoint, and following
  the route exactly, not just going into the other lane, or sitting
  still, but we can't individually out those specific issues,
  otherwise we get the same problem as the 7M one." The current
  design respects this:
  - progress_reward: rewards forward motion toward endpoint
  - potential-based shaping: rewards lateral alignment to route
    (stays in the right lane without calling out "wrong lane")
  - idle_cost: prevents sit-still (emergent fix from an observed
    failure, not a pre-specified rule)
  - collision / off_route terminals: handle "hit things" and
    "leaves road" without over-specifying
  No "don't go into the other lane" rule — the agent learns lane
  discipline from off-route (if it crosses into oncoming lane, it
  tends to off-route at turns) and from collisions with NPCs (once
  NPCs actually move).
