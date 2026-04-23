# PC-Side Claude — v2 Smoke Test Prompt

**Date written:** 2026-04-22 (revised 3x after deploy-side failures)
**Branch:** `claude/setup-av-training-VetPV`
**Tip commit:** `ddf0a8a` (fix(issue-4): inline per-worker spectator + cached TM + n_epochs 4->3)
**Purpose:** 10-minute smoke test of a full reward/policy/obs/infra rewrite before committing to a long training run. Goal is to verify the new code boots at 4 parallel CARLA envs, populates the beacon, and does not fire any reward/NaN guards OR the deadlock pattern from issues #3/#4.

**Revisions to date:**
1. First attempt: crashed with ENT_COEF_START UnboundLocalError (GitHub
   issue #2). Fixed in commit 79374e5 alongside a post-audit fix-pack
   (5 total bugs).
2. Second attempt (4 envs): training completed 1 PPO iteration then
   deadlocked (GitHub issue #3). Remote Claude initially diagnosed this
   as GPU contention at 4-env scale and fell back to 2 envs.
3. Third attempt (2 envs): ALSO deadlocked at iteration 3 (GitHub issue
   #4), proving the 4-env/GPU-contention diagnosis wrong. Real root cause
   identified via adversarial sub-agent audit with citations to CARLA's
   own issue tracker:
     - **CARLA #9172** (0.9.15-specific): lost-notify race in
       `TrafficManagerLocal.cpp`, fires preferentially when tick cadence
       slows. My v2 bumped `n_epochs=2→4`, doubling PPO's inter-rollout
       gap and reliably triggering the race.
     - **CARLA #1996 / #2239**: multi-client sync-mode contention. The
       external `scripts/spectator.py` was a secondary client on port
       2000, amplifying the race.
     - **CARLA #2789**: `get_trafficmanager()` hang path from re-
       registration under sync mode. v1 called this every reset.

**What changed in this (3rd) revision:**
- TrafficManager is now cached ONCE in env/carla_env.py `__init__`, not
  re-registered per reset (closes #2789 path).
- External scripts/spectator.py is NO LONGER auto-launched. Each CARLA
  server window follows its OWN worker's ego via an inline spectator
  transform update at the end of env/carla_env.py's `step()`. With 4
  envs the user sees 4 CARLA windows each tracking its own agent —
  NO external spectator process is needed.
- All remaining `world.tick()` call sites now pass `seconds=10.0` (three
  previously-unprotected sites fixed).
- `n_epochs=4 → 3` (closer to the 7M run's proven-safe `n_epochs=2`,
  shrinks the tick-gap that amplifies #9172 without fully giving up
  the extra gradient).
- Default `URBANZERO_N_ENVS` is back to **4**. The issue-#3 GPU-contention
  diagnosis was wrong; root cause was the CARLA TM race + multi-client
  contention, both now addressed.

---

## Copy-paste everything below into PC-side Claude Code as the first message

```text
I'm running UrbanZero (CARLA + PPO autonomous-driving RL). The remote
Claude has been iterating on branch `claude/setup-av-training-VetPV`; the
current tip is `ddf0a8a` and addresses three prior failures (GitHub
issues #2, #3, #4 — all closed). The v2 stack includes a CaRL-minimal
reward rewrite, a Traffic-Manager-cached env, an inline per-worker
spectator (no external process), and n_epochs=3 (tuned below the CARLA
#9172 deadlock threshold).

Your job is ONLY to:
  (1) pull the branch,
  (2) run a 10-minute smoke test with 4 parallel CARLA envs,
  (3) collect specific numeric signals,
  (4) paste the numbers back so the remote Claude can decide next steps.

=== HARD RULES — non-negotiable ===

1. DO NOT edit any Python file. Do not edit any shell script. Do not touch
   env/carla_env.py, agents/train.py, models/*.py, eval/*.py, or anything
   under scripts/. If you think a file needs editing, STOP and report that
   back in your final message. The remote Claude will do the edits.

2. DO NOT commit. DO NOT push. DO NOT tag. All git writes come from the
   remote side.

3. DO NOT attempt to resume from the 7M-step checkpoint. This is a fresh
   training run with different observation/action/reward shapes; resuming
   the old checkpoint will fail with shape-mismatch errors. The env var
   URBANZERO_AUTO_RESUME=0 in the launch command below takes care of this.

4. DO NOT hand-tune hyperparameters. If ent_coef or lr or anything else
   seems wrong, report it, don't change it.

5. If CARLA crashes or a worker times out, capture the last 100 lines of
   the log in your report. Do not attempt to fix it yourself.

=== STEP 1: sync the branch ===

Find the repo:
  REPO=$(find ~ -name carla_env.py -path "*/UrbanZero/*" 2>/dev/null | head -1 | xargs dirname | xargs dirname)
  cd "$REPO"

If that doesn't find anything, ask the user where the repo is. Do not guess.

Pull:
  git fetch origin
  git checkout claude/setup-av-training-VetPV
  git pull
  git log --oneline -6

Verify the current tip matches:

  ddf0a8a fix(issue-4): inline per-worker spectator + cached TM + n_epochs 4->3

If `ddf0a8a` is NOT the current HEAD, STOP and pull again. Running any
earlier tip will hit one of the already-fixed bugs:
  - before 79374e5 → crashes on ENT_COEF_START (issue #2)
  - before ddf0a8a → deadlocks via CARLA TM race (issues #3 and #4)

Also verify the external spectator tmux session is NOT running from a prior
attempt. If it is, kill it — the external spectator is what helped trigger
the issue-#4 deadlock:
  tmux kill-session -t spectator 2>/dev/null
  # should either succeed or say "can't find session: spectator"

If `scripts/spectator.py` is still alive as a Python process, kill that too:
  pkill -f "python3 scripts/spectator.py" 2>/dev/null

Make scripts executable:
  chmod +x scripts/preflight.py scripts/watchdog.sh scripts/start_training.sh

=== STEP 2: verify 4 CARLA instances on Windows ===

This test uses FOUR Windows CARLA servers running on ports 2000, 3000,
4000, and 5000. Prior issue-#3 attempt at 4 envs failed because of the
CARLA-internal #9172 race, not because of GPU contention at this env
count. The remote Claude's analysis + current fix pack (tip ddf0a8a)
addresses that race, so 4 envs is expected to be viable now.

From WSL, verify each port:

  for p in 2000 3000 4000 5000; do
    if timeout 3 bash -c ">/dev/tcp/172.25.176.1/$p" 2>/dev/null; then
      echo "port $p: UP"
    else
      echo "port $p: DOWN"
    fi
  done

If ALL FOUR ports are UP, proceed to STEP 3.

If any port is DOWN, STOP and paste the exact block below to the user.
Do NOT try to launch CARLA yourself from WSL — the paths and permissions
are fragile, and a failed launch half-holds a port which makes debugging
worse. The user runs these on the Windows side.

--- paste to user verbatim ---

"CARLA is not running on all four required ports (2000, 3000, 4000, 5000).
Please open FOUR separate Windows PowerShell windows and run the commands
below — one per window. Adjust the path if your CARLA install is somewhere
other than `C:\\Users\\aadit\\ECE-591\\CARLA_0.9.15`.

Launch them SEQUENTIALLY (wait ~20 seconds between each) — four simultaneous
UE4 boots are stressful on a 4080 and any single failure during launch is
harder to diagnose.

After ALL FOUR instances have reached the pre-game menu (you'll see a
CARLA logo and 'Press F1 for help' in each window), come back here and
say 'ready'. Leave ALL four windows OPEN for the duration of the smoke
test — closing one kills training on that worker. Each window will
automatically follow its own worker's ego vehicle once training starts
(inline spectator — no external spectator process needed).

Window 1 (port 2000):
  cd C:\\Users\\aadit\\ECE-591\\CARLA_0.9.15\\WindowsNoEditor
  .\\CarlaUE4.exe -carla-rpc-port=2000 -quality-level=Low -windowed -ResX=400 -ResY=300

Window 2 (port 3000):
  cd C:\\Users\\aadit\\ECE-591\\CARLA_0.9.15\\WindowsNoEditor
  .\\CarlaUE4.exe -carla-rpc-port=3000 -quality-level=Low -windowed -ResX=400 -ResY=300

Window 3 (port 4000):
  cd C:\\Users\\aadit\\ECE-591\\CARLA_0.9.15\\WindowsNoEditor
  .\\CarlaUE4.exe -carla-rpc-port=4000 -quality-level=Low -windowed -ResX=400 -ResY=300

Window 4 (port 5000):
  cd C:\\Users\\aadit\\ECE-591\\CARLA_0.9.15\\WindowsNoEditor
  .\\CarlaUE4.exe -carla-rpc-port=5000 -quality-level=Low -windowed -ResX=400 -ResY=300

Flags explained:
  -carla-rpc-port=<N>   : which port the CARLA server listens on (each
                          trainer worker connects to one of these)
  -quality-level=Low    : lowest render quality. Required at 4x concurrent
                          instances on 16 GB VRAM. Training uses semantic
                          seg not RGB, so render quality doesn't affect
                          the agent's observations.
  -windowed             : windowed mode, lets you see all four at once
  -ResX=400 -ResY=300   : tiny windows. Reduces VRAM and CPU cost.

If any instance fails to launch with 'server already running' or similar,
that port is held by a stale process. Kill it in Task Manager (look for
CarlaUE4.exe) and try again.

Once all four are at the pre-game menu, reply 'ready'."

--- end paste ---

After the user replies 'ready', re-run the port check loop. Do NOT proceed
until all four ports report UP.

If only 1-3 instances will launch (e.g., VRAM OOM on the 4th), STOP and
report back with the VRAM usage shown on the CARLA console — the remote
Claude may decide to downgrade to 3 envs for this run.

=== STEP 3: archive any prior experiment directory ===

Fresh run means fresh checkpoint dir. Prevent auto-resume ambiguity:

  EXPERIMENT=v2_rl
  if [ -d ~/urbanzero/checkpoints/$EXPERIMENT ]; then
    mv ~/urbanzero/checkpoints/$EXPERIMENT ~/urbanzero/checkpoints/$EXPERIMENT.pre-smoke-$(date +%s)
    echo "Archived prior $EXPERIMENT checkpoints"
  fi
  if [ -d ~/urbanzero/logs/$EXPERIMENT ]; then
    mv ~/urbanzero/logs/$EXPERIMENT ~/urbanzero/logs/$EXPERIMENT.pre-smoke-$(date +%s)
    echo "Archived prior $EXPERIMENT logs"
  fi

=== STEP 4: run preflight ===

  export URBANZERO_EXP=v2_rl
  export URBANZERO_N_ENVS=4
  export URBANZERO_BASE_PORT=2000
  python3 scripts/preflight.py

Expected output: all 12 checks show [ OK ] (9 base checks + 3 extra
"CARLA port NNNN" checks for ports 3000/4000/5000). If preflight FAILS,
do NOT bypass it — paste the failure lines and STOP.

=== STEP 5: launch the smoke test ===

Kill any prior training/watchdog sessions first:

  tmux kill-session -t urbanzero 2>/dev/null
  tmux kill-session -t wd 2>/dev/null
  tmux kill-session -t spectator 2>/dev/null

Launch with SMOKE-TEST-SPECIFIC env vars:

  URBANZERO_EXP=v2_rl \
  URBANZERO_N_ENVS=4 \
  URBANZERO_BASE_PORT=2000 \
  URBANZERO_TIMESTEPS=10000000 \
  URBANZERO_AUTO_RESUME=0 \
  URBANZERO_SEED=42 \
    bash scripts/start_training.sh

Note: URBANZERO_TIMESTEPS=10M is the eventual real target, but we will KILL
the smoke test after 10 minutes regardless. Do NOT use a small timesteps
value — a sub-10M cap would change the ent_coef anneal schedule (configured
for 10M in agents/train.py).

Attach briefly to confirm it started:
  tmux a -t urbanzero
  (then detach: Ctrl-b d)

You should see:
  - "=== UrbanZero Training ==="
  - "  Envs: 4"
  - "Device: cuda"
  - "Starting training for 10,000,000 timesteps..."
  - tqdm progress bar appearing
  - Each CARLA window's camera snaps to a bird's-eye view over its
    own Tesla ego (inline per-worker spectator — no external process)

If the process dies in the first 60 seconds, paste the last 50 lines:
  LOG=$(ls -t ~/urbanzero/logs/v2_rl/train_*.log | head -1)
  tail -50 "$LOG"
and STOP.

Watch specifically for `world.tick() raised ...` messages. The new
env/carla_env.py uses `world.tick(seconds=10.0)` at every tick site,
so any CARLA hang surfaces as a logged RuntimeError within 10s instead
of deadlocking silently (the failure mode from GitHub issues #3 and #4).

=== STEP 6: monitor for 10 minutes ===

At T+2min, T+5min, T+8min, T+10min, report:

  LOG=$(ls -t ~/urbanzero/logs/v2_rl/train_*.log | head -1)
  echo "=== beacon ==="
  cat ~/urbanzero/beacon.json | python3 -m json.tool
  echo "=== guard fires (should be 0) ==="
  grep -cE '\[NaN-GUARD\]|\[reward-guard\]' "$LOG"
  echo "=== EPISODE END distribution ==="
  grep "EPISODE END" "$LOG" | grep -oE 'reason=[A-Z_]+' | sort | uniq -c | sort -rn
  echo "=== recent PPO stats ==="
  tail -400 "$LOG" | grep -E 'ep_rew_mean|ep_len_mean|explained_variance|approx_kl|entropy_loss|std|ent_coef|rollout|timesteps' | tail -20

=== STEP 7: decide at T+10min ===

Kill the session regardless of outcome:
  tmux kill-session -t urbanzero

Then evaluate against these gates:

PASS ALL to declare smoke-test green:
  [P1] beacon "timesteps" >= 50000        (= aggregate >=83 FPS over 10 min)
  [P2] beacon "fps" >= 90                 (4-env aggregate target; issue-#3
                                           attempt at 4 envs measured 131
                                           FPS before the deadlock fired,
                                           so 90 is a conservative floor
                                           that reflects both new per-step
                                           overhead — inline spectator RPC,
                                           TM caching — and expected gains
                                           from removing the external
                                           spectator contention)
  [P3] grep count NaN-GUARD == 0
  [P4] grep count reward-guard == 0
  [P5] beacon "cumulative_reward_clip_hits" == 0
       (non-zero means reward hit the ±100 clip, which should never
        happen under the new design)
  [P6] beacon "termination_reasons" is a non-empty dict
       (at least 10 episodes have completed and their reasons tallied)
  [P7] beacon "policy_std" between 0.3 and 1.0
       (below 0.3 = entropy collapsing too fast; above 1.0 =
        LOG_STD_MAX clamp is firing — should not happen)
  [P8] beacon "approx_kl" is not None (SB3 has produced >=1 training pass)

FAIL signals (any one means stop and triage, not "try again"):
  [F1] fps < 70 aggregate: below the 4-env target. Report VRAM, GPU
       util, and the CARLA window count. The remote side may drop to
       3 envs. Do NOT change n_envs yourself.
  [F1a] process hangs / no log output for >60s — same pattern as issues
       #3 and #4. Should no longer happen at tip ddf0a8a (TM cached +
       inline spectator + 10s tick timeouts), but if it does:
         - Capture `py-spy dump --pid <trainer-pid>` if py-spy is
           installed. Worth installing: `pip install py-spy` before
           the test.
         - Else capture `ps auxf | grep -E "train.py|CarlaUE4"`
         - Grep log for any `world.tick() raised ...` messages. If
           PRESENT, the timeout fired — that tells us which tick site.
           If ABSENT, hang is elsewhere in the step pipeline.
  [F2] any NaN-GUARD or reward-guard line in the log.
  [F3] beacon "termination_reasons" heavily dominated by one reason
       (>80%) within the first 50 episodes. Note: 80% REALLY_STUCK
       would mean the agent isn't moving — but we expect this to
       surface LATER in training, not in the smoke window (10 min).
       In the smoke window OFF_ROUTE and COLLISION are expected
       dominant as the random-init policy explores.
  [F4] beacon "policy_std" above 0.95 OR below 0.3.
  [F5] CARLA worker timeouts (> 2 per minute in log).
  [F6] watchdog restart firing (check ~/urbanzero/watchdog.log).
       Watchdog should be idle — it's only running if you started it,
       and you should NOT start it for the smoke test.

=== STEP 8: report back ===

Paste this EXACT template back to the user, filling in every slot:

  === v2 SMOKE TEST REPORT (10 min) ===
  tip commit: <git rev-parse HEAD>
  CARLA ports: 2000/3000/4000/5000 UP: <yes/no>
  n_envs used: 4
  timesteps after 10 min: <n>
  aggregate FPS: <fps>
  policy_std: <value>
  approx_kl: <value>
  entropy_loss: <value>
  explained_variance: <value>
  ent_coef: <value>
  cumulative_reward_clip_hits: <n>
  NaN-GUARD fires: <n>
  reward-guard fires: <n>

  termination_reasons (rolling window 50):
  <paste dict>

  last EPISODE END distribution (cumulative):
  <paste uniq -c output>

  PASS gates (P1..P8):
  P1 >=50k timesteps : PASS / FAIL (<actual>)
  P2 >=90 FPS        : PASS / FAIL (<actual>)
  P3 NaN=0           : PASS / FAIL (<actual>)
  P4 reward-guard=0  : PASS / FAIL (<actual>)
  P5 clip_hits=0     : PASS / FAIL (<actual>)
  P6 reasons populated: PASS / FAIL (<actual>)
  P7 policy_std 0.3-1.0: PASS / FAIL (<actual>)
  P8 approx_kl present: PASS / FAIL (<actual>)

  Overall: PASS / FAIL

  Anomalies / surprises / user-visible behavior I noticed:
  <free text, anything unexpected>

  Log tail (last 100 lines):
  ```
  <paste>
  ```

Do NOT start the watchdog or the long training run. Do NOT launch with
different env vars to "try to fix" anything. Your job ends at pasting that
report. The remote Claude reads it, decides next steps, and may push
more commits.

If anything in these instructions is ambiguous or a command fails in an
unexpected way, ask the user BEFORE improvising. Silent improvisation is
what broke the 7M run.
```

---

## Notes for the user (not for pasting — just context)

- **Experiment name `v2_rl`.** This is a fresh namespace: checkpoints go to `~/urbanzero/checkpoints/v2_rl/`, logs to `~/urbanzero/logs/v2_rl/`. Your prior `shaped` / `phase1_notraffic` / `phase2_traffic` directories are untouched — you can still look at the old data if needed.

- **Inline per-worker spectator.** Each of the 4 CARLA windows will follow its own worker's Tesla Model 3 automatically — the env/carla_env.py `step()` sets each server's spectator transform once per tick. You should see 4 bird's-eye views, each tracking a different ego. No `scripts/spectator.py` process is launched. If you want to spot-check which window is which worker, look at the CARLA window title (CARLA appends the RPC port to the title bar in 0.9.15).

- **The agent has a goal every episode.** The route pipeline was NOT changed in v2: each episode `reset()` still picks a destination spawn point 200-800 m away, traces a waypoint-by-waypoint route with CARLA's `GlobalRoutePlanner`, and feeds the next 3 waypoints into the agent's state vector. The reward is the projection of ego motion onto that route, and ROUTE_COMPLETE (+50) fires when the ego reaches the final waypoint with speed < 3 m/s. What I changed in v2 was the *reward scale and shape*, not what the agent is trying to do.

- **Why 10 minutes and not 30.** The smoke test isn't trying to learn anything. It's proving the new stack doesn't crash, achieves target FPS on 4 envs, populates the new telemetry correctly, AND most importantly survives past iteration 3-4 where the issue-#4 deadlock used to fire. At 4 envs, `n_steps=256`, so a PPO rollout/train cycle is every `256 * 4 = 1024` env-steps. 10 minutes at 90+ FPS = ~54k env-steps = ~53 PPO iterations. If the deadlock pattern is truly dead, it will not recur in this window.

- **What happens if smoke test FAILS gate F1 (FPS too low).** Drop to 3 envs — i.e. shut down the CARLA instance on port 5000, set `URBANZERO_N_ENVS=3`, and re-run the smoke test. At 3 envs we lose ~25% sample diversity but VRAM pressure drops substantially.

- **What happens if smoke test FAILS gate F1a (hang despite the fixes).** That would mean CARLA #9172 fired anyway or some other deadlock source we haven't identified. The `world.tick(seconds=10.0)` timeouts I added should at least surface which tick site is blocking; look for `world.tick() raised` lines in the log and paste whatever does appear. If the log is truly silent, the hang is not in a tick call — it's in `apply_batch_sync`, a sensor dispatcher, or the SubprocVecEnv pipe, and we'll need the `py-spy dump` to find it.

- **What happens if smoke test FAILS gate F2 (NaN/reward-guard fires).** That's a code bug on the remote side. Paste the report back and I will investigate immediately.

- **After PASS.** I build P1 (BC data collection using `BehaviorAgent`) on a new branch, push, and write a similar PC-side prompt to smoke-test the data-collection script. Then P2 (BC training), P3 (value-head bootstrap), P4 (custom PPO with KL-to-BC), and finally the long run.

- **`4d` from Saturday deadline at time of writing (2026-04-22 Wed late afternoon).** P0 smoke test should take ~30 min of your and the PC-side Claude's time. Plan for P1-P4 to take ~4-6 hours of code work on my side + CARLA runtime for data collection and training. Total plan sits inside the 72-hour budget with buffer.
