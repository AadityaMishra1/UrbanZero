# PC-Side Claude — v2 Smoke Test Prompt

**Date written:** 2026-04-23 (revised 4x after deploy-side failures)
**Branch:** `claude/setup-av-training-VetPV`
**Tip commit:** (post-fix after issue #6 PC-side report — tip will be printed after the next push; see STEP 1 for verification)
**Purpose:** 10-minute smoke test of the v2 reward/obs/policy changes on TOP of v1's known-stable infrastructure. The 7M-step run used v1 infra at 2 envs with no special CARLA flags — this test reproduces exactly that infra setup and only layers the experiment changes on top.

**Revisions to date:**
1. Attempt 1: crashed with ENT_COEF_START UnboundLocalError (issue #2).
2. Attempt 2 (4 envs): deadlock at iter 1 (issue #3). Closed as "GPU contention" — that diagnosis was wrong.
3. Attempt 3 (2 envs): deadlock at iter 3 (issue #4). The `world.tick(seconds=10.0)` timeout never fired — hang was elsewhere in step().
4. Attempt 4 (2 envs, after adding TM-cache + inline spectator + n_epochs=3): deadlock at **iter 0** (issue #6). Those "fixes" made it worse.

**Root-cause finding from the latest diff audit:**
The earlier rounds of "fixes" layered multiple infrastructure changes
(TM caching in `__init__`, inline `world.get_spectator()` RPC in every
step, `seconds=10.0` timeout on every tick) on top of the experiment
changes. v1 did NONE of those things and ran 7M steps clean. This
revision removes ALL the infra changes and reverts to v1's exact
infrastructure behavior. Only the experiment changes (reward / obs /
policy hyperparameters) are kept.

**What's in this revision (post-issue-#6 fix):**
- v1 infra preserved: TM per-reset, no inline spectator, bare `world.tick()`.
- **`DummyVecEnv` instead of `SubprocVecEnv` in `agents/train.py`.** This
  is the real fix for issues #3/#4/#6. The BrokenPipeError in worker
  subprocesses that killed every prior attempt cannot happen with
  DummyVecEnv because there are no worker subprocesses. All envs run
  serially in the main process.
- Default `URBANZERO_N_ENVS=2`.
- Default CARLA launch flags (port only — no `-quality-level=Low`).
- Expected throughput: 90-110 FPS at 2 envs (down from the 147 FPS
  measured with SubprocVecEnv during the single surviving 20:03 run,
  but no BrokenPipeError risk).

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

  0a3f114 revert infra to v1 (834a8e0) behavior, keep v2 experiment changes

If `0a3f114` is NOT the current HEAD, STOP and pull again.

Kill EVERY lingering process from prior attempts before starting. Stale
processes and leaked CARLA state are a suspected contributor:

  tmux kill-session -t urbanzero 2>/dev/null
  tmux kill-session -t wd 2>/dev/null
  tmux kill-session -t spectator 2>/dev/null
  pkill -f "agents/train.py" 2>/dev/null
  pkill -f "scripts/spectator.py" 2>/dev/null
  pkill -f "scripts/watchdog.sh" 2>/dev/null

On the Windows side, ALSO kill every `CarlaUE4.exe` instance currently
running before launching fresh ones (Task Manager → details → sort by
name → end every CarlaUE4.exe process). Stale CARLA servers left over
from hung prior runs may be in a corrupted sync-mode state.

Make scripts executable:
  chmod +x scripts/preflight.py scripts/watchdog.sh scripts/start_training.sh

=== STEP 2: verify 2 CARLA instances on Windows ===

This test uses TWO Windows CARLA servers, ports 2000 and 3000 — exactly
the setup the user ran the prior 7M-step training on. Do NOT attempt
4 envs; the 4-env attempt exhausted VRAM on the 4080 Super (issue #5).

From WSL, verify each port:

  for p in 2000 3000; do
    if timeout 3 bash -c ">/dev/tcp/172.25.176.1/$p" 2>/dev/null; then
      echo "port $p: UP"
    else
      echo "port $p: DOWN"
    fi
  done

If BOTH ports are UP, proceed to STEP 3.

If either is DOWN, STOP and paste the exact block below to the user.
Do NOT try to launch CARLA yourself from WSL — the paths and permissions
are fragile, and a failed launch half-holds a port which makes debugging
worse. The user runs these on the Windows side.

--- paste to user verbatim ---

"CARLA is not running on both required ports (2000 and 3000). Please
kill any existing CarlaUE4.exe processes in Task Manager first (Details
tab, sort by name, end every CarlaUE4.exe). Then open TWO separate
Windows PowerShell windows and run the commands below — one per window.

Use the DEFAULT CARLA flags (no quality override, no forced resolution,
no windowed flag). This matches exactly how the 7M-step run was launched
that successfully trained for hours without deadlocks. Adjust the path
if your CARLA install is somewhere other than
`C:\\Users\\aadit\\ECE-591\\CARLA_0.9.15`.

After both instances have reached the pre-game menu (you'll see a CARLA
logo and 'Press F1 for help'), come back here and say 'ready'. Leave
both windows OPEN for the duration of the smoke test — closing one
kills training on that worker.

Window 1 (port 2000):
  cd C:\\Users\\aadit\\ECE-591\\CARLA_0.9.15\\WindowsNoEditor
  .\\CarlaUE4.exe -carla-rpc-port=2000

Window 2 (port 3000):
  cd C:\\Users\\aadit\\ECE-591\\CARLA_0.9.15\\WindowsNoEditor
  .\\CarlaUE4.exe -carla-rpc-port=3000

If either instance fails to launch with 'server already running', a
stale process is holding that port. Kill it in Task Manager and retry.

Once both are at the pre-game menu, reply 'ready'."

--- end paste ---

After the user replies 'ready', re-run the port check loop. Do NOT proceed
until both ports report UP.

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
  export URBANZERO_N_ENVS=2
  export URBANZERO_BASE_PORT=2000
  python3 scripts/preflight.py

Expected output: all 10 checks show [ OK ] (9 base checks + 1 extra
"CARLA port 3000" check for the second env). If preflight FAILS, do
NOT bypass it — paste the failure lines and STOP.

=== STEP 5: launch the smoke test ===

Kill any prior training/watchdog sessions first:

  tmux kill-session -t urbanzero 2>/dev/null
  tmux kill-session -t wd 2>/dev/null
  tmux kill-session -t spectator 2>/dev/null

Launch with SMOKE-TEST-SPECIFIC env vars:

  URBANZERO_EXP=v2_rl \
  URBANZERO_N_ENVS=2 \
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
  - "  Envs: 2"
  - "Device: cuda"
  - "Starting training for 10,000,000 timesteps..."
  - tqdm progress bar appearing

If the process dies in the first 60 seconds, paste the last 50 lines:
  LOG=$(ls -t ~/urbanzero/logs/v2_rl/train_*.log | head -1)
  tail -50 "$LOG"
and STOP.

If the process HANGS (no log output past iteration N, CARLA windows
freeze) — as happened in issues #3/#4/#6 — capture diagnostics before
killing it:

  # Install py-spy once if not installed:
  #   pip install py-spy
  TRAINER_PID=$(pgrep -f "agents/train.py" | head -1)
  # Dump stacks of the main process and all workers:
  py-spy dump --pid $TRAINER_PID > /tmp/trainer_main.stack
  for child in $(pgrep -P $TRAINER_PID); do
      py-spy dump --pid $child > /tmp/trainer_worker_$child.stack
  done
  ls -la /tmp/trainer_*.stack
  # Then paste the stack dumps in your report.

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
  [P1] beacon "timesteps" >= 40000        (= aggregate >=67 FPS over 10 min)
  [P2] beacon "fps" >= 70                 (2-env aggregate floor; the 7M
                                           run's prior 2-env measurement
                                           was ~100 FPS at commit 747a28a,
                                           so 70 is a conservative target)
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
  [F1] fps < 50 aggregate at 2 envs: below the 7M run's ~100 FPS by
       >50%. Report nvidia-smi VRAM, GPU util, CARLA window count.
  [F1a] process hangs / no log output for >60s — capture py-spy stack
       dumps for main + all workers as described in STEP 5 BEFORE
       killing anything. Without those dumps we cannot tell where in
       step() the hang is. Do NOT just kill and retry.
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
  CARLA ports: 2000/3000 UP: <yes/no>
  n_envs used: 2
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
  P1 >=40k timesteps : PASS / FAIL (<actual>)
  P2 >=70 FPS        : PASS / FAIL (<actual>)
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

- **No inline spectator.** The per-worker spectator RPC I added earlier was reverted. Each CARLA window just shows whatever its default spectator camera is. If you want per-worker bird's-eye tracking for debugging, you can run `URBANZERO_SPECTATOR_PORT=2000 python3 scripts/spectator.py` in a separate tmux pane (and similarly for port 3000) — but don't do this during the smoke test; it was a contributor to the issue-#4 deadlock chain.

- **The agent has a goal every episode.** The route pipeline was NOT changed in v2: each episode `reset()` still picks a destination spawn point 200-800 m away, traces a waypoint-by-waypoint route with CARLA's `GlobalRoutePlanner`, and feeds the next 3 waypoints into the agent's state vector. The reward is the projection of ego motion onto that route, and ROUTE_COMPLETE (+50) fires when the ego reaches the final waypoint with speed < 3 m/s. What I changed in v2 was the *reward scale and shape*, not what the agent is trying to do.

- **Why 10 minutes and not 30.** The smoke test isn't trying to learn anything. It's proving the new stack doesn't crash, reaches target FPS on 2 envs, populates the new telemetry correctly, and most importantly survives past iteration 3 where the issue-#4 deadlock used to fire. At 2 envs, `n_steps=512`, so a PPO rollout+train cycle is every `512 * 2 = 1024` env-steps. 10 minutes at 70+ FPS = ~42k env-steps ≈ 41 PPO iterations. If the revert cured the deadlock, it will not recur in that window.

- **What happens if smoke test FAILS gate F1 (FPS too low at 2 envs).** That's unexpected — the 7M run measured ~100 FPS at exactly this config. Report nvidia-smi, GPU util, CARLA window count, and any `[reset] world.tick() raised` lines in the log.

- **What happens if smoke test FAILS gate F1a (hang).** Capture `py-spy dump` for main + all workers BEFORE killing anything (commands in the PC-side prompt's STEP 5). Without those stacks we can't tell where in step()/reset()/init the hang is, and every retry without them is blind. py-spy install: `pip install py-spy`. Paste the full stack dumps in the report.

- **What happens if smoke test FAILS gate F2 (NaN/reward-guard fires).** That's a code bug on the remote side. Paste the report back and I will investigate immediately.

- **After PASS.** I build P1 (BC data collection using `BehaviorAgent`) on a new branch, push, and write a similar PC-side prompt to smoke-test the data-collection script. Then P2 (BC training), P3 (value-head bootstrap), P4 (custom PPO with KL-to-BC), and finally the long run.

- **`4d` from Saturday deadline at time of writing (2026-04-22 Wed late afternoon).** P0 smoke test should take ~30 min of your and the PC-side Claude's time. Plan for P1-P4 to take ~4-6 hours of code work on my side + CARLA runtime for data collection and training. Total plan sits inside the 72-hour budget with buffer.
