# PC-Side Claude — v2 Smoke Test Prompt

**Date written:** 2026-04-22 (revised twice after deploy-side failures)
**Branch:** `claude/setup-av-training-VetPV`
**Tip commit:** `<to-be-updated-after-this-commit>`
**Purpose:** 10-minute smoke test of a full reward/policy/obs/infra rewrite before committing to a long training run. Goal is to verify the new code boots at the TARGET env count, populates the beacon, and does not fire any reward/NaN guards.

**Revisions to date:**
1. First attempt: crashed with ENT_COEF_START UnboundLocalError (GitHub
   issue #2). Fixed in commit 79374e5 alongside a post-audit fix-pack
   (5 total bugs).
2. Second attempt: training completed 1 PPO iteration at 4 envs and then
   deadlocked (GitHub issue #3 — multi-CARLA sync deadlock, documented
   failure mode of SubprocVecEnv with multiple CARLA instances on a
   single GPU under PPO.train() load).

**What changed in the latest revision:**
- Default `URBANZERO_N_ENVS` for the smoke test is now **2**, matching
  the prior 7M run's empirically stable configuration (commit 747a28a
  measured ~100 FPS aggregate there).
- Added `world.tick(seconds=10.0)` in env/carla_env.py hot paths so any
  future hang surfaces as a clear RuntimeError within 10s instead of
  blocking the whole trainer.
- Do NOT run at 4 envs without explicit discussion with the remote
  Claude. The deadlock is infrastructure-level, not a code bug the next
  commit will fix; the literature-supported answer on this hardware is
  2 envs.

---

## Copy-paste everything below into PC-side Claude Code as the first message

```text
I'm running UrbanZero (CARLA + PPO autonomous-driving RL).
The remote Claude just pushed 5 commits (tip 2f9fe00) on branch
`claude/setup-av-training-VetPV` that rewrite the reward function, observation
space, policy clamping, hyperparameters, beacon telemetry, and infrastructure
scripts. These changes address specific failure modes identified by adversarial
review of the prior 7M-step run (which produced 0 route completions).

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

Verify the top 8 commits match (order is newest first — includes the
post-audit fix pack):

  79374e5 fix: post-deploy audit — 5 additional bugs the first smoke test would hit
  8d86025 docs: include explicit Windows CarlaUE4.exe launch commands in smoke prompt
  c46d3dc docs: paste-ready PC-side Claude smoke-test prompt for v2 stack
  2f9fe00 infra: watchdog seed-rand, preflight multi-port, BC-phase auto-resume guard
  0612756 beacon: add exploration/KL/termination_reason telemetry
  3e8845c train: ent_coef 0.02→0.01 anneal, RollingBest ckpt, PPO-default hypers
  800fb4d policy: soft upper log_std bound only (std<=1.0), no lower floor
  a0ac534 env: CaRL-minimal reward, 10-dim state, termination_reason telemetry

If `79374e5` (the fix-pack) is NOT the current HEAD, STOP and pull again —
running any earlier tip will crash on ENT_COEF_START within seconds of
launch (that's GitHub issue #2).

Make scripts executable:
  chmod +x scripts/preflight.py scripts/watchdog.sh scripts/start_training.sh

=== STEP 2: verify 2 CARLA instances on Windows ===

This test uses TWO Windows CARLA servers running on ports 2000 and 3000
(the empirically stable configuration from the prior 7M run — commit
747a28a confirmed ~100 FPS aggregate throughput there). Do NOT attempt
4 envs; that's what produced GitHub issue #3. From WSL, verify each:

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
open TWO separate Windows PowerShell windows and run the commands below
— one per window. Adjust the path if your CARLA install is somewhere
other than `C:\\Users\\aadit\\ECE-591\\CARLA_0.9.15`.

If you already have FOUR CARLA windows open from the prior attempt,
close the ones on ports 4000 and 5000 (Task Manager: CarlaUE4.exe,
pick the right PIDs) — running extra CARLA instances during 2-env
training wastes VRAM and risks GPU contention.

After BOTH port-2000 and port-3000 instances have reached the pre-game
menu (you'll see a CARLA logo and 'Press F1 for help'), come back here
and say 'ready'. Leave both windows OPEN for the duration of the smoke
test — closing one kills training on that worker.

Window 1 (port 2000):
  cd C:\\Users\\aadit\\ECE-591\\CARLA_0.9.15\\WindowsNoEditor
  .\\CarlaUE4.exe -carla-rpc-port=2000 -quality-level=Low -windowed -ResX=400 -ResY=300

Window 2 (port 3000):
  cd C:\\Users\\aadit\\ECE-591\\CARLA_0.9.15\\WindowsNoEditor
  .\\CarlaUE4.exe -carla-rpc-port=3000 -quality-level=Low -windowed -ResX=400 -ResY=300

Flags explained:
  -carla-rpc-port=<N>   : which port the CARLA server listens on (each
                          trainer worker connects to one of these)
  -quality-level=Low    : lowest render quality. Training uses semantic
                          seg not RGB, so render quality doesn't affect
                          the agent's observations.
  -windowed             : windowed mode (not fullscreen), lets you see
                          both at once to confirm they're running
  -ResX=400 -ResY=300   : tiny windows. Reduces VRAM and CPU cost of
                          the spectator render.

If either instance fails to launch with 'server already running' or
similar, that port is held by a stale process. Kill it in Task Manager
(look for CarlaUE4.exe) and try again.

Once both are at the pre-game menu, reply 'ready'."

--- end paste ---

After the user replies 'ready', re-run the port check loop. Do NOT proceed
until both ports report UP.

Why 2 envs, not 4: a prior 4-env attempt hung indefinitely after the
first PPO training pass (GitHub issue #3 — multi-CARLA sync deadlock).
Do NOT attempt 4 envs without explicit direction from the remote Claude.
If the user pushes for more envs, STOP and ask — the issue is
infrastructure-level, not something the smoke test can safely explore.

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

Watch specifically for `world.tick() raised ...` messages. The new
env/carla_env.py uses `world.tick(seconds=10.0)` so any CARLA hang
surfaces as a logged RuntimeError within 10s instead of deadlocking
silently (the failure mode from GitHub issue #3).

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
  [P2] beacon "fps" >= 70                 (2-env aggregate target; prior
                                           7M run at 2 envs measured ~100
                                           FPS so 70 is a conservative floor)
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
  [F1] fps < 50 aggregate at 2 envs: unexpected — prior 2-env runs hit
       ~100. Report VRAM, GPU util, and current CARLA window count to
       the remote side.
  [F1a] process hangs / no log output for >60s (same symptom as GitHub
       issue #3, but should no longer happen at 2 envs + 10s world.tick
       timeout). If it does, capture `py-spy dump --pid <trainer-pid>`
       if py-spy is available, else `ps auxf` showing trainer state.
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

- **Why 10 minutes and not 30.** The smoke test isn't trying to learn anything. It's proving the new stack doesn't crash, achieves target FPS on 2 envs, and populates the new telemetry correctly. 10 minutes is enough to see ~40k+ env-steps, ~50 episodes, and several PPO `train()` passes (which fire every `n_steps * n_envs = 512 * 2 = 1024` steps at 2 envs — same rollout size as before).

- **What happens if smoke test FAILS gate F1 (FPS too low).** Drop to 3 envs — i.e. shut down the CARLA instance on port 5000, set `URBANZERO_N_ENVS=3`, and re-run the smoke test. At 3 envs we lose ~20% sample diversity but VRAM pressure drops substantially.

- **What happens if smoke test FAILS gate F2 (NaN/reward-guard fires).** That's a code bug on the remote side. Paste the report back and I will investigate immediately.

- **After PASS.** I build P1 (BC data collection using `BehaviorAgent`) on a new branch, push, and write a similar PC-side prompt to smoke-test the data-collection script. Then P2 (BC training), P3 (value-head bootstrap), P4 (custom PPO with KL-to-BC), and finally the long run.

- **`4d` from Saturday deadline at time of writing (2026-04-22 Wed late afternoon).** P0 smoke test should take ~30 min of your and the PC-side Claude's time. Plan for P1-P4 to take ~4-6 hours of code work on my side + CARLA runtime for data collection and training. Total plan sits inside the 72-hour budget with buffer.
