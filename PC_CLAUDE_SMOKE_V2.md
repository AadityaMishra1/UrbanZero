# PC-Side Claude — v2 Smoke Test Prompt

**Date written:** 2026-04-22
**Branch:** `claude/setup-av-training-VetPV`
**Tip commit:** `2f9fe00` (infra: watchdog seed-rand, preflight multi-port, BC-phase auto-resume guard)
**Purpose:** 10-minute smoke test of a full reward/policy/obs/infra rewrite before committing to a long training run. Goal is to verify the new code boots, gets 4 CARLA envs running, populates the beacon, and does not fire any reward/NaN guards.

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

Verify the top 5 commits match (order is newest first):
  2f9fe00 infra: watchdog seed-rand, preflight multi-port, BC-phase auto-resume guard
  0612756 beacon: add exploration/KL/termination_reason telemetry
  3e8845c train: ent_coef 0.02→0.01 anneal, RollingBest ckpt, PPO-default hypers
  800fb4d policy: soft upper log_std bound only (std<=1.0), no lower floor
  a0ac534 env: CaRL-minimal reward, 10-dim state, termination_reason telemetry

If any SHA differs, STOP and ask the user — the remote Claude may have pushed more.

Make scripts executable:
  chmod +x scripts/preflight.py scripts/watchdog.sh scripts/start_training.sh

=== STEP 2: verify 4 CARLA instances on Windows ===

This test requires FOUR Windows CARLA servers running on ports 2000, 3000,
4000, 5000. From WSL, verify each:

  for p in 2000 3000 4000 5000; do
    if timeout 3 bash -c ">/dev/tcp/172.25.176.1/$p" 2>/dev/null; then
      echo "port $p: UP"
    else
      echo "port $p: DOWN"
    fi
  done

If any port is DOWN, STOP and tell the user:
  "CARLA on port <X> is not reachable. On Windows, launch an additional
   CarlaUE4.exe instance with: CarlaUE4.exe -carla-rpc-port=<X>
   -quality-level=Low. Recommended to launch all four at quality=Low so
   VRAM stays under 16 GB on the 4080 Super."

Do not proceed until all four ports are UP. Do not launch with fewer envs
on your own — the test is specifically of the 4-env configuration.

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

Expected output: all 9+ checks show [ OK ]. There should now be 3 extra
"CARLA port NNNN" checks (one per additional env). If preflight FAILS, do
NOT bypass it — paste the failure lines and STOP.

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

If the process dies in the first 60 seconds, paste the last 50 lines:
  LOG=$(ls -t ~/urbanzero/logs/v2_rl/train_*.log | head -1)
  tail -50 "$LOG"
and STOP.

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
  [P2] beacon "fps" >= 80                 (4-env aggregate target per
                                           remote Claude's hardware math)
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
  [F1] fps < 60 aggregate: 4-env config is too heavy. Report VRAM and
       drop to 3 envs on the remote side.
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
  CARLA ports: 2000/3000/4000/5000 all UP: <yes/no>
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
  P2 >=80 FPS        : PASS / FAIL (<actual>)
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

- **Why 10 minutes and not 30.** The smoke test isn't trying to learn anything. It's proving the new stack doesn't crash, achieves target FPS on 4 envs, and populates the new telemetry correctly. 10 minutes is enough to see ~40k+ env-steps, ~50 episodes, and at least one PPO `train()` pass (which fires at `n_steps * n_envs = 256 * 4 = 1024` steps).

- **What happens if smoke test FAILS gate F1 (FPS too low).** Drop to 3 envs — i.e. shut down the CARLA instance on port 5000, set `URBANZERO_N_ENVS=3`, and re-run the smoke test. At 3 envs we lose ~20% sample diversity but VRAM pressure drops substantially.

- **What happens if smoke test FAILS gate F2 (NaN/reward-guard fires).** That's a code bug on the remote side. Paste the report back and I will investigate immediately.

- **After PASS.** I build P1 (BC data collection using `BehaviorAgent`) on a new branch, push, and write a similar PC-side prompt to smoke-test the data-collection script. Then P2 (BC training), P3 (value-head bootstrap), P4 (custom PPO with KL-to-BC), and finally the long run.

- **`4d` from Saturday deadline at time of writing (2026-04-22 Wed late afternoon).** P0 smoke test should take ~30 min of your and the PC-side Claude's time. Plan for P1-P4 to take ~4-6 hours of code work on my side + CARLA runtime for data collection and training. Total plan sits inside the 72-hour budget with buffer.
