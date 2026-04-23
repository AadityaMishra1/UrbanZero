# PC-Side Claude — Long Training Run Prompt

**Context:** The 10-min smoke test at tip `5814854` PASSED all 8 gates
(8/8 PASS, 122 FPS aggregate, no NaN/guard fires, 53 episodes, healthy
PPO stats). The DummyVecEnv fix eliminated the issues #3/#4/#6
deadlocks. Time to start the real training run.

**Tip to use:** `5814854` (or later if remote Claude has pushed more).
**Expected duration:** ~23 hours wall-clock for 10M env-steps at 122 FPS.
**Deadline:** Saturday 2026-04-25, end of day.

---

## Copy-paste everything below into PC-side Claude Code as your next message

```text
Smoke test passed. Time to launch the long training run.

Your job:
  (1) verify tip and environment state are the same as the smoke test
  (2) launch the long run (10M env-steps, expected ~23h)
  (3) launch the watchdog in a separate tmux session
  (4) monitor the beacon every 1-2 hours
  (5) report back when specific signals fire (green or red)

=== HARD RULES — same as the smoke test ===

1. DO NOT edit any Python file. Do not edit any shell script. All code
   authority is the remote Claude. Report issues, don't fix them.
2. DO NOT commit. DO NOT push.
3. DO NOT kill the long run unless a red-flag signal fires or the user
   tells you to.
4. DO NOT hand-tune any hyperparameter. If a metric looks off, report
   it with the current beacon snapshot.

=== STEP 1: verify state is identical to the passing smoke test ===

cd to the repo root (where env/carla_env.py lives).
  git fetch origin
  git checkout claude/setup-av-training-VetPV
  git pull
  git log --oneline -3

Expected top commit: 5814854 (or later "docs:" commit from remote Claude).
If the top commit is older than 5814854, STOP and re-pull.

Verify CARLA is still running on both ports from the smoke test. If
the user killed them, paste this block to them:

--- paste to user ---
"Please verify (or relaunch) both CARLA instances on ports 2000 and
3000. Same commands as before:

Window 1 (port 2000):
  cd C:\\Users\\aadit\\ECE-591\\CARLA_0.9.15\\WindowsNoEditor
  .\\CarlaUE4.exe -carla-rpc-port=2000

Window 2 (port 3000):
  cd C:\\Users\\aadit\\ECE-591\\CARLA_0.9.15\\WindowsNoEditor
  .\\CarlaUE4.exe -carla-rpc-port=3000

Leave both windows open for the full training run (~23 hours). Closing
one kills that worker's training. Reply 'ready' once both are at the
pre-game menu."
--- end paste ---

After 'ready', verify:
  for p in 2000 3000; do
    if timeout 3 bash -c ">/dev/tcp/172.25.176.1/$p" 2>/dev/null; then
      echo "port $p: UP"
    else
      echo "port $p: DOWN"
    fi
  done

=== STEP 2: clean up any leftover state from the smoke test ===

  tmux kill-session -t urbanzero 2>/dev/null
  tmux kill-session -t wd 2>/dev/null
  tmux kill-session -t spectator 2>/dev/null
  pkill -f "agents/train.py" 2>/dev/null
  pkill -f "scripts/spectator.py" 2>/dev/null
  pkill -f "scripts/watchdog.sh" 2>/dev/null
  sleep 2

Archive the smoke-test logs so the long run starts clean:
  if [ -d ~/urbanzero/logs/v2_rl ]; then
    mv ~/urbanzero/logs/v2_rl ~/urbanzero/logs/v2_rl.smoke-$(date +%s)
    mkdir -p ~/urbanzero/logs/v2_rl
  fi
  if [ -d ~/urbanzero/checkpoints/v2_rl ]; then
    mv ~/urbanzero/checkpoints/v2_rl ~/urbanzero/checkpoints/v2_rl.smoke-$(date +%s)
    mkdir -p ~/urbanzero/checkpoints/v2_rl
  fi

=== STEP 3: launch the long run ===

Same env vars as the smoke test — only difference is you let it run:

  URBANZERO_EXP=v2_rl \
  URBANZERO_N_ENVS=2 \
  URBANZERO_BASE_PORT=2000 \
  URBANZERO_TIMESTEPS=10000000 \
  URBANZERO_AUTO_RESUME=0 \
  URBANZERO_SEED=42 \
    bash scripts/start_training.sh

Attach briefly to confirm it started:
  tmux a -t urbanzero
Then detach: Ctrl-b d.

=== STEP 4: launch the watchdog ===

The watchdog restarts training if the beacon goes stale (>3 min with
no updates) or if the trainer process dies. Launch it in a separate
tmux session so it stays alive after you close your terminal:

  tmux new -d -s wd 'bash scripts/watchdog.sh'
  tmux ls

You should see three sessions: `urbanzero` (training), `wd` (watchdog).
(No `spectator` session — inline/manual only.)

=== STEP 5: monitoring cadence ===

Sample the beacon every 1-2 hours for the first 6-8 hours, then every
4-6 hours overnight. Report metrics back to the user against this
expected trajectory:

  Hour  |  avg_speed  | collision_rate | REALLY_STUCK % | RC   | policy_std
  ------|-------------|----------------|----------------|------|------------
  ~1    |  2-4 m/s    |  ~0.4          |  ~30%          |  ~3% | 0.6-0.7
  ~3    |  4-6 m/s    |  ~0.5          |  ~15%          |  5-10%| 0.5-0.7
  ~6    |  5-7 m/s    |  ~0.6          |  ~8%           |  10-20%| 0.4-0.7
  ~12   |  6-8 m/s    |  ~0.4          |  ~5%           |  20-40%| 0.35-0.6
  ~22   |  6-8 m/s    |  ~0.25         |  ~3%           |  35-55%| 0.3-0.5

At each check, produce a one-line report like:

  [T+3h] ts=<n> fps=<n> speed=<n> coll=<n> stuck%=<n> RC=<n> std=<n>
         approx_kl=<n> ent_coef=<n> termination_reasons={<dict>}

Command to produce that:
  cat ~/urbanzero/beacon.json | python3 -c "
  import json,sys
  b=json.load(sys.stdin)
  tr=b.get('termination_reasons',{})
  tot=sum(tr.values()) or 1
  stuck_pct=100*tr.get('REALLY_STUCK',0)/tot
  print(f'ts={b[\"timesteps\"]} fps={b[\"fps\"]} '
        f'speed={b[\"rolling_avg_speed_ms\"]} '
        f'coll={b[\"rolling_collision_rate\"]} '
        f'stuck%={stuck_pct:.1f} '
        f'RC={b[\"rolling_route_completion\"]} '
        f'std={b[\"policy_std\"]} approx_kl={b.get(\"approx_kl\")} '
        f'ent_coef={b.get(\"ent_coef\")} '
        f'reasons={tr}')
  "

=== STEP 6: red flags — STOP and report immediately ===

Any of these means stop sampling on the hourly schedule and tell the
user right away:

  [RED-1] policy_std < 0.3 before timesteps=3_000_000
          (entropy collapsing; ent_coef schedule not holding)

  [RED-2] explained_variance > 0.99 sustained for >= 3 consecutive
          checks (critic fit a deterministic bad policy)

  [RED-3] REALLY_STUCK percentage stays > 30% past timesteps=1_000_000
          (reward signal not breaking through; agent not learning
           forward motion)

  [RED-4] rolling_route_completion flat at 0% at timesteps=3_000_000
          (trigger for the BC-warmstart fallback per PROJECT_NOTES.md §6.2)

  [RED-5] NaN-GUARD or reward-guard fires appear in the log at any time
            grep -cE '\[NaN-GUARD\]|\[reward-guard\]' \
              $(ls -t ~/urbanzero/logs/v2_rl/train_*.log | head -1)
          Non-zero is a problem. Paste the surrounding log lines.

  [RED-6] watchdog restarts more than 3 times in 1 hour
            tail -50 ~/urbanzero/watchdog.log
          (training is unstable; something is wrong)

  [RED-7] rolling_ep_return is a persistent NaN or ±inf in the beacon

=== STEP 7: green flags — report these as milestones ===

  [GREEN-1] First `ROUTE_COMPLETE` episode in the log (grep for
            'reason=ROUTE_COMPLETE'). This is the gate that proves
            the agent has learned the full task. Expected at
            timesteps = 1-3M. Paste the full EPISODE END line with the
            step count so the user can see when it happened.

  [GREEN-2] rolling_route_completion crosses 10%. Report the timestep.

  [GREEN-3] rolling_route_completion crosses 25%. Report the timestep.

  [GREEN-4] RollingBestCallback saves a new best_by_rc.zip
            (look for `[rolling-best] new best rolling RC` in the log).
            Report each time it fires.

=== STEP 8: when training finishes ===

At timesteps >= 10_000_000, the training loop exits cleanly and writes
`~/urbanzero/checkpoints/v2_rl/final_model.zip`. The rolling-best
checkpoint is at `~/urbanzero/checkpoints/v2_rl/best_by_rc.zip`.

Report the final state:
  - total wall-clock hours
  - final rolling_route_completion
  - peak rolling_route_completion and timestep it was reached at
  - final termination_reasons distribution
  - which checkpoint the user should use for eval/demo (best_by_rc.zip,
    not final_model.zip — they're likely different, and best_by_rc is
    the one to ship)

Kill the watchdog so it doesn't try to restart a finished trainer:
  tmux kill-session -t wd

Do NOT kick off an eval or a demo-video run yourself — the remote Claude
will write an eval prompt once the training is confirmed done.

=== SPECTATOR (optional) ===

You noticed in the smoke test that DummyVecEnv steps envs serially, so
each CARLA window only ticks every other step. That's cosmetic and the
user seems OK with it. If they want smoother per-port spectator views,
they can manually launch:
  tmux new-session -d -s spec0 \
    "source ~/urbanzero_env/bin/activate && \
     cd ~/UrbanZero && URBANZERO_SPECTATOR_PORT=2000 python3 scripts/spectator.py"
  tmux new-session -d -s spec1 \
    "source ~/urbanzero_env/bin/activate && \
     cd ~/UrbanZero && URBANZERO_SPECTATOR_PORT=3000 python3 scripts/spectator.py"

But understand: these are secondary clients on sync-mode servers, the
pattern that was suspected in issue #4's chain. The smoke test ran clean
WITHOUT them. Don't start them unless the user explicitly asks — prefer
the user uses remote desktop to watch the CARLA windows directly.

=== SUMMARY OF THE COMMITTED CONFIG ===

- n_envs = 2, DummyVecEnv (not SubprocVecEnv)
- tip commit 5814854 or later
- URBANZERO_TIMESTEPS=10_000_000
- URBANZERO_AUTO_RESUME=0 (fresh start, no resume from smoke test)
- ent_coef schedule 0.02 -> 0.01 over 10M steps
- log_std_init=-0.5, upper-only clamp std<=1.0
- CaRL-minimal reward (progress + annealed carrot + ±50 terminals)
- REALLY_STUCK at 1500 steps (DO NOT try to change this; 38% rate during
  smoke was diagnostic of early random-policy perpendicular behavior,
  NOT a problem; expected to drop to ~5% by hour 10)

DO NOT deviate from this config. If a red-flag fires, report to user
with full context — the remote Claude decides the intervention.
```

---

## Notes for the user (not to paste — just context)

- **Expected wall-clock**: ~23h for 10M steps at 122 FPS. Starting at ~21:37 Thursday puts finish around 20:30 Friday. Leaves all day Saturday for eval, demo video, report writing.

- **If the 3M-step gate fails** (rolling_route_completion flat at 0%), the `PROJECT_NOTES.md` §6.2 BC-warmstart pivot is the pre-agreed fallback. That adds ~6-8 hours for data collection + BC training + PPO fine-tune. Timeline: stop pure RL at 3M steps ≈ 7 hours in ≈ Friday 4 AM, pivot finishes ≈ Friday afternoon, leaves Saturday for eval/demo.

- **If things go well** (rolling RC climbs smoothly, first ROUTE_COMPLETE by 1-2M steps), the final state should be somewhere in the 35-55% rolling RC range. That's a legit scientific result: the 7M run got 0%, this run on the same hardware with the new reward gets tens-of-percent.

- **Monitoring from your laptop**: the beacon.json can be read via `ssh <PC> cat ~/urbanzero/beacon.json | jq .` if SSH is set up; otherwise just ask the PC-side Claude for a status report.

- **When the run ends, ship `best_by_rc.zip`, not `final_model.zip`.** The 7M run's failure was partly that there was no rolling-best and the regressed final model got shipped. `best_by_rc.zip` is what the eval/demo should use.
