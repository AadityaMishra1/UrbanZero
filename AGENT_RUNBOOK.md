# Agent Runbook — Hands-Off Training Bring-Up

This file is for you (the human) to copy-paste into a Claude Code session
running on the **training PC** (the one with CARLA + GPU). It walks an
agent through pulling the latest hardened branch, smoke-testing it for
30 min against pre-defined PASS/FAIL criteria, and either kicking off a
long unattended run or stopping for triage.

The runbook is intentionally narrow: explicit success/fail signals, no
judgment calls about hyperparameters or code edits. If you want the
agent to debug a failure, you start a fresh conversation after seeing
its report.

## Why a runbook?

The hardening commit on `claude/setup-av-training-VetPV` removes the
classic "weird behaviors" that forced manual intervention (figure-8
circling, wrong-way spawns, NaN reward, abrupt episode ends), adds a
`~/urbanzero/beacon.json` heartbeat, an upgraded `scripts/watchdog.sh`
that detects hung trainers (not just dead ones), and `scripts/preflight.py`
that refuses to launch the trainer when CARLA / GPU / disk aren't ready.

The dangerous remaining failure modes are runtime ones I can't validate
from a sandbox: a perf regression from per-step world-actor scans, a
CARLA RSS leak over 24-48h, an unanticipated reward pathology from the
new +10 ROUTE_COMPLETE terminal. The smoke test below catches all three
in 30 minutes.

## How to use this file

1. Open Claude Code on the training PC.
2. Copy everything inside the fenced block below into the chat as your
   first message.
3. Don't interject while it works — let it follow the runbook.
4. If the smoke test FAILS, the agent will stop and dump diagnostics.
   Read them, fix the root cause, then start a new session and paste
   the prompt again.

## The prompt — paste this into PC-side Claude Code

```text
I'm running UrbanZero (CARLA + PPO autonomous-driving RL).
A teammate just pushed branch `claude/setup-av-training-VetPV` with
hardening fixes for figure-8 circling, wrong-way spawns, NaN reward
explosions, and adds a hands-off watchdog + beacon. I need you to:
(1) sync that branch, (2) run a 30-min smoke test, (3) decide based
on smoke-test signals whether to launch the 24-48h training run.

The repo lives at the directory containing `agents/train.py`. Find it
with: `find ~ -name carla_env.py -path "*/UrbanZero/*" 2>/dev/null`
and cd to its parent's parent. If you can't find it, ask me where it is.

CARLA is running on Windows on this same PC, reachable from WSL at
172.25.176.1:2000. Verify it's up before doing anything by running:
  timeout 3 bash -c '>/dev/tcp/172.25.176.1/2000' && echo OK || echo CARLA_DOWN
If CARLA_DOWN, stop and tell me to start CarlaUE4.exe on Windows first.

=== STEP 1: sync the branch ===
git fetch origin
git checkout claude/setup-av-training-VetPV
git pull
chmod +x scripts/preflight.py scripts/watchdog.sh scripts/start_training.sh

=== STEP 2: 30-min smoke test ===
Edit nothing. Just run:
  bash scripts/start_training.sh

The script runs preflight first. If preflight fails, READ the failure
message and fix it (most likely CARLA_PYTHONAPI env var or disk space).
Don't bypass preflight unless I say so.

Then it launches in tmux session `urbanzero`. Attach with `tmux a -t urbanzero`.
Detach with Ctrl+b then d. The training will run forever until killed —
for the smoke test we let it run ~30 min then stop it.

=== STEP 3: monitor for 30 min and report ===
Every 5 min, check:
  cat ~/urbanzero/beacon.json | python3 -m json.tool

Then grep the latest log for episode reasons:
  LOG=$(ls -t ~/urbanzero/logs/train_*.log | head -1)
  grep "EPISODE END" "$LOG" | awk '{print $3}' | sort | uniq -c | sort -rn

Tell me the distribution every 10 min.

=== STEP 4: judge after 30 min ===
PASS criteria (all must hold):
  - beacon.json `timesteps` increased to >20000
  - beacon.json `fps` >= 5
  - NO lines containing `[NaN-GUARD]` in the log:
      grep -c "NaN-GUARD" "$LOG"     # must be 0
  - NO lines containing `[reward-guard]` in the log:
      grep -c "reward-guard" "$LOG"  # must be 0
  - Episode-end distribution: CIRCLING + STAGNATION + REALLY_STUCK
    combined < 30% of total episodes
  - At least one episode of any type completed (rolling_ep_count > 0
    in beacon.json)

FAIL signals (any one means stop and investigate, don't launch long run):
  - fps < 3 (almost certainly _is_blocked_by_vehicle perf hit; report and
    we'll cache actor list)
  - any [NaN-GUARD] or [reward-guard] line (real bug worth finding)
  - >50% episodes ending CIRCLING (something still wrong with
    early-episode behavior)
  - prog_clamp_hits in any episode-end line > 5 (projection bug
    the clamp is hiding)
  - timesteps not advancing between two 5-min checks (trainer hung;
    watchdog should restart it but verify)

=== STEP 5: if smoke test PASSES ===
Kill the smoke test:
  tmux kill-session -t urbanzero

Launch the long run with the watchdog in a separate tmux session:
  bash scripts/start_training.sh
  tmux new -d -s wd 'bash scripts/watchdog.sh'

Confirm both sessions are alive:
  tmux ls

Tell me the start timestamp and where the log file is, then exit.
DON'T sit and watch — the whole point is hands-off.

=== STEP 6: if smoke test FAILS ===
Stop everything:
  tmux kill-session -t urbanzero 2>/dev/null
  tmux kill-session -t wd 2>/dev/null
  tmux kill-session -t spectator 2>/dev/null

Paste back the FAIL signal you saw + the last 100 lines of the log:
  tail -100 "$LOG"
And the full beacon.json. Don't try to debug the training code yourself —
report and wait.

Don't take any actions outside this runbook. Don't edit Python files.
Don't change hyperparameters. Don't disable the watchdog or preflight.
If something is unclear, ask me before improvising.
```

## Quick reference for you (the human)

While the long run is going, you can check health without attaching to
tmux:

```bash
# One-shot health glance
cat ~/urbanzero/beacon.json | python3 -m json.tool

# Live watch
watch -n 5 'cat ~/urbanzero/beacon.json | python3 -m json.tool'

# Episode-end reason distribution from current run
LOG=$(ls -t ~/urbanzero/logs/train_*.log | head -1)
grep "EPISODE END" "$LOG" | awk '{print $3}' | sort | uniq -c | sort -rn

# Watchdog activity
tail -f ~/urbanzero/watchdog.log

# Anything bad?
grep -E '\[NaN-GUARD\]|\[reward-guard\]' "$LOG"
```

What to glance at every 4-6 hours during the long run:

- `timesteps` is advancing (i.e. trainer isn't dead and watchdog isn't
  in a restart loop)
- `rolling_collision_rate` is trending **down** over hours, not flat
- `rolling_route_completion` is trending **up** over hours, not flat
- `rolling_ep_len` is trending **up** (longer survival)
- No `[NaN-GUARD]` lines in the log

If all of those hold for the first 12 hours, the run is on track.
