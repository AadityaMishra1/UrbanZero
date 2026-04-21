# PC-Claude Prompt — Next Run (commit c9cd303+)

**Purpose**: after PC-Claude has been iterating on reward terms locally,
this runbook gets the training back on clean footing with the
corrected reward (commit `c9cd303`) and instructs the local agent not
to iterate further until we have fresh data.

## How to use

Copy everything inside the fenced block below and paste it as your
first message to PC-Claude. It's designed to be self-contained — the
local agent shouldn't need any additional context.

---

```text
Pull the latest from claude/setup-av-training-VetPV (tip commit c9cd303). Then
do a clean restart of training — don't resume from any existing checkpoint.
Three reward-shape changes since the checkpoint was saved invalidate its
VecNormalize stats.

Exact steps:

1. Sync:
   cd ~/UrbanZero
   git fetch origin
   git pull origin claude/setup-av-training-VetPV

2. Kill anything currently running:
   tmux kill-session -t urbanzero 2>/dev/null
   tmux kill-session -t wd 2>/dev/null
   tmux kill-session -t spectator 2>/dev/null
   pkill -9 -f agents/train.py 2>/dev/null

3. Move old checkpoints aside so start_training.sh can't auto-resume into
   stale VecNormalize stats:
   mv ~/urbanzero/checkpoints/shaped ~/urbanzero/checkpoints/shaped_$(date +%Y%m%d_%H%M%S)_OLD

4. Launch fresh:
   bash scripts/start_training.sh
   tmux new -d -s wd 'bash scripts/watchdog.sh'

5. Verify both sessions are alive:
   tmux ls

6. Let it run 30 minutes untouched. Do NOT add any new reward terms or
   termination rules during this window. Do NOT kill and restart. The
   previous commits piled on penalties reactively whenever the agent
   collapsed — this created the red-light-avoidance pathology the user
   is seeing. We want to see whether the current reward structure
   (commit c9cd303) actually works without bandages.

7. After 30 minutes, report back these three things only:

   LOG=$(ls -t ~/urbanzero/logs/train_*.log | head -1)

   # a. episode-end reason distribution
   grep "EPISODE END" "$LOG" | grep -oP 'reason=\S+' | sort | uniq -c | sort -rn

   # b. beacon snapshot
   cat ~/urbanzero/beacon.json | python3 -m json.tool

   # c. any safety-net fires
   grep -cE '\[NaN-GUARD\]|\[reward-guard\]' "$LOG"

   Report the numbers. Don't analyze, don't propose changes.

RULES while waiting for that report:
- Do NOT edit env/carla_env.py, agents/train.py, or any reward/termination logic
- Do NOT add new CLI args or env vars
- Do NOT kill training to iterate on reward weights
- If training crashes or the watchdog restarts it, just note the event
  in your report — don't code around it

If the user asks you to change the reward, refuse and tell them to talk
to the other Claude first. Too many reward tweaks from different heads
have made this hard to reason about.

Expected healthy distribution after 30 min:
- COLLISION dominant (30-60%)
- OFF_ROUTE second (20-40%)
- MAX_STEPS / STAGNATION / REALLY_STUCK combined under ~25%
- At least one ROUTE_COMPLETE after ~100k steps is great but not required
- No NaN-GUARD or reward-guard lines at all

If STAGNATION + REALLY_STUCK dominate (>50%), something is still wrong with
the reward and we need to look at telemetry before changing anything.
```

---

## Notes for you (the human)

- The "do not iterate on reward" rule is the important one. Left alone,
  PC-Claude will keep adding penalty terms reactively every time it
  sees the agent collapse. That's how the red-light-avoidance
  pathology crept in across commits `c770d1d`, `ac48a98`, `8791388`,
  `4fdd384`.
- If PC-Claude reports a distribution where STAGNATION + REALLY_STUCK
  dominate (>50% combined), that's actual diagnostic data worth acting
  on — send that back here and we'll debug before any more changes
  ship.
- If the distribution looks healthy, just let it cook. Budget 24-48h
  of wall-clock and expect 30-50% route completion by the end, per
  the README's own progression chart for single-env training.
