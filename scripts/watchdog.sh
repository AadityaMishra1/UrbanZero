#!/usr/bin/env bash
# Watchdog: restarts training in tmux from latest checkpoint if it crashes.
CARLA_PATH=/mnt/c/Users/aadit/ECE-591/CARLA_0.9.15/WindowsNoEditor/PythonAPI/carla
CKPT_DIR=~/urbanzero/checkpoints/shaped
LOGFILE=~/urbanzero/watchdog.log

echo "=== Watchdog Started $(date) ===" >> "$LOGFILE"

while true; do
    sleep 120

    # Check if training is running in tmux
    if tmux has-session -t urbanzero 2>/dev/null; then
        PANE_PID=$(tmux list-panes -t urbanzero -F '#{pane_pid}' 2>/dev/null)
        if [ -n "$PANE_PID" ] && pgrep -P "$PANE_PID" -f "python" > /dev/null 2>&1; then
            continue  # training is running, all good
        fi
        echo "[$(date)] Python process died in tmux, restarting..." >> "$LOGFILE"
    else
        echo "[$(date)] tmux session gone, restarting..." >> "$LOGFILE"
    fi

    CKPT=$(ls -t "$CKPT_DIR"/autosave_*_steps.zip "$CKPT_DIR"/ppo_urbanzero_*_steps.zip 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "[$(date)] ERROR: no checkpoint found!" >> "$LOGFILE"
        continue
    fi

    echo "[$(date)] Resuming from: $CKPT" >> "$LOGFILE"
    bash ~/urbanzero/scripts/start_training.sh "$CKPT"
    echo "[$(date)] Training + spectator restarted" >> "$LOGFILE"
    sleep 300  # wait 5 min before next check
done
