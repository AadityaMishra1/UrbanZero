#!/bin/bash
# Start training + spectator in a single tmux session.
# Usage: bash scripts/start_training.sh [checkpoint_path]
#
# Honors env vars (set these instead of editing this file):
#   CARLA_PYTHONAPI  path to CARLA's PythonAPI/carla dir
#   CARLA_HOST       (default 172.25.176.1) — Windows-WSL bridge IP
#   CARLA_PORT       (default 2000)
#   URBANZERO_EXP    experiment name (default 'shaped')
#   URBANZERO_VENV   path to venv activate script (default ~/urbanzero_env/bin/activate)
#   URBANZERO_REPO   path to UrbanZero repo (default ~/UrbanZero)
#   URBANZERO_HOME   work dir for logs/checkpoints (default ~/urbanzero)
#   URBANZERO_SKIP_PREFLIGHT  if set, skip preflight (use only when debugging)

set -uo pipefail

CARLA_PYTHONAPI="${CARLA_PYTHONAPI:-/mnt/c/Users/aadit/ECE-591/CARLA_0.9.15/WindowsNoEditor/PythonAPI/carla}"
EXPERIMENT="${URBANZERO_EXP:-shaped}"
VENV="${URBANZERO_VENV:-$HOME/urbanzero_env/bin/activate}"
REPO="${URBANZERO_REPO:-$HOME/UrbanZero}"
HOME_DIR="${URBANZERO_HOME:-$HOME/urbanzero}"
CKPT_DIR="$HOME_DIR/checkpoints/$EXPERIMENT"
LOG_DIR="$HOME_DIR/logs"

mkdir -p "$CKPT_DIR" "$LOG_DIR"

ACTIVATE="source '$VENV' && export PYTHONPATH=\$PYTHONPATH:'$CARLA_PYTHONAPI' && cd '$REPO'"

# Find checkpoint: use arg, or latest autosave/ppo/emergency.
CKPT="${1:-$(ls -t "$CKPT_DIR"/autosave_*_steps.zip "$CKPT_DIR"/ppo_urbanzero_*_steps.zip "$CKPT_DIR"/emergency_*_steps.zip 2>/dev/null | head -1)}"
RESUME=""
if [ -n "$CKPT" ] && [ -f "$CKPT" ]; then
    RESUME="--resume '$CKPT'"
    echo "Resuming from: $CKPT"
fi

# Pre-flight: refuse to launch if CARLA/disk/GPU/deps aren't ready.
# Saves the user from staring at a tmux pane that died 3 seconds in.
if [ -z "${URBANZERO_SKIP_PREFLIGHT:-}" ]; then
    echo "=== Running pre-flight ==="
    if ! bash -c "$ACTIVATE && URBANZERO_EXP='$EXPERIMENT' python3 '$REPO/scripts/preflight.py'"; then
        echo "Pre-flight FAILED. Set URBANZERO_SKIP_PREFLIGHT=1 to bypass." >&2
        exit 1
    fi
    echo
fi

# Kill old training session (NOT spectator — that's separate).
tmux kill-session -t urbanzero 2>/dev/null
sleep 1

# Training session.
TS="$(date +%Y%m%d_%H%M%S)"
tmux new-session -d -s urbanzero -x 200 -y 50
tmux send-keys -t urbanzero:0.0 \
  "$ACTIVATE && python3 -u agents/train.py --experiment '$EXPERIMENT' $RESUME 2>&1 | tee '$LOG_DIR/train_${TS}.log'" Enter

# Spectator runs in its own session — start only if not already running.
if ! tmux has-session -t spectator 2>/dev/null; then
    sleep 5
    tmux new-session -d -s spectator -x 80 -y 20
    tmux send-keys -t spectator "$ACTIVATE && python3 scripts/spectator.py" Enter
fi

echo "Training in tmux 'urbanzero', spectator in tmux 'spectator'"
echo "Log: $LOG_DIR/train_${TS}.log"
echo "Beacon: $HOME_DIR/beacon.json   (watch with: watch -n 5 'cat $HOME_DIR/beacon.json')"
