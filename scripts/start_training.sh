#!/bin/bash
# Start training + spectator in a single tmux session.
# Usage: bash scripts/start_training.sh [checkpoint_path]
CARLA_PATH=/mnt/c/Users/aadit/ECE-591/CARLA_0.9.15/WindowsNoEditor/PythonAPI/carla
CKPT_DIR=~/urbanzero/checkpoints/shaped
ACTIVATE="source ~/urbanzero_env/bin/activate && export PYTHONPATH=\$PYTHONPATH:$CARLA_PATH && cd ~/urbanzero"

# Find checkpoint: use arg, or latest
CKPT="${1:-$(ls -t "$CKPT_DIR"/autosave_*_steps.zip "$CKPT_DIR"/ppo_urbanzero_*_steps.zip 2>/dev/null | head -1)}"
RESUME=""
if [ -n "$CKPT" ] && [ -f "$CKPT" ]; then
    RESUME="--resume $CKPT"
    echo "Resuming from: $CKPT"
fi

# Kill old training session (NOT spectator — that's separate)
tmux kill-session -t urbanzero 2>/dev/null
sleep 1

# Training session
tmux new-session -d -s urbanzero -x 200 -y 50
tmux send-keys -t urbanzero:0.0 "$ACTIVATE && python3 -u agents/train.py $RESUME 2>&1 | tee ~/urbanzero/logs/train_\$(date +%Y%m%d_%H%M%S).log" Enter

# Spectator runs in its own session — start only if not already running
if ! tmux has-session -t spectator 2>/dev/null; then
    sleep 5
    tmux new-session -d -s spectator -x 80 -y 20
    tmux send-keys -t spectator "$ACTIVATE && python3 scripts/spectator.py" Enter
fi

echo "Training in tmux 'urbanzero', spectator in tmux 'spectator'"
