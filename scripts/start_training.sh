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

# Kill old session
tmux kill-session -t urbanzero 2>/dev/null
sleep 1

# Create session with training in left pane
tmux new-session -d -s urbanzero -x 200 -y 50
tmux send-keys -t urbanzero:0.0 "$ACTIVATE && python3 -u agents/train.py $RESUME 2>&1 | tee ~/urbanzero/logs/train_\$(date +%Y%m%d_%H%M%S).log" Enter

# Wait for training to connect to CARLA before starting spectator
sleep 5

# Spectator in right pane
tmux split-window -t urbanzero:0.0 -h
tmux send-keys -t urbanzero:0.1 "$ACTIVATE && python3 scripts/spectator.py" Enter

echo "Training + spectator running in tmux session 'urbanzero'"
echo "  Left pane:  training"
echo "  Right pane: spectator"
