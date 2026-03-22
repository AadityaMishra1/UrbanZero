#!/usr/bin/env bash
# UrbanZero — launch all training infrastructure in tmux panes.
#
# Usage:  ./run.sh          (fresh start)
#         ./run.sh resume   (resume from latest checkpoint)
#
# Prerequisites:
#   - CARLA must already be running on Windows
#   - tmux installed:  sudo apt install tmux

set -euo pipefail

SESSION="urbanzero"
SETUP='source ~/urbanzero_env/bin/activate && export PYTHONPATH=$PYTHONPATH:/mnt/c/Users/aadit/ECE-591/CARLA_0.9.15/WindowsNoEditor/PythonAPI/carla'

# Build training command
if [ "${1:-}" = "resume" ]; then
    TRAIN="$SETUP && cd ~/urbanzero && CKPT=\$(ls ~/urbanzero/checkpoints/shaped/*.zip 2>/dev/null | grep -v vecnormalize | sort | tail -1) && python3 agents/train.py --resume \"\$CKPT\""
else
    TRAIN="$SETUP && cd ~/urbanzero && python3 agents/train.py"
fi

# Kill existing session
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x 200 -y 50

# Pane 0: Training
tmux send-keys -t "$SESSION" "$TRAIN" C-m
tmux rename-window -t "$SESSION" "train"

# Pane 1: TensorBoard
tmux split-window -t "$SESSION" -h
tmux send-keys -t "$SESSION" "source ~/urbanzero_env/bin/activate && tensorboard --logdir ~/urbanzero/logs/shaped --host 0.0.0.0" C-m

# Pane 2: ROS2 node
tmux split-window -t "$SESSION" -v
tmux send-keys -t "$SESSION" "$SETUP && source /opt/ros/humble/setup.bash && python3 ~/urbanzero/ros/urbanzero_node.py" C-m

# Pane 3: Spectator camera
tmux select-pane -t "$SESSION:0.0"
tmux split-window -t "$SESSION" -v
tmux send-keys -t "$SESSION" "$SETUP && python3 ~/urbanzero/scripts/spectator.py" C-m

# Pane 4: Watchdog
tmux select-pane -t "$SESSION:0.1"
tmux split-window -t "$SESSION" -v
tmux send-keys -t "$SESSION" "bash ~/urbanzero/scripts/watchdog.sh" C-m

# Arrange and attach
tmux select-layout -t "$SESSION" tiled
tmux select-pane -t "$SESSION:0.0"
tmux attach -t "$SESSION"
