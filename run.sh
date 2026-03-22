#!/usr/bin/env bash
# UrbanZero — launch all training infrastructure in tmux panes.
#
# Usage:  ./run.sh          (fresh start)
#         ./run.sh resume   (resume from latest checkpoint)
#
# Prerequisites:
#   - CARLA must already be running on Windows
#     (CarlaUE4.exe from C:\Users\aadit\ECE-591\CARLA_0.9.15\WindowsNoEditor\)
#   - tmux must be installed:  sudo apt install tmux

set -euo pipefail

SESSION="urbanzero"
VENV="source ~/urbanzero_env/bin/activate"
CARLA_PY="export PYTHONPATH=\$PYTHONPATH:/mnt/c/Users/aadit/ECE-591/CARLA_0.9.15/WindowsNoEditor/PythonAPI/carla"

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -x 200 -y 50

# ── Pane 0: Training ────────────────────────────────────────────────
TRAIN_CMD="$VENV && $CARLA_PY && cd ~/urbanzero"
if [ "${1:-}" = "resume" ]; then
    CKPT=\$(ls ~/urbanzero/checkpoints/shaped/*.zip 2>/dev/null | grep -v vecnormalize | sort | tail -1)
    TRAIN_CMD="$TRAIN_CMD && CKPT=\$(ls ~/urbanzero/checkpoints/shaped/*.zip 2>/dev/null | grep -v vecnormalize | sort | tail -1) && echo \"Resuming from \$CKPT\" && python3 agents/train.py --resume \"\$CKPT\""
else
    TRAIN_CMD="$TRAIN_CMD && python3 agents/train.py"
fi
tmux send-keys -t "$SESSION" "$TRAIN_CMD" C-m
tmux rename-window -t "$SESSION" "train"

# ── Pane 1: TensorBoard ─────────────────────────────────────────────
tmux split-window -t "$SESSION" -h
tmux send-keys -t "$SESSION" "$VENV && tensorboard --logdir ~/urbanzero/logs/shaped --host 0.0.0.0" C-m

# ── Pane 2: ROS2 node ───────────────────────────────────────────────
tmux split-window -t "$SESSION" -v
tmux send-keys -t "$SESSION" "$VENV && $CARLA_PY && source /opt/ros/humble/setup.bash && python3 ~/urbanzero/ros/urbanzero_node.py" C-m

# ── Pane 3: Spectator camera ────────────────────────────────────────
tmux select-pane -t "$SESSION:0.0"
tmux split-window -t "$SESSION" -v
tmux send-keys -t "$SESSION" "$VENV && $CARLA_PY && python3 -c \"
import carla, os, time
client = carla.Client(os.environ.get('CARLA_HOST', '172.25.176.1'), 2000)
client.set_timeout(10.0)
world = client.get_world()
spectator = world.get_spectator()
print('Spectator following ego vehicle...')
while True:
    vehicles = world.get_actors().filter('vehicle.tesla.model3')
    if vehicles:
        v = list(vehicles)[0]
        loc = v.get_location()
        spectator.set_transform(carla.Transform(
            carla.Location(x=loc.x, y=loc.y, z=loc.z+40),
            carla.Rotation(pitch=-90)
        ))
    time.sleep(0.05)
\"" C-m

# ── Pane 4: Watchdog ────────────────────────────────────────────────
tmux select-pane -t "$SESSION:0.1"
tmux split-window -t "$SESSION" -v
tmux send-keys -t "$SESSION" "while true; do
    if ! pgrep -f 'agents/train.py' > /dev/null; then
        echo \"\$(date): Training not running, restarting...\" | tee -a ~/urbanzero/watchdog.log
        cd ~/urbanzero
        $VENV
        $CARLA_PY
        CKPT=\$(ls ~/urbanzero/checkpoints/shaped/*.zip 2>/dev/null | grep -v vecnormalize | sort | tail -1)
        if [ -n \"\$CKPT\" ]; then
            echo \"Resuming from \$CKPT\" | tee -a ~/urbanzero/watchdog.log
            python3 agents/train.py --resume \"\$CKPT\" >> ~/urbanzero/watchdog.log 2>&1 &
        else
            echo \"Fresh start\" | tee -a ~/urbanzero/watchdog.log
            python3 agents/train.py >> ~/urbanzero/watchdog.log 2>&1 &
        fi
    fi
    sleep 60
done" C-m

# Arrange panes and attach
tmux select-layout -t "$SESSION" tiled
tmux select-pane -t "$SESSION:0.0"
tmux attach -t "$SESSION"
