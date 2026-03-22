#!/usr/bin/env bash
# Watchdog: restarts training from latest checkpoint if it crashes.
source ~/urbanzero_env/bin/activate
export PYTHONPATH=$PYTHONPATH:/mnt/c/Users/aadit/ECE-591/CARLA_0.9.15/WindowsNoEditor/PythonAPI/carla
cd ~/urbanzero

while true; do
    if ! pgrep -f "agents/train.py" > /dev/null; then
        echo "$(date): Training not running, restarting..." | tee -a ~/urbanzero/watchdog.log
        CKPT=$(ls ~/urbanzero/checkpoints/shaped/*.zip 2>/dev/null | grep -v vecnormalize | sort | tail -1)
        if [ -n "$CKPT" ]; then
            echo "Resuming from $CKPT" | tee -a ~/urbanzero/watchdog.log
            python3 agents/train.py --resume "$CKPT" >> ~/urbanzero/watchdog.log 2>&1 &
        else
            echo "Fresh start" | tee -a ~/urbanzero/watchdog.log
            python3 agents/train.py >> ~/urbanzero/watchdog.log 2>&1 &
        fi
    fi
    sleep 60
done
