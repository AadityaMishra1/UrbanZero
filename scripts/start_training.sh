#!/bin/bash
# Start training + spectator in a single tmux session.
# Usage:
#   bash scripts/start_training.sh [checkpoint_path] [--no-traffic] [--no-weather]
# First positional arg (if it's a .zip path) becomes --resume. Subsequent args
# are forwarded verbatim to agents/train.py.
#
# Honors env vars (set these instead of editing this file):
#   CARLA_PYTHONAPI  path to CARLA's PythonAPI/carla dir
#   CARLA_HOST       (default 172.25.176.1) — Windows-WSL bridge IP
#   CARLA_PORT       (default 2000)
#   URBANZERO_EXP    experiment name (default 'shaped')
#   URBANZERO_VENV   path to venv activate script (default ~/urbanzero_env/bin/activate)
#   URBANZERO_REPO   path to UrbanZero repo (default ~/UrbanZero)
#   URBANZERO_HOME   work dir for logs/checkpoints (default ~/urbanzero)
#   URBANZERO_N_ENVS parallel CARLA envs (default 1). Each env uses
#                    base_port + i*1000, so launch CARLA on 2000, 3000, ...
#                    n=2 is recommended for RTX 4080 Super (safe VRAM
#                    budget, ~1.8x throughput). n>=3 risks OOM.
#   URBANZERO_BASE_PORT  base CARLA port (default 2000)
#   URBANZERO_TIMESTEPS  total training steps (default 10_000_000)
#   URBANZERO_SEED   RNG seed (default 42)
#   URBANZERO_EXTRA_ARGS  free-form args passed through to train.py (e.g.
#                         "--no-traffic --no-weather")
#   URBANZERO_SKIP_PREFLIGHT  if set, skip preflight (use only when debugging)

set -uo pipefail

CARLA_PYTHONAPI="${CARLA_PYTHONAPI:-/mnt/c/Users/aadit/ECE-591/CARLA_0.9.15/WindowsNoEditor/PythonAPI/carla}"
EXPERIMENT="${URBANZERO_EXP:-shaped}"
VENV="${URBANZERO_VENV:-$HOME/urbanzero_env/bin/activate}"
REPO="${URBANZERO_REPO:-$HOME/UrbanZero}"
HOME_DIR="${URBANZERO_HOME:-$HOME/urbanzero}"
N_ENVS="${URBANZERO_N_ENVS:-1}"
BASE_PORT="${URBANZERO_BASE_PORT:-2000}"
TIMESTEPS="${URBANZERO_TIMESTEPS:-10000000}"
SEED="${URBANZERO_SEED:-42}"
CKPT_DIR="$HOME_DIR/checkpoints/$EXPERIMENT"
LOG_DIR="$HOME_DIR/logs"

mkdir -p "$CKPT_DIR" "$LOG_DIR"

ACTIVATE="source '$VENV' && export PYTHONPATH=\$PYTHONPATH:'$CARLA_PYTHONAPI' && export URBANZERO_SEED='$SEED' && cd '$REPO'"

# Parse args: first positional arg is an optional checkpoint path. Anything
# else is passed verbatim to train.py (e.g. --no-traffic, --no-weather).
CKPT=""
EXTRA_ARGS="${URBANZERO_EXTRA_ARGS:-}"
if [ "$#" -ge 1 ]; then
    if [ -f "$1" ] && [[ "$1" == *.zip ]]; then
        CKPT="$1"; shift
    fi
    # Remaining args (if any) append to EXTRA_ARGS.
    if [ "$#" -gt 0 ]; then
        EXTRA_ARGS="$EXTRA_ARGS $*"
    fi
fi
# Fall back to latest checkpoint in this experiment's dir if none specified.
if [ -z "$CKPT" ]; then
    CKPT="$(ls -t "$CKPT_DIR"/autosave_*_steps.zip "$CKPT_DIR"/ppo_urbanzero_*_steps.zip "$CKPT_DIR"/emergency_*_steps.zip 2>/dev/null | head -1)"
fi
RESUME=""
if [ -n "$CKPT" ] && [ -f "$CKPT" ]; then
    RESUME="--resume '$CKPT'"
    echo "Resuming from: $CKPT"
fi
if [ -n "$EXTRA_ARGS" ]; then
    echo "Extra train.py args: $EXTRA_ARGS"
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

# Pre-flight the additional CARLA ports if multi-env.
if [ "$N_ENVS" -gt 1 ]; then
    echo "=== Multi-env (n=$N_ENVS): verifying CARLA on ports $BASE_PORT..."
    for i in $(seq 0 $((N_ENVS - 1))); do
        port=$((BASE_PORT + i * 1000))
        if ! timeout 3 bash -c ">/dev/tcp/${CARLA_HOST:-172.25.176.1}/${port}" 2>/dev/null; then
            echo "ERROR: CARLA not reachable on port $port. Launch additional CARLA server(s):" >&2
            echo "  On Windows: start separate CarlaUE4.exe instances with -carla-rpc-port=$port" >&2
            exit 1
        fi
        echo "  CARLA on port $port reachable"
    done
fi

# Training session.
TS="$(date +%Y%m%d_%H%M%S)"
tmux new-session -d -s urbanzero -x 200 -y 50
tmux send-keys -t urbanzero:0.0 \
  "$ACTIVATE && python3 -u agents/train.py --experiment '$EXPERIMENT' --n-envs '$N_ENVS' --base-port '$BASE_PORT' --timesteps '$TIMESTEPS' $RESUME $EXTRA_ARGS 2>&1 | tee '$LOG_DIR/train_${TS}.log'" Enter

# Spectator runs in its own session — start only if not already running.
if ! tmux has-session -t spectator 2>/dev/null; then
    sleep 5
    tmux new-session -d -s spectator -x 80 -y 20
    tmux send-keys -t spectator "$ACTIVATE && python3 scripts/spectator.py" Enter
fi

echo "Training in tmux 'urbanzero', spectator in tmux 'spectator'"
echo "Log: $LOG_DIR/train_${TS}.log"
echo "Beacon: $HOME_DIR/beacon.json   (watch with: watch -n 5 'cat $HOME_DIR/beacon.json')"
