#!/usr/bin/env bash
# scripts/run_bc_pipeline.sh — End-to-end BC warmstart pipeline for UrbanZero.
#
# Runs all three stages in sequence:
#   Stage 1: collect_bc_data.py   — CARLA rollouts with BehaviorAgent
#   Stage 2: train_bc.py          — Gaussian NLL BC training
#   Stage 3: train.py             — PPO fine-tuning from BC weights
#
# Usage:
#   bash scripts/run_bc_pipeline.sh [--port PORT] [--n_frames N] [--epochs E]
#
# Honors env vars (same convention as start_training.sh):
#   CARLA_PYTHONAPI   path to CARLA PythonAPI/carla dir
#   CARLA_HOST        (default 172.25.176.1)
#   URBANZERO_VENV    path to venv activate script (default ~/urbanzero_env/bin/activate)
#   URBANZERO_REPO    path to repo (default ~/UrbanZero)
#   URBANZERO_HOME    work dir for logs/checkpoints (default ~/urbanzero)
#   URBANZERO_EXP     experiment name (default 'bc_warmstart')
#   URBANZERO_N_ENVS  parallel CARLA envs for PPO phase (default 2)
#   URBANZERO_BC_OUTPUT  override output .npz and .zip paths
#
# Exits nonzero if any stage fails.

set -euo pipefail

# -----------------------------------------------------------------------
# Parse flags
# -----------------------------------------------------------------------
PORT=2000
N_FRAMES=150000
EPOCHS=30

_usage() {
    echo "Usage: $0 [--port PORT] [--n_frames N] [--epochs E]"
    echo "  --port      CARLA RPC port for data collection (default: 2000)"
    echo "  --n_frames  number of expert frames to collect (default: 150000)"
    echo "  --epochs    BC training epochs (default: 30)"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)     PORT="$2";     shift 2 ;;
        --n_frames) N_FRAMES="$2"; shift 2 ;;
        --epochs)   EPOCHS="$2";   shift 2 ;;
        --help|-h)  _usage ;;
        *) echo "Unknown argument: $1"; _usage ;;
    esac
done

# -----------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------
CARLA_PYTHONAPI="${CARLA_PYTHONAPI:-/mnt/c/Users/aadit/ECE-591/CARLA_0.9.15/WindowsNoEditor/PythonAPI/carla}"
VENV="${URBANZERO_VENV:-$HOME/urbanzero_env/bin/activate}"
REPO="${URBANZERO_REPO:-$HOME/UrbanZero}"
HOME_DIR="${URBANZERO_HOME:-$HOME/urbanzero}"
EXPERIMENT="${URBANZERO_EXP:-bc_warmstart}"
N_ENVS="${URBANZERO_N_ENVS:-2}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BC_DATA_DIR="$HOME_DIR/bc_data"
BC_CKPT_DIR="$HOME_DIR/checkpoints"
LOG_DIR="$HOME_DIR/logs"

mkdir -p "$BC_DATA_DIR" "$BC_CKPT_DIR" "$LOG_DIR"

BC_NPZ="${URBANZERO_BC_OUTPUT:-$BC_DATA_DIR/bc_data_${TIMESTAMP}.npz}"
BC_ZIP="${URBANZERO_BC_OUTPUT:-$BC_CKPT_DIR/bc_pretrain_${TIMESTAMP}.zip}"
# If URBANZERO_BC_OUTPUT is set, derive .npz and .zip from it.
if [[ -n "${URBANZERO_BC_OUTPUT:-}" ]]; then
    BC_NPZ="${URBANZERO_BC_OUTPUT%.zip}.npz"
    BC_ZIP="${URBANZERO_BC_OUTPUT%.npz}.zip"
fi

# Activate venv and set PYTHONPATH
if [[ ! -f "$VENV" ]]; then
    echo "ERROR: venv not found at $VENV" >&2
    echo "  Set URBANZERO_VENV to the path of your venv's activate script." >&2
    exit 1
fi

# Source the venv in this shell for all subsequent python3 calls
# shellcheck disable=SC1090
source "$VENV"
export PYTHONPATH="${PYTHONPATH:-}:$CARLA_PYTHONAPI"
cd "$REPO"

echo "========================================================"
echo " UrbanZero BC warmstart pipeline"
echo "  CARLA port   : $PORT"
echo "  n_frames     : $N_FRAMES"
echo "  epochs       : $EPOCHS"
echo "  BC data      : $BC_NPZ"
echo "  BC model     : $BC_ZIP"
echo "  Experiment   : $EXPERIMENT"
echo "  venv         : $VENV"
echo "  repo         : $REPO"
echo "========================================================"
echo ""

# -----------------------------------------------------------------------
# Helper: print elapsed time
# -----------------------------------------------------------------------
_elapsed() {
    local secs=$1
    printf "%dm%02ds" $((secs / 60)) $((secs % 60))
}

# -----------------------------------------------------------------------
# Stage 1: Data collection
# -----------------------------------------------------------------------
echo "[pipeline] === Stage 1/3: Collecting ${N_FRAMES} expert frames ==="
STAGE1_START="$(date +%s)"

python3 -u "$REPO/scripts/collect_bc_data.py" \
    --port "$PORT" \
    --n_frames "$N_FRAMES" \
    --output "$BC_NPZ" \
    2>&1 | tee "$LOG_DIR/bc_collect_${TIMESTAMP}.log"

STAGE1_STATUS="${PIPESTATUS[0]}"
STAGE1_END="$(date +%s)"
STAGE1_ELAPSED=$(( STAGE1_END - STAGE1_START ))

if [[ "$STAGE1_STATUS" -ne 0 ]]; then
    echo "[pipeline] FAILED: Stage 1 exited with status $STAGE1_STATUS" >&2
    exit "$STAGE1_STATUS"
fi
echo "[pipeline] Stage 1 done in $(_elapsed $STAGE1_ELAPSED). Data: $BC_NPZ"
echo ""

# Verify output file exists
if [[ ! -f "$BC_NPZ" ]]; then
    echo "[pipeline] FAILED: $BC_NPZ not found after collection" >&2
    exit 1
fi

# -----------------------------------------------------------------------
# Stage 2: BC training
# -----------------------------------------------------------------------
echo "[pipeline] === Stage 2/3: BC training for ${EPOCHS} epochs ==="
STAGE2_START="$(date +%s)"

python3 -u "$REPO/agents/train_bc.py" \
    --data "$BC_NPZ" \
    --output "$BC_ZIP" \
    --epochs "$EPOCHS" \
    2>&1 | tee "$LOG_DIR/bc_train_${TIMESTAMP}.log"

STAGE2_STATUS="${PIPESTATUS[0]}"
STAGE2_END="$(date +%s)"
STAGE2_ELAPSED=$(( STAGE2_END - STAGE2_START ))

if [[ "$STAGE2_STATUS" -ne 0 ]]; then
    echo "[pipeline] FAILED: Stage 2 exited with status $STAGE2_STATUS" >&2
    exit "$STAGE2_STATUS"
fi
echo "[pipeline] Stage 2 done in $(_elapsed $STAGE2_ELAPSED). Model: $BC_ZIP"
echo ""

if [[ ! -f "$BC_ZIP" ]]; then
    echo "[pipeline] FAILED: $BC_ZIP not found after BC training" >&2
    exit 1
fi

# -----------------------------------------------------------------------
# Stage 3: PPO fine-tuning from BC weights
# -----------------------------------------------------------------------
echo "[pipeline] === Stage 3/3: PPO fine-tuning from BC warmstart ==="
echo "[pipeline] Set URBANZERO_BC_WEIGHTS=$BC_ZIP — train.py will detect and load."
STAGE3_START="$(date +%s)"

export URBANZERO_BC_WEIGHTS="$BC_ZIP"
export URBANZERO_EXP="$EXPERIMENT"
export URBANZERO_N_ENVS="$N_ENVS"

python3 -u "$REPO/agents/train.py" \
    --experiment "$EXPERIMENT" \
    --n-envs "$N_ENVS" \
    --base-port "$PORT" \
    2>&1 | tee "$LOG_DIR/ppo_finetune_${TIMESTAMP}.log"

STAGE3_STATUS="${PIPESTATUS[0]}"
STAGE3_END="$(date +%s)"
STAGE3_ELAPSED=$(( STAGE3_END - STAGE3_START ))

if [[ "$STAGE3_STATUS" -ne 0 ]]; then
    echo "[pipeline] FAILED: Stage 3 exited with status $STAGE3_STATUS" >&2
    exit "$STAGE3_STATUS"
fi

TOTAL_ELAPSED=$(( STAGE3_END - STAGE1_START ))
echo ""
echo "========================================================"
echo " Pipeline complete."
echo "  Stage 1 (collect): $(_elapsed $STAGE1_ELAPSED)"
echo "  Stage 2 (BC train): $(_elapsed $STAGE2_ELAPSED)"
echo "  Stage 3 (PPO):     $(_elapsed $STAGE3_ELAPSED)"
echo "  Total:             $(_elapsed $TOTAL_ELAPSED)"
echo "  BC data    : $BC_NPZ"
echo "  BC model   : $BC_ZIP"
echo "  Checkpoints: $BC_CKPT_DIR/$EXPERIMENT/"
echo "========================================================"
