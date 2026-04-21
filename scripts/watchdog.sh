#!/usr/bin/env bash
# UrbanZero watchdog v2 — checks 3 health signals every 60s:
#   1. CARLA RPC port reachable                  -> if not, log + wait (we
#      don't auto-restart CARLA because it lives on the Windows host).
#   2. Trainer process alive in tmux             -> if not, restart from
#      latest checkpoint via start_training.sh.
#   3. Beacon JSON mtime <  STALE_SEC            -> if stale, the trainer
#      is hung even though its process is alive. Kill it, then restart.
#
# After any restart, sleep WARMUP_SEC before resuming normal checks (gives
# VecNormalize, world.tick, first reset, etc. time to come up).

set -uo pipefail

CARLA_HOST="${CARLA_HOST:-172.25.176.1}"
CARLA_PORT="${CARLA_PORT:-2000}"
EXPERIMENT="${URBANZERO_EXP:-shaped}"
CKPT_DIR="$HOME/urbanzero/checkpoints/$EXPERIMENT"
BEACON="$HOME/urbanzero/beacon.json"
LOGFILE="$HOME/urbanzero/watchdog.log"
START_SCRIPT="$HOME/UrbanZero/scripts/start_training.sh"

CHECK_INTERVAL=60       # base poll interval (sec)
STALE_SEC=180           # beacon older than 3 min => hung
WARMUP_SEC=600          # wait 10 min after restart before re-checking
CARLA_BACKOFF_SEC=120   # wait this long when CARLA is down

mkdir -p "$(dirname "$LOGFILE")"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOGFILE"; }
log "=== Watchdog v2 started (CARLA $CARLA_HOST:$CARLA_PORT, exp=$EXPERIMENT) ==="

restart_trainer() {
    log "Restarting trainer..."
    # Kill trainer cleanly (start_training.sh kills the urbanzero session
    # itself, but be explicit for hung processes).
    tmux kill-session -t urbanzero 2>/dev/null
    pkill -TERM -f "agents/train.py" 2>/dev/null
    sleep 5
    pkill -KILL -f "agents/train.py" 2>/dev/null
    bash "$START_SCRIPT"
    log "Trainer restart issued. Sleeping ${WARMUP_SEC}s for warmup."
    sleep "$WARMUP_SEC"
}

while true; do
    sleep "$CHECK_INTERVAL"

    # 1. CARLA RPC port — bash's /dev/tcp probe (no nc dependency).
    if ! timeout 3 bash -c ">/dev/tcp/${CARLA_HOST}/${CARLA_PORT}" 2>/dev/null; then
        log "CARLA $CARLA_HOST:$CARLA_PORT NOT reachable; backing off ${CARLA_BACKOFF_SEC}s."
        sleep "$CARLA_BACKOFF_SEC"
        continue
    fi

    # 2. Trainer process check — must exist as the actual python process,
    # not just the tmux pane (TensorBoard etc. also spawn python).
    if ! pgrep -f "agents/train.py" >/dev/null 2>&1; then
        log "Trainer process not found (pgrep agents/train.py)."
        if [ ! -f "$START_SCRIPT" ]; then
            log "ERROR: $START_SCRIPT not found, cannot restart."
            continue
        fi
        restart_trainer
        continue
    fi

    # 3. Beacon staleness — strongest signal that the trainer is hung.
    if [ -f "$BEACON" ]; then
        now=$(date +%s)
        mtime=$(stat -c %Y "$BEACON" 2>/dev/null || stat -f %m "$BEACON" 2>/dev/null || echo "$now")
        age=$(( now - mtime ))
        if [ "$age" -gt "$STALE_SEC" ]; then
            log "Beacon STALE (${age}s > ${STALE_SEC}s); trainer is hung. Restarting."
            restart_trainer
            continue
        fi
    else
        log "Beacon $BEACON missing (trainer may still be warming up)."
    fi
done
