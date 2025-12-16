#!/usr/bin/env bash
set -euo pipefail

# Round-robin watchdog for Test1.py, Test2.py, Test3.py
# ----------------------------------------------------
# Runs ONE job at a time:
#   Test1.py  → Test2.py → Test3.py → Test1.py → ...
#
# Each job has:
#   - cmd       : python3 -u TestN.py
#   - logfile   : runN.log
#   - checkpoint: /tmp/mab_stateN.pkl
#
# When the current job:
#   - exits, OR
#   - stalls (no log activity for IDLE_TIMEOUT seconds), OR
#   - starts zero/O spam in the logs,
# the watchdog:
#   - kills it,
#   - waits GUI_SHUTDOWN_WAIT seconds (so Qt GUI can close properly),
#   - switches to the next TestN in round robin.

# ===================== JOB CONFIG =======================

CMDS=(
  "python3 -u Test1.py"
  "python3 -u Test2.py"
  "python3 -u Test3.py"
)

LOGS=(
  "run1.log"
  "run2.log"
  "run3.log"
)

CHECKPOINTS=(
  "/tmp/mab_state1.pkl"
  "/tmp/mab_state2.pkl"
  "/tmp/mab_state3.pkl"
)

num_cmds=${#CMDS[@]}
num_logs=${#LOGS[@]}
num_ckpt=${#CHECKPOINTS[@]}

# Sanity check: all arrays must have same length
if (( num_cmds != num_logs || num_cmds != num_ckpt )); then
  echo "FATAL: CMDS/LOGS/CHECKPOINTS length mismatch: cmds=$num_cmds logs=$num_logs ckpt=$num_ckpt" >&2
  exit 1
fi

num_jobs=$num_cmds

# How long to sleep after killing a job so the Qt GUI can close properly.
GUI_SHUTDOWN_WAIT=1   # seconds (you can increase to e.g. 10 if windows are slow)

# =================== WATCHDOG TUNABLES ==================

POLL_INTERVAL=2       # seconds between checks
IDLE_TIMEOUT=12       # seconds: no log writes -> treat as stall

PATTERN_REPEAT=3
TAIL_WINDOW=180
ZERO_MIN_TIME=3
ZERO_MAX_TIME=7
ZERO_COUNT_N=4

WRITE_RUNLOG=1
RESTARTING_FLAG=0
FIRST_ZERO_TS=0
ZERO_COUNT=0

# These will be updated per job in the main loop
PYCMD="Test1.py"
LOGFILE="run1.log"
CHECKPOINT="/tmp/mab_state1.pkl"

timestamp(){ date -u +"%Y-%m-%dT%H:%M:%SZ"; }

logput(){
  local msg="$1"
  if (( WRITE_RUNLOG == 1 )); then
    echo "[$(timestamp)] [WATCHDOG] $msg" | tee -a "$LOGFILE"
  else
    echo "[$(timestamp)] [WATCHDOG] $msg"
  fi
}

start_proc(){
  logput "Starting flowgraph: cmd='$PYCMD' checkpoint='$CHECKPOINT' RESTARTING_FLAG=$RESTARTING_FLAG"
  export MAB_CHECKPOINT="$CHECKPOINT"
  export MAB_LOAD_ON_RESTART="$RESTARTING_FLAG"
  # For normal runs set to 0; during debug set to 1 (we default to 1 to capture per-sample logs)
  export MAB_JSON_PER_SAMPLE="${MAB_JSON_PER_SAMPLE:-1}"
  export PYTHONUNBUFFERED=1
  export PYTHONIOENCODING=utf-8

  # Start process (stdout and stderr appended to LOGFILE)
  $PYCMD >> "$LOGFILE" 2>&1 &
  PID=$!
  echo $PID > /tmp/flowgraph.pid
  logput "Started PID=$PID"
  RESTARTING_FLAG=1
  FIRST_ZERO_TS=0
  ZERO_COUNT=0
}

stop_proc(){
  if [ -f /tmp/flowgraph.pid ]; then
    PID=$(cat /tmp/flowgraph.pid)
    if ps -p "$PID" > /dev/null 2>&1; then
      logput "Stopping PID=$PID"
      kill "$PID" || true
      sleep 1
      if ps -p "$PID" > /dev/null 2>&1; then
        logput "Killing PID=$PID (force)"
        kill -9 "$PID" || true
      fi
    fi
    rm -f /tmp/flowgraph.pid
  fi
  FIRST_ZERO_TS=0
  ZERO_COUNT=0
}

check_idle(){
  # If LOGFILE is empty/unset, just say "no idle detected"
  if [ -z "${LOGFILE:-}" ]; then
    return 1
  fi

  if [ ! -f "$LOGFILE" ]; then
    return 1
  fi

  local last_mod now
  last_mod=$(stat -c %Y "$LOGFILE")
  now=$(date +%s)
  if (( now - last_mod > IDLE_TIMEOUT )); then
    return 0
  else
    return 1
  fi
}

check_zero_spam(){
  # If LOGFILE isn't set or doesn't exist yet, nothing to check
  if [ -z "${LOGFILE:-}" ] || [ ! -f "$LOGFILE" ]; then
    return 1
  fi

  local matches now elapsed
  # grep series of zeros/Os repeated; count occurrences
  matches=$(tail -n "$TAIL_WINDOW" "$LOGFILE" | grep -E -o "0{$PATTERN_REPEAT,}|O{$PATTERN_REPEAT,}" | wc -l || true)
  now=$(date +%s)

  if (( matches > 0 )); then
    if (( FIRST_ZERO_TS == 0 )); then
      FIRST_ZERO_TS=$now
      ZERO_COUNT=$matches
      logput "Zero-like spam first seen: count=$ZERO_COUNT time=$(date -u -d @$FIRST_ZERO_TS +%FT%TZ)"
    else
      ZERO_COUNT=$((ZERO_COUNT + matches))
      elapsed=$(( now - FIRST_ZERO_TS ))
      logput "Zero-like spam observed: total_count=$ZERO_COUNT elapsed=${elapsed}s"
    fi
  fi

  if (( FIRST_ZERO_TS > 0 )); then
    elapsed=$(( now - FIRST_ZERO_TS ))
    if (( elapsed >= ZERO_MAX_TIME )); then
      logput "Zero spam elapsed >= ZERO_MAX_TIME ($elapsed >= $ZERO_MAX_TIME) -> switch job"
      return 0
    fi
    if (( elapsed >= ZERO_MIN_TIME )) && (( ZERO_COUNT >= ZERO_COUNT_N )); then
      logput "Zero spam: elapsed >= Tmin and count >= N -> switch job (elapsed=$elapsed count=$ZERO_COUNT)"
      return 0
    fi
  fi
  return 1
}

logput "Round-robin watchdog main loop starting (jobs=$num_jobs)"

job_idx=0

while true; do
  # -------- Select current job in round-robin --------
  PYCMD="${CMDS[$job_idx]}"
  LOGFILE="${LOGS[$job_idx]}"
  CHECKPOINT="${CHECKPOINTS[$job_idx]}"

  logput "=== Switching to job index $job_idx ==="
  logput "Command   : $PYCMD"
  logput "Log file  : $LOGFILE"
  logput "Checkpoint: $CHECKPOINT"

  start_proc
  sleep 2
  reason=""

  # --------------- Inner monitoring loop --------------
  while true; do
    sleep "$POLL_INTERVAL"

    if [ -f /tmp/flowgraph.pid ]; then
      PID=$(cat /tmp/flowgraph.pid)
      if ! ps -p "$PID" > /dev/null 2>&1; then
        logput "Flowgraph PID $PID exited; will switch to next job"
        reason="exit"
        break
      fi
    else
      logput "PID file missing; will switch to next job"
      reason="pid_missing"
      break
    fi

    if check_idle; then
      logput "Stall detected — no log activity for $IDLE_TIMEOUT s"
      reason="idle"
      break
    fi

    if check_zero_spam; then
      reason="zero_spam"
      break
    fi
  done
  # ----------------------------------------------------

  stop_proc

  logput "Job index $job_idx finished (reason: $reason)"
  logput "Waiting $GUI_SHUTDOWN_WAIT s for GUI to close properly before switching job"
  sleep "$GUI_SHUTDOWN_WAIT"

  # Move to next job in round-robin
  job_idx=$(( (job_idx + 1) % num_jobs ))
  logput "Next job will be index $job_idx"
done
