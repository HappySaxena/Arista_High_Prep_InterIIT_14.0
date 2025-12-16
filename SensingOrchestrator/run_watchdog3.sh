#!/usr/bin/env bash
set -euo pipefail

# Watchdog launcher for Test.py
# - Uses unbuffered python (-u)
# - Exports diagnostic env vars
# - Detects "zero-spam" and idle stalls
# - Writes run.log in working dir
# - Periodically soft-reboots Pluto SDR and waits for it to come back

PYCMD="python3 -u Test3.py"
LOGFILE="run3.log"
CHECKPOINT="/tmp/mab_state3.pkl"
POLL_INTERVAL=2
IDLE_TIMEOUT=12

PATTERN_REPEAT=4
TAIL_WINDOW=300
ZERO_MIN_TIME=3
ZERO_MAX_TIME=12
ZERO_COUNT_N=6

WRITE_RUNLOG=1
RESTARTING_FLAG=0
FIRST_ZERO_TS=0
ZERO_COUNT=0

# ----------------- Pluto reboot config -----------------
PLUTO_IP="192.168.4.1"
PLUTO_PASSWORD="analog"             # Pluto default root password
PLUTO_REBOOT_INTERVAL_MIN=30        # <<< T minutes here
PLUTO_REBOOT_INTERVAL=$((PLUTO_REBOOT_INTERVAL_MIN * 60))  # seconds
LAST_REBOOT_TS=$(date +%s)          # last time we rebooted Pluto
# -------------------------------------------------------

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
  logput "Starting flowgraph (checkpoint=$CHECKPOINT) RESTARTING_FLAG=$RESTARTING_FLAG"
  export MAB_CHECKPOINT="$CHECKPOINT"
  export MAB_LOAD_ON_RESTART="$RESTARTING_FLAG"
  # For normal runs set to 0; during debug set to 1 (we default to 1 to capture per-sample logs)
  export MAB_JSON_PER_SAMPLE="${MAB_JSON_PER_SAMPLE:-1}"
  export PYTHONUNBUFFERED=1
  export PYTHONIOENCODING=utf-8

  # Start process (stdout and stderr appended to run.log)
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
  if [ ! -f "$LOGFILE" ]; then return 1; fi
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
  if [ ! -f "$LOGFILE" ]; then return 1; fi
  local matches now elapsed
  # grep series of zeros/Os repeated; count occurrences
  matches=$(tail -n "$TAIL_WINDOW" "$LOGFILE" | egrep -o "0{$PATTERN_REPEAT,}|O{$PATTERN_REPEAT,}" | wc -l || true)
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
      logput "Zero spam elapsed >= ZERO_MAX_TIME ($elapsed >= $ZERO_MAX_TIME) -> restart"
      return 0
    fi
    if (( elapsed >= ZERO_MIN_TIME )) && (( ZERO_COUNT >= ZERO_COUNT_N )); then
      logput "Zero spam: elapsed >= Tmin and count >= N -> restart (elapsed=$elapsed count=$ZERO_COUNT)"
      return 0
    fi
  fi
  return 1
}

# ------------- NEW: Pluto reboot helpers ----------------

pluto_wait_up(){
  # Wait for ping + SSH port 22 to be available
  logput "Waiting for Pluto ($PLUTO_IP) to respond to ping..."
  while ! ping -c1 -W1 "$PLUTO_IP" >/dev/null 2>&1; do
    sleep 2
  done

  logput "Pluto responds to ping; waiting for SSH port..."
  while ! nc -z "$PLUTO_IP" 22 >/dev/null 2>&1; do
    sleep 2
  done

  logput "Pluto is back (SSH reachable)"
}

reboot_pluto(){
  LAST_REBOOT_TS=$(date +%s)
  logput "Soft rebooting Pluto at $PLUTO_IP via SSH"

  # Non-interactive SSH:
  #  - sshpass provides the password
  #  - StrictHostKeyChecking=no avoids yes/no prompt
  if ! sshpass -p "$PLUTO_PASSWORD" \
       ssh -o StrictHostKeyChecking=no \
           -o UserKnownHostsFile=/dev/null \
           root@"$PLUTO_IP" "pluto_reboot reset"; then
    logput "WARNING: SSH command to Pluto failed; still waiting for device to reappear"
  fi

  pluto_wait_up
}
# --------------------------------------------------------

logput "Watchdog main loop starting"

while true; do
  start_proc
  sleep 2
  reason=""

  while true; do
    sleep "$POLL_INTERVAL"

    # -------- NEW: periodic Pluto reboot decision --------
    now_ts=$(date +%s)
    if (( now_ts - LAST_REBOOT_TS >= PLUTO_REBOOT_INTERVAL )); then
      logput "Periodic Pluto reboot due (interval ${PLUTO_REBOOT_INTERVAL}s) -> scheduling restart"
      reason="pluto_reboot"
      break
    fi
    # -----------------------------------------------------

    if [ -f /tmp/flowgraph.pid ]; then
      PID=$(cat /tmp/flowgraph.pid)
      if ! ps -p "$PID" > /dev/null 2>&1; then
        logput "Flowgraph PID $PID exited; will restart"
        reason="exit"
        break
      fi
    else
      logput "PID file missing; will restart"
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

  stop_proc

  # If this restart was triggered by the periodic Pluto reboot,
  # reboot and wait for the device before starting the flowgraph again.
  if [[ "$reason" == "pluto_reboot" ]]; then
    reboot_pluto
  fi

  logput "Restarting flowgraph (reason: $reason) — next start will load checkpoint"
  sleep 1
done

