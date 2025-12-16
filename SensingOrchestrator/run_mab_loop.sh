#!/usr/bin/env bash
# run_mab_loop.sh
# Supervisor for Test.py: logs, restarts on non-zero exit, preserves terminal output via tee.

set -euo pipefail

PY_SCRIPT="/home/happy/Test.py"
LOGFILE="/home/happy/mab_runtime.log"
# marker file the python code can check to decide whether to reload saved state.
AUTO_RELOAD_FLAG="/home/happy/.mab_autoreload"

# create logfile if missing
mkdir -p "$(dirname "$LOGFILE")"
touch "$LOGFILE"

# export a flag so Test.py can detect "launched by supervisor" (auto restart mode)
export MAB_SUPERVISOR=1
export MAB_AUTO_RELOAD=1

echo "[$(date -Iseconds)] [MAB Supervisor] Starting run loop..." | tee -a "$LOGFILE"
echo "[$(date -Iseconds)] [MAB Supervisor] Logging to $LOGFILE" | tee -a "$LOGFILE"
echo "[$(date -Iseconds)] [MAB Supervisor] PID $$" | tee -a "$LOGFILE"

run=0
# infinite loop
while true; do
  run=$((run+1))
  echo "[$(date -Iseconds)] [MAB Supervisor] Launching $PY_SCRIPT (run:$run)" | tee -a "$LOGFILE"

  # Run python and keep live terminal + log (tee). Capture python exit code correctly using PIPESTATUS.
  python3 -u "$PY_SCRIPT" 2>&1 | tee -a "$LOGFILE"
  rc=${PIPESTATUS[0]:-126}   # exit code of python (if pipe exists). default to 126 if unusual.

  echo "[$(date -Iseconds)] [MAB Supervisor] $PY_SCRIPT exited with code $rc" | tee -a "$LOGFILE"

  # if exit code == 0 => clean termination. Stop restarting.
  if [ "$rc" -eq 0 ]; then
    echo "[$(date -Iseconds)] [MAB Supervisor] Clean exit (code 0). Not restarting." | tee -a "$LOGFILE"
    break
  fi

  # Unexpected exit -> restart after a short delay
  echo "[$(date -Iseconds)] [MAB Supervisor] Unexpected termination (code $rc). Will restart in 2s." | tee -a "$LOGFILE"

  # Save a short excerpt for quick debugging
  tail -n 400 "$LOGFILE" > "${LOGFILE}.last" || true

  sleep 2
done

echo "[$(date -Iseconds)] [MAB Supervisor] Supervisor exiting." | tee -a "$LOGFILE"
exit 0
