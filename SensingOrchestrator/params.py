import json
import os
from collections import defaultdict, deque
import numpy as np
import time

# Persistent state for dynamic behavior
GLOBAL_STATE = defaultdict(lambda: {
    "interference_history": deque(),
    "dfs_history": deque(),
    "sense_timestamps": deque(),
    "call_count": 0
})

THRESHOLD = 0.001  # interference threshold

def to_dBm(power):
    """Convert linear power to dBm."""
    if power <= 0:
        return -100  # avoid log(0) and handle zero/negative power
    return 10 * np.log10(power * 1000)

def physics_adjustment(mean, variance, sensing_gap, Max_ap_range):
    """
    Physics/statistics-based smoothing for rarely sensed channels.
    Larger sensing gap => uncertainty grows.
    """
    K = 0.05  # tuning constant
    return mean + K * variance * (sensing_gap / Max_ap_range)

def params(path, buffer_length, Max_ap_range=10, mean_method="simple"):
    """
    Analyzes log data from the given path to produce a summary of channel states.
    It simulates a system dynamically processing incoming log entries.
    """
    # Load last N entries from JSON log
    # Read the entire file content as a single JSON object.
    # This assumes the file content is always a valid JSON array of objects.
    with open(path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {path}. Returning empty summary.")
            return {}

    # Keep only the last 'buffer_length' entries from the loaded data for processing
    data = data[-buffer_length:]

    # Update GLOBAL_STATE with new data
    now = time.time()  # Using current real time for "sensing time" in this context
    for entry in data:
        ch = entry["next_channel"]
        interf_pwr = entry.get("interference_power", 0)
        dfs_state = entry.get("Channel_State", "UNKNOWN")

        # Ensure buffer length management (deque handles maxlen implicitly if set at creation)
        # We need to re-initialize deques if buffer_length changes or GLOBAL_STATE is not persistent
        # For this simulation, GLOBAL_STATE is reset at the start, then persists for the simulation.
        # Maxlen of deque needs to be managed if buffer_length changes during simulation, but here it's fixed.
        if not isinstance(GLOBAL_STATE[ch]["interference_history"], deque) or \
           GLOBAL_STATE[ch]["interference_history"].maxlen != buffer_length:
            GLOBAL_STATE[ch]["interference_history"] = deque(maxlen=buffer_length)
            GLOBAL_STATE[ch]["dfs_history"] = deque(maxlen=buffer_length)
            GLOBAL_STATE[ch]["sense_timestamps"] = deque(maxlen=buffer_length)

        hist = GLOBAL_STATE[ch]["interference_history"]
        dfs_hist = GLOBAL_STATE[ch]["dfs_history"]
        ts_hist = GLOBAL_STATE[ch]["sense_timestamps"]

        # Append new samples
        hist.append(interf_pwr)
        dfs_hist.append(dfs_state)
        ts_hist.append(now) # All entries processed in one params call get the same timestamp

        GLOBAL_STATE[ch]["call_count"] += 1

    # Compute summary
    summary = {}

    for ch, info in GLOBAL_STATE.items():
        hist_np = np.array(list(info["interference_history"]), dtype=float)

        if len(hist_np) == 0:
            continue

        # Mean method
        if mean_method == "simple":
            mean_val = np.mean(hist_np)
        elif mean_method == "ema":  # exponential smoothing
            alpha = 0.3
            ema = hist_np[0]
            for x in hist_np[1:]:
                ema = alpha * x + (1 - alpha) * ema
            mean_val = ema
        else:
            mean_val = np.mean(hist_np)

        # Deviation based prediction for rarely sensed channels
        variance = np.var(hist_np) if len(hist_np) > 1 else 0.0
        if len(info["sense_timestamps"]) >= 2:
            sensing_gap = info["sense_timestamps"][-1] - info["sense_timestamps"][0] # Gap between oldest and newest
        else:
            sensing_gap = 0

        adjusted_mean = physics_adjustment(mean_val, variance, sensing_gap, Max_ap_range)

        # Interference rate
        # Counting how many entries in the history buffer exceed the THRESHOLD
        interference_count = np.sum(hist_np > THRESHOLD)
        # Using the actual number of samples in the history for rate calculation
        interference_rate = interference_count / max(1, len(hist_np))


        # DFS state (majority)
        dfs_mode = "UNKNOWN"
        if info["dfs_history"]:
            dfs_mode = max(set(info["dfs_history"]), key=list(info["dfs_history"]).count)

        # Assign summary
        summary[f"ch{ch}"] = {
            "interference_power_dBm": to_dBm(adjusted_mean),
            "interference_rate": round(float(interference_rate), 4), # Convert numpy float to native float
            "DFS_state": dfs_mode,
            "samples_in_buffer": len(hist_np)
        }
    return summary


