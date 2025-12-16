#!/usr/bin/env python3
import subprocess
import json
import time
import re

# ---------------------------------------
# CONFIG
# ---------------------------------------
# Command to run your passive_rtt script
# Remove "sudo" if you run this as root already.
PASSIVE_RTT_CMD = ["sudo", "python3", "passive_rtt.py"]

# JSON output path
JSON_PATH = "rtt_metrics.json"

# Interval between measurements (seconds)
INTERVAL_SEC = 40  # Call passive_rtt every 40 seconds
TIMEOUT_SEC = 30   # Timeout for passive_rtt to give output in 30 seconds

# CUSUM parameters (for P95 RTT spikes)
CUSUM_DRIFT = 1.0        # small drift to ignore minor noise (ms)
CUSUM_THRESHOLD = 50.0   # threshold for spike detection (ms)
# ---------------------------------------

# CUSUM state (kept across iterations)
cusum_pos = 0.0
baseline_p95 = None


def run_passive_rtt():
    """
    Run passive_rtt.py and return its stdout as text.
    It will return empty output if the script doesn't complete within 30 seconds.
    """
    try:
        proc = subprocess.Popen(
            PASSIVE_RTT_CMD,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Set the timeout for the process to 30 seconds
        out, err = proc.communicate(timeout=TIMEOUT_SEC)

        # If there is an error in stderr, print it
        if err.strip():
            print("[passive_rtt stderr]:", err.strip())

        return out
    except subprocess.TimeoutExpired:
        print("[INFO] passive_rtt timed out after 30 seconds.")
        return ""  # Return empty output if the process exceeds timeout
    except Exception as e:
        print("[ERROR] running passive_rtt:", e)
        return ""


def parse_passive_rtt_output(text):
    """
    Parse the output of passive_rtt.py and extract metrics.

    Expected lines (from your script):
      Samples       : N
      RTT Median    : XX.XX ms
      RTT P95       : YY.YY ms
      RTT Jitter    : ZZ.ZZ ms
      Loss rate     : AA.AA %
      Loss variance : 0.000000
    """
    metrics = {
        "samples": None,
        "rtt_median_ms": None,
        "rtt_p95_ms": None,
        "rtt_jitter_ms": None,
        "loss_rate": None,        # as fraction 0..1
        "loss_variance": None,
    }

    for line in text.splitlines():
        line = line.strip()

        m = re.match(r"Samples\s*:\s*(\d+)", line)
        if m:
            metrics["samples"] = int(m.group(1))
            continue

        m = re.match(r"RTT Median\s*:\s*([0-9.]+)", line)
        if m:
            metrics["rtt_median_ms"] = float(m.group(1))
            continue

        m = re.match(r"RTT P95\s*:\s*([0-9.]+)", line)
        if m:
            metrics["rtt_p95_ms"] = float(m.group(1))
            continue

        m = re.match(r"RTT Jitter\s*:\s*([0-9.]+)", line)
        if m:
            metrics["rtt_jitter_ms"] = float(m.group(1))
            continue

        m = re.match(r"Loss rate\s*:\s*([0-9.]+)", line)
        if m:
            # convert from percent to fraction
            metrics["loss_rate"] = float(m.group(1)) / 100.0
            continue

        m = re.match(r"Loss variance\s*:\s*([0-9.]+)", line)
        if m:
            metrics["loss_variance"] = float(m.group(1))
            continue

    return metrics


def update_cusum(p95):
    """
    One-sided CUSUM on P95 RTT to detect sudden spikes.
    Returns (spike_detected, baseline_p95, cusum_pos).
    """
    global baseline_p95, cusum_pos

    # If we don't have a valid P95, no update
    if p95 is None:
        return False, baseline_p95, cusum_pos

    # Initialize baseline with first measurement
    if baseline_p95 is None:
        baseline_p95 = p95
        cusum_pos = 0.0
        return False, baseline_p95, cusum_pos

    # Deviation from current baseline
    deviation = p95 - baseline_p95

    # Positive (upward) CUSUM: accumulate only when above baseline
    # minus a small drift to avoid triggering on small noise
    cusum_pos = max(0.0, cusum_pos + deviation - CUSUM_DRIFT)

    # Optional: slowly adapt baseline over time (EWMA)
    alpha = 0.05  # smoothing factor
    baseline_p95 = (1 - alpha) * baseline_p95 + alpha * p95

    spike = False
    if cusum_pos > CUSUM_THRESHOLD:
        spike = True
        # Reset CUSUM after detection (so next spike must build up again)
        cusum_pos = 0.0

    return spike, baseline_p95, cusum_pos


def write_json(metrics):
    """
    Overwrite the JSON file with the latest metrics.
    NOTE: No timestamp is written, as requested.
    """
    with open(JSON_PATH, "w") as f:
        json.dump(metrics, f, indent=2)


def main_loop():
    while True:
        print("[INFO] Running passive_rtt...")
        out = run_passive_rtt()

        if out == "":  # If no output is produced, skip this iteration
            print("[INFO] No output received, skipping this iteration.")
            time.sleep(INTERVAL_SEC)
            continue

        metrics = parse_passive_rtt_output(out)

        p95 = metrics.get("rtt_p95_ms")
        spike, baseline, cusum_value = update_cusum(p95)

        # Add CUSUM-related fields (still no explicit timestamp)
        metrics["cusum_baseline_p95_ms"] = baseline
        metrics["cusum_value"] = cusum_value
        metrics["p95_spike_detected"] = spike

        write_json(metrics)
        print("[INFO] Updated", JSON_PATH, "with latest RTT + CUSUM stats")

        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main_loop()
