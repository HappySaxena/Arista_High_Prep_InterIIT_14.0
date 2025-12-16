import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_PATH = "rrm_controller_log.jsonl"
BASELINE_WARMUP_STEPS = 3  # same as your controller

steps = []
median_thr = []
rewards = []

# --- Parse log ---
with open(LOG_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("event") != "kpi_snapshot":
            continue

        step = rec.get("step")
        data = rec.get("data", {})
        kpis = data.get("kpis", {})
        r = data.get("reward")

        steps.append(step)
        median_thr.append(float(kpis.get("median_edge_thr", 0.0)))
        rewards.append(float(r) if r is not None else 0.0)

if not steps:
    raise SystemExit("No kpi_snapshot events in log; run controller first.")

steps = np.array(steps)
median_thr = np.array(median_thr)
rewards = np.array(rewards)

# --- Baseline median (first few steps) ---
warmup_mask = steps < BASELINE_WARMUP_STEPS
if warmup_mask.sum() == 0:
    # fallback: use first 3 samples
    baseline_median = float(median_thr[:3].mean())
else:
    baseline_median = float(median_thr[warmup_mask].mean())

# --- QoE lift: relative change vs baseline median ---
eps = 1e-6
qoe_lift = (median_thr - baseline_median) / max(baseline_median, eps)

# For plotting, mask pre-warmup steps so the line starts after baseline
qoe_lift_plot = qoe_lift.copy()
qoe_lift_plot[warmup_mask] = np.nan

# --- Plot 1: Throughput vs step ---
plt.figure()
plt.plot(steps, median_thr, label="Median edge throughput index")
plt.axvline(BASELINE_WARMUP_STEPS, linestyle="--", label="End of baseline warmup")
plt.xlabel("Slow-loop step")
plt.ylabel("Throughput index (normalized)")
plt.grid(True)
plt.legend()
plt.title("Edge throughput vs time")
plt.tight_layout()
plt.savefig("throughput_vs_time.png")

# --- Plot 2: QoE lift vs baseline ---
plt.figure()
plt.plot(steps, qoe_lift_plot, label="QoE lift vs baseline")
plt.axhline(0.0, linestyle="--", label="No change")
plt.xlabel("Slow-loop step")
plt.ylabel("QoE lift (relative to baseline)")
plt.grid(True)
plt.legend()
plt.title(f"QoE lift vs baseline (baseline_median={baseline_median:.3f})")
plt.tight_layout()
plt.savefig("qoe_lift_vs_time.png")

# --- Plot 3: Reward vs time ---
plt.figure()
plt.plot(steps, rewards, label="Reward")
plt.axvline(BASELINE_WARMUP_STEPS, linestyle="--", label="End of baseline warmup")
plt.xlabel("Slow-loop step")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()
plt.title("Reward vs time")
plt.tight_layout()
plt.savefig("reward_vs_time.png")
