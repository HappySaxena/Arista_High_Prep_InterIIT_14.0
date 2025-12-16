"""
Intermittent Interference Injector (3 DFS types) with DFS scheduler + JSONL logging.

Port mapping (inputs, in order):
  - in0 : complex64  -> ip1 : DFS Type 1
  - in1 : complex64  -> ip2 : DFS Type 2
  - in2 : complex64  -> ip3 : DFS Type 1 + 2
  - in3 : complex64  -> dfs_in : DFS-monitor signal used to detect radar (power measured)

Output:
  - out  : complex64 combined interference (or zeros when idle/hold-off)

Logging:
  - Newline-delimited JSON appended to log_path
  - Events: injection_start, injection_terminated_by_dfs, dfs_detected
"""

import numpy as np
import json
import os
import math
from datetime import datetime, timezone
from gnuradio import gr

# Human-readable labels for the three interference inputs
INTERFERENCE_MAP = {
    0: "DFS_Type_1",        # in0 / ip1
    1: "DFS_Type_2",        # in1 / ip2
    2: "DFS_Type_1_and_2"   # in2 / ip3
}


class blk(gr.sync_block):
    def __init__(self,
                 example_param=1.0,

                 # timing / scheduling
                 sample_rate=20e6,
                 mean_interval_seconds=2.0,
                 inject_duration_seconds=0.1,

                 # pure injection (occasional long single-source injection)
                 pure_inject_enabled=True,
                 pure_injection_probability=0.02,
                 pure_inject_duration_seconds=20.0,

                 # DFS scheduler options
                 dfs_monitor_enabled=True,
                 dfs_detection_threshold=1e-6,
                 dfs_holdoff_seconds=60.0,

                 # logging / output
                 log_path="/tmp/interference_log.jsonl",
                 log_every_injection=True,
                 output_when_idle="zeros"):
        """
        Constructor parameters appear in GRC as block parameters. All have defaults.
        """
        # Inputs: 3 interferers + 1 dfs monitor
        gr.sync_block.__init__(
            self,
            name='Interference Injector (3 DFS types) with DFS scheduler',
            in_sig=[np.complex64, np.complex64, np.complex64, np.complex64],
            out_sig=[np.complex64]
        )

        # Exposed parameters
        self.example_param = example_param
        self.sample_rate = float(sample_rate)
        self.mean_interval_seconds = float(mean_interval_seconds)
        self.inject_duration_seconds = float(inject_duration_seconds)

        self.pure_inject_enabled = bool(pure_inject_enabled)
        self.pure_injection_probability = float(pure_injection_probability)
        self.pure_inject_duration_seconds = float(pure_inject_duration_seconds)

        self.dfs_monitor_enabled = bool(dfs_monitor_enabled)
        self.dfs_detection_threshold = float(dfs_detection_threshold)
        self.dfs_holdoff_seconds = float(dfs_holdoff_seconds)

        self.log_path = str(log_path)
        self.log_every_injection = bool(log_every_injection)
        self.output_when_idle = str(output_when_idle)

        # Internal state
        self._time_accumulator_seconds = 0.0
        self._injection_remaining_seconds = 0.0
        self._next_wait_seconds = self._sample_next_wait()
        self._current_coeffs = None  # numpy array length 3 when injecting
        self._injection_id = 0

        self._is_current_pure = False
        self._pure_source_index = None  # 0..2

        # DFS state
        self._dfs_holdoff_remaining = 0.0
        self._last_dfs_detect_time = None

        # Ensure log folder exists
        log_dir = os.path.dirname(os.path.abspath(self.log_path))
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
            except Exception:
                pass

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _sample_next_wait(self):
        mean = max(1e-12, float(self.mean_interval_seconds))
        return np.random.exponential(mean)

    def _write_log(self, record):
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            # Do not let logging failures crash the flowgraph
            pass

    def _start_injection(self, force_pure=False):
        """
        Begin a new injection (mixed or pure). Logs injection start if enabled.
        """
        self._injection_id += 1
        self._is_current_pure = False
        self._pure_source_index = None

        will_be_pure = False
        if self.pure_inject_enabled:
            if force_pure:
                will_be_pure = True
            else:
                if np.random.random() < self.pure_injection_probability:
                    will_be_pure = True

        if will_be_pure:
            src_idx = int(np.random.randint(0, 3))
            self._pure_source_index = src_idx
            self._is_current_pure = True
            coeffs = np.zeros(3, dtype=float)
            coeffs[src_idx] = 1.0
            self._current_coeffs = coeffs
            self._injection_remaining_seconds = float(self.pure_inject_duration_seconds)
        else:
            coeffs = np.random.random(3)  # three random floats in [0,1)
            self._current_coeffs = coeffs.astype(float)
            self._injection_remaining_seconds = float(self.inject_duration_seconds)

        # Log injection start
        if self.log_every_injection:
            dominant_idx = int(np.argmax(self._current_coeffs))
            record = {
                "event": "injection_start",
                "injection_id": int(self._injection_id),
                "start_time_utc": datetime.now(timezone.utc).isoformat(),
                "is_pure_injection": bool(self._is_current_pure),
                "pure_source_index": None if self._pure_source_index is None else int(self._pure_source_index),
                "pure_source_type": None if self._pure_source_index is None else INTERFERENCE_MAP.get(self._pure_source_index),
                "mean_interval_seconds": float(self.mean_interval_seconds),
                "inject_duration_seconds": float(self._injection_remaining_seconds),
                "coefficients": { INTERFERENCE_MAP[i]: float(self._current_coeffs[i]) for i in range(3) },
                "dominant_index": dominant_idx,
                "dominant_type": INTERFERENCE_MAP.get(dominant_idx, f"input_{dominant_idx}")
            }
            self._write_log(record)

        # reset wait accumulator and sample next wait for after injection
        self._time_accumulator_seconds = 0.0
        self._next_wait_seconds = self._sample_next_wait()

    def _handle_dfs_detection(self, dfs_power):
        """
        Called when DFS detection threshold is exceeded.
        Terminates any active injection, enters hold-off, and logs the event.
        """
        # If an injection was active, log its termination
        if self._injection_remaining_seconds > 0.0 and self.log_every_injection:
            record_term = {
                "event": "injection_terminated_by_dfs",
                "injection_id": int(self._injection_id),
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "remaining_injection_seconds": float(self._injection_remaining_seconds)
            }
            self._write_log(record_term)

        # Clear injection state
        self._injection_remaining_seconds = 0.0
        self._current_coeffs = None
        self._is_current_pure = False
        self._pure_source_index = None

        # Enter hold-off
        self._dfs_holdoff_remaining = float(self.dfs_holdoff_seconds)
        self._last_dfs_detect_time = datetime.now(timezone.utc)

        # Log DFS detection
        record = {
            "event": "dfs_detected",
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "dfs_power": float(dfs_power),
            "holdoff_seconds": float(self._dfs_holdoff_remaining)
        }
        self._write_log(record)

        # Reset waiting accumulator so schedule restarts after hold-off
        self._time_accumulator_seconds = 0.0
        self._next_wait_seconds = self._sample_next_wait()

    # ---------------------------
    # Main processing
    # ---------------------------
    def work(self, input_items, output_items):
        # Inputs: in0 (DFS Type 1), in1 (DFS Type 2), in2 (DFS Type 1+2), dfs_in (monitor)
        in0 = input_items[0]
        in1 = input_items[1]
        in2 = input_items[2]
        dfs_in = input_items[3]

        # samples to process
        n = min(len(in0), len(in1), len(in2), len(dfs_in))
        if n <= 0:
            return 0

        elapsed_seconds = float(n) / max(1.0, float(self.sample_rate))
        out = np.empty(n, dtype=np.complex64)

        samples_filled = 0
        remaining_samples = n
        local_time = 0.0

        # Process by segments (idle / injection / hold-off boundaries)
        while remaining_samples > 0:
            # Handle DFS hold-off first
            if self._dfs_holdoff_remaining > 0.0:
                seg_seconds = min(self._dfs_holdoff_remaining, elapsed_seconds - local_time)
                seg_samples = int(math.floor(seg_seconds * self.sample_rate + 1e-12))
                seg_samples = min(seg_samples, remaining_samples)
                if seg_samples <= 0:
                    seg_samples = 1

                # Idle output during hold-off
                out[samples_filled:samples_filled+seg_samples] = 0

                used_seconds = float(seg_samples) / self.sample_rate
                self._dfs_holdoff_remaining -= used_seconds
                local_time += used_seconds
                remaining_samples -= seg_samples
                samples_filled += seg_samples

                # Continue, still monitor dfs_in
                continue

            # If not currently injecting, we are waiting
            if self._injection_remaining_seconds <= 0.0:
                time_until_next = self._next_wait_seconds - self._time_accumulator_seconds
                time_until_next = max(0.0, time_until_next)

                # If next injection starts within this chunk
                if local_time + time_until_next < elapsed_seconds:
                    seg_seconds = time_until_next
                    seg_samples = int(math.floor(seg_seconds * self.sample_rate + 1e-12))
                    seg_samples = min(seg_samples, remaining_samples)
                    if seg_samples <= 0:
                        seg_samples = 1

                    # Idle output
                    out[samples_filled:samples_filled+seg_samples] = 0

                    # Check DFS power over this idle segment
                    dfs_segment = dfs_in[samples_filled:samples_filled+seg_samples]
                    dfs_power = float(np.mean(np.abs(dfs_segment) ** 2)) if seg_samples > 0 else 0.0

                    used_seconds = float(seg_samples) / self.sample_rate
                    self._time_accumulator_seconds += used_seconds
                    local_time += used_seconds
                    remaining_samples -= seg_samples
                    samples_filled += seg_samples

                    # DFS detection?
                    if self.dfs_monitor_enabled and dfs_power >= self.dfs_detection_threshold:
                        self._handle_dfs_detection(dfs_power)
                        continue

                    # Wait finished -> start injection if not suppressed
                    if self._time_accumulator_seconds >= self._next_wait_seconds - 1e-12:
                        if self._dfs_holdoff_remaining <= 0.0:
                            self._start_injection()
                        else:
                            pass
                else:
                    # Entire remaining chunk is idle
                    seg_samples = remaining_samples
                    out[samples_filled:samples_filled+seg_samples] = 0

                    dfs_segment = dfs_in[samples_filled:samples_filled+seg_samples]
                    dfs_power = float(np.mean(np.abs(dfs_segment) ** 2)) if seg_samples > 0 else 0.0

                    used_seconds = float(seg_samples) / self.sample_rate
                    self._time_accumulator_seconds += used_seconds
                    local_time += used_seconds
                    remaining_samples -= seg_samples
                    samples_filled += seg_samples

                    if self.dfs_monitor_enabled and dfs_power >= self.dfs_detection_threshold:
                        self._handle_dfs_detection(dfs_power)
                        continue
                    break
            else:
                # Injection active
                seg_seconds = min(self._injection_remaining_seconds, elapsed_seconds - local_time)
                seg_samples = int(math.floor(seg_seconds * self.sample_rate + 1e-12))
                seg_samples = min(seg_samples, remaining_samples)
                if seg_samples <= 0:
                    seg_samples = 1

                start_idx = samples_filled
                end_idx = samples_filled + seg_samples

                # Monitor DFS inside injection window
                dfs_segment = dfs_in[start_idx:end_idx]
                dfs_power = float(np.mean(np.abs(dfs_segment) ** 2)) if seg_samples > 0 else 0.0

                if self.dfs_monitor_enabled and dfs_power >= self.dfs_detection_threshold:
                    # Handle DFS detection: terminate injection immediately
                    self._handle_dfs_detection(dfs_power)
                    out[start_idx:end_idx] = 0
                    used_seconds = float(seg_samples) / self.sample_rate
                    local_time += used_seconds
                    remaining_samples -= seg_samples
                    samples_filled += seg_samples
                    continue

                # No DFS detection -> produce injection output
                if self._is_current_pure and self._pure_source_index is not None:
                    if self._pure_source_index == 0:
                        out[start_idx:end_idx] = in0[start_idx:end_idx]
                    elif self._pure_source_index == 1:
                        out[start_idx:end_idx] = in1[start_idx:end_idx]
                    elif self._pure_source_index == 2:
                        out[start_idx:end_idx] = in2[start_idx:end_idx]
                    else:
                        out[start_idx:end_idx] = 0
                else:
                    a, b, c = self._current_coeffs.tolist() if self._current_coeffs is not None else (0.0, 0.0, 0.0)
                    out[start_idx:end_idx] = (a * in0[start_idx:end_idx]
                                              + b * in1[start_idx:end_idx]
                                              + c * in2[start_idx:end_idx])

                used_seconds = float(seg_samples) / self.sample_rate
                self._injection_remaining_seconds -= used_seconds
                local_time += used_seconds
                remaining_samples -= seg_samples
                samples_filled += seg_samples

                # If injection ended, reset state
                if self._injection_remaining_seconds <= 0.0:
                    self._current_coeffs = None
                    self._is_current_pure = False
                    self._pure_source_index = None
                    self._time_accumulator_seconds = 0.0
                    self._next_wait_seconds = self._sample_next_wait()

        # write output buffer
        output_items[0][:n] = out
        return n
