#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Intermittent Interference Injector
- Supports mixed + pure injections
- Creates its log file automatically
- Safe logging & fallback
"""

import numpy as np
import json
import os
import math
import sys
import traceback
from datetime import datetime, timezone
from gnuradio import gr


INTERFERENCE_MAP = {
    0: "Microwave",
    1: "Zigbee",
    2: "BLE_burst",
    3: "FHSS"
}


class blk(gr.sync_block):

    def __init__(self,
                 example_param=1.0,
                 sample_rate=1e6,
                 mean_interval_seconds=1.0,
                 inject_duration_seconds=0.1,
                 log_path="/tmp/Injection_2_4.jsonl",
                 log_every_injection=True,
                 output_when_idle="zeros",
                 pure_inject_enabled=True,
                 pure_injection_probability=0.01,
                 pure_inject_duration_seconds=20.0):

        gr.sync_block.__init__(
            self,
            name='Intermittent Interference Injector (stable)',
            in_sig=[np.complex64, np.complex64, np.complex64, np.complex64],
            out_sig=[np.complex64]
        )

        self.example_param = example_param
        self.sample_rate = float(sample_rate)
        self.mean_interval_seconds = float(mean_interval_seconds)
        self.inject_duration_seconds = float(inject_duration_seconds)
        self.log_path = str(log_path)
        self.log_every_injection = bool(log_every_injection)
        self.output_when_idle = str(output_when_idle)

        self.pure_inject_enabled = bool(pure_inject_enabled)
        self.pure_injection_probability = float(pure_injection_probability)
        self.pure_inject_duration_seconds = float(pure_inject_duration_seconds)

        self._time_accumulator_seconds = 0.0
        self._injection_remaining_seconds = 0.0
        self._next_wait_seconds = self._sample_next_wait()
        self._current_coeffs = None
        self._injection_id = 0
        self._is_current_pure = False
        self._pure_source_index = None

        self._prepare_log_file()

    # -------------------------------------------------------------------------
    # LOG FILE SETUP
    # -------------------------------------------------------------------------
    def _prepare_log_file(self):
        """Ensures directory exists and log file is creatable."""

        try:
            log_dir = os.path.dirname(os.path.abspath(self.log_path))
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

        except Exception:
            print("[Injector] Failed to create log directory:", log_dir, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

        # test create the file
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write("")     # touch file
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            print("[Injector] Cannot create log file at:", self.log_path, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

            # fallback to /tmp
            fallback = "/tmp/Injection_2_4.jsonl"
            print("[Injector] Switching to fallback:", fallback, file=sys.stderr)
            self.log_path = fallback

            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write("")
                f.flush()

    # -------------------------------------------------------------------------
    def _sample_next_wait(self):
        return np.random.exponential(max(1e-12, self.mean_interval_seconds))

    def _write_log(self, record):
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
        except Exception:
            print("[Injector] Log write failed at:", self.log_path, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    # -------------------------------------------------------------------------
    def _start_injection(self):

        self._injection_id += 1
        self._is_current_pure = False
        self._pure_source_index = None

        will_pure = False
        if self.pure_inject_enabled:
            if np.random.random() < self.pure_injection_probability:
                will_pure = True

        # PURE INJECTION
        if will_pure:
            idx = int(np.random.randint(0, 4))
            self._pure_source_index = idx
            self._is_current_pure = True

            coeffs = np.zeros(4)
            coeffs[idx] = 1.0
            self._current_coeffs = coeffs

            self._injection_remaining_seconds = float(self.pure_inject_duration_seconds)

        else:
            # MIXED INJECTION
            coeffs = np.random.random(4)
            s = np.sum(coeffs)
            coeffs = coeffs / max(s, 1e-12)
            self._current_coeffs = coeffs

            self._injection_remaining_seconds = float(self.inject_duration_seconds)

        # Log
        if self.log_every_injection:
            dom = int(np.argmax(self._current_coeffs))
            record = {
                "injection_id": self._injection_id,
                "start_time_utc": datetime.now(timezone.utc).isoformat(),
                "is_pure_injection": self._is_current_pure,
                "pure_source_index": None if not self._is_current_pure else int(self._pure_source_index),
                "pure_source_type": None if not self._is_current_pure else INTERFERENCE_MAP[self._pure_source_index],
                "duration_seconds": float(self._injection_remaining_seconds),
                "coefficients": {
                    "Microwave": float(self._current_coeffs[0]),
                    "Zigbee": float(self._current_coeffs[1]),
                    "BLE_burst": float(self._current_coeffs[2]),
                    "FHSS": float(self._current_coeffs[3])
                },
                "dominant_type": INTERFERENCE_MAP.get(dom, "unknown")
            }
            self._write_log(record)

        # prepare next wait
        self._time_accumulator_seconds = 0.0
        self._next_wait_seconds = self._sample_next_wait()

    # -------------------------------------------------------------------------
    # MAIN WORK FUNCTION
    # -------------------------------------------------------------------------
    def work(self, input_items, output_items):

        in0, in1, in2, in3 = input_items
        n = min(len(in0), len(in1), len(in2), len(in3))
        if n <= 0:
            return 0

        elapsed_seconds = n / self.sample_rate
        out = np.zeros(n, dtype=np.complex64)

        samples_done = 0
        remaining = n
        local_time = 0.0

        while remaining > 0:

            # ------------------ IDLE STATE ------------------
            if self._injection_remaining_seconds <= 0.0:

                wait_left = self._next_wait_seconds - self._time_accumulator_seconds
                wait_left = max(0.0, wait_left)

                if local_time + wait_left < elapsed_seconds:
                    seg_sec = wait_left
                    seg_samples = max(1, int(seg_sec * self.sample_rate))
                    seg_samples = min(seg_samples, remaining)

                    # idle: passthrough or zeros
                    if self.output_when_idle == "passthrough":
                        out[samples_done:samples_done+seg_samples] = (
                            in0[samples_done:samples_done+seg_samples] +
                            in1[samples_done:samples_done+seg_samples] +
                            in2[samples_done:samples_done+seg_samples] +
                            in3[samples_done:samples_done+seg_samples]
                        ) / 4.0
                    else:
                        out[samples_done:samples_done+seg_samples] = 0

                    time_used = seg_samples / self.sample_rate
                    self._time_accumulator_seconds += time_used
                    local_time += time_used
                    samples_done += seg_samples
                    remaining -= seg_samples

                    if self._time_accumulator_seconds >= self._next_wait_seconds:
                        self._start_injection()

                else:
                    # rest entire block idle
                    if self.output_when_idle == "passthrough":
                        out[samples_done:samples_done+remaining] = (
                            in0[samples_done:samples_done+remaining] +
                            in1[samples_done:samples_done+remaining] +
                            in2[samples_done:samples_done+remaining] +
                            in3[samples_done:samples_done+remaining]
                        ) / 4.0
                    else:
                        out[samples_done:samples_done+remaining] = 0

                    time_used = remaining / self.sample_rate
                    self._time_accumulator_seconds += time_used
                    local_time += time_used
                    samples_done += remaining
                    remaining = 0
                    break

            # ------------------ ACTIVE INJECTION ------------------
            else:
                seg_sec = min(self._injection_remaining_seconds,
                              elapsed_seconds - local_time)
                seg_samples = max(1, int(seg_sec * self.sample_rate))
                seg_samples = min(seg_samples, remaining)

                a, b, c, d = self._current_coeffs

                if self._is_current_pure:
                    idx = self._pure_source_index
                    if idx == 0:
                        out[samples_done:samples_done+seg_samples] = in0[samples_done:samples_done+seg_samples]
                    elif idx == 1:
                        out[samples_done:samples_done+seg_samples] = in1[samples_done:samples_done+seg_samples]
                    elif idx == 2:
                        out[samples_done:samples_done+seg_samples] = in2[samples_done:samples_done+seg_samples]
                    else:
                        out[samples_done:samples_done+seg_samples] = in3[samples_done:samples_done+seg_samples]

                else:
                    out[samples_done:samples_done+seg_samples] = (
                        a * in0[samples_done:samples_done+seg_samples] +
                        b * in1[samples_done:samples_done+seg_samples] +
                        c * in2[samples_done:samples_done+seg_samples] +
                        d * in3[samples_done:samples_done+seg_samples]
                    )

                used = seg_samples / self.sample_rate
                self._injection_remaining_seconds -= used
                local_time += used
                samples_done += seg_samples
                remaining -= seg_samples

                if self._injection_remaining_seconds <= 0.0:
                    self._current_coeffs = None
                    self._is_current_pure = False
                    self._pure_source_index = None
                    self._time_accumulator_seconds = 0.0

        output_items[0][:n] = out
        return n
