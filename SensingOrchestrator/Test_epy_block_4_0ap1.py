#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defensive DataLogger (fixed): robust handling of parent.timestamps and parent.count types.
Does NOT terminate the process; only flushes and saves state and emits a RESTART_POINT marker.
"""

import os
import json
import time
import traceback
import numpy as np
from gnuradio import gr

# Config via env
LOG_PATH = os.environ.get("DATALOGGER_LOG_PATH", "/home/happy/Downloads/interference_log1_2ndrun.json")
RESTART_EVERY = int(os.environ.get("DATALOGGER_RESTART_EVERY", "40"))
FLUSH_EVERY = int(os.environ.get("DATALOGGER_FLUSH_EVERY", "100"))

LABELS = ['BLE', 'ZIGBEE', 'CW', 'FHSS', 'MICROWAVE', 'NONE']

class blk(gr.sync_block):
    def __init__(self, parent=None):
        super().__init__(
            name="Data_Logger",
            in_sig=[
                np.uint8,          # 0: change_flag
                np.float32,        # 1: duty_cycle
                np.complex64,      # 2: next_channel (real)
                (np.int16, 6),     # 3: interference vector
                np.float32,        # 4: reward
                np.float32,        # 5: oracle
                np.float32,        # 6: regret
                np.complex64,      # 7: est_center + j*bandwidth
                np.float32,        # 8: confidence
                np.float32         # 9: interference_power
            ],
            out_sig=[]
        )

        self.parent = parent
        # buffer + flush configuration
        self._write_buf = []
        self._flush_every = FLUSH_EVERY
        self._log_path = LOG_PATH

        # ensure parent fields exist and have safe types
        if self.parent is not None:
            # safe count: make sure it's an int
            try:
                cnt = getattr(self.parent, "count", None)
                if cnt is None:
                    self.parent.count = 0
                else:
                    # coerce to int if possible
                    try:
                        self.parent.count = int(cnt)
                    except Exception:
                        self.parent.count = 0
            except Exception:
                self.parent.count = 0

            # safe timestamps: ensure it's a list or replace with empty list
            try:
                ts = getattr(self.parent, "timestamps", None)
                if ts is None:
                    self.parent.timestamps = []
                elif not isinstance(ts, list):
                    # try convert iterable -> list, else reset to empty
                    try:
                        self.parent.timestamps = list(ts)
                    except Exception:
                        self.parent.timestamps = []
            except Exception:
                self.parent.timestamps = []

        # load existing log file if possible
        if os.path.exists(self._log_path):
            try:
                with open(self._log_path, "r") as f:
                    self.log = json.load(f)
            except Exception:
                # if file corrupted, start fresh but keep original file
                self.log = []
        else:
            self.log = []

    def _atomic_write(self, path, data):
        """Safe JSON write (atomic replace)."""
        try:
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, path)
        except Exception as e:
            print("[DataLogger] Atomic write error:", e, flush=True)

    def _flush(self):
        """Flush write buffer into disk, appending to existing JSON array."""
        if not self._write_buf:
            return
        try:
            if os.path.exists(self._log_path):
                with open(self._log_path, "r") as f:
                    try:
                        existing = json.load(f)
                    except Exception:
                        existing = []
            else:
                existing = []
            existing.extend(self._write_buf)
            self._atomic_write(self._log_path, existing)
            self.log = existing
            self._write_buf = []
        except Exception as e:
            print("[DataLogger] Flush error:", e, flush=True)
            traceback.print_exc()

    def _safe_inc_count(self):
        """Increment parent.count safely and return the new value."""
        if self.parent is None:
            return -1
        try:
            cnt = getattr(self.parent, "count", 0)
            try:
                cnt_i = int(cnt)
            except Exception:
                cnt_i = 0
            cnt_i += 1
            self.parent.count = cnt_i
            return cnt_i
        except Exception:
            try:
                self.parent.count = 1
                return 1
            except Exception:
                return -1

    def _safe_append_timestamp(self, ts):
        """Append ts to parent.timestamps safely (coerce to list if necessary)."""
        if self.parent is None:
            return
        try:
            existing = getattr(self.parent, "timestamps", None)
            if existing is None:
                self.parent.timestamps = [ts]
                return
            if isinstance(existing, list):
                existing.append(ts)
                return
            # if it's convertible to list (like tuple), attempt conversion
            try:
                newlist = list(existing)
                newlist.append(ts)
                self.parent.timestamps = newlist
                return
            except Exception:
                # fallback: overwrite with single-element list
                self.parent.timestamps = [ts]
                return
        except Exception:
            try:
                self.parent.timestamps = [ts]
            except Exception:
                pass

    def work(self, input_items, output_items):
        if not input_items or len(input_items[0]) == 0:
            return 0

        n = len(input_items[0])

        try:
            for i in range(n):
                change_flag = bool(input_items[0][i] > 0)
                duty_cycle = float(input_items[1][i])
                next_channel = int(input_items[2][i].real)

                vec = list(input_items[3][i])
                interferences = [LABELS[j] for j, v in enumerate(vec) if v == 1] or ["NONE"]

                reward = float(input_items[4][i])
                oracle = float(input_items[5][i])
                regret = float(input_items[6][i])

                est_raw = input_items[7][i]
                center_freq = float(est_raw.real)
                bandwidth   = float(est_raw.imag)

                confidence = float(input_items[8][i])
                interference_power = float(input_items[9][i])

                # parent state
                selected_channel = getattr(self.parent, "current_ch", -1)

                try:
                    DFS_state = self.parent.DFS_State
                except Exception:
                    DFS_state = {}

                if next_channel in DFS_state:
                    state = f"{DFS_state[next_channel]} : DFS Channel"
                else:
                    state = "Non-DFS Channel"

                # timestamp & counter
                now = int(time.time())

                # update parent.timestamp (most recent) and timestamps list safely
                if self.parent is not None:
                    try:
                        self.parent.timestamp = now
                    except Exception:
                        pass
                    # safe append / coerce
                    self._safe_append_timestamp(now)
                    # increment safe counter
                    cnt_val = self._safe_inc_count()
                else:
                    cnt_val = -1

                entry = {
                    "ts": now,
                    "count": cnt_val,
                    "next_channel": next_channel,
                    "selected_channel": selected_channel,
                    "reward": round(reward, 6),
                    "oracle": round(oracle, 6),
                    "regret": round(regret, 6),
                    "interferences": interferences,
                    "center_freq": center_freq,
                    "bandwidth": bandwidth,
                    "confidence": round(confidence, 3),
                    "interference_power": interference_power,
                    "Channel_State": state
                }

                # print JSON line for easy monitoring (watchdog uses run.log)
                try:
                    print(json.dumps(entry), flush=True)
                except Exception:
                    pass

                # buffer entry
                self._write_buf.append(entry)

                # flush occasionally
                if len(self._write_buf) >= self._flush_every:
                    self._flush()

                # Save checkpoint / emit marker at configured intervals, but DO NOT terminate process.
                if self.parent is not None and RESTART_EVERY > 0 and (getattr(self.parent, "count", 0) % RESTART_EVERY == 0):
                    try:
                        self._flush()
                    except Exception:
                        pass

                    if hasattr(self.parent, "save_state"):
                        try:
                            self.parent.save_state()
                        except Exception as e:
                            print("[DataLogger] save_state error:", e, flush=True)

                    # emit a clear, parseable marker so the external watchdog can detect a restart point
                    try:
                        print("[DataLogger] RESTART_POINT count={}".format(getattr(self.parent, "count", -1)), flush=True)
                    except Exception:
                        pass

        except Exception as e:
            print("[DataLogger] DataLogger exception:", e, flush=True)
            traceback.print_exc()

        return n
