#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time, json, warnings
import numpy as np
from gnuradio import gr

# ---------------------------------------------------------------------
#  1D Kalman smoother (kept unchanged)
# ---------------------------------------------------------------------
class KalmanLayer:
    def __init__(self, process_var=0.01, meas_var=0.1):
        self.x = None
        self.P = 1.0
        self.Q = process_var
        self.R = meas_var

    def update(self, z):
        z = float(z)
        if self.x is None:
            self.x = z
            return z
        P_pred = self.P + self.Q
        K = P_pred / (P_pred + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * P_pred
        return self.x


# ---------------------------------------------------------------------
#  Updated CUSUM + Duty Cycle block (fixed)
# ---------------------------------------------------------------------
class blk(gr.sync_block):
    """
    Channel Change Detection + Duty Cycle (stream JSON printing, ONLY sends what next block needs)
    Fixed issues:
      - use absolute sample counter (self.sample_counter) for cooldown checks
      - explicit output dtype for change flag (np.uint8)
      - robust slope calculation (suppresses RankWarning)
    """

    def __init__(self, alpha=0.1, base_thresh=3.0, cooldown=20, min_delta=1.0, slope_window=10):
        super().__init__(
            name='Change_Detect_Duty',
            in_sig=[np.float32],
            out_sig=[np.uint8, np.float32]     # change: use uint8 for flags
        )

        # change detection params
        self.ewma = None
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.alpha = float(alpha)
        self.base_thresh = float(base_thresh)
        self.cooldown = int(cooldown)
        self.min_delta = float(min_delta)
        self.slope_window = int(slope_window)
        self.last_trigger_index = None

        # hold recent raw values for rolling stats
        self.recent_vals = []

        # duty cycle storage (boolean history)
        self.power_hist = []

        # absolute sample counter across work() calls
        self.sample_counter = 0

    def work(self, input_items, output_items):
        vals = input_items[0]
        out_chg  = output_items[0]
        out_duty = output_items[1]

        n = len(vals)

        # Process each sample in the buffer
        for j in range(n):
            v = float(vals[j])

            # ----- Duty cycle calculation -----
            self.power_hist.append(1 if v > 0 else 0)
            if len(self.power_hist) > 50:
                # maintain sliding window of last 50 booleans
                del self.power_hist[0]

            duty = float(sum(self.power_hist) / len(self.power_hist)) if self.power_hist else 0.0
            out_duty[j] = duty

            # ----- Change detection using CUSUM -----
            self.recent_vals.append(v)
            if len(self.recent_vals) > 50:
                del self.recent_vals[0]

            rolling_std = float(np.std(self.recent_vals)) if len(self.recent_vals) >= 5 else 1.0
            thresh = float(self.base_thresh * rolling_std)

            if self.ewma is None:
                self.ewma = v
                out_chg[j] = 0
                self.sample_counter += 1
                continue

            # EWMA update
            self.ewma = self.alpha * v + (1 - self.alpha) * self.ewma
            delta = float(v - self.ewma)

            if abs(delta) < self.min_delta:
                out_chg[j] = 0
                self.sample_counter += 1
                continue

            # CUSUM update
            self.cusum_pos = max(0.0, self.cusum_pos + delta)
            self.cusum_neg = min(0.0, self.cusum_neg + delta)

            # slope test (use recent window)
            slope = 0.0
            if len(self.recent_vals) >= self.slope_window:
                try:
                    # suppress RankWarning from polyfit on flat data
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', np.RankWarning)
                        x = np.arange(self.slope_window)
                        y = self.recent_vals[-self.slope_window:]
                        slope = float(np.polyfit(x, y, 1)[0])
                except Exception:
                    slope = 0.0

            # Use absolute sample index for cooldown checking
            trigger = False
            if (self.cusum_pos > thresh) or (abs(self.cusum_neg) > thresh) or (abs(slope) > 0.5):
                if (self.last_trigger_index is None) or ((self.sample_counter - self.last_trigger_index) > self.cooldown):
                    self.last_trigger_index = self.sample_counter
                    trigger = True
                    self.cusum_pos = 0.0
                    self.cusum_neg = 0.0

            out_chg[j] = 1 if trigger else 0

            # increment global sample counter
            self.sample_counter += 1

            # Note: I intentionally removed streaming prints from inside the loop.
            # If you still want the JSON stream printed on each sample, re-enable below (careful: it's verbose).
            if trigger:
                print(json.dumps({"change_flag":"true","duty_cycle":round(duty,3)}), flush=True)
    

        return n
