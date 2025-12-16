import numpy as np
from gnuradio import gr
from collections import deque

class blk(gr.sync_block):
    """
    Raw-signal interference power estimator.
    Input:  complex64 IQ samples
    Output: float32 per-sample interference power (Pi)

    Pi = ShortTermPower - NoiseFloor
    """

    def __init__(self,
                 short_win=256,      # short-term energy window
                 noise_win=4096,     # long-term noise estimator window
                 alpha=0.05):        # smoothing factor
        gr.sync_block.__init__(
            self,
            name='RAW_InterferenceEstimator',
            in_sig=[np.complex64],
            out_sig=[np.float32]
        )

        # Parameters
        self.short_win = int(short_win)
        self.noise_win = int(noise_win)
        self.alpha = float(alpha)

        # Buffers
        self.short_buf = deque(maxlen=self.short_win)
        self.noise_buf = deque(maxlen=self.noise_win)

        # State variables
        self.noise_floor = 0.0
        self.last_Pi = 0.0

    def work(self, input_items, output_items):
        iq = input_items[0]
        out = output_items[0]
        n = len(iq)

        for i in range(n):
            s = iq[i]

            # instantaneous magnitude squared
            p_inst = float((s.real * s.real) + (s.imag * s.imag))

            # short-term window power
            self.short_buf.append(p_inst)
            Pshort = float(np.mean(self.short_buf)) if len(self.short_buf) > 1 else p_inst

            # update long-term noise estimate (slow varying)
            self.noise_buf.append(p_inst)
            Pnoise_raw = float(np.median(self.noise_buf))

            # EMA smoothing noise floor
            self.noise_floor = (1 - self.alpha) * self.noise_floor + self.alpha * Pnoise_raw

            # Interference power
            Pi = Pshort - self.noise_floor
            if Pi < 0:        # clamp numerical negatives
                Pi = 0.0

            self.last_Pi = Pi
            out[i] = Pi

        return n
