"""
DFS Monitor Signal Generator (STANDARD DFS TEST WAVEFORM)

Generates radar-like pulses used as the DFS detection input.
Follows standard DFS radar characteristics:
- Pulse width: 1 µs
- PRF: 300 Hz
- Amplitude: 0.1
- Noise floor: 1e-7
- Sample rate: 20 MHz
"""

import numpy as np
from gnuradio import gr
import math

class blk(gr.sync_block):
    def __init__(self,
                 sample_rate=20e6,          # STANDARD DFS test bandwidth
                 pulse_width_us=1.0,        # 1 µs pulse width
                 prf_hz=300.0,              # Pulse repetition frequency
                 pulse_amplitude=0.1,       # Radar peak amplitude
                 noise_power=1e-7):         # Noise floor power
        gr.sync_block.__init__(
            self,
            name="DFS Monitor (Standard Radar)",
            in_sig=None,
            out_sig=[np.complex64]
        )

        self.sample_rate = float(sample_rate)
        self.pulse_width_seconds = float(pulse_width_us) * 1e-6
        self.prf_hz = float(prf_hz)
        self.pulse_amplitude = float(pulse_amplitude)
        self.noise_power = float(noise_power)

        # Derived parameters
        self.pulse_samples = int(self.pulse_width_seconds * self.sample_rate)
        self.samples_per_pulse_period = int(self.sample_rate / self.prf_hz)

        # Internal counter
        self._sample_counter = 0

    def work(self, input_items, output_items):
        out = output_items[0]
        n = len(out)

        # Time vector
        t = (self._sample_counter + np.arange(n)) / self.sample_rate

        # Base noise floor
        sigma = math.sqrt(self.noise_power / 2.0)
        noise = (np.random.normal(scale=sigma, size=n) +
                 1j * np.random.normal(scale=sigma, size=n))

        # Start with noise only
        sig = noise.astype(np.complex64)

        # Radar pulse periodicity logic
        for i in range(n):
            sample_index = self._sample_counter + i
            pos_in_period = sample_index % self.samples_per_pulse_period

            if pos_in_period < self.pulse_samples:
                # Insert radar pulse (simple CW pulse)
                sig[i] += self.pulse_amplitude

        # Output
        out[:] = sig
        self._sample_counter += n
        return n
