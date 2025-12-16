import numpy as np
from gnuradio import gr

class blk(gr.sync_block):
    """Realistic Microwave Oven Interference - Dynamic FFT Shaping v4"""

    def __init__(self,
                 samp_rate=1e6,
                 burst_rate=100,
                 duty_cycle=0.40,
                 noise_amp=1.0,
                 wobble_freq=3.0,
                 wobble_depth=0.12,
                 hump_width=0.55,     # <<< STRONG microwave-like width
                 hump_gain=15.0):     # <<< STRONG microwave-like intensity

        gr.sync_block.__init__(
            self,
            name='Microwave Oven (FFT-Shaped Dynamic v4)',
            in_sig=None,
            out_sig=[np.complex64]
        )

        # Store parameters
        self.samp_rate = samp_rate
        self.burst_rate = burst_rate
        self.duty_cycle = duty_cycle
        self.noise_amp = noise_amp
        self.wobble_freq = wobble_freq
        self.wobble_depth = wobble_depth
        self.hump_width = hump_width
        self.hump_gain = hump_gain

        # Burst timing
        self.samples_per_period = int(samp_rate / burst_rate)
        self.samples_on = int(self.samples_per_period * duty_cycle)
        self.counter = 0
        self.time_idx = 0
        self.phase = 0


    def work(self, input_items, output_items):
        out = output_items[0]
        n = len(out)

        # Time vector
        t = (np.arange(n) + self.time_idx) / self.samp_rate

        # 1 — Base noise (complex)
        noise = (np.random.normal(0, self.noise_amp, n) +
                 1j*np.random.normal(0, self.noise_amp, n))

        # 2 — FFT-shaping (dynamic, based on buffer size)
        F = np.fft.fftshift(np.fft.fft(noise))

        # Frequency axis for shaping curve
        freqs = np.linspace(-1, 1, n)

        # Microwave hump (Gaussian bell shape across spectrum)
        shape = 1 + self.hump_gain * np.exp(-(freqs / self.hump_width)**2)

        # 3 — Add MANY cavity-mode spikes (microwave textures)
        spikes = np.zeros(n)
        for _ in range(120):     # <<< Increased for realism
            pos = np.random.randint(0, n)
            spikes[pos] = np.random.uniform(0.4, 1.0)

        # Smooth the spikes (spread them a little)
        spikes = np.convolve(spikes, np.hanning(61), mode="same")

        # Combine with hump
        shape += spikes

        # Apply shaping window
        F *= shape

        # Back to time-domain
        shaped = np.fft.ifft(np.fft.ifftshift(F))

        # 4 — Magnetron wobble (cause center drift)
        wobble = self.wobble_depth * np.sin(2*np.pi*self.wobble_freq * t)
        phase_drift = np.cumsum(2*np.pi * wobble)
        shaped *= np.exp(1j * phase_drift)

        # 5 — Burst gating
        for i in range(n):
            if self.counter < self.samples_on:
                out[i] = np.complex64(shaped[i])
            else:
                out[i] = 0

            self.counter += 1
            if self.counter >= self.samples_per_period:
                self.counter = 0

        # Update time
        self.time_idx += n
        return n

