import numpy as np
from gnuradio import gr

class blk(gr.sync_block):
    """
    Realistic DFS Radar Type-2 (FCC / ETSI Standard)
    - Pulse width: 5 us
    - PRI: 200 us
    - Pulses per burst: 10
    - Burst interval: 250 ms
    - Bandwidth ~6 MHz
    """

    def __init__(self,
                 samp_rate=20e6,          # 20 MHz required for 5 us pulse shaping
                 center_freq=5260e6,      # visual center frequency
                 pulse_width_us=5.0,      # DFS Type-2 official value: 5 us
                 pri_us=200.0,            # PRI = 200 us
                 pulses_per_burst=10,     # Type-2: 8–20; default = 10
                 burst_interval_ms=250,   # DFS standard: 250 ms burst interval
                 radar_bw=6e6,            # ~6 MHz radar spectral width
                 amplitude=1.0,           # pulse amplitude
                 phase_mod=True):         # phase jitter for realism

        gr.sync_block.__init__(
            self,
            name='DFS Radar Type-2 (Realistic Default)',
            in_sig=None,
            out_sig=[np.complex64]
        )

        self.samp_rate = float(samp_rate)
        self.center_freq = float(center_freq)
        self.pulse_width = max(1, int((pulse_width_us / 1e6) * self.samp_rate))
        self.pulses_per_burst = int(pulses_per_burst)
        self.pri = max(1, int((pri_us / 1e6) * self.samp_rate))
        self.burst_interval = max(1, int((burst_interval_ms / 1000.0) * self.samp_rate))
        self.radar_bw = float(radar_bw)
        self.amplitude = float(amplitude)
        self.phase_mod = bool(phase_mod)

        self.t = 0  # time counter

    def _make_pulse(self, length):
        """Generate a realistic 5-us radar pulse with FM jitter and shaping."""
        win = np.hanning(length * 2)[0:length]
        tvec = np.arange(length) / self.samp_rate

        # Tiny chirp inside pulse
        chirp_bw = self.radar_bw * 0.15  # small fraction of radar BW
        k = chirp_bw / (length / self.samp_rate)
        phase = 2*np.pi * (0.5 * k * tvec**2)

        if self.phase_mod:
            phase += np.random.randn(length) * 0.02

        pulse = (self.amplitude * win) * np.exp(1j * phase)

        return pulse.astype(np.complex64)

    def work(self, input_items, output_items):
        out = output_items[0]
        n = len(out)
        buf = np.zeros(n, dtype=np.complex64)

        t_local = self.t + np.arange(n)
        cycle_pos = t_local % self.burst_interval

        for p in range(self.pulses_per_burst):
            start = p * self.pri
            end = start + self.pulse_width

            mask = np.logical_and(cycle_pos >= start, cycle_pos < end)
            if not np.any(mask):
                continue

            pulse = self._make_pulse(self.pulse_width)
            idx = np.where(mask)[0]
            pos = (cycle_pos[idx] - start).astype(int)
            buf[idx] += pulse[pos]

        # add light clutter noise (band-limited)
        noise = (np.random.normal(0, 0.005, n) +
                 1j * np.random.normal(0, 0.005, n))

        F = np.fft.fftshift(np.fft.fft(noise))
        freqs = np.linspace(-0.5, 0.5, n, endpoint=False) * self.samp_rate
        sigma = self.radar_bw / 2
        mask = np.exp(-0.5 * (freqs / sigma)**2)
        F *= mask
        clutter = np.fft.ifft(np.fft.ifftshift(F))

        buf += 0.05 * clutter.astype(np.complex64)

        out[:] = buf
        self.t += n
        return n

