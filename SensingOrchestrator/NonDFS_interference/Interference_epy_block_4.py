import numpy as np
from gnuradio import gr

class blk(gr.sync_block):
    """
    Realistic DFS Radar Signal (Type-1)
    Simulation-only radar pulses for visualization in QT sinks.
    DEFAULT PARAMETERS IMPLEMENT FCC/ETSI DFS RADAR TYPE-1.
    """

    def __init__(self,
                 samp_rate=20e6,          # 20 MHz required for 1 µs resolution
                 pulse_width_us=1.0,      # Pulse width (1 µs)
                 pulses_per_burst=9,      # DFS Type 1 = 9 pulses
                 pri_us=1428.0,           # Pulse Repetition Interval (1428 µs)
                 burst_interval_ms=1000,  # One burst every 1 second
                 radar_bw=3e6,            # ~3 MHz radar bandwidth
                 gain=1.0):               # amplitude
        gr.sync_block.__init__(
            self,
            name='DFS Radar (Realistic Type-1)',
            in_sig=None,
            out_sig=[np.complex64]
        )

        # Save defaults
        self.samp_rate = samp_rate
        self.pulse_width = int((pulse_width_us / 1e6) * samp_rate)
        self.pulses_per_burst = pulses_per_burst
        self.pri = int((pri_us / 1e6) * samp_rate)
        self.burst_interval = int((burst_interval_ms / 1000.0) * samp_rate)
        self.radar_bw = radar_bw
        self.gain = gain

        # Internal time index
        self.t = 0


    def work(self, input_items, output_items):
        out = output_items[0]
        n = len(out)

        # Output buffer
        radar = np.zeros(n, dtype=np.complex64)

        # Local time vector
        t_local = np.arange(n)

        # Current time inside repeating burst cycle
        cycle_pos = (self.t + t_local) % self.burst_interval

        # For each pulse in the burst
        for p in range(self.pulses_per_burst):
            # Pulse start time in samples
            pulse_start = p * self.pri
            pulse_end = pulse_start + self.pulse_width

            # Boolean mask for pulse presence
            pulse_mask = np.logical_and(cycle_pos >= pulse_start,
                                        cycle_pos < pulse_end)

            # Radar pulse = complex amplitude (carrier is baseband)
            radar[pulse_mask] = self.gain + 0j

        # BANDWIDTH SHAPING (Radar ≈ 3 MHz)
        if self.radar_bw > 0:
            # Create colored noise
            noise = (np.random.normal(0, 0.05, n) +
                     1j * np.random.normal(0, 0.05, n)).astype(np.complex64)

            # Shape noise with Gaussian spectral mask
            freqs = np.linspace(-self.samp_rate/2, self.samp_rate/2, n)
            mask = np.exp(-0.5 * (freqs / (self.radar_bw/2))**2)

            F = np.fft.fftshift(np.fft.fft(noise))
            F *= mask
            shaped_noise = np.fft.ifft(np.fft.ifftshift(F))

            radar += shaped_noise.astype(np.complex64)

        # Write output
        out[:] = radar
        self.t += n
        return n

