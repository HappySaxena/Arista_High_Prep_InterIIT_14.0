import numpy as np
from gnuradio import gr

class blk(gr.sync_block):
    """
    Multi-FHSS Interference Generator (Simulation Only)
    - Works with QT sinks for visualization
    - All defaults handled internally
    """

    def __init__(self,
                 samp_rate=4e6,        # Default: 4 MHz
                 num_hoppers=5,        # Default: 5 FHSS devices
                 hop_rate=1000,        # Default: 1000 hops/sec
                 num_channels=40,      # Default: 40 channels
                 tone_bw=300e3,        # Default: 300 kHz spike width
                 gain=1.0):            # Default: amplitude

        gr.sync_block.__init__(
            self,
            name='Multi-FHSS Interference (Default)',
            in_sig=None,
            out_sig=[np.complex64]
        )

        # Store defaults
        self.samp_rate = samp_rate
        self.num_hoppers = num_hoppers
        self.hop_rate = hop_rate
        self.num_channels = num_channels
        self.tone_bw = tone_bw
        self.gain = gain

        # Hop interval in samples
        self.hop_interval = max(1, int(self.samp_rate / self.hop_rate))

        # FHSS channel positions normalized (-0.5 to +0.5)
        self.channels = np.linspace(-0.5, 0.5, self.num_channels)

        # Per-hopper random states
        self.current_channel = np.random.randint(0, self.num_channels, self.num_hoppers)
        self.t = 0


    def _make_mask(self, nfft, pos):
        """Gaussian spike mask at normalized channel position 'pos'"""
        f = np.linspace(-0.5, 0.5, nfft)
        bw = self.tone_bw / self.samp_rate
        sigma = bw / 3
        return np.exp(-0.5 * ((f - pos) / sigma) ** 2)


    def work(self, input_items, output_items):
        out = output_items[0]
        n = len(out)

        result = np.zeros(n, dtype=np.complex64)

        # Generate interference from each FHSS device
        for h in range(self.num_hoppers):

            # Hop on boundary
            if self.t % self.hop_interval == 0:
                self.current_channel[h] = np.random.randint(0, self.num_channels)

            # Base noise
            noise = (np.random.normal(0, 1, n) +
                     1j * np.random.normal(0, 1, n))

            # Frequency shaping
            F = np.fft.fftshift(np.fft.fft(noise))
            mask = self._make_mask(n, self.channels[self.current_channel[h]])
            F *= mask * self.gain
            shaped = np.fft.ifft(np.fft.ifftshift(F))

            result += shaped

        # Normalize
        m = np.max(np.abs(result))
        if m > 0:
            result /= m

        out[:] = result.astype(np.complex64)
        self.t += n
        return n

