import numpy as np
from gnuradio import gr

class blk(gr.sync_block):
    """BLE Burst FHSS Interference Generator (with full default values)"""

    def __init__(self,
                 samp_rate=4e6,       # Default: 4 MHz (works perfectly on PlutoSDR)
                 ble_bw=1e6,          # Default BLE channel width
                 hop_rate=1600,       # Default BLE hop rate (1600 hops/sec)
                 burst_ms=4,          # Default burst length = 4 ms
                 gain=0.8):           # Default interference strength

        gr.sync_block.__init__(
            self,
            name='BLE FHSS Burst Interference',
            in_sig=None,
            out_sig=[np.complex64]
        )

        # Save defaults
        self.samp_rate = samp_rate
        self.ble_bw = ble_bw
        self.hop_rate = hop_rate
        self.burst_ms = burst_ms
        self.gain = gain

        # Compute burst length in samples
        self.burst_len = int((burst_ms / 1000.0) * samp_rate)

        # Normalize BLE hopping positions (40 BLE channels across spectrum)
        self.channels = np.linspace(-0.5, 0.5, 40)

        self.t = 0                 # time index
        self.current_channel = 0   # current BLE hop index


    def _make_ble_mask(self, n, channel_pos):
        """Create BLE GFSK-like 1 MHz spectral mask centered at channel_pos."""
        freqs = np.linspace(-0.5, 0.5, n)
        bw_norm = self.ble_bw / self.samp_rate
        rolloff = 0.25

        mask = np.zeros(n)
        for i, f in enumerate(freqs):
            fshift = abs(f - channel_pos)

            if fshift <= (bw_norm/2)*(1-rolloff):
                mask[i] = 1.0
            elif fshift <= (bw_norm/2)*(1+rolloff):
                x = (fshift - (bw_norm/2)*(1-rolloff)) / (bw_norm*rolloff)
                mask[i] = 0.5 * (1 + np.cos(np.pi * x))

        return mask


    def work(self, input_items, output_items):
        out = output_items[0]
        n = len(out)

        #
        # 1) Burst ON/OFF pattern
        #
        if (self.t // self.burst_len) % 2 == 0:
            active = True
        else:
            active = False

        #
        # 2) BLE hopping logic
        #
        hop_interval_samples = int(self.samp_rate / self.hop_rate)
        if hop_interval_samples < 1:
            hop_interval_samples = 1

        if self.t % hop_interval_samples == 0:
            self.current_channel = np.random.randint(0, 40)

        #
        # 3) Generate white noise base
        #
        noise = (np.random.normal(0, 1, n) +
                 1j * np.random.normal(0, 1, n))

        #
        # 4) If burst active → shape interference into BLE channel
        #
        if active:
            F = np.fft.fftshift(np.fft.fft(noise))

            channel_pos = self.channels[self.current_channel]
            mask = self._make_ble_mask(n, channel_pos) * self.gain

            F *= mask
            shaped = np.fft.ifft(np.fft.ifftshift(F))
        else:
            shaped = np.zeros(n, dtype=np.complex64)

        out[:] = shaped.astype(np.complex64)
        self.t += n
        return n

