import numpy as np
from gnuradio import gr

class blk(gr.sync_block):
    """Continuous ZigBee-like interference for 4 MHz sample rate"""

    def __init__(self,
                 samp_rate=4e6,        # Works with 4 MHz
                 center_freq=2.44e9,   # ZigBee Ch.18
                 chip_rate=2e6,        # ZigBee chips
                 gain=0.8):            # Interference power

        gr.sync_block.__init__(
            self,
            name='ZigBee Interference (Continuous)',
            in_sig=None,
            out_sig=[np.complex64]
        )

        self.samp_rate = samp_rate
        self.center_freq = center_freq
        self.chip_rate = chip_rate
        self.gain = gain

        # ZigBee 32-chip sequence (one example)
        self.sequence = np.array(
            [1,1,1,-1,1,1,-1,1,1,-1,-1,-1,-1, 1,-1, 1, 1,1,-1,-1,
             1,-1,-1, 1,-1, 1,-1,-1,-1, 1,1,1]
        )

        # samples per chip ( = 2 at 4MHz SR)
        self.sps = max(1, int(self.samp_rate / self.chip_rate))

        # pre-generate continuous baseband chipstream
        self.offset = 0


    def work(self, input_items, output_items):
        out = output_items[0]
        n = len(out)

        # Repeat chips enough times to fill n samples
        repeated = np.repeat(self.sequence, self.sps)

        # Continuous wraparound indexing
        data = np.zeros(n, dtype=np.complex64)
        for i in range(n):
            idx = (self.offset + i) % len(repeated)
            data[i] = repeated[idx]

        self.offset = (self.offset + n) % len(repeated)

        # random Q-phase (simple OQPSK-like)
        phase = np.exp(1j * np.pi * (np.random.rand() * 2 - 1))

        out[:] = (data * phase * self.gain).astype(np.complex64)
        return n
	
