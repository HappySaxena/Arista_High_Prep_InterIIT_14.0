import numpy as np
from gnuradio import gr

class blk(gr.sync_block):
    """OFDM Raw Generator – continuous complex symbols"""

    def __init__(self, samp_rate=20e6):
        gr.sync_block.__init__(
            self,
            name='OFDM Raw Src',
            in_sig=None,
            out_sig=[np.complex64]
        )
        self.samp_rate = samp_rate
        self.phase = 0.0

    def work(self, input_items, output_items):
        out = output_items[0]
        Nout = len(out)            # how many samples GNURadio wants this call

        n = np.arange(Nout)
        freq_offset = 100e3
        symbol = np.exp(1j * (2*np.pi*freq_offset*n/self.samp_rate + self.phase))

        self.phase += 0.1

        out[:] = symbol.astype(np.complex64)
        return Nout
