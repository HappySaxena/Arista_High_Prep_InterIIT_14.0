import numpy as np
from gnuradio import gr

class blk(gr.sync_block):
    """Takes a vector and outputs its mean"""

    def __init__(self):
        gr.sync_block.__init__(
            self,
            name="Vector Mean",
            in_sig=[(np.float32,  2048)],  # complex vector of length 2048
            out_sig=[np.float32]            # single float mean value
        )

    def work(self, input_items, output_items):
        vec = input_items[0][0]              # 2048 complex samples
        power = np.abs(vec) ** 2             # convert to magnitude^2
        mean_val = np.mean(power)            # compute mean

        output_items[0][0] = mean_val        # output single value
        return 1
