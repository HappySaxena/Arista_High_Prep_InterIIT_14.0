"""
Embedded Python Blocks:

Each time this file is saved, GRC will instantiate the first class it finds
to get ports and parameters of your block. The arguments to __init__  will
be the parameters. All of them are required to have default values!
"""

import numpy as np
from gnuradio import gr


class blk(gr.sync_block):  # other base classes are basic_block, decim_block, interp_block
    """Embedded Python Block example - a simple multiply const"""

    def __init__(self, example_param=1.0):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.sync_block.__init__(
            self,
            name='Embedded Python Block',   # will show up in GRC
            in_sig=[
                np.uint8,          # 0: change_flag
                np.float32,        # 1: duty_cycle
                np.complex64,      # 2: next_channel (real)
                (np.int16, 6),     # 3: interference vector
                np.float32,        # 4: reward
                np.float32,        # 5: oracle
                np.float32,        # 6: regret
                np.complex64,      # 7: est_center + j*bandwidth
                np.float32,        # 8: confidence
                np.float32         # 9: interference_power
            ],
            out_sig=[     # 1: duty_cycle
                ]
        )
        # if an attribute with the same name as a parameter is found,
        # a callback is registered (properties work, too).
        self.example_param = example_param

    def work(self, input_items, output_items):
        """example: multiply with constant"""
        output_items[0][:] = 1
        return 1
