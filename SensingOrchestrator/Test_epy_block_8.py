#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from gnuradio import gr

class blk(gr.sync_block):
    """
    Smooth Stream → Vector Accumulator
    ----------------------------------
    Eliminates GR scheduler batching.

    - Accepts a raw stream of complex64 samples
    - Maintains an internal buffer
    - Emits vectors of length vec_len as soon as enough samples arrive
    - Never stalls upstream or downstream
    """

    def __init__(self, vec_len=2048):
        gr.sync_block.__init__(
            self,
            name="smooth_stream_to_vector",
            in_sig=[np.complex64],
            out_sig=[(np.complex64, vec_len)]
        )

        self.vec_len = vec_len
        self.buffer = np.zeros(0, dtype=np.complex64)

    def work(self, input_items, output_items):
        inp = input_items[0]
        out = output_items[0]

        nin = len(inp)
        nout = len(out)

        # append new samples to buffer
        if nin > 0:
            self.buffer = np.concatenate((self.buffer, inp))

        produced = 0

        # output as many full vectors as available
        while len(self.buffer) >= self.vec_len and produced < nout:
            out[produced][:] = self.buffer[:self.vec_len]
            self.buffer = self.buffer[self.vec_len:]
            produced += 1

        return produced
