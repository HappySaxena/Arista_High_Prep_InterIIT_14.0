#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust VecResize: safely convert frames with input length VAR -> 255x255 -> resized 224x224
"""

import numpy as np
from PIL import Image
from gnuradio import gr
import sys, traceback, time

IMG_W_IN = 255
IMG_H_IN = 255
IMG_IN = IMG_W_IN * IMG_H_IN
IMG_W_OUT = 224
IMG_H_OUT = 224
IMG_OUT = IMG_W_OUT * IMG_H_OUT

def safe_work(func):
    def wrapper(self, input_items, output_items):
        try:
            return func(self, input_items, output_items)
        except Exception as e:
            try:
                print(f"[{self.__class__.__name__}] CRASH in work(): {e}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
            except Exception:
                pass
            time.sleep(0.01)
            return 0
    return wrapper

class blk(gr.sync_block):
    def __init__(self):
        gr.sync_block.__init__(
            self,
            name="VecResize_255_to_224",
            in_sig=[(np.uint8, IMG_IN)],
            out_sig=[(np.uint8, IMG_OUT)]
        )
        self._calls = 0
        self._multi_cnt = 0
        self._last_warn = 0

    @safe_work
    def work(self, input_items, output_items):
        inbuf = input_items[0]
        outbuf = output_items[0]
        n_avail = len(inbuf)
        n_space = len(outbuf) if outbuf.ndim == 1 else outbuf.shape[0]
        m = min(n_avail, n_space)
        if m == 0:
            return 0
        self._calls += 1
        if (self._calls % 1000) == 0:
            try: print(f"[VecResize] work called: n_avail={n_avail}, n_space={n_space}, calls={self._calls}", flush=True)
            except Exception: pass
        for i in range(m):
            vec = inbuf[i]
            try:
                arr = np.asarray(vec).ravel().astype(np.uint8)
            except Exception:
                try:
                    arr = np.frombuffer(bytes(vec), dtype=np.uint8).ravel()
                except Exception:
                    arr = np.zeros(IMG_IN, dtype=np.uint8)
            if arr.size >= IMG_IN:
                if arr.size != IMG_IN:
                    self._multi_cnt += 1
                    now = time.time()
                    if now - self._last_warn > 5.0:
                        try: print(f"[VecResize] warning: incoming length {arr.size} (expected {IMG_IN}) — using first frame", flush=True)
                        except Exception: pass
                        self._last_warn = now
                frame = arr[:IMG_IN]
            else:
                pad = np.zeros(IMG_IN, dtype=np.uint8)
                pad[:arr.size] = arr
                frame = pad
            # reshape robustly
            try:
                img_in = frame.reshape((IMG_H_IN, IMG_W_IN))
            except Exception:
                tmp = np.resize(frame, IMG_IN).astype(np.uint8)
                img_in = tmp.reshape((IMG_H_IN, IMG_W_IN))
            # resize with PIL
            try:
                pil = Image.fromarray(img_in, mode='L')
                pil_resized = pil.resize((IMG_W_OUT, IMG_H_OUT), resample=Image.BILINEAR)
                out_frame = np.asarray(pil_resized, dtype=np.uint8).ravel()
            except Exception:
                try:
                    arr2d = img_in
                    ys = np.linspace(0, IMG_H_IN - 1, IMG_H_OUT).astype(int)
                    xs = np.linspace(0, IMG_W_IN - 1, IMG_W_OUT).astype(int)
                    small = arr2d[np.ix_(ys, xs)]
                    out_frame = np.asarray(small, dtype=np.uint8).ravel()
                except Exception:
                    out_frame = np.zeros(IMG_OUT, dtype=np.uint8)
            if out_frame.size != IMG_OUT:
                if out_frame.size > IMG_OUT:
                    out_frame = out_frame[:IMG_OUT]
                else:
                    tmp = np.zeros(IMG_OUT, dtype=np.uint8)
                    tmp[:out_frame.size] = out_frame
                    out_frame = tmp
            # write to outbuf
            if outbuf.ndim == 2:
                try:
                    outbuf[i, :IMG_OUT] = out_frame
                except Exception:
                    for k in range(min(IMG_OUT, outbuf.shape[1])):
                        outbuf[i, k] = out_frame[k]
            else:
                start = i * IMG_OUT
                end = start + IMG_OUT
                total_len = outbuf.shape[0]
                if end <= total_len:
                    outbuf[start:end] = out_frame
                else:
                    fit = max(0, total_len - start)
                    if fit > 0:
                        outbuf[start:start+fit] = out_frame[:fit]
        return m
