import numpy as np
from gnuradio import gr

class blk(gr.sync_block):
    """
    DFS Radar Combo (Type-1 + Type-2)
    REALISTIC DEFAULT VALUES (FCC/ETSI)
    - Alternates Type-1 and Type-2 radar every 1 second
    - Type-1: 1us PW, 9 pulses, 1428us PRI, 1s interval
    - Type-2: 5us PW, 10 pulses, 200us PRI, 250ms interval
    - Simulation-only, safe for QT sinks
    """

    def __init__(self,
                 samp_rate=20e6,             # REQUIRED for microsecond pulses
                 switch_interval_s=1.0,      # switch radar mode every 1 second

                 # ---------- TYPE 1 DEFAULTS ----------
                 t1_pulse_width_us=1.0,      # 1 µs
                 t1_pulses_per_burst=9,      # 9 pulses
                 t1_pri_us=1428.0,           # 1428 µs PRI
                 t1_burst_interval_ms=1000,  # burst every 1 sec
                 t1_bw=3e6,                  # 3 MHz radar BW

                 # ---------- TYPE 2 DEFAULTS ----------
                 t2_pulse_width_us=5.0,      # 5 µs
                 t2_pulses_per_burst=10,     # 10 pulses
                 t2_pri_us=200.0,            # 200 µs PRI
                 t2_burst_interval_ms=250,   # burst every 0.25 sec
                 t2_bw=6e6,                  # 6 MHz radar BW

                 amplitude=1.0):              # default power
        gr.sync_block.__init__(
            self,
            name='DFS Radar Combo (Type1+Type2)',
            in_sig=None,
            out_sig=[np.complex64]
        )

        self.samp_rate = float(samp_rate)
        self.switch_interval = int(samp_rate * switch_interval_s)
        self.amplitude = float(amplitude)

        # Store Type-1 parameters
        self.t1_pw = int((t1_pulse_width_us/1e6)*samp_rate)
        self.t1_ppb = int(t1_pulses_per_burst)
        self.t1_pri = int((t1_pri_us/1e6)*samp_rate)
        self.t1_burst = int((t1_burst_interval_ms/1000.0)*samp_rate)
        self.t1_bw = float(t1_bw)

        # Store Type-2 parameters
        self.t2_pw = int((t2_pulse_width_us/1e6)*samp_rate)
        self.t2_ppb = int(t2_pulses_per_burst)
        self.t2_pri = int((t2_pri_us/1e6)*samp_rate)
        self.t2_burst = int((t2_burst_interval_ms/1000.0)*samp_rate)
        self.t2_bw = float(t2_bw)

        self.t = 0  # internal time counter

    # --------------------------------------------------------------------
    def _make_pulse(self, length, bw):
        """Generate a shaped radar pulse with small chirp."""
        win = np.hanning(length*2)[0:length]
        tvec = np.arange(length) / self.samp_rate

        # small chirp for realism
        chirp_bw = bw * 0.1
        k = chirp_bw / (length / self.samp_rate)
        phase = 2*np.pi * (0.5*k*tvec*tvec)

        pulse = (self.amplitude * win) * np.exp(1j*phase)

        # tiny noise for realism
        pulse += (np.random.randn(length)*0.01 + 1j*np.random.randn(length)*0.01)

        return pulse.astype(np.complex64)

    # --------------------------------------------------------------------
    def _place_radar(self, buf, t_local, pw, ppb, pri, burst_interval, bw):
        """Place burst pulses into the buffer for one radar type."""
        cycle_pos = t_local % burst_interval
        template = self._make_pulse(pw, bw)

        for p in range(ppb):
            start = p * pri
            end = start + pw

            mask = np.logical_and(cycle_pos >= start, cycle_pos < end)
            if not np.any(mask):
                continue

            idx = np.where(mask)[0]
            pos = ((cycle_pos[idx]-start).astype(int))
            pos = np.clip(pos, 0, pw-1)

            buf[idx] += template[pos]

        return buf

    # --------------------------------------------------------------------
    def work(self, input_items, output_items):
        out = output_items[0]
        n = len(out)
        t_local = self.t + np.arange(n)

        buf = np.zeros(n, dtype=np.complex64)

        # Decide current mode (Type1 or Type2)
        mode = (self.t // self.switch_interval) % 2

        if mode == 0:
            # ---------------- TYPE 1 ACTIVE ----------------
            buf = self._place_radar(
                buf, t_local,
                self.t1_pw, self.t1_ppb,
                self.t1_pri, self.t1_burst, self.t1_bw
            )
        else:
            # ---------------- TYPE 2 ACTIVE ----------------
            buf = self._place_radar(
                buf, t_local,
                self.t2_pw, self.t2_ppb,
                self.t2_pri, self.t2_burst, self.t2_bw
            )

        out[:] = buf
        self.t += n
        return n

