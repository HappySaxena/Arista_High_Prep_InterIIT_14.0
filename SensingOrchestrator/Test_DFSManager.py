
import numpy as np

class DFSManagerFull:
    """DFS channel NOP/CAC manager (Wi-Fi style)"""
    def __init__(self, dfs_channels, init_mix=0.5, cac_min=10, cac_max=60, idle_to_avail_prob=0.3, nop=1800):
        self.channels = list(dfs_channels)
        self.state = {}
        self.timer = {}
        self.idle_to_avail_prob = idle_to_avail_prob
        self.cac_min = cac_min
        self.cac_max = cac_max
        self.nop = nop

        for c in self.channels:
            if np.random.rand() < init_mix:
                self.state[c] = "IDLE"
                self.timer[c] = np.random.randint(cac_min, cac_max)
            else:
                self.state[c] = "AVAILABLE"
                self.timer[c] = 0

    def step(self):
        for c in self.channels:
            if self.state[c] == "RADAR":
                self.timer[c] -= 1
                if self.timer[c] <= 0:
                    self.state[c] = "IDLE"
                    self.timer[c] = np.random.randint(self.cac_min, self.cac_max)
            elif self.state[c] == "IDLE":
                self.timer[c] -= 1
                if self.timer[c] <= 0 or np.random.rand() < self.idle_to_avail_prob:
                    self.state[c] = "AVAILABLE"
                    self.timer[c] = 0
        return self.snapshot()

    def set_radar(self, c, nop=None):
        self.state[c] = "RADAR"
        self.timer[c] = self.nop if nop is None else nop

    def snapshot(self):
        return dict(self.state)
# this module will be imported in the into your flowgraph
