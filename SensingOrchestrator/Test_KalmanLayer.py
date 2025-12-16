# this module will be imported in the into your flowgraph
import numpy as np

class Kalman1D:
    """Kalman 1D smoother"""
    def __init__(self, process_var=0.01, meas_var=0.1):
        self.x = None
        self.P = 1.0
        self.Q = process_var
        self.R = meas_var

    def update(self, z):
        if self.x is None:
            self.x = z
            return z

        x_pred = self.x
        P_pred = self.P + self.Q
        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K*(z - x_pred)
        self.P = (1 - K)*P_pred

        return self.x
