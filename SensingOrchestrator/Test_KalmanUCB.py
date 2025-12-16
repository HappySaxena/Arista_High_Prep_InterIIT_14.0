# this module will be imported in the into your flowgraph
import numpy as np

class KalmanUCB:
    """Kalman UCB multi-arm selector"""
    def __init__(self, K, process_var=0.01, meas_var=0.1, beta=2.5):
        self.K = K
        self.beta = beta
        self.mu = np.zeros(K)
        self.P  = np.ones(K)
        self.Q = process_var
        self.R = meas_var

    def select(self, allowed=None):
        if allowed is None:
            allowed = list(range(self.K))
        allowed = list(allowed)
        if not allowed:
            return np.random.randint(0, self.K)

        ucbs = {i: self.mu[i] + self.beta*np.sqrt(self.P[i]) for i in allowed}
        best_val = max(ucbs.values())
        best_arms = [i for i,v in ucbs.items() if v==best_val]
        return int(np.random.choice(best_arms))

    def update(self, i, reward):
        P_pred = self.P[i] + self.Q
        K = P_pred / (P_pred + self.R)
        self.mu[i] = self.mu[i] + K*(reward - self.mu[i])
        self.P[i]  = (1 - K)*P_pred
        return self.mu[i]
