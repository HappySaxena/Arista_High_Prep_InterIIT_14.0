# this module will be imported in the into your flowgraph
import numpy as np

def rbf_kernel(A, B, g):
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    AA = np.sum(A * A, axis=1)[:, None]
    BB = np.sum(B * B, axis=1)[None, :]
    AB = A.dot(B.T)
    d2 = AA + BB - 2 * AB
    return np.exp(-g * d2)

class RBFNReward:
    """RBFN-based reward predictor (callable in GRC after import)"""

    def __init__(self, centers=None, gamma=1.0, ridge_lambda=1e-6, ridge_dim=None):
        # Default centers so GRC accepts this as a parameter
        # You can override centers later from training/initial decision block
        if centers is None:
            if ridge_dim is None:
                ridge_dim = 4  # fallback
            self.centers = np.zeros((ridge_dim, ridge_dim), dtype=np.float32)
        else:
            self.centers = np.array(centers, dtype=np.float32)

        self.gamma = float(gamma)
        self.ridge_lambda = float(ridge_lambda)

    def train(self, X, y):
        X = np.atleast_2d(X)
        y = np.array(y, dtype=np.float32)

        Kmat = rbf_kernel(X, self.centers, self.gamma)
        ridge_dim = Kmat.shape[1]
        A_mat = Kmat.T.dot(Kmat) + self.ridge_lambda * np.eye(ridge_dim)
        b_vec = Kmat.T.dot(y)

        try:
            self.w = np.linalg.solve(A_mat, b_vec)
        except np.linalg.LinAlgError as e:
            print("⚠ RBFN train solve failed:", e)
            self.w = None

        return self.w

    def predict(self, Xq):
        Xq = np.atleast_2d(Xq)
        Kx = rbf_kernel(Xq, self.centers, self.gamma)
        if hasattr(self, "w") and self.w is not None:
            out = Kx.dot(self.w).flatten()
        else:
            out = np.zeros(Kx.shape[0], dtype=np.float32)

        # Return a reward vector that GRC can handle if connected to message/stream block
        return out.tolist()


# --- Standalone functional interface for direct import/use outside class (optional) ---

def build_predictor(X, centers, gamma, ridge_lambda, y):
    """Train once and return predictor function"""
    X = np.atleast_2d(X)
    Kmat = rbf_kernel(X, centers, gamma)
    dim = Kmat.shape[1]
    A_mat = Kmat.T @ Kmat + ridge_lambda * np.eye(dim)
    b_vec = Kmat.T @ y

    try:
        w = np.linalg.solve(A_mat, b_vec)
    except np.linalg.LinAlgError as e:
        print("⚠ Solve failed:", e)
        return None

    def predictor(Xq):
        Xq = np.atleast_2d(Xq)
        Kx = rbf_kernel(Xq, centers, gamma)
        return Kx.dot(w).flatten()

    return predictor
