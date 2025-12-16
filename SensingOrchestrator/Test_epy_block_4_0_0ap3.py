#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAB_Controller_Clean.py

Cleaned-up GNU Radio epy MAB controller:
 - Kalman-smoothed UCB bandit over channels
 - Safe model loading (pickle / dill / cloudpickle fallback)
 - Checkpoint save/load (atomic, periodic autosave)
 - Optional per-sample JSON log (MAB_JSON_PER_SAMPLE=1)
 - NO external JSON override of reward / oracle / regret
"""

import os
import time
import math
import json
import pickle
import traceback

import numpy as np
from gnuradio import gr

LOG_PATH = os.environ.get("MAB_JSON_LOG3", "/tmp/mab_samples3.jsonl")


# ---------------------------
# Small helpers
# ---------------------------

def map_14_to_5(x):
    """
    Map the 14-dim feature vector coming from GNU Radio into a
    smaller 5-dim feature vector for the RBN / reward model.

    For now we simply take the first 5 elements and pad with zeros
    if needed. You can customize this mapping later.
    """
    arr = np.asarray(x, dtype=float).reshape(-1)
    out = np.zeros(5, dtype=float)
    n = min(5, arr.size)
    out[:n] = arr[:n]
    return out


class SimpleKalman1D:
    """Very small scalar Kalman filter to smooth noisy rewards."""
    def __init__(self, q=0.1, r=1.0, x0=0.0, p0=1.0):
        self.q = float(q)
        self.r = float(r)
        self.x = float(x0)
        self.p = float(p0)

    def update(self, z):
        z = float(z)
        # predict
        self.p += self.q
        # update
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (z - self.x)
        self.p = (1.0 - k) * self.p
        return self.x


class KalmanUCB:
    """
    Simple UCB on top of a smoothed reward per-arm.
    mu[a] ~ mean reward of arm a (already smoothed by Kalman).
    """
    def __init__(self, K, beta=1.0):
        self.K = int(K)
        self.beta = float(beta)
        self.mu = np.zeros(self.K, dtype=float)
        self.n = np.zeros(self.K, dtype=float)
        self._t = 0.0

    def select(self, allowed_arms=None):
        if allowed_arms is None or len(allowed_arms) == 0:
            allowed_arms = list(range(self.K))

        # try each arm at least once
        for a in allowed_arms:
            if self.n[a] == 0:
                return a

        self._t += 1.0
        t = max(self._t, 1.0)
        best_arm = allowed_arms[0]
        best_val = -1e30

        for a in allowed_arms:
            if self.n[a] <= 0:
                ucb = 1e9
            else:
                bonus = math.sqrt(self.beta * math.log(t) / self.n[a])
                ucb = self.mu[a] + bonus
            if ucb > best_val:
                best_val = ucb
                best_arm = a
        return best_arm

    def update(self, arm, reward):
        a = int(arm)
        r = float(reward)
        if a < 0 or a >= self.K:
            return
        self.n[a] += 1.0
        eta = 1.0 / self.n[a]
        self.mu[a] = (1.0 - eta) * self.mu[a] + eta * r


class RBNPredictor:
    """
    Small wrapper so that legacy pickled models that expect
    a callable object still work. You can replace this with
    your own model class.
    """
    def __init__(self, w=None, b=None):
        self.w = np.asarray(w) if w is not None else None
        self.b = np.asarray(b) if b is not None else None

    def __call__(self, x):
        try:
            xr = np.asarray(x)
            if xr.ndim == 1:
                xr = xr.reshape(1, -1)
            if hasattr(self, "predict") and callable(self.predict):
                return self.predict(xr)
            if self.w is not None:
                out = xr.dot(self.w)
                if self.b is not None:
                    out = out + self.b
                return out
        except Exception:
            pass
        return np.zeros((xr.shape[0],), dtype=float)


# ---------------------------
# MAB controller block
# ---------------------------

class MAB_Controller(gr.sync_block):
    def __init__(self, model_path="", hop_delay=0.5, parent=None,
                 checkpoint_path="/tmp/mab_state3.pkl",
                 autosave_every_secs=3, autosave_every_samples=20,
                 print_every=200, summary_interval=2.0):

        gr.sync_block.__init__(
            self,
            name="MAB_Controller_Clean",
            in_sig=[(np.float32, 14)],
            out_sig=[np.complex64, np.float32, np.float32, np.float32],
        )

        self.parent = parent
        self.hop_delay = float(hop_delay)
        self._gc_interval = 50

        # ------------- Channel / arm mapping -------------
        try:
            nd24 = list(parent.NON_DFS_2_4_GHz)
        except Exception:
            nd24 = []
        try:
            nd5 = list(parent.NON_DFS_5_GHz)
        except Exception:
            nd5 = []
        try:
            dfs_keys = sorted(map(int, getattr(parent, "DFS_State", {}).keys()))
        except Exception:
            dfs_keys = []

        self.arm_to_ch = list(nd24) + list(nd5) + list(dfs_keys)
        if not self.arm_to_ch:
            self.arm_to_ch = [1, 6, 11]  # fallback
        self.K = max(1, len(self.arm_to_ch))
        self.ch_to_arm = {ch: i for i, ch in enumerate(self.arm_to_ch)}

        # DFS stub (you can wire to a real DFS manager later)
        self.dfs = type(
            "dfs",
            (),
            {
                "state": {c: "AVAILABLE" for c in self.arm_to_ch},
                "step": lambda self=None: None,
                "filter_sense_log": lambda self, chs: chs,
            },
        )()

        # ------------- Bandit state -------------
        self.ucb = KalmanUCB(self.K, beta=1.0)
        self.kf_scalar = [SimpleKalman1D(q=0.1, r=1.0, x0=0.0, p0=1.0) for _ in range(self.K)]

        # High-level controller state
        self.oracle = 0.0
        self.init_sweep = True
        self.sweep = 0

        # ------------- Checkpoint / autosave -------------
        self._checkpoint_path = os.environ.get("MAB_CHECKPOINT", checkpoint_path)
        self._autosave_secs = float(autosave_every_secs)
        self._autosave_samples = int(autosave_every_samples)
        self._last_autosave_time = time.time()
        self._samples_since_autosave = 0

        # ------------- Logging / summaries -------------
        self._print_tick = 0
        self._print_every = int(print_every)
        self._last_summary_time = 0.0
        self._summary_interval = float(summary_interval)

        # ------------- Optional JSON per-sample log -------------
        self._json_per_sample = int(os.environ.get("MAB_JSON_PER_SAMPLE", "0")) == 1

        # ------------- Model load -------------
        self.rbn = None
        self._model_path = model_path or ""

        # Expose RBNPredictor in __main__ so old pickles can unpickle
        try:
            import __main__ as _main
            if not hasattr(_main, "RBNPredictor"):
                _main.RBNPredictor = RBNPredictor
        except Exception:
            pass

        self._attempt_model_load(self._model_path)

        # ------------- Optional checkpoint restore -------------
        _load_flag = os.environ.get("MAB_LOAD_ON_RESTART", "0").strip()
        if _load_flag == "1":
            try:
                if os.path.exists(self._checkpoint_path):
                    self.load_state(self._checkpoint_path)
                    print("[MAB] Loaded checkpoint (restored previous state)", flush=True)
            except Exception as e:
                print("[MAB] checkpoint load failed:", e, flush=True)
        else:
            print("[MAB] Cold start (not restoring checkpoint).", flush=True)

    # ---------------------------
    # Model load
    # ---------------------------
    def _attempt_model_load(self, model_path):
        if not model_path:
            return
        try:
            # plain pickle
            try:
                with open(model_path, "rb") as fh:
                    self.rbn = pickle.load(fh)
                    print("[MAB] Loaded model (pickle):", model_path, flush=True)
                    return
            except Exception:
                pass

            # dill
            try:
                import dill
                with open(model_path, "rb") as fh:
                    self.rbn = dill.load(fh)
                    print("[MAB] Loaded model (dill):", model_path, flush=True)
                    return
            except Exception:
                pass

            # cloudpickle
            try:
                import cloudpickle
                with open(model_path, "rb") as fh:
                    self.rbn = cloudpickle.load(fh)
                    print("[MAB] Loaded model (cloudpickle):", model_path, flush=True)
                    return
            except Exception:
                pass

            # fallback: plain pickle again
            with open(model_path, "rb") as fh:
                self.rbn = pickle.load(fh)
                print("[MAB] Loaded model (fallback pickle):", model_path, flush=True)
                return
        except Exception as e:
            print("[MAB] model load failed, continuing without rbn:", e, flush=True)
            self.rbn = None

    # ---------------------------
    # Atomic checkpoint write
    # ---------------------------
    def _atomic_write(self, path, obj):
        tmp = path + ".tmp"
        try:
            with open(tmp, "wb") as fh:
                pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
            return True
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            except Exception:
                pass
            return False

    # ---------------------------
    # Checkpoint save/load
    # ---------------------------
    def save_state(self, path=None):
        if path is None:
            path = self._checkpoint_path
        try:
            state = {
                "mu": getattr(self.ucb, "mu", None),
                "n": getattr(self.ucb, "n", None),
                "beta": getattr(self.ucb, "beta", None),
                "kf_scalar_state": [getattr(k, "x", None) for k in self.kf_scalar],
                "oracle": getattr(self, "oracle", None),
                "sweep": getattr(self, "sweep", None),
                "init_sweep": getattr(self, "init_sweep", None),
                "arm_to_ch": getattr(self, "arm_to_ch", None),
                "dfs_state": getattr(self.dfs, "state", None),
                "timestamp": time.time(),
            }
            ok = self._atomic_write(path, state)
            if ok:
                print("[MAB] state saved ->", path, flush=True)
            else:
                print("[MAB] state save failed (atomic write error)", flush=True)
            return ok
        except Exception as e:
            print("[MAB] save_state exception:", e, flush=True)
            return False

    def load_state(self, path):
        try:
            with open(path, "rb") as fh:
                state = pickle.load(fh)

            if "mu" in state and state["mu"] is not None:
                try:
                    self.ucb.mu = np.array(state["mu"], dtype=float)
                except Exception:
                    pass
            if "n" in state and state["n"] is not None:
                try:
                    self.ucb.n = np.array(state["n"], dtype=float)
                except Exception:
                    pass
            if "beta" in state and state["beta"] is not None:
                try:
                    self.ucb.beta = float(state["beta"])
                except Exception:
                    pass

            if "kf_scalar_state" in state and state["kf_scalar_state"] is not None:
                for k, v in zip(self.kf_scalar, state["kf_scalar_state"]):
                    try:
                        k.x = float(v) if v is not None else k.x
                    except Exception:
                        pass

            if "oracle" in state:
                try:
                    self.oracle = float(state["oracle"])
                except Exception:
                    pass
            if "sweep" in state:
                try:
                    self.sweep = int(state["sweep"])
                except Exception:
                    pass
            if "init_sweep" in state:
                try:
                    self.init_sweep = bool(state["init_sweep"])
                except Exception:
                    pass
            if "arm_to_ch" in state and state["arm_to_ch"] is not None:
                try:
                    self.arm_to_ch = list(state["arm_to_ch"])
                    self.K = max(1, len(self.arm_to_ch))
                    self.ch_to_arm = {ch: i for i, ch in enumerate(self.arm_to_ch)}
                except Exception:
                    pass
            if "dfs_state" in state and state["dfs_state"] is not None:
                try:
                    self.dfs.state = dict(state["dfs_state"])
                except Exception:
                    pass

            print("[MAB] successfully restored checkpoint fields", flush=True)
            return True
        except Exception as e:
            print("[MAB] load_state exception:", e, flush=True)
            return False

    def autosave(self):
        now = time.time()
        self._samples_since_autosave += 1
        if (self._samples_since_autosave >= self._autosave_samples) or (
            now - self._last_autosave_time >= self._autosave_secs
        ):
            try:
                self.save_state()
            except Exception:
                pass
            self._last_autosave_time = now
            self._samples_since_autosave = 0

    # ---------------------------
    # Reward prediction
    # ---------------------------
    def predict_reward(self, feat, arm):
        """
        Compute reward for the given feature vector + arm.

        - Uses the RBN model if available.
        - Otherwise uses a simple, stable fallback based on the first feature.
        """
        try:
            arm = int(arm)
            if arm < 0 or arm >= self.K:
                arm = 0
            x5 = map_14_to_5(feat)
            if self.rbn is not None:
                try:
                    v = self.rbn(x5.reshape(1, -1))
                    arr = np.asarray(v)
                    if arr.size == 0:
                        return 0.0
                    return float(arr.reshape(-1)[0])
                except Exception:
                    pass

            # Fallback reward: larger when |x0| is small
            xs0 = float(x5[0])
            return float(1.0 / (1.0 + abs(xs0)))
        except Exception as e:
            print("[MAB] predict_reward internal error; fallback=0.0:", str(e), flush=True)
            return 0.0

    # ---------------------------
    # Optional JSON logging
    # ---------------------------
    def _log_sample_json(self, entry):
        if not self._json_per_sample:
            return
        try:
            line = json.dumps(entry, separators=(",", ":")) + "\n"
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass

    # ---------------------------
    # Main work loop
    # ---------------------------
    def work(self, input_items, output_items):
        try:
            if not input_items or len(input_items[0]) == 0:
                self._hb = getattr(self, "_hb", 0) + 1
                if (self._hb % 1000) == 0:
                    print(f"[MAB] heartbeat (no input) tick={self._hb}", flush=True)
                return 0

            inbuf = input_items[0]
            out_ch, out_rew, out_orc, out_reg = output_items
            n_avail = len(inbuf)
            n_space = min(len(out_ch), len(out_rew), len(out_orc), len(out_reg))
            m = min(n_avail, n_space)
            if m == 0:
                self._starve = getattr(self, "_starve", 0) + 1
                if (self._starve % 200) == 0:
                    print(f"[MAB] output starved; starve_tick={self._starve}", flush=True)
                return 0

            for i in range(m):
                feat = inbuf[i]

                # ---- Select arm / channel ----
                allowed_arms = list(range(self.K))
                try:
                    # if you later integrate real DFS state, filter here
                    allowed_arms = [
                        a for a in allowed_arms
                        if self.dfs.state.get(self.arm_to_ch[a], "AVAILABLE") == "AVAILABLE"
                    ]
                    if not allowed_arms:
                        allowed_arms = list(range(self.K))
                except Exception:
                    allowed_arms = list(range(self.K))

                arm = self.ucb.select(allowed_arms)
                arm = int(arm)
                if arm < 0 or arm >= self.K:
                    arm = 0
                next_ch = int(self.arm_to_ch[arm])

                try:
                    if self.parent is not None:
                        self.parent.current_ch = next_ch
                except Exception:
                    pass

                # ---- Reward / oracle / regret ----
                rew = self.predict_reward(feat, arm)

                # First-ever sample: initialize oracle
                if getattr(self, "_t", 0) == 0 and self.oracle == 0.0:
                    self.oracle = rew
                    regret = 0.0
                else:
                    if rew > self.oracle:
                        self.oracle = rew
                    regret = self.oracle - rew

                # ---- UCB update: use reward directly (no sign flip) ----
                rs = self.kf_scalar[arm].update(rew)
                self.ucb.update(arm, rs)

                # ---- Autosave / logging ----
                self.autosave()

                self._print_tick += 1
                if (self._print_tick % self._print_every) == 0:
                    try:
                        print(
                            f"[MAB] ch={next_ch} reward={rew:.4f} "
                            f"oracle={self.oracle:.4f} reg={regret:.4f}",
                            flush=True,
                        )
                    except Exception:
                        pass

                entry = {
                    "ts": time.time(),
                    "idx": int(getattr(self, "_t", 0)),
                    "channel": int(next_ch),
                    "arm": int(arm),
                    "reward": float(rew),
                    "oracle": float(self.oracle),
                    "regret": float(regret),
                }
                self._log_sample_json(entry)
                self._t = getattr(self, "_t", 0) + 1

                # ---- Write outputs ----
                try:
                    out_ch[i] = np.complex64(complex(next_ch, 0.0))
                    out_rew[i] = np.float32(rew)
                    out_orc[i] = np.float32(self.oracle)
                    out_reg[i] = np.float32(regret)
                except Exception:
                    pass

                try:
                    self.dfs.step()
                except Exception:
                    pass

                # Occasionally free memory
                self._mem_tick = getattr(self, "_mem_tick", 0) + 1
                if (self._mem_tick % self._gc_interval) == 0:
                    import gc
                    gc.collect()

            return m

        except Exception:
            print("[MAB] Exception in work():", flush=True)
            traceback.print_exc()
            return 0
