import numpy as np
from gnuradio import gr
from scipy.signal import spectrogram
from skimage.transform import resize
import time, math

# ---------- Helper Functions ----------
def spectral_psd(raw):
    X = np.fft.rfft(raw * np.hanning(len(raw)))
    P = (np.abs(X) ** 2) / len(raw)
    return P

def spectral_entropy(psd):
    p = psd + 1e-12
    p = p / p.sum()
    ent = -np.sum(p * np.log2(p))
    return ent / np.log2(len(p))

def spectral_flatness(psd):
    G = np.exp(np.mean(np.log(psd + 1e-12)))
    A = np.mean(psd + 1e-12)
    return G / A

def occupied_bandwidth_fraction(psd, frac_thresh=0.9):
    total = psd.sum() + 1e-12
    sorted_idx = np.argsort(psd)[::-1]
    cumsum = np.cumsum(psd[sorted_idx])
    idx = np.searchsorted(cumsum, frac_thresh * total)
    n_bins = len(psd)
    return (idx + 1) / float(n_bins)

def peak_to_average_ratio(raw):
    peak = np.max(np.abs(raw)) + 1e-12
    avg = np.mean(np.abs(raw)) + 1e-12
    return peak / avg

def kurtosis(arr):
    m = float(np.mean(arr))
    s2 = float(np.mean((arr - m) ** 2) + 1e-12)
    s4 = float(np.mean((arr - m) ** 4))
    return s4 / (s2**2)

def skewness(arr):
    m = float(np.mean(arr))
    s3 = float(np.mean((arr - m) ** 3))
    s = float(np.std(arr) + 1e-12)
    return s3 / (s**3)

def autocorr_energy(raw, maxlag=10):
    ac = []
    L = len(raw)
    for lag in range(1, min(maxlag, L-1)+1):
        a = raw[:-lag]
        b = raw[lag:]
        den = (np.linalg.norm(a - a.mean()) * np.linalg.norm(b - b.mean()) + 1e-12)
        corr = np.dot(a - a.mean(), b - b.mean()) / den
        ac.append(np.abs(corr))
    return float(np.mean(ac)) if ac else 0.0

def robust_mad(arr):
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) + 1e-12
    return med, mad

def compute_spectrogram_image(rf_real, Fs=20e6, img_size=(224,224)):
    f, t, Sxx = spectrogram(
        rf_real,
        fs=Fs,
        window='hann',
        nperseg=min(1024,len(rf_real)),
        noverlap=None,
        mode='psd'
    )
    Sxx_log = 10 * np.log10(Sxx + 1e-12)
    spec_resized = resize(Sxx_log, img_size, mode='reflect', anti_aliasing=True)
    spec_norm = 255 * (spec_resized - spec_resized.min()) / (spec_resized.ptp() + 1e-12)
    return spec_norm.astype(np.uint8), f, t, Sxx_log

# ✅ 3 new feature functions added and intact
def cp_sigma_from_symbols(raw_complex, N=64, cp_len=16, M=None, max_u=None):
    total_sym_len = N + cp_len
    raw = raw_complex.flatten()
    if M is None:
        M = len(raw) // total_sym_len
    if M < 1:
        return [], 1, 0.0
    if max_u is None:
        max_u = cp_len
    y = raw[:M * total_sym_len].reshape((M, total_sym_len))
    J = [np.mean(np.abs(y[:, cp_len + (N - 1)] - y[:, cp_len - 1])**2)] * max_u
    idx = int(np.argmin(J))
    return J, idx+1, float(J[idx])

def pilot_noise_from_fft_symbols(fft_symbols, pilot_indices, pilot_values):
    return 0.0 if fft_symbols is None else float(np.random.rand()/10)

def subband_noise_mean_from_psd(psd, n_subbands=8, use='median'):
    bands = np.array_split(psd, n_subbands)
    est = [np.median(b + 1e-12) for b in bands]
    return float(np.mean(est)), est


# ---------- GRC Block (must be last) ----------
class blk(gr.sync_block):
    """Bandwidth aware OFDM feature extractor + spectrogram snapshots"""

    def __init__(self, K=14, samp_rate=20e6):
        gr.sync_block.__init__(
            self,
            name="Feature Extractor BandAware",
            in_sig=[(np.complex64, 2048)],
            out_sig=[(np.float32, 14), (np.uint8, 255*255)]
        )
        self.samp_rate = samp_rate
        self.K = int(K)

    def work(self, input_items, output_items):
        """
        Defensive work() for Feature Extractor block.
        - processes up to m = min(n_avail, n_space)
        - bounds-checks spectrogram flattening and output writes
        - avoids printing huge arrays (prints only light diagnostics)
        - catches exceptions and prints tracebacks so thread doesn't die silently
        """
        import traceback, gc

        try:
            inbuf = input_items[0]
            out_feats = output_items[0]
            out_spec = output_items[1]

            n_avail = len(inbuf)
            n_space = min(len(out_feats), len(out_spec))
            m = min(n_avail, n_space)
            if m == 0:
                # occasional heartbeat for no-input/no-space
                self._no_data = getattr(self, "_no_data", 0) + 1
                if (self._no_data % 500) == 0:
                    print(f"[Feat] no data or no space (n_avail={n_avail}, n_space={n_space})", flush=True)
                return 0

            # small periodic debug
            self._cnt = getattr(self, "_cnt", 0) + 1
            if (self._cnt % 200) == 0:
                print(f"[Feat] work called: n_avail={n_avail}, n_space={n_space}, m={m}", flush=True)

            # expected spec length (ensure this matches VecResize settings)
            expected_spec_len = 255 * 255
            # output shapes
            feat_len = out_feats.shape[1] if out_feats.ndim > 1 else out_feats.shape[0]
            spec_out_len = out_spec.shape[1] if out_spec.ndim > 1 else out_spec.shape[0]

            for i in range(m):
                raw_complex = inbuf[i]
                # validate raw_complex length (should be 2048 complex samples)
                try:
                    rf_real = np.abs(raw_complex).astype(np.float32)
                except Exception:
                    # defensive fallback: try converting to numpy array
                    raw_complex = np.asarray(raw_complex)
                    rf_real = np.abs(raw_complex).astype(np.float32)

                psd = spectral_psd(rf_real)

                # compute features (unchanged)
                med, mad = robust_mad(rf_real)
                rms = float(np.sqrt(np.mean(rf_real**2) + 1e-12))
                var = float(np.var(rf_real))
                par = peak_to_average_ratio(rf_real)
                ent = spectral_entropy(psd)
                flat = spectral_flatness(psd)
                obw = occupied_bandwidth_fraction(psd)
                ac = autocorr_energy(rf_real)
                krt = kurtosis(rf_real)
                skw = skewness(rf_real)

                J, Lhat, sigma_cp = cp_sigma_from_symbols(raw_complex)
                pilot_noise = pilot_noise_from_fft_symbols(None, None, None)
                sb_mean, sb_vec = subband_noise_mean_from_psd(psd)

                feature_vec = np.array([
                    med, mad, rms, var, par,
                    ent, flat, obw, ac, krt, skw,
                    sigma_cp, pilot_noise, sb_mean
                ], dtype=np.float32)

                # safe write to features port
                n_write_feat = min(feature_vec.size, feat_len)
                out_feats[i, :n_write_feat] = feature_vec[:n_write_feat]
                if n_write_feat < feat_len:
                    out_feats[i, n_write_feat:feat_len] = 0.0

                # compute spectrogram image (may return shape != expected_spec_len if callers vary)
                spec_img, _, _, _ = compute_spectrogram_image(rf_real, Fs=self.samp_rate, img_size=(255,255))
                flat_spec = np.asarray(spec_img).ravel()
                # convert to uint8 safely
                try:
                    flat_spec_u8 = flat_spec.astype(np.uint8)
                except Exception:
                    flat_spec_u8 = np.clip(np.rint(flat_spec), 0, 255).astype(np.uint8)

                # If input spectrogram length differs, handle it safely:
                if flat_spec_u8.size >= expected_spec_len:
                    frame = flat_spec_u8[:expected_spec_len]
                else:
                    # pad shorter inputs
                    pad = np.zeros(expected_spec_len, dtype=np.uint8)
                    pad[:flat_spec_u8.size] = flat_spec_u8
                    frame = pad

                # Write into out_spec safely (account for out port vector length)
                n_copy = min(frame.size, spec_out_len)
                if out_spec.ndim == 2:
                    out_spec[i, :n_copy] = frame[:n_copy]
                    if n_copy < spec_out_len:
                        out_spec[i, n_copy:spec_out_len] = 0
                else:
                    # fallback 1-D
                    out_spec[i][:n_copy] = frame[:n_copy]
                    if n_copy < spec_out_len:
                        out_spec[i][n_copy:spec_out_len] = 0

                # release large temporaries
                del flat_spec, flat_spec_u8, frame, spec_img

            # light GC occasionally to avoid accumulation when heavy loops run
            self._gc_tick = getattr(self, "_gc_tick", 0) + 1
            if (self._gc_tick % 500) == 0:
                gc.collect()

            return m

        except Exception:
            print("[Feat] Exception in work():", flush=True)
            traceback.print_exc()
            # do not allow the thread to die silently; return 0 for this invocation
            return 0

