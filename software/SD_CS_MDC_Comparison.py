#!/usr/bin/env python
"""
Compare two MDC matrix generation methods:
1. Current: Distance function + MDC_UMDC_Gen
2. Adaptive sigma: iterative sigma tuning to hit target M=16
Compares SNDR, actual CR, and runtime.
"""
import numpy as np, time, os, warnings, math
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, 'saved_results')
DATABASE_DIR = os.environ.get('SPIKE_DATA_DIR', os.path.join(BASE_DIR, '..', 'data'))
WINDOW_SIZE = 50
SEG_LEN = 2 * WINDOW_SIZE
CR_TARGET = 0.84
TARGET_M = int(SEG_LEN * (1 - CR_TARGET))  # 16
P_SIGMA = 0.4

# ====== MDC functions (Current method) ======
def Distance(input_signal, CR, p):
    _max, _min = np.max(input_signal), np.min(input_signal)
    N = len(input_signal)
    M = N * (1 - CR)
    ave_width = p * (_max - _min) / M
    return (_max - _min - (M - 1) * ave_width) / M

def MDC_UMDC_Gen(input_signal, sigma):
    N = len(input_signal)
    MDC = np.empty([1, N]); i = 0
    cluster = np.zeros(N); others = np.zeros(N)
    while i < N:
        if i == 0:
            core_data = input_signal[0]
            for j in range(N):
                if sigma - np.abs(core_data - input_signal[j]) >= 0:
                    cluster[j] = input_signal[j]
            others = input_signal - cluster
            MDC = cluster.reshape(1, N)
        else:
            core_data = core_data_next
            for j in core_data_ind:
                if sigma - np.abs(core_data - input_signal[j]) >= 0:
                    cluster[j] = input_signal[j]
            others = input_signal_new - cluster
            MDC = np.append(MDC, cluster)
        if not others.any(): break
        core_data_ind = np.where(others != 0)[0]
        core_data_next = input_signal[core_data_ind[0]]
        input_signal_new = others; i += 1
        cluster = np.zeros(N); others = np.zeros(N)
    return MDC.reshape(MDC.size // N, N)

def norm2(x): return np.linalg.norm(x, 2)

def CS_IRLS(y, T_Mat, m):
    hat_x_tp = T_Mat.T.dot(y)
    epsilong = 1; p = 1; times = 1
    max_iter = max(5, int(len(y) / 4))
    while (epsilong > 10e-9) and (times < max_iter):
        AA = hat_x_tp * hat_x_tp + epsilong
        BB = np.ones(AA.shape) * (p / 2 - 1)
        weight = AA ** BB
        n_y = T_Mat.shape[1]
        CC = np.ones((n_y, 1)) / weight
        Q_Mat = np.zeros((n_y, n_y))
        CC_flat = CC.flatten()
        for i in range(n_y):
            Q_Mat[i, i] = CC_flat[i] if i < len(CC_flat) else CC_flat[-1]
        A = Q_Mat.dot(T_Mat.T)
        B = T_Mat.dot(A)
        C = np.linalg.pinv(B).dot(y)
        hat_x = A.dot(C)
        if norm2(hat_x - hat_x_tp) < math.sqrt(epsilong) / 100:
            epsilong /= 10
        hat_x_tp = hat_x; times += 1
    return hat_x

def compute_sndr(re_signal, input_signal):
    error = input_signal - re_signal
    return 20 * math.log(norm2(input_signal) / norm2(error), 10)

# ====== Method 1: Current ======
def method_current(signal):
    t0 = time.time()
    sigma = Distance(signal, CR_TARGET, P_SIGMA)
    MDC = MDC_UMDC_Gen(signal, sigma)
    m = MDC.shape[0]
    com = np.dot(MDC, signal)
    recon = CS_IRLS(com.reshape(m, 1), MDC, SEG_LEN).flatten()
    sndr = compute_sndr(recon.reshape(-1, 1), signal.reshape(-1, 1))
    actual_cr = 1 - m / SEG_LEN
    elapsed = time.time() - t0
    return sndr, actual_cr, elapsed, m

# ====== Method 2: Adaptive sigma ======
def method_adaptive(signal, target_m=TARGET_M):
    t0 = time.time()
    sigma = 0.1
    max_iter = 100
    for _ in range(max_iter):
        MDC = MDC_UMDC_Gen(signal, sigma)
        m = MDC.shape[0]
        if m == target_m:
            break
        elif m > target_m:
            sigma += 0.05  # too many clusters, widen threshold
        else:
            sigma -= 0.05  # too few clusters, narrow threshold
        if sigma <= 0:
            sigma = 0.01
    com = np.dot(MDC, signal)
    recon = CS_IRLS(com.reshape(m, 1), MDC, SEG_LEN).flatten()
    sndr = compute_sndr(recon.reshape(-1, 1), signal.reshape(-1, 1))
    actual_cr = 1 - m / SEG_LEN
    elapsed = time.time() - t0
    return sndr, actual_cr, elapsed, m, sigma

# ====== Load data ======
import scipy.io as sio

# Single-channel: C_Easy1_noise005
print("Loading C_Easy1_noise005...")
mat = sio.loadmat(os.path.join(DATABASE_DIR, 'C_Easy1_noise005.mat'))
raw = mat['data'].flatten()
st = mat['spike_times'][0, 0].flatten().astype(int) - 1
sc_signals = []
for t in st:
    if t - WINDOW_SIZE >= 0 and t + WINDOW_SIZE < len(raw):
        sc_signals.append(raw[t - WINDOW_SIZE:t + WINDOW_SIZE].astype(np.float64))
print(f"  {len(sc_signals)} spikes")

# Multi-channel: 3000 spikes from best_channel_data
print("Loading multi-channel data (3000 spikes)...")
mc_data = np.load(os.path.join(SAVE_DIR, 'best_channel_data.npz'), allow_pickle=True)
mc_signals = mc_data['orig_best'][:3000].astype(np.float64)
print(f"  {len(mc_signals)} spikes")

# ====== Test ======
results = {}
for label, signals in [('SC', sc_signals), ('MC', mc_signals)]:
    print(f"\n{'='*60}")
    print(f"Testing {label} ({len(signals)} spikes)")
    print(f"{'='*60}")
    m1_sndr, m1_cr, m1_time = [], [], []
    m2_sndr, m2_cr, m2_time, m2_m, m2_sigma = [], [], [], [], []
    
    for i, sig in enumerate(signals):
        # Method 1
        s1, c1, t1, m1 = method_current(sig)
        m1_sndr.append(s1); m1_cr.append(c1); m1_time.append(t1)
        
        # Method 2
        s2, c2, t2, m2, sg2 = method_adaptive(sig)
        m2_sndr.append(s2); m2_cr.append(c2); m2_time.append(t2)
        m2_m.append(m2); m2_sigma.append(sg2)
        
        if (i+1) % 500 == 0:
            print(f"  ... {i+1}/{len(signals)}")
    
    results[label] = {
        'm1': {'sndr': m1_sndr, 'cr': m1_cr, 'time': m1_time},
        'm2': {'sndr': m2_sndr, 'cr': m2_cr, 'time': m2_time, 'm': m2_m, 'sigma': m2_sigma},
    }
    
    m1_s = np.array(m1_sndr); m2_s = np.array(m2_sndr)
    m1_c = np.array(m1_cr); m2_c = np.array(m2_cr)
    
    # Apply <50dB filter
    m1_s_f = m1_s[(m1_s < 50) & np.isfinite(m1_s)]
    m2_s_f = m2_s[(m2_s < 50) & np.isfinite(m2_s)]
    m1_c_f = m1_c[(m1_s < 50) & np.isfinite(m1_s)]
    m2_c_f = m2_c[(m2_s < 50) & np.isfinite(m2_s)]
    
    def print_row(label2, m1_val, m2_val):
        print(f"  {label2:<20} {m1_val:<20} {m2_val:<20}")
    
    print(f"\n  Results for {label} (SNDR < 50dB filtered):")
    print(f"  {'Metric':<20} {'Method 1 (Current)':<20} {'Method 2 (Adaptive)':<20}")
    print(f"  {'-'*60}")
    print_row('Mean SNDR (dB)', f'{np.mean(m1_s_f):.2f}', f'{np.mean(m2_s_f):.2f}')
    print_row('Median SNDR (dB)', f'{np.median(m1_s_f):.2f}', f'{np.median(m2_s_f):.2f}')
    print_row('Max SNDR (dB)', f'{np.max(m1_s_f):.2f}', f'{np.max(m2_s_f):.2f}')
    print_row('Min SNDR (dB)', f'{np.min(m1_s_f):.2f}', f'{np.min(m2_s_f):.2f}')
    print_row('Std SNDR (dB)', f'{np.std(m1_s_f):.2f}', f'{np.std(m2_s_f):.2f}')
    print_row('Mean CR (%)', f'{np.mean(m1_c_f)*100:.2f}', f'{np.mean(m2_c_f)*100:.2f}')
    print_row('Max CR (%)', f'{np.max(m1_c_f)*100:.2f}', f'{np.max(m2_c_f)*100:.2f}')
    print_row('Min CR (%)', f'{np.min(m1_c_f)*100:.2f}', f'{np.min(m2_c_f)*100:.2f}')
    print_row('Spikes kept', f'{len(m1_s_f)}/{len(m1_s)}', f'{len(m2_s_f)}/{len(m2_s)}')
    
    if label == 'MC':
        print(f"\n  Method 2 sigma stats: mean={np.mean(m2_sigma):.4f}, "
              f"min={np.min(m2_sigma):.4f}, max={np.max(m2_sigma):.4f}")

print(f"\n{'='*60}")
print("Done!")
