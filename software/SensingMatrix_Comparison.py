#!/usr/bin/env python
"""比较6种感知矩阵的CS重构SNDR (单通道, IRLS重构)"""
import numpy as np, os, warnings, math, scipy.io as sio
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, 'saved_results')
DB_DIR = os.environ.get('SPIKE_DATA_DIR', os.path.join(BASE_DIR, '..', 'data'))
WINDOW_SIZE = 50
SEG_LEN = 2 * WINDOW_SIZE
CR_TARGET = 0.84
TARGET_M = int(SEG_LEN * (1 - CR_TARGET))  # 16
P_SIGMA = 0.4

# ========== 工具函数 ==========
def norm2(x): return np.linalg.norm(x, 2)

def compute_sndr(recon, orig):
    err = orig - recon
    return 20 * math.log(norm2(orig) / norm2(err), 10)

def CS_IRLS(y, T_Mat, max_iter=100):
    """IRLS重构 (原始版本，对MDC有效)。"""
    hat_x = T_Mat.T @ y
    epsilong = 1
    for _ in range(max_iter):
        w = 1.0 / (np.abs(hat_x) + epsilong)
        W = np.diag(w.flatten())
        A = W @ T_Mat.T
        B = T_Mat @ A
        x_new = A @ np.linalg.pinv(B) @ y
        if norm2(x_new - hat_x) < 1e-10:
            break
        if norm2(x_new - hat_x) < math.sqrt(epsilong) / 100:
            epsilong /= 10
        hat_x = x_new
    return hat_x

def reconstruct(signal, sensing_mat):
    """通用CS重构"""
    m = sensing_mat.shape[0]
    com = sensing_mat @ signal
    recon = CS_IRLS(com.reshape(m, 1), sensing_mat).flatten()
    return recon

# ========== 稀疏基函数 ==========

def dct_matrix(N):
    """DCT-II矩阵 (N×N)，正变换：θ = D·x，综合：x = Dᵀ·θ"""
    D = np.zeros((N, N))
    for k in range(N):
        for n in range(N):
            D[k, n] = math.cos(math.pi / N * (n + 0.5) * k)
        D[k, :] *= math.sqrt(2 / N) if k > 0 else math.sqrt(1 / N)
    return D  # DCT矩阵: θ = D @ x

def gabor_wavelet_matrix(N):
    """Gabor Wavelet字典 (N×N)，每个原子是高斯窗函数调制的正弦波。
    频率从低到高均匀分布，加上不同时移。"""
    Psi = np.zeros((N, N))
    n_freq = 10  # 频率数
    n_shift = 10  # 时移数
    col = 0
    sigma = N / (2 * n_freq)  # 高斯窗宽
    for f_idx in range(n_freq):
        freq = (f_idx + 1) * 0.5 / n_freq  # 归一化频率
        for s_idx in range(n_shift):
            shift = int(s_idx * N / n_shift)
            if col >= N: break
            for n in range(N):
                t = n - shift
                window = math.exp(-0.5 * (t / sigma) ** 2)
                Psi[col, n] = window * math.cos(2 * math.pi * freq * t)
            # 归一化原子
            norm = np.linalg.norm(Psi[col, :])
            if norm > 0:
                Psi[col, :] /= norm
            col += 1
    # 补齐剩余列（若不足N列，用噪声填充）
    while col < N:
        Psi[col, :] = np.random.randn(N)
        Psi[col, :] /= np.linalg.norm(Psi[col, :])
        col += 1
    return Psi  # 综合矩阵: x = Ψᵀ·θ (每行是一个原子)

def reconstruct_with_basis(signal, sensing_mat, basis):
    """CS重构 with sparsity basis: y = Φ·Ψ·θ, recover θ, then x = Ψ·θ"""
    Phi_eff = sensing_mat @ basis.T  # 有效感知矩阵 = Φ @ Ψ (Ψ为综合矩阵)
    m = Phi_eff.shape[0]
    com = sensing_mat @ signal
    theta_hat = CS_IRLS(com.reshape(m, 1), Phi_eff).flatten()
    recon = basis.T @ theta_hat  # x = Ψ·θ
    return recon

# ========== 6种感知矩阵生成方法 ==========

# ----- Method 1: Current MDC (Distance + MDC_UMDC_Gen) -----
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

def gen_mdc_current(signal):
    sigma = Distance(signal, CR_TARGET, P_SIGMA)
    MDC = MDC_UMDC_Gen(signal, sigma)
    recon = reconstruct(signal, MDC)
    sndr = compute_sndr(recon, signal)
    actual_cr = 1 - MDC.shape[0] / SEG_LEN
    return sndr, actual_cr

# ----- Method 2: Adaptive MDC -----
def gen_mdc_adaptive(signal, target_m=TARGET_M):
    sigma = 0.1
    for _ in range(100):
        MDC = MDC_UMDC_Gen(signal, sigma)
        m = MDC.shape[0]
        if m == target_m: break
        elif m > target_m: sigma += 0.05
        else: sigma -= 0.05
        if sigma <= 0: sigma = 0.01
    recon = reconstruct(signal, MDC)
    sndr = compute_sndr(recon, signal)
    actual_cr = 1 - MDC.shape[0] / SEG_LEN
    return sndr, actual_cr

# ----- Method 3: Zhao SBM (Sparse Binary Matrix) -----
def gen_zhao_sbm(N, M, d=5):
    """Zhao et al. 2018 Sparse Binary Matrix (列归一化)."""
    Phi = np.zeros((M, N), dtype=np.float64)
    for j in range(N):
        rows = np.random.choice(M, d, replace=False)
        Phi[rows, j] = 1.0
    Phi = Phi / np.linalg.norm(Phi, axis=0, keepdims=True)
    return Phi

# ----- Method 4: Zhao STM (Sparse Ternary Matrix) -----
def gen_zhao_stm(N, M, d=5):
    """Zhao et al. 2018 Sparse Ternary Matrix (列归一化)."""
    Phi = np.zeros((M, N), dtype=np.float64)
    for j in range(N):
        rows = np.random.choice(M, d, replace=False)
        vals = np.random.choice([-1.0, 1.0], d)
        Phi[rows, j] = vals
    Phi = Phi / np.linalg.norm(Phi, axis=0, keepdims=True)
    return Phi

# ----- Method 5: Binary Random Matrix -----
def gen_binary_random(N, M, p=0.5):
    Phi = np.random.binomial(1, p, (M, N)).astype(np.float64)
    Phi = Phi / np.linalg.norm(Phi, axis=0, keepdims=True)
    return Phi

# ----- Method 6: Bernoulli Random Matrix -----
def gen_bernoulli_random(N, M):
    Phi = np.random.choice([-1.0, 1.0], (M, N))
    Phi = Phi / np.linalg.norm(Phi, axis=0, keepdims=True)
    return Phi

# ========== 数据加载 ==========
print("Loading Quiroga datasets...")
dataset_names = [
    'C_Easy1_noise005','C_Easy1_noise01','C_Easy1_noise015','C_Easy1_noise02',
    'C_Easy1_noise025','C_Easy1_noise03','C_Easy1_noise035','C_Easy1_noise04',
    'C_Easy2_noise005','C_Easy2_noise01','C_Easy2_noise015','C_Easy2_noise02',
    'C_Difficult1_noise005','C_Difficult1_noise01','C_Difficult1_noise015','C_Difficult1_noise02',
    'C_Difficult2_noise005','C_Difficult2_noise01','C_Difficult2_noise015','C_Difficult2_noise02'
]

all_signals = []  # 所有spike segment
MAX_SPIKES = 30  # 限制总数
total = 0
for name in dataset_names:
    mat = sio.loadmat(os.path.join(DB_DIR, f'{name}.mat'))
    raw = mat['data'].flatten().astype(np.float64)
    st = mat['spike_times'][0, 0].flatten().astype(int) - 1
    for t in st:
        if total >= MAX_SPIKES: break
        if t - WINDOW_SIZE >= 0 and t + WINDOW_SIZE < len(raw):
            all_signals.append(raw[t - WINDOW_SIZE:t + WINDOW_SIZE])
            total += 1
    if total >= MAX_SPIKES: break

print(f"  Total spikes: {len(all_signals)}")

# ========== CR扫描对比 ==========
CR_VALUES = [0.60, 0.66, 0.72, 0.78, 0.84, 0.90]
N_spikes = len(all_signals)
PLOT_ONLY = False

method_names = [
    '1) Current MDC',
    '2) Adaptive MDC',
    '3) Zhao SBM',
    '4) Zhao STM',
    '5) Binary Random',
    '6) Bernoulli Random'
]

# 对每个方法、每个CR，收集SNDR
results = {name: {'cr': [], 'sndr_mean': [], 'sndr_std': []} for name in method_names}

if not PLOT_ONLY:
    print(f"\nSweeping CR over {N_spikes} spikes...")
    for cr in CR_VALUES:
        M_target = max(1, int(SEG_LEN * (1 - cr)))
        print(f"\nCR={cr:.2f} (M={M_target}):")
        
        # 预生成固定矩阵（随CR变化）
        np.random.seed(42)
        Phi_sbm = gen_zhao_sbm(SEG_LEN, M_target, d=min(5, M_target))
        Phi_stm = gen_zhao_stm(SEG_LEN, M_target, d=min(5, M_target))
        Phi_bin = gen_binary_random(SEG_LEN, M_target)
        Phi_bern = gen_bernoulli_random(SEG_LEN, M_target)
        
        sweep_phis = [(Phi_sbm, method_names[2]), (Phi_stm, method_names[3]),
                       (Phi_bin, method_names[4]), (Phi_bern, method_names[5])]
        
        cr_sndrs = {name: [] for name in method_names}
        
        for i, sig in enumerate(all_signals):
            # M1: Current MDC
            try:
                s1, c1 = gen_mdc_current(sig)
            except Exception:
                s1, c1 = np.nan, np.nan
            cr_sndrs[method_names[0]].append(s1)
            
            # M2: Adaptive MDC (target M)
            try:
                s2, c2 = gen_mdc_adaptive(sig, M_target)
            except Exception:
                s2, c2 = np.nan, np.nan
            cr_sndrs[method_names[1]].append(s2)
            
            # M3-6: 固定矩阵 + 时域IRLS
            for phi, name in sweep_phis:
                try:
                    recon = reconstruct(sig, phi)
                    s = compute_sndr(recon, sig)
                except Exception:
                    s = np.nan
                cr_sndrs[name].append(s)
            
            if (i+1) % 10 == 0:
                print(f"    spike {i+1}/{N_spikes}", flush=True)
        
        for name in method_names:
            s = np.array(cr_sndrs[name])
            finite = s[np.isfinite(s) & (s < 50)]
            if len(finite) > 0:
                results[name]['cr'].append(cr)
                results[name]['sndr_mean'].append(np.mean(finite))
                results[name]['sndr_std'].append(np.std(finite))
                print(f"  {name:<20}: mean={np.mean(finite):.1f}dB, std={np.std(finite):.1f}dB")
            else:
                print(f"  {name:<20}: no valid data")
    
    # 保存结果
    np.savez(os.path.join(SAVE_DIR, 'cr_sweep_results.npz'),
             cr_values=np.array(CR_VALUES),
             **{f'{k}_mean': np.array(v['sndr_mean']) for k, v in results.items() if len(v['sndr_mean']) > 0},
             **{f'{k}_cr': np.array(v['cr']) for k, v in results.items() if len(v['cr']) > 0})
    print(f"\nResults saved.")

else:
    # PLOT_ONLY: 从npz加载
    data = np.load(os.path.join(SAVE_DIR, 'cr_sweep_results.npz'))
    print("Loaded saved results")

# ========== 绘图 ==========
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

colors = ['#2E8B57', '#FF8C00', '#348ABD', '#E24A33', '#988ED5', '#404040']
markers = ['o', 's', '^', 'v', 'D', 'x']

fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=300)
fig.subplots_adjust(left=0.14, right=0.97, bottom=0.14, top=0.92)

for idx, name in enumerate(method_names):
    if len(results[name]['cr']) == 0:
        continue
    crs = results[name]['cr']
    means = results[name]['sndr_mean']
    stds = results[name]['sndr_std']
    ax.errorbar(crs, means, yerr=stds, label=name, color=colors[idx],
                marker=markers[idx], markersize=4, linewidth=1.0, capsize=2,
                markerfacecolor=colors[idx], markeredgecolor='black', markeredgewidth=0.3)

ax.set_xlabel('Compression Ratio (CR)', fontsize=8)
ax.set_ylabel('Mean SNDR (dB)', fontsize=8)
ax.set_title('SNDR vs CR (IRLS Reconstruction)', fontsize=9, fontweight='bold')
ax.legend(fontsize=5.5, loc='upper left', framealpha=0.8, edgecolor='gray')
ax.tick_params(labelsize=7)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0.45, 0.96)

fig.savefig(os.path.join(BASE_DIR, 'SNDR_vs_CR_Comparison.png'), dpi=300, bbox_inches='tight')
print("Saved: SNDR_vs_CR_Comparison.png")
print("\nDone!")
