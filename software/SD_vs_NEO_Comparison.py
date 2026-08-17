#!/usr/bin/env python
"""比较 SD (SpikeDetection.py的方法) 与 NEO (Non-linear Energy Operator) 尖峰检测方法"""
import numpy as np, os, warnings, scipy.io as sio
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.environ.get('SPIKE_DATA_DIR', os.path.join(BASE_DIR, '..', 'data'))
SAVE_DIR = os.path.join(BASE_DIR, 'saved_results')
os.makedirs(SAVE_DIR, exist_ok=True)

# ====== Plot-Only Mode ======
PLOT_ONLY = True  # True=只画图, False=重新计算
# ============================

# 与 SD_CS_SpikeDetection.py 完全一致的参数
SEARCH_WIN = 50
TOL_MATCH = 30
MERGE_DIST = 50
GUARD = 100

dataset_names = [
    'C_Easy1_noise005','C_Easy1_noise01','C_Easy1_noise015','C_Easy1_noise02',
    'C_Easy1_noise025','C_Easy1_noise03','C_Easy1_noise035','C_Easy1_noise04',
    'C_Easy2_noise005','C_Easy2_noise01','C_Easy2_noise015','C_Easy2_noise02',
    'C_Difficult1_noise005','C_Difficult1_noise01','C_Difficult1_noise015','C_Difficult1_noise02',
    'C_Difficult2_noise005','C_Difficult2_noise01','C_Difficult2_noise015','C_Difficult2_noise02'
]

# ==================== SD方法 (与SD_CS_SpikeDetection.py完全一致) ====================

def compute_noise_max(signal, gt_times):
    """计算远离所有GT尖峰的噪声区p99.9绝对幅值。"""
    mask = np.ones(len(signal), dtype=bool)
    for t in gt_times:
        lo, hi = max(0, t - GUARD), min(len(signal), t + GUARD)
        if lo < hi:
            mask[lo:hi] = False
    noise = np.abs(signal[mask])
    if len(noise) == 0:
        return 0
    return float(np.percentile(noise, 99.9))

def compute_gt_true_peaks(signal, gt_times):
    """对每个GT，在±SEARCH_WIN内找真正peak的位置和幅值。"""
    pos, amp = [], []
    for t in gt_times:
        lo = max(0, t - SEARCH_WIN)
        hi = min(len(signal), t + SEARCH_WIN + 1)
        seg = signal[lo:hi]
        idx = np.argmax(np.abs(seg))
        pos.append(lo + idx)
        amp.append(np.abs(seg[idx]))
    return np.array(pos), np.array(amp)

def detect_peaks(signal, threshold):
    """检测局部极值点：|signal| > threshold 且中间点大于两边（向量化加速）。"""
    above = np.abs(signal) > threshold
    # 局部极大值或极小值
    local_max = (signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:])
    local_min = (signal[1:-1] < signal[:-2]) & (signal[1:-1] < signal[2:])
    peaks = np.where(above[1:-1] & (local_max | local_min))[0] + 1
    return peaks.astype(int)

def merge_peaks(peaks, signal):
    """合并<MERGE_DIST点内的peak，保留幅值最大的（快速实现）。"""
    if len(peaks) == 0:
        return np.array([], dtype=int)
    # 用diff找分组边界
    gaps = np.diff(peaks)
    split_points = np.where(gaps >= MERGE_DIST)[0] + 1
    groups = np.split(peaks, split_points)
    merged = np.array([g[np.argmax(np.abs(signal[g]))] for g in groups])
    return merged.astype(int)

def evaluate_detection(detected, gt_true_pos):
    """匹配检测到的peak和GT真实peak位置。"""
    used = set()
    tp = 0
    for d in detected:
        for gi, gp in enumerate(gt_true_pos):
            if gi not in used and abs(gp - d) <= TOL_MATCH:
                tp += 1
                used.add(gi)
                break
    fp = len(detected) - tp
    fn = len(gt_true_pos) - tp
    prec = tp/(tp+fp)*100 if tp+fp else 0
    rec = tp/(tp+fn)*100 if tp+fn else 0
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
    return tp, fp, fn, prec, rec, f1

def set_threshold(min_gt, max_noise, margin_big=0.002, margin_small=0.001):
    """阈值：基于max_noise加固定余量。"""
    gap = min_gt - max_noise
    if gap > margin_big:
        th = max_noise + margin_big
    elif gap > margin_small:
        th = max_noise + margin_small
    else:
        th = max_noise
    return th, gap

def detect_sd(signal, gt_times):
    """SD方法：与SD_CS_SpikeDetection.py单通道部分完全一致的检测流程。"""
    gt_true_pos, gt_true_amp = compute_gt_true_peaks(signal, gt_times)
    min_gt = np.min(gt_true_amp)
    max_noise = compute_noise_max(signal, gt_times)
    th, gap = set_threshold(min_gt, max_noise, margin_big=0.002, margin_small=0.001)
    peaks = detect_peaks(signal, th)
    peaks = merge_peaks(peaks, signal)
    return peaks, th, gt_true_pos

# ==================== NEO 方法 ====================
def compute_neo(signal):
    """Non-linear Energy Operator: ψ[n] = x[n]² - x[n-1]·x[n+1]"""
    x = np.asarray(signal, dtype=np.float64)
    psi = np.zeros_like(x)
    psi[1:-1] = x[1:-1]**2 - x[:-2] * x[2:]
    psi[0] = 0
    psi[-1] = 0
    return psi

def detect_neo(signal, k_neo=2.0):
    """
    NEO-based spike detection.
    1. Compute Teager energy ψ[n]
    2. th = mean(ψ_est) + k * std(ψ_est)
      其中 ψ_est 取信号首尾5%段（假设无spike）
    """
    psi = compute_neo(signal)
    n = len(psi)
    n_noise = n // 20  # 5%
    psi_noise = np.concatenate([psi[:n_noise], psi[-n_noise:]])
    psi_pos = psi_noise[psi_noise > 0]
    if len(psi_pos) < 10:
        psi_pos = psi_noise
    neo_mean = np.mean(psi_pos)
    neo_std = np.std(psi_pos)
    threshold = neo_mean + k_neo * neo_std
    
    # 在NEO域检测峰值
    peaks = detect_peaks(psi, threshold)
    peaks = merge_peaks(peaks, signal)  # 用原始signal幅值合并
    return peaks, threshold

def detect_neo_hybrid(signal, k_neo=2.0):
    """
    NEO + 幅值验证: NEO检测候选位置后，用原始信号幅值过滤噪声。
    """
    psi = compute_neo(signal)
    n = len(psi)
    n_noise = n // 20
    psi_noise = np.concatenate([psi[:n_noise], psi[-n_noise:]])
    psi_pos = psi_noise[psi_noise > 0]
    if len(psi_pos) < 10:
        psi_pos = psi_noise
    neo_mean = np.mean(psi_pos)
    neo_std = np.std(psi_pos)
    threshold = neo_mean + k_neo * neo_std
    
    neo_peaks = detect_peaks(psi, threshold)
    neo_peaks = merge_peaks(neo_peaks, signal)
    
    # 过滤：保留幅值 ≥ 中位数30%的peak
    if len(neo_peaks) > 0:
        signal_amp = np.abs(signal[neo_peaks])
        amp_th = np.median(signal_amp) * 0.3
        neo_peaks = neo_peaks[signal_amp >= amp_th]
    
    return neo_peaks, threshold

# ==================== 主循环 ====================
RESULTS_PATH = os.path.join(SAVE_DIR, 'sd_vs_neo_results.npz')
K_NEO = 1.0
K_HYB = 1.0

if PLOT_ONLY:
    # 从保存的结果加载
    data = np.load(RESULTS_PATH)
    results = []
    for i in range(len(data['names'])):
        results.append((data['names'][i], data['gt_counts'][i],
                        data['sd_f1'][i], data['sd_prec'][i], data['sd_rec'][i],
                        data['neo_f1'][i], data['neo_prec'][i], data['neo_rec'][i],
                        data['hyb_f1'][i], data['hyb_prec'][i], data['hyb_rec'][i]))
    print(f"Loaded {len(results)} results from npz")
else:
    print("=" * 100)
    print(f"{'Dataset':<28} {'GT':>4} {'SD_F1':>7} {'SD_Prec':>9} {'SD_Rec':>8}"
          f" {'NEO_F1':>7} {'NEO_Prec':>9} {'NEO_Rec':>8}"
          f" {'Hyb_F1':>7} {'Hyb_Prec':>9} {'Hyb_Rec':>8}")
    print("=" * 100)

    print(f"  Using fixed k: NEO(k={K_NEO:.0f}), Hybrid(k={K_HYB:.0f})")

    results = []
    for name in dataset_names:
        mat = sio.loadmat(os.path.join(DB_DIR, f'{name}.mat'))
        signal = mat['data'].flatten().astype(np.float64)
        gt_times = mat['spike_times'][0,0].flatten().astype(int) - 1
        gt_true_pos, _ = compute_gt_true_peaks(signal, gt_times)
        
        peaks_sd, th_sd, _ = detect_sd(signal, gt_times)
        tp_s, fp_s, fn_s, prec_s, rec_s, f1_s = evaluate_detection(peaks_sd, gt_true_pos)
        
        peaks_neo, th_neo = detect_neo(signal, K_NEO)
        tp_n, fp_n, fn_n, prec_n, rec_n, f1_n = evaluate_detection(peaks_neo, gt_true_pos)
        
        peaks_hyb, th_hyb = detect_neo_hybrid(signal, K_HYB)
        tp_h, fp_h, fn_h, prec_h, rec_h, f1_h = evaluate_detection(peaks_hyb, gt_true_pos)
        
        results.append((name, len(gt_times), f1_s, prec_s, rec_s, f1_n, prec_n, rec_n,
                        f1_h, prec_h, rec_h))
        
        print(f"  {name:<28} {len(gt_times):>4}"
              f" {f1_s:>6.1f}% {prec_s:>8.1f}% {rec_s:>7.1f}%"
              f" {f1_n:>6.1f}% {prec_n:>8.1f}% {rec_n:>7.1f}%"
              f" {f1_h:>6.1f}% {prec_h:>8.1f}% {rec_h:>7.1f}%")
    
    np.savez(os.path.join(SAVE_DIR, 'sd_vs_neo_results.npz'),
             names=np.array([r[0] for r in results]),
             gt_counts=np.array([r[1] for r in results]),
             sd_f1=np.array([r[2] for r in results]),
             sd_prec=np.array([r[3] for r in results]),
             sd_rec=np.array([r[4] for r in results]),
             neo_f1=np.array([r[5] for r in results]),
             neo_prec=np.array([r[6] for r in results]),
             neo_rec=np.array([r[7] for r in results]),
             hyb_f1=np.array([r[8] for r in results]),
             hyb_prec=np.array([r[9] for r in results]),
             hyb_rec=np.array([r[10] for r in results]))
    print(f"Results saved to sd_vs_neo_results.npz")
    
# ====== 分组汇总 ======
print("\n" + "=" * 100)
print("Group Average:")
print("=" * 100)
groups = {'E1':[], 'E2':[], 'D1':[], 'D2':[]}
for r in results:
    g = r[0].split('_')[1]
    key = {'Easy1':'E1','Easy2':'E2','Difficult1':'D1','Difficult2':'D2'}[g]
    groups[key].append(r)

for g in ['E1','E2','D1','D2']:
    if not groups[g]: continue
    vals = groups[g]
    f1s_s = np.mean([v[2] for v in vals])
    f1s_n = np.mean([v[5] for v in vals])
    f1s_h = np.mean([v[8] for v in vals])
    print(f"  {g}: SD F1={f1s_s:.1f}% | NEO F1={f1s_n:.1f}% | Hybrid F1={f1s_h:.1f}%")

f1s_s = np.mean([r[2] for r in results])
f1s_n = np.mean([r[5] for r in results])
f1s_h = np.mean([r[8] for r in results])
print(f"\n  Overall: SD F1={f1s_s:.1f}% | NEO F1={f1s_n:.1f}% | Hybrid F1={f1s_h:.1f}%")
print(f"  Optimal k: NEO(k={K_NEO:.0f}), Hybrid(k={K_HYB:.0f})")
print("\nCalculation done!")

# ==================== 绘图部分 ====================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

FS_TITLE = 9
FS_LABEL = 7
FS_LEGEND = 6
FS_TICK = 5.5
FS_ANNOT = 5.5
FS_VAL = 4.5

box_color = '#7A9DCA'   # 统一颜色

sd_f1_vals = np.array([r[2] for r in results])
neo_f1_vals = np.array([r[5] for r in results])

# 硬件开销定义（每样本运算次数）
categories = ['Abs', 'Multiply', 'Add/Sub', 'Compare']
sd_ops = [1, 0, 0, 3]   # SD: 1 abs + 3 comparisons
neo_ops = [1, 2, 1, 3]   # NEO: 1 abs + 2 mult + 1 add/sub + 3 comparisons
cat_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']  # 彩色

width = 3.5
fig, axes = plt.subplots(1, 2, figsize=(width, width * 0.45), dpi=300)
fig.subplots_adjust(left=0.11, right=0.97, bottom=0.16, top=0.85, wspace=0.35)

# ===== (a) F1 Box Plot =====
ax = axes[0]
bp = ax.boxplot([sd_f1_vals, neo_f1_vals], positions=[0, 1], widths=0.4,
                patch_artist=True, showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='white',
                               markeredgecolor='black', markersize=4),
                medianprops=dict(color='white', linewidth=1.2),
                boxprops=dict(alpha=0.7, edgecolor='black', linewidth=0.5),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                flierprops=dict(marker='o', markersize=2.5, alpha=0.5))

# 设置统一颜色
bp['boxes'][0].set_facecolor(box_color)
bp['boxes'][1].set_facecolor(box_color)
for i in range(2):
    bp['whiskers'][i*2].set_color(box_color)
    bp['whiskers'][i*2+1].set_color(box_color)
    bp['caps'][i*2].set_color(box_color)
    bp['caps'][i*2+1].set_color(box_color)
    bp['fliers'][i].set_markerfacecolor(box_color)
    bp['fliers'][i].set_markeredgecolor(box_color)

ax.set_xticks([0, 1])
ax.set_xticklabels(['STD', 'NEO'], fontsize=FS_TICK)
ax.set_ylabel('Accuracy', fontsize=FS_LABEL)
ax.set_ylim(30, 100)
ax.tick_params(labelsize=FS_TICK)

# 标注均值（右侧0.2处，水平对齐）
for i, data in enumerate([sd_f1_vals, neo_f1_vals]):
    mean_val = np.mean(data)
    ax.text(i + 0.2, mean_val, f'{mean_val:.1f}%', fontsize=FS_ANNOT,
            color=box_color, va='center', fontweight='bold')

ax.set_title('(a) Detection Accuracy', fontsize=FS_TITLE, fontweight='bold')

# ===== (b) Stacked Bar: Hardware Complexity =====
ax2 = axes[1]
x_pos = [0, 1]
bar_width = 0.35

for idx, (ops, label) in enumerate([(sd_ops, 'SD'), (neo_ops, 'NEO')]):
    bottom = 0
    for cat_idx in range(len(categories)):
        val = ops[cat_idx]
        if val > 0:
            ax2.bar(x_pos[idx], val, bar_width, bottom=bottom,
                color=cat_colors[cat_idx], edgecolor='black', linewidth=0.3)
            ax2.text(x_pos[idx], bottom + val/2, f'{val}',
                     ha='center', va='center', fontsize=FS_ANNOT, color='white' if cat_idx >= 2 else 'black',
                     fontweight='bold')
            bottom += val

ax2.set_xticks(x_pos)
ax2.set_xticklabels(['STD', 'NEO'], fontsize=FS_TICK)
ax2.set_ylabel('Operations / Sample', fontsize=FS_LABEL)
ax2.set_ylim(0, 9)
ax2.tick_params(labelsize=FS_TICK)

# 图例（放在顶部中间）
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=cat_colors[i], label=categories[i]) for i in range(len(categories))]
ax2.legend(handles=legend_elements, fontsize=FS_LEGEND, loc='lower center',
           bbox_to_anchor=(0.5, 0.70), framealpha=0.8, edgecolor='gray', ncol=2)

ax2.set_title('(b) Hardware Complexity', fontsize=FS_TITLE, fontweight='bold')

fig.savefig(os.path.join(BASE_DIR, 'SD_vs_NEO_Combined.png'), dpi=300, bbox_inches='tight')
print("Saved: SD_vs_NEO_Combined.png")
print("\nAll plots done!")
