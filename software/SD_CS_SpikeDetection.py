#!/usr/bin/env python
"""Verify spike detection accuracy for single-channel and multi-channel data."""
import numpy as np, os, warnings, scipy.io as sio
from scipy.signal import butter, filtfilt
from collections import defaultdict
warnings.filterwarnings('ignore')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, 'saved_results')
DB_DIR = os.environ.get('SPIKE_DATA_DIR', os.path.join(BASE_DIR, '..', 'data'))
SEARCH_WIN = 50    # 尖峰窗口半宽
TOL_MATCH = 30     # 匹配容差
MERGE_DIST = 50   # peak合并距离（覆盖99.9%+的同尖峰波谷-反弹峰间隙）
GUARD = 100       # 尖峰前后排除区域
FS = 30000        # Neuropixel采样率 30kHz
BP_LOW = 300      # 带通下限 Hz
BP_HIGH = 3000    # 带通上限 Hz

def _design_bpf():
    """设计零相位带通滤波器 (b, a 格式)。"""
    b, a = butter(4, [BP_LOW/FS*2, BP_HIGH/FS*2], btype='band')
    return b, a

_BPF_BA = None

def smooth_signal(signal):
    """零相位带通滤波（300-5000Hz）降噪。"""
    global _BPF_BA
    if _BPF_BA is None:
        _BPF_BA = _design_bpf()
    b, a = _BPF_BA
    return filtfilt(b, a, signal)

def compute_noise_max(signal, gt_times):
    """计算远离所有GT尖峰的噪声区p99.9绝对幅值（排除极端离群值）。"""
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
    """检测局部极值点：|signal| > threshold 且中间点大于两边。"""
    peaks = []
    for i in range(1, len(signal)-1):
        if abs(signal[i]) > threshold:
            if (signal[i] > signal[i-1] and signal[i] > signal[i+1]) or \
               (signal[i] < signal[i-1] and signal[i] < signal[i+1]):
                peaks.append(i)
    return np.array(peaks, dtype=int)

def merge_peaks(peaks, signal):
    """合并<100点内的peak，保留幅值最大的。"""
    if len(peaks) == 0:
        return np.array([], dtype=int)
    merged = []
    i = 0
    while i < len(peaks):
        cluster = [peaks[i]]
        j = i + 1
        while j < len(peaks) and peaks[j] - peaks[j-1] < MERGE_DIST:
            cluster.append(peaks[j])
            j += 1
        best = cluster[np.argmax([np.abs(signal[t]) for t in cluster])]
        merged.append(best)
        i = j
    return np.array(merged)

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

def evaluate_detection_neighborhood(detected, gt_self, gt_by_ch, ch, neighbor_radius=NEIGHBOR_RADIUS):
    """Evaluate detection with neighborhood matching.
    - TP: detection matches any GT in neighborhood [ch-R, ch+R]
    - FP: detection matches no GT in neighborhood (real false positive)
    - FN: self-GT spikes not matched by any detection
    """
    # Build ordered list of all (gt_time, gt_channel) in neighborhood
    all_nb_gt = []
    for nch in range(max(0, ch - neighbor_radius), ch + neighbor_radius + 1):
        if nch in gt_by_ch:
            for t in gt_by_ch[nch]:
                all_nb_gt.append((int(t), nch))

    if len(all_nb_gt) == 0:
        fp, fn = len(detected), len(gt_self)
        return 0, fp, fn, 0.0, 0.0, 0.0

    # Match detections to neighborhood GT (greedy, each GT used at most once)
    used_gt = set()
    tp = 0
    for d in detected:
        for gi, (gt, _) in enumerate(all_nb_gt):
            if gi not in used_gt and abs(gt - d) <= TOL_MATCH:
                tp += 1
                used_gt.add(gi)
                break

    fp = len(detected) - tp

    # Count how many self-GT were matched (for FN / recall)
    self_gt_set = set(int(t) for t in gt_self)
    self_matched = 0
    for gi in used_gt:
        gt_time, gt_ch = all_nb_gt[gi]
        if gt_ch == ch and gt_time in self_gt_set:
            self_matched += 1

    fn = len(gt_self) - self_matched

    prec = tp / (tp + fp) * 100 if (tp + fp) else 0.0
    rec = self_matched / len(gt_self) * 100 if len(gt_self) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    return tp, fp, fn, prec, rec, f1

def set_threshold(min_gt, max_noise, margin_big=0.002, margin_small=0.001):
    """阈值：基于max_noise加固定余量。
    margin_big/margin_small 根据数据单位设置：
    - 单通道(V级): 0.002/0.001
    - 多通道(mV级): 2.0/1.0
    """
    gap = min_gt - max_noise
    if gap > margin_big:
        th = max_noise + margin_big
    elif gap > margin_small:
        th = max_noise + margin_small
    else:
        th = max_noise
    return th, gap

# 设为True使用 min_gt 作为阈值（直接等于最小peak）
USE_MIN_GT_TH = False
SKIP_SINGLE_CHANNEL = True  # 设为True跳过单通道部分（已确认旧阈值正确）

if not SKIP_SINGLE_CHANNEL:
    # ====== Single-Channel ======
    print("=" * 70)
    print("Single-Channel Spike Detection Accuracy")
    print("=" * 70)
    dataset_names = [
        'C_Easy1_noise005','C_Easy1_noise01','C_Easy1_noise015','C_Easy1_noise02',
        'C_Easy1_noise025','C_Easy1_noise03','C_Easy1_noise035','C_Easy1_noise04',
        'C_Easy2_noise005','C_Easy2_noise01','C_Easy2_noise015','C_Easy2_noise02',
        'C_Difficult1_noise005','C_Difficult1_noise01','C_Difficult1_noise015','C_Difficult1_noise02',
        'C_Difficult2_noise005','C_Difficult2_noise01','C_Difficult2_noise015','C_Difficult2_noise02'
    ]
    print(f"{'Name':<30} {'GT':>4} {'Det':>5} {'TP':>4} {'FP':>4} {'FN':>4}"
          f" {'Prec':>6} {'Rec':>6} {'F1':>6}  {'minG':>7}  {'th':>7}")
    sc_results = []
    for name in dataset_names:
        mat = sio.loadmat(os.path.join(DB_DIR, f'{name}.mat'))
        signal = mat['data'].flatten().astype(np.float64)
        gt_times = mat['spike_times'][0,0].flatten().astype(int) - 1

        gt_true_pos, gt_true_amp = compute_gt_true_peaks(signal, gt_times)
        min_gt = np.min(gt_true_amp)
        if USE_MIN_GT_TH:
            th = min_gt
            max_noise = 0
            gap = 0
        else:
            max_noise = compute_noise_max(signal, gt_times)
            th, gap = set_threshold(min_gt, max_noise)

        peaks = detect_peaks(signal, th)
        peaks = merge_peaks(peaks, signal)
        tp, fp, fn, prec, rec, f1 = evaluate_detection(peaks, gt_true_pos)

        sc_results.append((name, prec, rec, f1, len(gt_times), len(peaks), tp, fp, fn))
        print(f"  {name:<28} {len(gt_times):>4} {len(peaks):>5} {tp:>4} {fp:>4} {fn:>4}"
              f" {prec:>5.1f}% {rec:>5.1f}% {f1:>5.1f}%"
              f"  minG={min_gt:.4f} th={th:.4f}")

    # Group summary
    groups = {'E1':[], 'E2':[], 'D1':[], 'D2':[]}
    for name, prec, rec, f1, ng, nd, tp, fp, fn in sc_results:
        g = name.split('_')[1]
        if g == 'Easy1': groups['E1'].append(f1)
        elif g == 'Easy2': groups['E2'].append(f1)
        elif g == 'Difficult1': groups['D1'].append(f1)
        elif g == 'Difficult2': groups['D2'].append(f1)
    print(f"\n  Group average F1:")
    for g, vals in groups.items():
        print(f"    {g}: {np.mean(vals):.1f}%" if vals else f"    {g}: N/A")

# ====== Multi-Channel (with neighborhood matching) ======
NEIGHBOR_RADIUS = 8  # 每个spike扩散约16个通道，±8

def build_expected_spikes(gt_by_ch, n_radius=NEIGHBOR_RADIUS):
    """构建每个通道的期望尖峰集：考虑邻域扩散。
    若spike的best_ch=Y，则通道X（|X-Y|≤n_radius）也期望检测到该spike。
    """
    expected = defaultdict(set)
    for ch, times in gt_by_ch.items():
        for t in times:
            lo = max(0, ch - n_radius)
            hi = ch + n_radius + 1
            for nch in range(lo, hi):
                expected[nch].add(t)
    return expected

print("\n" + "=" * 70)
print("Multi-Channel Spike Detection Accuracy (neighborhood matching)")
print("=" * 70)
# 用mmap加载仅读取需要的数据，避免OOM
raw = np.load(os.path.join(BASE_DIR, 'transfer', 'dataSample_BPF_300_5000.npy'), mmap_mode='r')
bcd = np.load(os.path.join(SAVE_DIR, 'best_channel_data.npz'))
best_ch_phys = bcd['best_ch_phys']
best_spike_times = bcd['spike_times']

# Only 1 second of data (30000 samples at 30kHz)
T_1S = 30000
raw_1s = np.array(raw[:, :T_1S])  # load only 1s into memory

# Group GT spikes by best channel (only those in first 1s)
ch_to_gt = defaultdict(list)
for i in range(len(best_spike_times)):
    t = int(best_spike_times[i])
    if t < T_1S:
        ch_to_gt[int(best_ch_phys[i])].append(t)

all_chs_with_gt = sorted(ch_to_gt.keys())
# Sample up to 20 channels with most spikes
top_chs = sorted(all_chs_with_gt, key=lambda ch: -len(ch_to_gt[ch]))[:20]

mc_results_nb = []   # 邻域匹配
mc_results_self = [] # 仅自GT匹配（baseline对比）
print(f"{'Ch':>4} {'Self':>5} {'Det':>5} {'TP':>4} {'FP':>4} {'FN':>4}"
      f" {'Prec':>6} {'Rec':>6} {'F1':>6}  {'minG':>7}  {'th':>7}")
for ch in top_chs:
    signal = raw_1s[ch].astype(np.float64)
    signal_sm = smooth_signal(signal)

    # 阈值：仅用自GT的peak幅值
    gt_on_ch = np.array(sorted(ch_to_gt[ch]))
    if len(gt_on_ch) == 0:
        continue
    max_noise = compute_noise_max(signal_sm, gt_on_ch)
    gt_true_pos_self, gt_true_amp_self = compute_gt_true_peaks(signal_sm, gt_on_ch)
    min_gt = np.min(gt_true_amp_self)
    # 多通道(mV级)：margin使用2.0/1.0 mV
    th, gap = set_threshold(min_gt, max_noise, margin_big=2.0, margin_small=1.0)

    peaks = detect_peaks(signal_sm, th)
    peaks = merge_peaks(peaks, signal_sm)

    # 新方法：邻域匹配（TP=匹配任何邻域GT, FN=仅自GT漏检）
    tp, fp, fn, prec, rec, f1 = evaluate_detection_neighborhood(
        peaks, gt_on_ch, ch_to_gt, ch)

    # Baseline：仅自GT匹配
    tp_self, fp_self, fn_self, prec_self, rec_self, f1_self = evaluate_detection(
        peaks, gt_true_pos_self)

    mc_results_nb.append((ch, prec, rec, f1, len(gt_on_ch), len(peaks), tp, fp, fn))
    mc_results_self.append((ch, prec_self, rec_self, f1_self, len(gt_on_ch), len(peaks)))
    print(f"  Ch{ch:>3} {len(gt_on_ch):>5} {len(peaks):>5} {tp:>4} {fp:>4} {fn:>4}"
          f" {prec:>5.1f}% {rec:>5.1f}% {f1:>5.1f}%"
          f"  minG={min_gt:.4f} th={th:.4f}")

if mc_results_nb:
    f1s_nb = [r[3] for r in mc_results_nb]
    f1s_self = [r[3] for r in mc_results_self]
    print(f"\n  === 对比 ===")
    print(f"  自GT匹配（baseline）: Avg F1 = {np.mean(f1s_self):.1f}%")
    print(f"  邻域匹配（TP含邻域spillover, FN仅自GT）: Avg F1 = {np.mean(f1s_nb):.1f}%")
print("\nDone!")
